// Convert the 16 installed AAS catalog skills to Zcode skill format, then verify the result.
//
// Deterministic gates (run as node commands, exit code decides):
//   node .zcode/workflow-drafts/convert-skills.mjs          — performs the conversion
//     (.agents/<id>/ -> .zcode/skills/<id>/, rewritten SKILL.md frontmatter, byte-exact body)
//     and self-verifies name match, description length, body bytes, and aux-file sha256 parity.
//   node .zcode/workflow-drafts/convert-skills.mjs --verify — read-only re-verification of bytes
//     and inventory, re-parsed independently from disk (validated against the real conversion in a
//     sandbox before submission, including negative tests).
//
// Subagents then verify format fidelity per skill (read-only), an independent reviewer checks
// shadowing and deep-reads samples, and every finding goes through an independent confirmer
// before it is reported. The deliverable is published as the "conversion-report" artifact.

interface GateLine {
  /** Skill id (directory name under .agents/). */
  id: string;
  /** Whether the gate passed this skill. */
  ok: boolean;
  /** Converted description length in characters. */
  descChars: number;
  /** Auxiliary file count beside SKILL.md in the converted folder. */
  auxCount: number;
}

interface VerifierResult {
  /** Skill id, e.g. "007". */
  id: string;
  /** true when no blocking issues were found. */
  ok: boolean;
  /** Short issue list, empty when ok. */
  issues: string[];
}

interface Finding {
  /** Workspace-relative path, with a line when it applies: ".zcode/skills/007/SKILL.md". */
  where: string;
  /** One sentence: what is wrong, or what was found. */
  what: string;
  /** What showed it: the lines read, or the command and the output that proved it. */
  evidence: string;
  /** "verified" when an independent subagent or a deterministic check confirmed it; "unconfirmed" when confirmation failed or was not attempted. */
  status: "verified" | "unconfirmed";
  /** How much it matters. Reserve "high" for data loss, a crash, or a wrong result. */
  severity: "low" | "medium" | "high";
}

interface WorkflowReport {
  /** Two or three sentences answering what the user asked for. */
  conclusion: string;
  findings: Finding[];
  /** What the run checked and how: the commands it ran, the files it covered. */
  verified: string[];
  /** What the run did not look at or could not check, and why. */
  notCovered: string[];
}

function isGateLine(value: unknown): value is GateLine {
  if (typeof value !== "object" || value === null) return false;
  const v = value as Record<string, unknown>;
  return (
    typeof v.id === "string" &&
    typeof v.ok === "boolean" &&
    typeof v.descChars === "number" &&
    typeof v.auxCount === "number"
  );
}

/** Parse the per-skill JSON lines the conversion gate prints on stdout; the summary line has no id. */
function parseGateLines(stdout: string): GateLine[] {
  const lines: GateLine[] = [];
  for (const raw of stdout.split("\n")) {
    const t = raw.trim();
    if (!t.startsWith("{")) continue;
    let parsed: unknown;
    try {
      parsed = JSON.parse(t);
    } catch {
      continue;
    }
    if (isGateLine(parsed)) {
      lines.push({ id: parsed.id, ok: parsed.ok, descChars: parsed.descChars, auxCount: parsed.auxCount });
    }
  }
  return lines;
}

const ids = [
  "007",
  "address-github-comments",
  "agent-evaluation",
  "agent-memory-systems",
  "agents-md",
  "ai-agents-architect",
  "architecture",
  "architecture-decision-records",
  "autonomous-agents",
  "brooks-audit",
  "brooks-test",
  "dependency-management-deps-audit",
  "diagnosing-bugs",
  "game-design",
  "invariant-guard",
  "langgraph",
];

phase("Convert the 16 skills to Zcode format");
const convert = await world.run("node", [".zcode/workflow-drafts/convert-skills.mjs"]);
if (convert.exitCode !== 0) {
  const finding: Finding = {
    where: ".zcode/workflow-drafts/convert-skills.mjs",
    what: "The conversion gate failed, so the Zcode-format conversion is not complete.",
    evidence: `node .zcode/workflow-drafts/convert-skills.mjs exited ${convert.exitCode}. stderr:\n${convert.stderr.slice(0, 2000)}`,
    status: "verified",
    severity: "high",
  };
  report(finding);
  return {
    conclusion:
      "Conversion stopped at the first gate: node .zcode/workflow-drafts/convert-skills.mjs exited nonzero, so the converted skill set is not complete. Nothing downstream ran. Fix the gate failure named in the finding, then run this workflow again.",
    findings: [finding],
    verified: [],
    notCovered: [
      "Everything after the conversion gate did not run: independent byte/inventory re-verification, per-skill format verification, independent review, and the conversion report.",
      "Runtime skill discovery in a fresh Zcode session is not verifiable statically and was not tested.",
      "Antigravity-side loading of the .agents rule entry points was not covered.",
    ],
  };
}
const gateRows = parseGateLines(convert.stdout);
for (const row of gateRows) {
  log(`converted ${row.id}: description ${row.descChars} chars, ${row.auxCount} aux files, body preserved`);
}

phase("Independently re-verify bytes and inventory");
const reverify = await world.run("node", [".zcode/workflow-drafts/convert-skills.mjs", "--verify"]);
if (reverify.exitCode !== 0) {
  const finding: Finding = {
    where: ".zcode/workflow-drafts/convert-skills.mjs --verify",
    what: "The independent byte/inventory re-verification failed; the converted set is not proven faithful.",
    evidence: `node .zcode/workflow-drafts/convert-skills.mjs --verify exited ${reverify.exitCode}. stderr:\n${reverify.stderr.slice(0, 2000)}`,
    status: "verified",
    severity: "high",
  };
  report(finding);
  return {
    conclusion:
      "The conversion itself ran, but its byte/inventory re-verification gate failed, so the converted skills are not proven byte-faithful. Format verification and review did not run. Fix the failure named in the finding (stderr shows the failing check), then run this workflow again.",
    findings: [finding],
    verified: [`node .zcode/workflow-drafts/convert-skills.mjs (exit 0, ${gateRows.length} skills converted)`],
    notCovered: [
      "Per-skill format verification, the independent review, and the conversion report did not run after this gate failure.",
      "Runtime skill discovery in a fresh Zcode session is not verifiable statically and was not tested.",
      "Antigravity-side loading of the .agents rule entry points was not covered.",
    ],
  };
}

phase("Verify each skill's format and fidelity");
const verifierResults: VerifierResult[] = await Promise.all(
  ids.map((id) =>
    agent(`format-verifier-${id}`).ask<VerifierResult>(
      `You are a READ-ONLY format verifier for one converted skill: ${id}. Do not edit, create, move, or delete any file, and do not run commands that write. Do NOT re-run byte-level or sha256 checks — deterministic gates already decided those.

Read both files:
- converted: .zcode/skills/${id}/SKILL.md
- source: .agents/${id}/SKILL.md

Zcode SKILL.md rules to check against: the frontmatter is a "---"-delimited block of FLAT "key: value" lines; the recognized keys are exactly name, description, when_to_use, license, metadata; name and description are required; description must be a single line of at most 1024 characters; multi-line values must use ">" or "|"; values containing colons should be double-quoted.

Check:
1. The converted frontmatter contains exactly name, description, when_to_use (plus a license line only when the SOURCE frontmatter had a license key); the name equals "${id}".
2. The description is faithful to the source description. The source value may be quoted or folded across continuation lines; the converted value must be the unfolded single-line form with whitespace collapsed — same meaning, no truncation, no rewording, at most 1024 characters.
3. when_to_use is sensible and consistent with what the source skill is for, with no garbled or injected tokens.
4. No anomalies: mojibake, corrupted text, wrong name, missing or damaged body content.

Return VerifierResult with id "${id}", ok true when there are no blocking issues, otherwise ok false with a short issues list (one sentence per issue, citing what you read).`,
    ),
  ),
);
const verifierCandidates: Finding[] = [];
for (const result of verifierResults) {
  if (!result.ok) {
    verifierCandidates.push({
      where: `.zcode/skills/${result.id}/SKILL.md`,
      what:
        result.issues.length > 0 ? result.issues.join("; ") : "format verifier reported failure without details",
      evidence: `format-verifier-${result.id} reported: ${result.issues.join(" | ")}`,
      status: "unconfirmed",
      severity: "medium",
    });
  }
}
const passedVerifiers = verifierResults.filter((r) => r.ok).length;
log(`${passedVerifiers} of ${ids.length} format verifiers passed cleanly.`);

phase("Independent review of the converted set");
const reviewerFindings: Finding[] = await agent("independent-reviewer").ask<Finding[]>(
  `You are an independent, READ-ONLY reviewer of a finished skill-format conversion. Do not edit, create, move, or delete any file, and do not run commands that write.

Background: 16 skills were converted from .agents/<id>/SKILL.md (AAS catalog frontmatter) to .zcode/skills/<id>/SKILL.md (Zcode frontmatter: a "---"-delimited block of FLAT "key: value" lines; recognized keys are exactly name, description, when_to_use, license, metadata; name and description are required; description must be a single line of at most 1024 characters; multi-line values must use ">" or "|"; values containing colons should be double-quoted; the markdown body after the frontmatter is preserved byte-exact; aux files are copied byte-exact). Deterministic gates already verified bodies byte-for-byte and aux-file sha256 parity — do NOT redo byte checks.

Do three things:
1. Shadowing: list the directory entries under C:\\Users\\darf3\\.zcode\\skills and C:\\Users\\darf3\\.agents\\skills (Glob or a read-only directory listing). If any of these ids already exist there — ${ids.join(", ")} — that is a finding: a same-name skill at a higher-precedence location than the workspace .zcode/skills copy.
2. Deep-read three converted skills against their sources and the rules above: 007, langgraph, and game-design (read .zcode/skills/<id>/SKILL.md and .agents/<id>/SKILL.md for each). Check the frontmatter shape, name equals the id, description faithful to the source (the source may be quoted or folded; the converted value must be the unfolded single-line form, meaning preserved), when_to_use sensible and consistent with the skill's purpose, license kept only when the source had one, and no mojibake or corrupted text.
3. Hunt for anything a byte-level gate cannot see: meaning drift, a wrong or incoherent when_to_use, encoding damage, a body that lost content.

Return a JSON array of Finding objects, each with status "unconfirmed" (a separate confirmer will reproduce each one). Use workspace-relative paths in "where" where the subject is inside the workspace; cite what you read in "evidence". Return [] if you find nothing.`,
);
const candidates: Finding[] = [...verifierCandidates, ...reviewerFindings];
const findings: Finding[] = [];
if (candidates.length > 0) {
  const verdicts = await Promise.all(
    candidates.map(async (candidate, i) => {
      const verdict = await agent(`finding-confirmer-${i + 1}`).ask<Finding>(
        `You are an independent, READ-ONLY confirmer. Do not edit, create, move, or delete any file, and do not run commands that write.

Reproduce this finding from its evidence alone — read the files it names yourself and check the claim:
${JSON.stringify(candidate)}

If you can reproduce it, return the Finding with status "verified" and add the proof you saw (exact quotes or the command and its output) to "evidence". If you cannot reproduce it from the evidence, return it with status "unconfirmed" and say what you checked and what did not hold in "evidence". Keep "severity" as given unless your evidence justifies changing it. Never report a check you did not run.`,
      );
      report(verdict);
      return verdict;
    }),
  );
  findings.push(...verdicts);
}
const confirmedCount = findings.filter((f) => f.status === "verified").length;
const unconfirmedCount = findings.length - confirmedCount;
log(`${confirmedCount} of ${candidates.length} findings confirmed independently.`);

// Publish the deliverable. Folded into the review phase on purpose: per the compiler's phase
// rules a phase must contain an ask or world.run, and plain script logic plus an artifact
// publish is not a stage (and a reader on top of already-confirmed findings is redundant).
const rows: string[] = [];
for (const id of ids) {
  const gate = gateRows.find((r) => r.id === id);
  const verifier = verifierResults.find((r) => r.id === id);
  const descChars = gate === undefined ? -1 : gate.descChars;
  const auxCount = gate === undefined ? -1 : gate.auxCount;
  const verifierOk = verifier === undefined ? false : verifier.ok;
  const issues: string[] = [];
  if (verifier !== undefined && !verifier.ok) {
    issues.push(...verifier.issues);
  }
  for (const f of findings) {
    if (f.where.startsWith(`.zcode/skills/${id}/`)) {
      issues.push(`${f.status}: ${f.what}`);
    }
  }
  const issuesCell =
    issues.length === 0 ? "—" : issues.join("; ").replace(/\|/g, "/").replace(/\r?\n/g, " ");
  rows.push(
    `| ${id} | ${descChars < 0 ? "?" : String(descChars)} | ${auxCount < 0 ? "?" : String(auxCount)} | ${
      verifierOk ? "yes" : "NO"
    } | ${issuesCell} |`,
  );
}
const findingLines = findings.map((f) => `- **${f.severity}** (${f.status}) ${f.where} — ${f.what}`);
const md = [
  "# Zcode-format conversion report",
  "",
  `Converted ${ids.length} AAS catalog skills from .agents/<id>/ to .zcode/skills/<id>/ (Zcode frontmatter: name, description, when_to_use, optional license; body preserved byte-exact).`,
  "",
  "| id | description chars | aux files | verifier ok | issues |",
  "| --- | --- | --- | --- | --- |",
  ...rows,
  "",
  "## Findings",
  "",
  findingLines.length === 0 ? "None." : findingLines.join("\n"),
].join("\n");
await artifact.markdown("conversion-report", md, {
  title: "Zcode-format conversion report",
  description: "Per-skill conversion table and independently confirmed findings.",
  primary: true,
});

const confirmed = findings.filter((f) => f.status === "verified");
const blocking = confirmed.filter((f) => f.severity === "high");
let conclusion: string;
if (blocking.length > 0) {
  conclusion = `All 16 skills were converted to .zcode/skills/<id>/ and passed both gates, but ${blocking.length} confirmed high-severity issue(s) need attention before the skills are used. ${
    confirmed.length - blocking.length
  } lower-severity finding(s) were confirmed and ${unconfirmedCount} remained unconfirmed. See the conversion-report artifact for the per-skill table.`;
} else if (findings.length > 0) {
  conclusion = `All 16 skills were converted to .zcode/skills/<id>/ and passed the conversion gate, the independent byte/inventory re-verification, ${passedVerifiers}/16 format verifiers, and an independent review. ${confirmed.length} lower-severity finding(s) were confirmed and ${unconfirmedCount} remained unconfirmed; none block use. See the conversion-report artifact.`;
} else {
  conclusion = `All 16 skills were converted to .zcode/skills/<id>/ and passed the conversion gate, the independent byte/inventory re-verification, ${passedVerifiers}/16 format verifiers, and an independent review with no findings. Runtime loading in a fresh Zcode session was not tested — verification was static.`;
}

return {
  conclusion,
  findings,
  verified: [
    `node .zcode/workflow-drafts/convert-skills.mjs (exit 0): converted all ${ids.length} skills and self-verified name match, single-line description of at most 1024 chars, byte-exact body, and sha256 aux-inventory parity`,
    "node .zcode/workflow-drafts/convert-skills.mjs --verify (exit 0): re-verified bytes and inventory from disk",
    `${passedVerifiers}/${ids.length} read-only format verifiers compared each converted SKILL.md against its source`,
    "independent reviewer checked higher-precedence shadowing and deep-read 007, langgraph, and game-design; every finding went through an independent confirmer",
  ],
  notCovered: [
    "Runtime skill discovery in a fresh Zcode session was not tested — verification here is static (files, bytes, frontmatter).",
    "Antigravity-side loading of the .agents rule entry points was not covered.",
  ],
};
