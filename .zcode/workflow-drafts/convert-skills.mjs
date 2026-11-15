// convert-skills.mjs — deterministic converter from AAS catalog skill format to Zcode skill format.
//
// Source: .agents/<id>/SKILL.md (+ aux files)   Dest: .zcode/skills/<id>/SKILL.md (+ aux files)
//
// Modes (mutually exclusive):
//   node .zcode/workflow-drafts/convert-skills.mjs             real run: copy each .agents/<id>/ to
//                                                              .zcode/skills/<id>/ and rewrite SKILL.md
//                                                              frontmatter; self-verify afterwards.
//   node .zcode/workflow-drafts/convert-skills.mjs --dry-run   read-only: parse + preview, no writes.
//   node .zcode/workflow-drafts/convert-skills.mjs --verify    read-only: re-verify an existing
//                                                              conversion (bytes + inventory + shape).
//
// Output (stdout): one JSON line per skill {id, ok, descChars, auxCount, bodyPreserved[, preview]}
// followed by one summary JSON line. Failure details go to stderr. Exit 0 iff all 16 pass.
//
// Conversion contract:
//   - New frontmatter: --- delimited, flat "key: value" lines, double-quoted values:
//       name: "<id>"
//       description: "<source description unfolded to one line, whitespace collapsed>"
//       when_to_use: "<contract value for id>"
//       license: "<value>"   (only when the source frontmatter had a license key)
//   - Source keys other than name/description/license are dropped
//     (risk, source, date_added, source_repo, source_type, license_source, author, tags, tools, ...).
//   - The body (everything after the frontmatter-closing --- line, including the exact bytes after it)
//     is preserved byte-exact. Aux files are copied byte-exact (verified by sha256).

import fs from "node:fs";
import path from "node:path";
import crypto from "node:crypto";
import { fileURLToPath } from "node:url";

const scriptDir = path.dirname(fileURLToPath(import.meta.url));
const repoRoot = path.resolve(scriptDir, "..", "..");
const SRC_BASE = path.join(repoRoot, ".agents");
const DEST_BASE = path.join(repoRoot, ".zcode", "skills");

const IDS = [
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

// Contract data: when_to_use per skill id (clean, deduplicated values).
// dependency-management-deps-lint is part of the contract list but not among the
// current 16 ids; it is kept here for completeness and is simply unused.
const WHEN_TO_USE = {
  "007":
    "Use when auditing security, hardening, threat modeling, OWASP review, or code security review of any project.",
  "address-github-comments":
    "Use when addressing review or issue comments on an open GitHub pull request with the gh CLI.",
  "agent-evaluation":
    "Use when testing, benchmarking, or evaluating LLM agents: behavioral testing, capability assessment, reliability metrics.",
  "agent-memory-systems":
    "Use when designing or reviewing agent memory: short-term vs long-term memory, vector stores, retrieval.",
  "agents-md": "Use when creating or updating AGENTS.md or CLAUDE.md agent instructions.",
  "ai-agents-architect":
    "Use when designing autonomous AI agents: tool use, memory systems, planning strategies, multi-agent orchestration.",
  architecture:
    "Use when making architecture decisions, analyzing system design, or documenting requirements, trade-offs, and ADRs.",
  "architecture-decision-records":
    "Use when creating, maintaining, or managing Architecture Decision Records (ADRs).",
  "autonomous-agents":
    "Use when building or reviewing autonomous agent loops, goal decomposition, reflection patterns, or agent reliability and guardrails.",
  "brooks-audit":
    "Use when auditing architecture, reviewing folder/module structure, checking for circular imports, or mapping module dependencies.",
  "brooks-test":
    "Use when reviewing test-suite quality: brittleness, mock overuse, and structural test problems.",
  "dependency-management-deps-audit":
    "Use when auditing dependency vulnerabilities, licenses, or supply-chain risk, e.g. for hash-locked requirements profiles.",
  "dependency-management-deps-lint":
    "Use when auditing dependency vulnerabilities, licenses, or supply-chain risk, e.g. for hash-locked requirements profiles.",
  "diagnosing-bugs":
    "Use when diagnosing hard bugs, regressions, crashes, or performance issues root-cause-first.",
  "game-design":
    "Use when designing or balancing a simulation/game: core loop, progression, difficulty, player psychology.",
  "invariant-guard":
    "Use when writing or reviewing correctness-critical code: function contracts, loop invariants, edge cases first.",
  langgraph:
    "Use when building or reviewing LangGraph/LangChain agent graphs, state management, checkpointers, or ReAct patterns.",
};

// ---------- robust source frontmatter parsing ----------

/** Split a raw SKILL.md into { front, body, closingHadNewline, bom, crlf }. */
function splitFrontmatter(raw) {
  let bom = "";
  let work = raw;
  if (work.charCodeAt(0) === 0xfeff) {
    bom = work[0];
    work = work.slice(1);
  }
  if (!work.startsWith("---")) {
    throw new Error("missing opening '---' frontmatter marker");
  }
  const firstNl = work.indexOf("\n");
  if (firstNl === -1) {
    throw new Error("unterminated opening '---' line");
  }
  let pos = firstNl + 1;
  for (;;) {
    const nl = work.indexOf("\n", pos);
    const lineEnd = nl === -1 ? work.length : nl;
    let le = lineEnd;
    if (le > pos && work[le - 1] === "\r") {
      le -= 1;
    }
    const line = work.slice(pos, le);
    if (line === "---") {
      return {
        front: work.slice(firstNl + 1, pos),
        // Body = everything after the closing '---' line, exact bytes included.
        body: nl === -1 ? "" : work.slice(nl + 1),
        closingHadNewline: nl !== -1,
        bom,
        crlf: work.includes("\r\n"),
      };
    }
    if (nl === -1) {
      throw new Error("missing closing '---' frontmatter marker");
    }
    pos = nl + 1;
  }
}

/** Index just past the closing unescaped '"' of a double-quoted scalar, or -1. s starts with '"'. */
function scanDoubleQuotedEnd(s) {
  let i = 1;
  while (i < s.length) {
    const c = s[i];
    if (c === "\\") {
      i += 2;
      continue;
    }
    if (c === '"') {
      return i + 1;
    }
    i += 1;
  }
  return -1;
}

/** Index just past the closing "'" of a single-quoted scalar (handling '' escapes), or -1. */
function scanSingleQuotedEnd(s) {
  let i = 1;
  while (i < s.length) {
    const c = s[i];
    if (c === "'") {
      if (s[i + 1] === "'") {
        i += 2;
        continue;
      }
      return i + 1;
    }
    i += 1;
  }
  return -1;
}

function unescapeDoubleQuoted(content) {
  let out = "";
  for (let i = 0; i < content.length; i += 1) {
    const c = content[i];
    if (c === "\\" && i + 1 < content.length) {
      const n = content[i + 1];
      if (n === '"') {
        out += '"';
        i += 1;
        continue;
      }
      if (n === "\\") {
        out += "\\";
        i += 1;
        continue;
      }
      if (n === "n") {
        out += "\n";
        i += 1;
        continue;
      }
      if (n === "t") {
        out += "\t";
        i += 1;
        continue;
      }
    }
    out += c;
  }
  return out;
}

/** Extract the value of a double-quoted scalar that may span continuation lines. */
function readDoubleQuoted(t, lines, iRef, key) {
  let combined = t;
  for (;;) {
    const endIdx = scanDoubleQuotedEnd(combined);
    if (endIdx !== -1) {
      return unescapeDoubleQuoted(combined.slice(1, endIdx - 1));
    }
    if (iRef.i >= lines.length) {
      throw new Error(`unterminated double-quoted value for key "${key}"`);
    }
    const l = lines[iRef.i];
    if (!(l.trim() === "" || /^[ \t]/.test(l))) {
      throw new Error(
        `unterminated double-quoted value for key "${key}" (hit non-indented line ${JSON.stringify(l)})`,
      );
    }
    iRef.i += 1;
    combined += ` ${l.trim()}`;
  }
}

/** Extract the value of a single-quoted scalar that may span continuation lines. */
function readSingleQuoted(t, lines, iRef, key) {
  let combined = t;
  for (;;) {
    const endIdx = scanSingleQuotedEnd(combined);
    if (endIdx !== -1) {
      return combined.slice(1, endIdx - 1).replace(/''/g, "'");
    }
    if (iRef.i >= lines.length) {
      throw new Error(`unterminated single-quoted value for key "${key}"`);
    }
    const l = lines[iRef.i];
    if (!(l.trim() === "" || /^[ \t]/.test(l))) {
      throw new Error(
        `unterminated single-quoted value for key "${key}" (hit non-indented line ${JSON.stringify(l)})`,
      );
    }
    iRef.i += 1;
    combined += ` ${l.trim()}`;
  }
}

/**
 * Parse AAS catalog frontmatter text into a Map of key -> string value.
 * Handles quoted single-line values, quoted values folded across indented
 * continuation lines, plain folded scalars, ">"/">-"/"|"/"|-" block scalars,
 * and YAML sequences (skipped; only name/description/license are consumed).
 */
function parseSourceFrontmatter(frontText) {
  const lines = frontText.split("\n").map((l) => (l.endsWith("\r") ? l.slice(0, -1) : l));
  const map = new Map();
  const iRef = { i: 0 };
  const gatherIndented = () => {
    const cont = [];
    while (iRef.i < lines.length) {
      const l = lines[iRef.i];
      if (l.trim() === "" || /^[ \t]/.test(l)) {
        cont.push(l);
        iRef.i += 1;
        continue;
      }
      break;
    }
    return cont;
  };
  while (iRef.i < lines.length) {
    const line = lines[iRef.i];
    if (line.trim() === "") {
      iRef.i += 1;
      continue;
    }
    const keyMatch = /^([A-Za-z0-9_-]+):[ \t]?(.*)$/.exec(line);
    if (!keyMatch) {
      // YAML sequence entry (e.g. "- security" list items) or stray content: skip it.
      if (/^[ \t]*-[ \t]/.test(line) || /^[ \t]*-[ \t]*$/.test(line)) {
        iRef.i += 1;
        continue;
      }
      throw new Error(`unparseable frontmatter line: ${JSON.stringify(line)}`);
    }
    const key = keyMatch[1];
    const t = keyMatch[2].trim();
    iRef.i += 1;
    let value;
    if (/^[>|][\d+-]*$/.test(t)) {
      // Block scalar: marker stripped, lines folded with single spaces.
      const parts = gatherIndented().map((l) => l.replace(/^[ \t]+/, ""));
      while (parts.length && parts[parts.length - 1] === "") {
        parts.pop();
      }
      value = parts.join(" ").trim();
    } else if (t.startsWith('"')) {
      value = readDoubleQuoted(t, lines, iRef, key);
    } else if (t.startsWith("'")) {
      value = readSingleQuoted(t, lines, iRef, key);
    } else {
      // Plain scalar; plain folded scalars continue on indented lines.
      const parts = [];
      if (t !== "") {
        parts.push(t);
      }
      for (const l of gatherIndented()) {
        const trimmed = l.trim();
        if (trimmed !== "") {
          parts.push(trimmed);
        }
      }
      value = parts.join(" ");
    }
    map.set(key, value);
  }
  return map;
}

// ---------- Zcode-format frontmatter (output side) ----------

/** Double-quote a value for the flat Zcode frontmatter, escaping backslashes and quotes. */
function zq(value) {
  return `"${value.replace(/\\/g, "\\\\").replace(/"/g, '\\"')}"`;
}

function buildFrontmatter({ name, description, whenToUse, license, eol, closingHadNewline }) {
  const lines = [
    `name: ${zq(name)}`,
    `description: ${zq(description)}`,
    `when_to_use: ${zq(whenToUse)}`,
  ];
  if (license !== undefined) {
    lines.push(`license: ${zq(license)}`);
  }
  return `---${eol}${lines.join(eol)}${eol}---${closingHadNewline ? eol : ""}`;
}

/** Re-parse a written Zcode frontmatter (flat "key: value" lines, quoted values). */
function parseZcodeFrontmatter(frontText) {
  const lines = frontText
    .split("\n")
    .map((l) => (l.endsWith("\r") ? l.slice(0, -1) : l))
    .filter((l) => l.trim() !== "");
  const map = new Map();
  for (const line of lines) {
    const m = /^([A-Za-z0-9_-]+): (.*)$/.exec(line);
    if (!m) {
      throw new Error(`converted frontmatter is not a flat "key: value" line: ${JSON.stringify(line)}`);
    }
    let v = m[2];
    const dq = /^"(.*)"$/s.exec(v);
    if (dq) {
      v = unescapeDoubleQuoted(dq[1]);
    } else if (/^'.*'$/.test(v)) {
      v = v.slice(1, -1).replace(/''/g, "'");
    }
    map.set(m[1], v);
  }
  return map;
}

// ---------- inventory ----------

function walkFiles(dir) {
  const out = [];
  for (const ent of fs.readdirSync(dir, { withFileTypes: true })) {
    const p = path.join(dir, ent.name);
    if (ent.isDirectory()) {
      out.push(...walkFiles(p));
    } else if (ent.isFile() || ent.isSymbolicLink()) {
      out.push(p);
    }
  }
  return out.sort();
}

/** Map of workspace-style relative path -> sha256 for every file except SKILL.md. */
function inventory(dir) {
  const map = new Map();
  for (const f of walkFiles(dir)) {
    const rel = path.relative(dir, f).split(path.sep).join("/");
    if (rel === "SKILL.md") {
      continue;
    }
    map.set(rel, crypto.createHash("sha256").update(fs.readFileSync(f)).digest("hex"));
  }
  return map;
}

function sha256(text) {
  return crypto.createHash("sha256").update(text, "utf8").digest("hex");
}

// ---------- per-skill pipeline ----------

function analyzeSource(id) {
  const srcDir = path.join(SRC_BASE, id);
  const srcSkill = path.join(srcDir, "SKILL.md");
  if (!fs.existsSync(srcSkill)) {
    throw new Error(`source skill not found: ${srcSkill}`);
  }
  const srcRaw = fs.readFileSync(srcSkill, "utf8");
  const split = splitFrontmatter(srcRaw);
  const fm = parseSourceFrontmatter(split.front);
  const srcName = fm.get("name");
  const descRaw = fm.get("description");
  const description = (descRaw === undefined ? "" : descRaw).replace(/\s+/g, " ").trim();
  const licensePresent = fm.has("license");
  const license = fm.get("license");
  const wtu = WHEN_TO_USE[id];
  const eol = split.crlf ? "\r\n" : "\n";
  const newFrontmatter = buildFrontmatter({
    name: id,
    description,
    whenToUse: wtu === undefined ? "" : wtu,
    license,
    eol,
    closingHadNewline: split.closingHadNewline,
  });
  return {
    id,
    srcDir,
    srcRaw,
    split,
    fm,
    srcName,
    description,
    descChars: description.length,
    licensePresent,
    license,
    wtu,
    eol,
    newFrontmatter,
    auxInv: inventory(srcDir),
  };
}

/** Checks that apply to the source parse alone (all modes). */
function sourceProblems(a) {
  const problems = [];
  if (a.srcName !== a.id) {
    problems.push(`source name ${JSON.stringify(a.srcName)} !== id ${JSON.stringify(a.id)}`);
  }
  if (a.descChars === 0) {
    problems.push("description is empty after unfolding");
  }
  if (a.descChars > 1024) {
    problems.push(`description is ${a.descChars} chars (> 1024); Zcode would drop this skill`);
  }
  if (a.wtu === undefined) {
    problems.push("no when_to_use contract entry for this id");
  }
  return problems;
}

/** Verify the destination folder against the source-derived expectations. Read-only. */
function checkDest(a, destDir) {
  const problems = [];
  const destSkill = path.join(destDir, "SKILL.md");
  if (!fs.existsSync(destSkill)) {
    return ["converted skill not found: run the real conversion first"];
  }
  const destRaw = fs.readFileSync(destSkill, "utf8");
  let destSplit;
  try {
    destSplit = splitFrontmatter(destRaw);
  } catch (err) {
    return [`converted SKILL.md unparseable: ${err instanceof Error ? err.message : String(err)}`];
  }
  // (a) body byte-exact (compared as UTF-8 text and by sha256 of that encoding)
  if (destSplit.body !== a.split.body) {
    problems.push("converted body differs from source body");
  }
  if (sha256(destSplit.body) !== sha256(a.split.body)) {
    problems.push("converted body sha256 differs from source body");
  }
  // (b) frontmatter shape + round-trip of every emitted value
  let zfm;
  try {
    zfm = parseZcodeFrontmatter(destSplit.front);
  } catch (err) {
    problems.push(`converted frontmatter rejected: ${err instanceof Error ? err.message : String(err)}`);
    return problems;
  }
  if (zfm.get("name") !== a.id) {
    problems.push(`converted name ${JSON.stringify(zfm.get("name"))} !== id`);
  }
  if (zfm.get("description") !== a.description) {
    problems.push("converted description does not round-trip to the expected unfolded value");
  }
  if (zfm.get("when_to_use") !== a.wtu) {
    problems.push("converted when_to_use does not match the contract value");
  }
  if (a.licensePresent && zfm.get("license") !== a.license) {
    problems.push("converted license does not match the source license value");
  }
  if (!a.licensePresent && zfm.has("license")) {
    problems.push("converted frontmatter has a license line but the source had none");
  }
  const keys = [...zfm.keys()];
  const expectedKeys = a.licensePresent
    ? ["name", "description", "when_to_use", "license"]
    : ["name", "description", "when_to_use"];
  if (keys.join("|") !== expectedKeys.join("|")) {
    problems.push(
      `converted frontmatter keys are ${JSON.stringify(keys)}, expected ${JSON.stringify(expectedKeys)}`,
    );
  }
  // (c) aux file inventory identical (relative paths + sha256)
  const destInv = inventory(destDir);
  const rels = new Set([...a.auxInv.keys(), ...destInv.keys()]);
  for (const rel of rels) {
    if (!a.auxInv.has(rel)) {
      problems.push(`extra file in destination: ${rel}`);
    } else if (!destInv.has(rel)) {
      problems.push(`missing file in destination: ${rel}`);
    } else if (a.auxInv.get(rel) !== destInv.get(rel)) {
      problems.push(`sha256 mismatch for ${rel}`);
    }
  }
  return problems;
}

function convertOne(id, mode) {
  const a = analyzeSource(id);
  const problems = sourceProblems(a);
  const destDir = path.join(DEST_BASE, id);
  const destSkill = path.join(destDir, "SKILL.md");

  let bodyPreserved = null;
  if (mode === "convert") {
    fs.rmSync(destDir, { recursive: true, force: true });
    fs.cpSync(a.srcDir, destDir, { recursive: true });
    fs.writeFileSync(destSkill, a.split.bom + a.newFrontmatter + a.split.body, "utf8");
    const destProblems = checkDest(a, destDir);
    problems.push(...destProblems);
    bodyPreserved = !destProblems.some((p) => p.includes("body") || p.includes("not found"));
  } else if (mode === "verify") {
    const destProblems = checkDest(a, destDir);
    problems.push(...destProblems);
    bodyPreserved = !destProblems.some((p) => p.includes("body") || p.includes("not found"));
  }

  const line = {
    id,
    ok: problems.length === 0,
    descChars: a.descChars,
    auxCount: a.auxInv.size,
    bodyPreserved,
  };
  if (mode === "dry-run") {
    line.preview = {
      name: id,
      description: a.description,
      when_to_use: a.wtu === undefined ? null : a.wtu,
      license: a.licensePresent ? a.license : null,
    };
  }
  return { line, problems };
}

// ---------- main ----------

function main() {
  const argv = process.argv.slice(2);
  let mode = "convert";
  if (argv.includes("--dry-run") && argv.includes("--verify")) {
    console.error("--dry-run and --verify are mutually exclusive");
    process.exit(1);
  }
  if (argv.includes("--dry-run")) {
    mode = "dry-run";
  } else if (argv.includes("--verify")) {
    mode = "verify";
  }
  const unknown = argv.filter((arg) => arg !== "--dry-run" && arg !== "--verify");
  if (unknown.length > 0) {
    console.error(`unknown arguments: ${unknown.join(" ")}`);
    process.exit(1);
  }

  if (!Array.isArray(IDS) || IDS.length !== 16) {
    console.error(`expected exactly 16 ids, found ${IDS.length}`);
    process.exit(1);
  }
  const sorted = [...IDS].sort();
  if (sorted.some((v, idx) => v !== IDS[idx])) {
    console.error("IDS is not in alphabetical order");
    process.exit(1);
  }
  const missingContract = IDS.filter((id) => WHEN_TO_USE[id] === undefined);
  if (missingContract.length > 0) {
    console.error(`missing when_to_use contract entries: ${missingContract.join(", ")}`);
    process.exit(1);
  }

  let passed = 0;
  let failed = 0;
  for (const id of IDS) {
    let line;
    let problems;
    try {
      ({ line, problems } = convertOne(id, mode));
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      line = { id, ok: false, descChars: -1, auxCount: -1, bodyPreserved: null };
      problems = [message];
    }
    console.log(JSON.stringify(line));
    if (line.ok) {
      passed += 1;
    } else {
      failed += 1;
      console.error(`[FAIL] ${id}: ${problems.join("; ")}`);
    }
  }

  const allPresent =
    mode === "dry-run" ||
    IDS.every((id) => fs.existsSync(path.join(DEST_BASE, id, "SKILL.md")));
  const ok = failed === 0 && allPresent;
  console.log(JSON.stringify({ ok, mode, total: IDS.length, passed, failed, allIdsPresent: allPresent }));
  process.exit(ok ? 0 : 1);
}

main();
