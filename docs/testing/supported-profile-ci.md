# Supported-profile Python CI

The **Python CI** workflow runs on pull requests targeting `main` and pushes to
`main`. Its three profile jobs must all succeed for **Python CI required** to
pass. AI review automation is separate. Adding this workflow does not itself
make its status a branch requirement: enforcement must be configured and read
back after the first successful hosted rollout.

## Profiles and local reproduction

Use CPython 3.12 on Linux x86-64. Hosted jobs select Ubuntu 24.04, with the
resolved Python patch version recorded in the evidence. Each job creates a new
virtual environment, installs the selected complete hash-locked wheel closure,
and runs `pip check`. Pip download caches are keyed by selected lock contents;
venvs, corpora and model assets are not cached.

| Profile | Installation | Checks |
| --- | --- | --- |
| `core` | Root requirements | 21 action/GOAP/event cases, 19 memory/headless/minimal cases and three actual CLI probes |
| `dev` | Development lock; no visual/AI packages | CI-runner failure/cleanup regressions and two public CLI smoke cases |
| `visual` | Development and visual locks | Four real SDL-dummy lifecycle, 24 graph, 27 cache and 13 subprocess-capture cases |
| `nlp` | NLP lock, own environment | Two actual spaCy/basic-runtime and controlled missing-model cases |
| `llm-cpu` | CPU-LLM lock, own environment | Real FAISS index and installed-AI/basic-runtime import guards |

NLP and CPU-LLM run **only through Actions → Python ML profiles → Run workflow**;
choose `nlp`, `llm-cpu` or `both` (default). There is no schedule. These workflows
become available for manual dispatch after they reach the default branch. A
failed manual check remains a failed check, but is not part of the required PR
gate. AI is required in the final product under D015; these offline package
checks do not establish live inference, semantic-model or gameplay readiness.

Example from the repository root after installing the corresponding profile:

```bash
python -m pip check
python -B scripts/run_ci_checks.py --profile core --artifacts .test-artifacts/ci/local-core-1
```

Use a new artifact directory for each attempt. The runner refuses to overwrite
an existing summary or batch record. It audits installed package presence and
rejects rendering/AI packages in core/dev; dependency installation and `pip check`
remain explicit workflow steps. Reported package versions let reviewers verify
which environment actually ran.

## Results and deadlines

The standard-library parent runs explicit batches in separate owned POSIX
sessions. A scoped registry records supervisor-created nested sessions before
their workloads execute, so a worker watchdog exit cannot orphan them. Closed
owners reject late startup; Linux process identities guard against PID reuse. It captures stdout/stderr to durable files, uses a 120-second parent
batch deadline, and terminates owned descendants on completion, timeout or
cancellation. Aggregate unittest workers emit diagnostics at 110 seconds;
pytest smoke keeps 115/120. Existing inner graph/runtime checks keep 20/25 and
cache checks 25/30. Standalone CLI probes use the existing durable capture helper
with 20/25. Cleanup has a separate two-second TERM grace and up to one second to confirm
SIGKILL quiescence; zombies awaiting host reaping are not running work. Unrelated processes
are not signalled. Workflow ceilings are 20 minutes for required profiles and
40 minutes for manual ML; superseded PR runs are cancelled, main runs are not.

`summary.json` records profile, source revision, Python/platform, selected lock
SHA-256 hashes, installed package versions, batch commands, real exit statuses,
actual test counts and CLI probe counts. Each batch has `run.json`, result JSON,
stdout/stderr and, where applicable, JUnit XML. Existing cold-child metadata and
watchdog output remain under `children/`. Unittest methods and pytest cases are
counted as executed; failing subtests are reported as failures without inflating
method counts. Pytest reporting hooks also retain XFAIL/XPASS outcomes that
JUnit alone does not distinguish; neither can silently pass the gate. Controlled failures inside runner regressions are test fixtures,
not failed application checks.

A profile passes only when all selected batches exit successfully with complete,
nonempty structured results and no skips/failures/errors/expected failures or
unexpected successes. The invalid CLI probe deliberately expects exit 2; its
worker succeeds only when that exact result is observed. Missing evidence,
zero-test selections, import errors, nonzero worker exits and timeouts fail.
An unfinished record is incomplete, never passing. The batch supervisor can
retain diagnostic files even if the worker exits before writing its report.

Actions uploads the selected profile evidence after success or failure, retaining
it for 30 days. No model cache or environment dump is uploaded. Hard runner loss
or forced job termination can prevent upload; missing evidence is not a success
claim. Review raw logs before sharing them outside the project.

## Required-check activation and human handoff

**Current rollout:** Complete; #726 is closed. #789 is merged, and main/both
manual profiles pass 132 cases + three CLI probes with verified artifact digests.
After owner activation, API readback confirms `Python CI required` is effective
on main, bound to GitHub Actions (15368), with branches required up to date.
Existing rules/reviews/bypasses are preserved. See the
[rollout report](issue-726-ci-validation.md#post-merge-rollout--complete).
The steps below document the activation procedure for future maintenance.

After human review and merge of the CI PR:

1. Confirm a main-branch **Python CI** run succeeds with all three profiles.
2. Manually run **Python ML profiles**, selecting `both`, and retain both results.
3. In **Settings → Rules → Rulesets → tiny_ruleset**, add **Require status checks
   to pass → Python CI required**, bind its source to GitHub Actions, and enable
   **Require branches to be up to date before merging**. Keep existing review
   rules and bypass actors unchanged. Verify the actual check name and source
   against the successful hosted run before saving.
4. Read back the effective ruleset and confirm the required context, publisher
   and up-to-date policy. Existing bypass actors retain their current policy.

An authorized API update may perform step 3 instead. If integration permissions
reject it, the owner can use the UI above; do not replace credentials or claim
activation succeeded. Keep #726 open until hosted main/manual evidence and
required-check enforcement have all been verified. The implementation PR does
not auto-close #726. Existing human phase gates remain separate decisions.

This gate deliberately excludes whole-repository legacy suites, #260 prompt
coverage/linting targets, live model inference (#585), full gameplay (#751/#658),
GPU/native Windows/macOS claims, and human UX/aesthetic reviews (#727–#730).
