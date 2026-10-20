# #726 supported-profile CI validation

## Scope and baseline

Approved plan: [supported-profile CI](../superpowers/plans/2026-10-20-issue-726-supported-profile-ci.md).
D017 selects manual-only ML checks. D015 preserves required final-product AI.
No gameplay, live inference, GPU or native Windows/macOS claim is supplied.

Base main is `eb5f4c935e87abe53570d0d949ffd8a6f9dd5cfa` (#788 merged).
Post-merge verification passes 47 methods: 19 core runtime, four real Pygame
lifecycle and 24 graph lifecycle/ownership, plus actual CLI exit codes 0/0/2.
Both minimal invocations succeed; implicit visual/headless is an argument error.
Evidence: `/workspace/scratch/issue-726/baseline.json`, stdout/stderr and durable
cold-child records. This uses existing #654 environments, not new installations.

## Local implementation evidence

- New CI runner regressions failed before implementation: 12 failures because
  the runner did not exist (`runner-red.log`). The initial 12 then passed; an
  additional real subtest-count regression brings the selection to 13 passing.
- New core/dev/visual environments (`/workspace/.venvs/issue726-*`) were installed
  from the unchanged locks with `--require-hashes --only-binary=:all:`. Every
  install and `pip check` exits 0; exact receipts are in `installs.json`.
- All five runner profiles pass locally at initial runner revision `7eb2d70`:
  core 40 cases + three CLI probes, dev 15, visual 68, NLP 2, CPU-LLM 2. Total:
  **127 cases, three probes; zero profile skips, failures or timeouts.**
- NLP/CPU-LLM reuse the locked #725 environments; this is not a new optional
  installation claim. Fresh optional environments are specified in the manual
  hosted workflow. No weights or corpora were downloaded.
- Controlled runner fixtures intentionally exercise failing, empty, skipped,
  import-error, cancelled, timed-out and missing-report programs. These are
  passing regression scenarios, not hidden application failures.
- The cancellation fixture isolates GITHUB_STEP_SUMMARY so its intentional
  cancellation cannot add a false cancelled application row to a real job.
  All 13 runner methods pass after that test-only isolation change.

Commands are `python -B scripts/run_ci_checks.py --profile PROFILE --artifacts
NEW_DIRECTORY`, using the corresponding profile interpreter. Worker commands,
counts, statuses, package versions, Python/platform and lock hashes are retained
in each profile's summary and batch records. Actual local OS is Linux/cloud;
Ubuntu 24.04 hosting is not inferred from that local evidence.

Workflow validation uses SHA-256-verified actionlint 1.7.12 (archive
`8aca8db96f1b94770f1b0d72b6dddcb1ebb8123cb3712530b08cc387b349a3d8`):

```bash
actionlint -shellcheck= .github/workflows/python-ci.yml .github/workflows/python-ml-profiles.yml
```

Initial output rejected `runner.temp` at job-env scope; moved the isolated cache
assignment into a runner step. Both files now validate, exit 0. Original output
is preserved. This validates workflow structure, not hosted execution. No style
linting, dependency regeneration or full legacy-suite run was added.

## Independent review and final verification

One independent whole-branch review at `ab22ce2` found three Important issues,
no Critical findings, one Minor observation, and no declined findings. All three
Important issues were fixed in one pass in `586fbd1`:

1. Track nested supervisor-owned sessions durably so timeout/watchdog exit does
   not leave a TERM-resistant descendant running; preserve unrelated processes.
2. Count pytest XFAIL/XPASS via reporting hooks and reject unexpected successes.
3. Require complete integer probe exit codes; missing/null/boolean values fail.

Review regressions were observed RED: six failed observations across 17 methods
(including subcases), plus a separate pure-XPASS false-green regression. The
first fix run retained an asynchronous SIGKILL-delivery failure; bounded
quiescence now waits for live group members to stop. All 18 runner methods pass.

The complete five-profile selection was rerun on `586fbd1`: **132 cases and
three actual CLI probes pass**, no unexpected skips, profile failures or
timeouts. Counts: core 40 + three probes, dev 20, visual 68, NLP 2, CPU-LLM 2.
All five top-level commands exit 0. Workflow structural validation still exits 0.
No second independent review was requested. Original failures and the earlier
127-case matrix remain separate; neither is relabeled as a clean final run.

Published receipts: [merged baseline](evidence/issue-726-merged-baseline.json)
and [final profile commands/results/install receipts/source hashes](evidence/issue-726-local-results.json).
Full local batch/child artifacts remain in `/workspace/scratch/issue-726/`.

**Deferred minor:** one original test prewrites RUNNING before the supervisor
removes stale reports, duplicating missing-file coverage. It remains a redundant
assertion. A new child-written partial-report regression covers the actual
incomplete-result contract, so this is test cleanup rather than missing behavior.

## Hosted rollout and completion boundary

Hosted PR results, main-run results, manual dispatch evidence and actual ruleset
activation must be recorded as they become available. They are currently pending;
local pass results do not establish enforcement. The PR references #726 without
an automatic closing keyword. Keep it open until the post-merge rollout passes.

[Operator instructions](supported-profile-ci.md#required-check-activation-and-human-handoff)
name the stable check, GitHub Actions publisher, up-to-date policy and owner UI
handoff if API permissions deny configuration. Existing bypass/review rules stay
unchanged; later phase gates receive no approval here.
