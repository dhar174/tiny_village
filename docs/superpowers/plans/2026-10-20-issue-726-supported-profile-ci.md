# Required Python 3.12 CI implementation plan

**Status:** Approved by dhar174 on 2026-10-20; native implementation authorized.

> Use superpowers:executing-plans sequentially, with one independent whole-branch
> review before publication. The implementation ledger is TASK013.

**Goal:** Make supported offline runtime checks a reliable merge requirement,
with separate manually triggered NLP/CPU-LLM checks. AI remains required for the
final product under D015. Package optionality is a testing boundary.

**Architecture:** GitHub Actions uses fresh verified profile environments and a
shared bounded verification runner. One stable aggregate check represents the
required profiles. Spec: #726 and the approved #724 support contract.

## Global constraints

- Ubuntu 24.04, Linux x86-64, CPython 3.12; record the resolved patch version.
- Preserve dependency locks; install hashed wheels and run pip check each time.
- ML workflows are manual only, with no schedule. No weights/corpora downloads.
- No gameplay/inference/GPU/native-platform readiness or automatic phase approval.
- Keep #260's broader prompt coverage and #727/#728 design work separate.
- Preserve original failed observations and existing subprocess deadlines.

## Task 1 — Merged-main baseline

- [ ] Branch from merged main and record revision; save plan/task/decision.
- [ ] Verify 47 methods (19 core runtime, 4 visual, 24 graph), both minimal CLI
  commands and invalid implicit visual/headless exit 2. Retain actual evidence.
- [ ] Investigate failures without weakening checks or increasing deadlines.

## Task 2 — Shared bounded runner

- [ ] Implement scripts/run_ci_checks.py with --profile selecting core, dev,
  visual, nlp or llm-cpu and --artifacts DIRECTORY.
- [ ] Use explicit selections, owned process groups, diagnostic deadlines and
  bounded descendant cleanup. Reuse existing cold-process diagnostics.
- [ ] Retain commands, exit status, source/lock hashes, actual test counts,
  failures/errors/skips, stdout/stderr, timeout and incomplete results.
- [ ] Write initial incomplete records; never infer success from a banner.
- [ ] Fail on empty/skipped suites, import errors, nonzero exits, timeout or
  missing/incomplete reports. Add failing regressions before implementing.

## Task 3 — Required PR/main checks

- [ ] Create .github/workflows/python-ci.yml: PRs targeting main and main pushes,
  no path filters; fresh core-only, dev-without-visual, and dev+visual environments.
- [ ] Core: 21 action/GOAP/event and 19 runtime cases plus bounded CLI probes.
- [ ] Dev: two existing parametrized CLI cases and new runner regressions.
- [ ] Visual: 4 real SDL lifecycle, 24 graph, 27 cache and 13 capture cases.
- [ ] Cache pip downloads keyed by selected locks, never venv/model caches.
- [ ] Verify package absence in core/dev; offline models, owned isolated caches,
  SDL dummy drivers and two-thread numerical libraries.
- [ ] contents:read; checkout persist-credentials:false; no secrets/elevated PR
  execution. Upload selected evidence on success/failure, retention 30 days.
- [ ] Stable aggregate Python CI required: all three jobs must succeed; failed,
  skipped or cancelled dependencies fail. Cancel superseded PR runs only.
- [ ] Pin checkout v7.0.1 to 3d3c42e5aac5ba805825da76410c181273ba90b1,
  setup-python v7.0.0 to 5fda3b95a4ea91299a34e894583c3862153e4b97 and
  upload-artifact v7.0.2 to cf430e030ddbb5b0abf93d22962f4752f3646cd9.

## Task 4 — Manual ML checks

- [ ] Create .github/workflows/python-ml-profiles.yml, workflow_dispatch only;
  choices nlp/llm-cpu/both, default both. Separate fresh environment per profile.
- [ ] NLP: existing two installed-spaCy/basic and controlled missing-model cases.
- [ ] CPU-LLM: real FAISS index and guarded minimal with AI libraries installed.
- [ ] Same evidence/install conventions; no weights/corpora or required ML gate.

## Review focus and validation

- False greens: controlled failure/error/empty/skipped/nonzero-exit fixtures.
- Interrupted execution: partial records, owned descendant cleanup, unrelated
  process preserved; timeouts/cancellation never PASS.
- Missing evidence: absent/incomplete reports cannot PASS.
- Dependency leakage: warm pip cache cannot pollute fresh core/dev environments.
- Aggregation: only success/success/success satisfies Python CI required.

Cold probes: 20/25 seconds; cache retains 25/30; existing CLI smoke 115/120;
aggregate unittest diagnostic/parent budgets 110/120. Required jobs 20 minutes,
manual ML jobs 40 minutes. Existing required selections contain 110 cases,
before adding runner regressions and standalone CLI probes; report actual counts.

## Rollout and completion

- [ ] Verify locally, review whole branch, publish one focused PR referencing
  #726 without auto-closing it. Obtain human review; do not auto-merge.
- [ ] After merge, verify main workflow and trigger both manual profiles once.
- [ ] Add Python CI required, bound to GitHub Actions, to the existing active
  ruleset with branches current before merge; preserve review/bypass policy.
- [ ] Verify effective enforcement. If access is denied, provide exact owner
  steps and leave enforcement incomplete; never claim running means required.
- [ ] Close #726 only after hosted results and enforcement are verified.

Next planning block: #727 UX journeys and #728 aesthetic direction, with separate
human review. All human decisions stay in docs/decisions.md.
