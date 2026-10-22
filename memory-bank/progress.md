## What Works

- TASK021 / #734 complete under D027: #797 merged at 2c3bc57, approved
  publication independently verified and #734 closed. All 307 annotations
  preserve original bodies/assignees; 80 missing-parent links, seven narrow
  closures, #799/#800/#801 and seven blocker edges verified, zero cycles.
  #602/#609 and #733/#738 remain open. PR #798 retains current publication
  receipts separately from the frozen historical review package. No Project
  writes; configuration remains #737's pending handoff after retained HTTP 403.
- TASK022 / #735 complete: #703 merged at 323dad6 and #735 closed; #714
  remains superseded under D028. #798 merged at 11b6e93 with review fixes and
  retained publication receipts. Required merged-main checks pass.
- TASK023 / #736: reviewing nine old functional-test PRs against main 11b6e93
  and merged #349, one disposition at a time. #420's five original methods
  and one selected #417 utility method pass on current main. #420 choice selected under D029:
  recommend salvaging unique State/Action/Condition assertions into maintained
  suites, then superseding the old demonstration. D029 also records #417 utility salvage, #408 superseding, and #406 memory API salvage; also selected #404 obsolete/supersede; also selected #397 counting/supersede; reviewing #395 next; no
  implementation/closures.
  Remaining source findings are preliminary; #737 and human gate #738 follow.

- TASK020 / #732 is complete: dhar174 explicitly approved Phase 2 under D025.
  #732 and parent #723 are closed/completed with immutable decision links
  verified; all native Phase 2 children were closed first. #734's native H2
  dependency is closed. PR #795 merged at a7dea4c; Project fields remain
  unsynchronized after HTTP 403. Gate #738 retains its own later human decision.

- TASK019 / #731: Approved under D024 within the reviewed Python 3.12/Linux
  foundation, with all documented evidence limits. No new blocking remediation
  identified; current hosted/retained source evidence stands. Published in #795,
  #731 is closed/completed with its immutable approval link verified. See TASK019.

- D023 records dhar174's explicit approval of #727/#728/#729 planning and #730
  combined design review on 2026-10-21. TASK014–TASK018 are complete for their
  planning scope. All four issues are closed/completed; immutable decision
  links and states were verified after repository authentication recovered.
  Existing PR #795 carries the record and engineering handoff. Project 10
  fields remain unsynchronized after HTTP 403. #731/#732 and #723 stay open;
  do not repeat settled design choices or infer further human approval.

- TASK013 / #726 is complete and closed. #789 merged at ac0e25d; main and
  both manual ML profiles pass 132 cases + three CLI probes, all five artifact
  digests verified. Owner activated tiny_ruleset; API readback confirms the
  effective required check, GitHub Actions source and strict/up-to-date policy.
  Existing rules/reviews/bypasses are preserved. Evidence PR #790 merged at
  969f6b9. Next: review #727 UX journeys and plan #728 aesthetic direction, with
  separate human review. No phase gate is approved. See TASK013/validation report.

- The repository has established architecture and documentation for the core
  simulation loop: events, strategy updates, GOAP planning, action execution,
  graph state updates, and memory recording.
- Current docs describe working integration across core systems: gameplay
  control, event handling, strategy coordination, graph management,
  storytelling, checkpoints, and analytics.
- The repo has current contributor guidance (`AGENTS.md`), organized docs under
  `docs/`, a `tests/` directory with organized test coverage, and testing
  guidance that emphasizes realistic test objects over permissive mocking.
- The `tests/` directory and root-level `test_*.py` files provide integration
  and regression test coverage across many subsystems.

## What Is Still Incomplete

- Several Memory Bank core files contained stale content from an unrelated
  project (`langgraph_system_generator`) and are being corrected as part of
  TASK001.
- Task-level Memory Bank tracking has only just begun with TASK001; the index
  and task file need to be committed.
- Some current status docs still describe demo/runtime gaps such as display
  initialization edge cases or action execution compatibility concerns.

## Current Status

- As of 2026-10-20, TASK012 implements #654/#776 under D016 with real basic
  memory and headless construction; no production mocks in the minimal runner.
  111 distinct cases / 112 profile executions pass after five Important review
  corrections; original RED and cache-expectation failures are retained.
  Gameplay turns, live AI models and phase approval remain unproven. Independent review
  complete; PR #788 ready for human review. #785 merged at 8a899650 and #725 is closed.

- As of 2026-10-20, TASK011 / #725 has eight SHA-256 requirement locks and
  mode-aware checks, with full legacy dependencies preserved separately. Fresh
  core/dev+visual/NLP/CPU-LLM installs and unchanged repeats pass, as do pip
  checks, 85 existing methods and real dependency probes. GPU resolution is
  unexecuted; lightweight runtime/model/platform readiness remains separate.
  Setup and validation are documented in ready PR #785, targeting main. #725
  issue comment/status read-back is verified; issue closure follows merge.

- As of 2026-10-20, D014 accepts TASK010 / #724's MVP contract with the
  original expansive vision preserved. Scenario limits are fixture data;
  optional capabilities and later roadmap work remain intact. D013 confirms
  both presentations. Targets are not implemented or benchmarked guarantees.
  #782 merged to main before #783 merged into its old branch, so a focused
  integration PR was required. #784 subsequently merged as 8d6b254 and closed
  #724; TASK011 now implements the next task, #725, without feature removal.

- As of 2026-10-16, TASK009 verifies merged main 77e4b94 under D009. All 94
  method summaries report success and all three probes exit 0, with one
  lifecycle supervisor completion gap after an execution-server disconnect.
  Its 12-test OK summary and child exits remain intact; only NOT_RUN targets
  were continued once. 390 source hashes unchanged; 42 child artifact sets
  verified. Report: docs/testing/phase-1-post-merge-verification.md.
  On 2026-10-17 UTC, D010 accepts both residual uncertainties and #779 closes
  as unable to reproduce; diagnostics added. The subsequent D011 explicitly
  approves #721 for Linux/offline startup and #721 closes as completed. D012
  separately approves #722 and completes #717 on 2026-10-17 America/New_York.
  Phase 2 is unblocked with #724 next; no new runtime observations.
- As of 2026-10-16, TASK008 implementation is approved under D008. The new
  capture helper and integrations pass 11 new checks, 81 existing methods and
  three probes in the approved window. No failures/errors/skips/timeouts; 42 child
  artifact sets verified. Runtime/dependency behavior and zero-action limit unchanged.
  PR #781 now targets main after #780 merged. Its P2 parent-deadline review
  correction passes 13 focused checks, including two new red-to-green regressions.
  Earlier 81-method/probe evidence is historical; no new application window.
  #779 stays open; its historical cause is unknown. D008 budget consumed.
- As of 2026-10-15, TASK007 synchronized 18 issue bodies, verified original
  bodies/states, and linked native #779 under #721 while retaining blocker
  order. Draft docs PR #780 is published/attached. No issue closure, reopening,
  new runtime work or new human disposition supplied.
- As of 2026-10-15, TASK006's approved bounded investigation is complete.
  51/51 affected methods pass; all 16 diagnostic cases exit 0 with unchanged
  limits. Plot-cache cost is measured, but historical timeout cause remains
  unconfirmed; #779 is not resolved. No production/test/dependency edits.
  Report: docs/testing/issue-779-startup-investigation.md. Other 30/probes and
  full 81-method final verification were not rerun in this cycle.
- As of 2026-10-14, #718 is complete: #706 merged and the issue closed.
- TASK003 / #719 is complete: #777 merged and the issue closed. Post-merge
  verification passes 54/54 tests, no skips/timeouts, plus cold-getter, actual
  Goal fallback and minimal probes.
- As of 2026-10-15, TASK004 / #720 is complete: PR #778 merged at `93079fd`
  and #720 closed. Its 81 pre-merge checks and timing limits remain historical.
- Historical TASK005 verification at 93079fd (superseded by D011 at 77e4b94):
  Cold imports fail two timing subcases; lifecycle/ownership batches exceed
  aggregate deadlines after partial passes. Global-manager (9), named-target
  (6), actions (5), events (5) and GOAP errors (5) pass; minimal exits 0 with
  zero actions. Cache passes 26/27; both affected import orders pass subsequent
  instrumented diagnostics with unchanged limits, without explaining failures.
  Verification is complete with failures; #779 tracks diagnosis. #721/#722
  remain open. dhar174 confirmed Deferred on #721, recorded with evidence and
  verified native blocker #779. Project Decision field update is outstanding
  because Project API access returns HTTP 403. D011 subsequently supersedes
  Deferred for the scoped #721 review; Project Decision now needs Approved.
  No phase-gate approval supplied.
- Existing integration tests remain 9/11 and utility tests 16/26. #773/#774 retain
  those failures. #775/#776 retain missing Door effects and optional spaCy
  initialization limits; full Character/simulation readiness is not established.
- Evidence: `docs/testing/issue-719-graph-initialization-validation.md`.

- TinyVillage is documented as a modular Python simulation project with
  root-level runtime modules and optional advanced AI/LLM features.
- The Memory Bank is in the process of being corrected from stale template
  content into repo-specific, TinyVillage-accurate documentation.
- The current canonical contributor instructions emphasize minimal, reality-
  based changes and conservative test design.

## Known Issues and Limitations

- Documentation may contain inconsistencies across time; newer docs such as the
  root `README.md` and root `AGENTS.md` should generally be prioritized over
  older or archived status notes when they conflict.
- Optional LLM/NLP features require additional dependencies and should degrade
  gracefully when those dependencies are absent.
- Some testing history in the repo shows over-mocking pitfalls, especially
  around memory-related tests; Memory Bank guidance should preserve the
  repository preference for tests that fail on real behavior mismatches.
