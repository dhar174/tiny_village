# TASK022 — Consolidate utility PRs (#735)

**Status:** In Progress — implementation verified; awaiting #703 review/merge  
**Added:** 2026-10-21  
**Updated:** 2026-10-21

## Request and decision

After approving/merging #797, dhar174 requested forward progress and selected
“Keep #703; consolidate useful tests from #714 (recommended).” See D028 in the
single decision file, linked from the validation report. Do not merge #703 or
approve phase gate #738 automatically.

## Plan

1. Inspect current PR heads, differences, reviews and CI; reproduce numeric
   conversion failure and compare each existing candidate suite.
2. Rebase #703 onto merged main, keep one utility test module and import only
   useful behavior-oriented coverage from #714. Document retained numeric,
   interpolation and legacy score-scaling contracts without redesigning them.
3. Include the module in the existing core component selection, validate with
   the locked core environment and existing CI-runner tests, retain evidence.
4. Publish with a guarded lease to the selected existing branch; verify head,
   crosslink #735 and both PRs, and close #714 as superseded under D028 only
   after the selected result exists remotely. #735 closes after merge.

## Progress

- Original heads c1528b5 (#703), f99d49f9 (#714); both behind main 2c3bc57.
  Candidate suites pass 10/12 tests, and main reproduces two/one TypeError
  errors. All four bounded subprocess captures retained.
- Rebase complete; source commit 80df5fc combines the two PRs' useful coverage
  into 11 methods, documents existing contracts and adds the module to the
  existing core profile. No changes to interpolation/scaling behavior or locks.
- Local validation: 51 core tests + 3 CLI probes, 18 CI-runner tests pass.
  No unexpected failures, errors, skips or timeouts. Expected fault injection
  within runner tests is retained, not omitted from the evidence.
- [Validation and handoff](../../docs/testing/issue-735-utility-consolidation.md).
  #734 publication is completing separately; #735 remains open until merge.
