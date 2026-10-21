# #734 — Backlog triage for review

**The complete triage package is prepared. Issue publication and #734 completion await human review; no issue metadata, assignees, parents, blockers or states have been changed.**

At main `13b8e374d07e7e7ea6f8f562ea6a290fcab967f7`, the snapshot contains
475 total issues, 306 open issues and 37 open PRs. Replaying creation and
closed/reopened events at **2026-10-13T07:30:00Z** reconstructs exactly **261
original open issues**. Only original #654 is now closed. The ledger therefore
contains **307 issues: all 261 originals plus 46 later open issues**. This is
not an estimate of feature completion.

## Review the package

| Artifact | Purpose |
|---|---|
| [Grouped ledger](ledger.md) | Observable acceptance, ownership, disposition, evidence limits and order for every item |
| [CSV](ledger.csv) / [JSON](ledger.json) | Complete sortable records, original checklists, native parents/blockers and source/discussion links |
| [Publication review](publication-review.md) | Exact issue-change policy and seven specific closure proposals |
| [Proposed new issues](proposed-new-issues.md) | Three scoped remediation drafts, not yet created |
| [Audit evidence](audit-report.md) | Six mandatory audits, additional isolated findings, original failures and exact commands |
| [PR handoffs](pr-handoffs.md) | All 37 open PRs at pinned heads; subsequent reconciliation remains separate |
| [Validation receipt](validation.json) | Cohort, field completeness, preserved hierarchy, cycles and retained evidence checks |

Proposed dispositions are **105 implement, 73 verify, 127 defer and 2 duplicate**.
These classify work; they are not approvals, claims that every existing method
is broken, or permission to remove the broader product. Most rows have source
review only and explicitly do not claim executed behavioral verification.
Deferred rows retain their acceptance and activation conditions. D015 continues
to require AI in the final product.

## Decisions requested in the consolidated review

1. Review the ledger, accountable owners and order. GitHub assignees stay as they
   are; unassigned items have dhar174 as ledger owner. Canonical topic links do
   not replace an existing native parent.
2. Review **307 marked issue annotations and 80 additions of missing parents**.
   Existing bodies, assignments, native dependencies and historical parents
   remain intact. The 23 open children under historically closed parents are
   flagged, not silently reparented or reopened.
3. Review narrow completed-issue closures **#355, #482, #594, #596 and #597**.
   Their limits are explicit: queue compatibility, rainy-panel rendering,
   rendering-issue organization, timing collection and frame analysis. They
   do not establish healthy gameplay or measured performance budgets.
4. Specifically review duplicate closures **#351 → #350** and **#382 → #380**.
   Both pairs have identical original bodies; #350/#351 also reference the
   identical review comment. #507/#508 are deliberately separate findings in
   different UI classes despite their matching titles.
5. Review the three separate remediation drafts. Existing #609 owns the social
   reader/decay defect; it does not need another issue. Keep **#602 and #609
   open**, with their actual failing evidence preserved.

Record the eventual approval/changes/defer decision only in
[the single decision file](../../decisions.md). D026 approves execution of this
plan and the review-first/ledger-owner policy; it does **not** approve these
new results. Unapproved proposals remain pending.

## Ordering and boundaries

Finish review, merge this documentation and verify the approved issue updates
before closing #734. Then reconcile utility PRs under **#735** and old
functional-test PRs under **#736**, followed by **#737** for selected PR ownership,
merge order and Project configuration. Native blockers remain authoritative;
proposed execution prerequisites are separate text, not silently added edges.
Handoffs coordinate work and must not become circular sibling dependencies.

Human gate **#738** remains open and requires its own later decision. The new
remediation proposals are Phase 4 work after that gate. They do not become new
Phase 3 implementation requirements. Existing UX, pixel-art and UI decisions
are settled; this work neither repeats those choices nor implements the UI.

Project 10 still returns **HTTP 403**, retained in
[project-access.json](project-access.json). Issue/decision records remain
usable; #737 receives pending synchronization. No old roadmap publisher is run
and no Project field update is claimed.

## Reproduce the catalog checks

```bash
python docs/planning/issue-734/validate_bundle.py
```

This validates the frozen bundle, not a passing runtime. Every published input,
including Markdown review documents and capture logs, must have a matching
SHA-256 entry in `artifact-hashes.json`; missing, unlisted or changed artifacts
fail validation. The manifest itself, the derived `validation.json` output and
interpreter `__pycache__` files are excluded from this input set. All source
anchors are also checked against the recorded baseline commit. Raw issue/PR/
review-origin snapshots, source hashes and capture receipts are retained alongside the ledger.
`build_ledger.py` contains the explicit per-issue curation and can regenerate the
machine ledger/proposed annotations from those frozen inputs; it never writes
to GitHub. The Markdown review copy was rendered from that ledger.

Audit commands and profiles are documented in [audit-report.md](audit-report.md).
The package preserves **12 subprocess attempts**, including setup errors and
real failures. No runtime, dependency, CI, graphics or gameplay implementation
was changed. No whole-suite, live-model, native-platform, gameplay-episode or
release-readiness claim is supplied.
