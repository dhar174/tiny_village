# Phase 2 gate #732 readiness

Prepared 2026-10-21 (America/New_York). **Human phase disposition pending.**
The maintainer has approved the design planning/review under D023 and the
engineering review under D024. Those decisions do not themselves approve #732.

## Recommendation and meaning

Recommend **Approved** for Phase 2's actual scope: freeze the MVP contract,
reproducible supported dependencies/runtime/CI and reviewed design foundations.
Approval would complete parent #723 and unlock Phase 3 #733, starting with
#734's canonical backlog triage. It would not authorize arbitrary UI or gameplay
implementation, remove the wider village plans, or substitute for later gates.

## Work and evidence

| Required work | Completion / evidence | Scope retained |
| --- | --- | --- |
| Previous gate #722 | Closed; D012 records explicit Phase 1 approval. | Unknown historical timeout cause and retained supervisor evidence gap remain accepted limitations, not proven fixes. |
| #724 MVP contract | Closed; D013/D014 and the merged #784 contract. | Headless plus Pygame presentation, bounded fixture, support/budget targets; expansiveness preserved. Targets are not runtime achievements. |
| #725 dependencies | Closed; eight hashed profiles and locked resolver/target, install/pip-check evidence. | Python 3.12/Linux x86-64 supported foundation; no inferred native Windows/macOS/GPU readiness. |
| #654/#776 lightweight runtime | Closed; real graph/Characters/events/basic memory and corrected renderer/service boundaries. | Startup/event/memory smoke explicitly exercises zero economic gameplay actions; final-product AI remains required. |
| #726 supported-profile CI | Closed; hosted rollout 132 cases + three probes, verified artifact digests and effective required-check rules. | Core/dev/visual required; NLP/CPU-LLM manual under D017; live inference remains separate. |
| #727/#728/#729 design planning | Closed under D023; complete UX, pixel-art and UI specifications, D018–D021 and consolidated review. | Art/font/palette choices are trials; native font-fit/interaction/performance and production audit remain later implementation/release evidence. |
| #730 combined design review | Closed under D023; no required new design rework identified. | Review of planned behavior, not an executed human native-display test. |
| #731 engineering review | Approved under D024; closure and decision link are verified in the issue handoff. | Review report/receipt retain exact source revisions and limitations; no framework rewrite or new blocking remediation required. |

The [engineering review](issue-731-engineering-review.md) and
[receipt](evidence/issue-731-engineering-review.json) verify unchanged runtime,
tests, workflows and locks since the #726 rollout, fourteen prior source hashes,
and nineteen reviewed-source hashes. Existing #795 hosted artifacts contain
128 cases + three probes, all PASS with three ZIP digests verified. Its command
replay correction at `17bed6d` passed CI; exact download commands were replayed
and their SHA-256 values matched. Earlier manual ML counts remain separate.
No new runtime test or manual ML cycle was launched for this gate preparation.

PR #795 remains open for the approval-record/report publication; its head and
checks are reviewed, and no automatic merge is authorized. Decision records on
the branch and closed issues use immutable commit links. This report does not
pretend that an open PR has merged or that its decisions are already on main.

## Remaining limits and phase decision

Project 10 field synchronization still receives HTTP 403. The issue states and
single decision file are authoritative; Project Decision/Status fields must not
be reported as updated. The limitation is administrative, not missing runtime
validation, and remains visible for the gate reviewer.

The original legacy integration/utility/Door limitations retain their existing
issue-backed phasing (#773/#774/#775, D002). Native support/live inference,
atomic persistence, economic action/episode correctness and actual UI/human
release verification remain in their existing later tasks. Optional testing
dependencies do not make final-product AI optional (D015).

The human may choose **Approved**, **Changes requested**, or **Deferred**.
Any required new remediation must have its own issue/blocker; no such new
remediation was identified in this preparation. Record the selected disposition
in docs/decisions.md before closing #732/#723 or unlocking Phase 3. Until then,
both remain open and downstream work is not started.
