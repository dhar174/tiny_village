# Five PR dispositions for human review

These are recommendations, not applied decisions. All five PRs remain open at
their captured heads. D031 authorizes investigation and coordination; closure,
rewriting and later implementation still require the stated review/activation.

| PR | Recommendation | Evidence and retained work |
|---|---|---|
| #701 | Supersede as addressed by the merged owned-cache implementation. | #720/#778 replaced the shared temporary cache with validated owned directories; #654/#788 retained that integration. All 27 maintained cache tests pass on base4821378. The old patch conflicts and its reproduction deletes a shared test directory and executes fixed source lines; do not transplant it. |
| #704 | Supersede as addressed by the merged dependency-profile/runtime changes. | #725/#785 and #654/#788 provide mode-aware imports without FAISS/Pygame in minimal mode. A core-only probe confirms both absent and dependency checking succeeds; the canonical core run passes72 tests+3 CLI probes. Semantic FAISS functionality remains final-product scope. |
| #709 | Split useful assertions into #807; supersede only after the maintained replacement is verified and reviewed. | Real-character selection/filtering/independence passes. A controlled real-provider load failure leaves requested=True and status=disabled, reproducing an unresolved capability contract; the original MagicMock non-None assertions would not prove it. #807 is under #585, after #584/#768. |
| #711 | Retain the diagnostic intent in #808; supersede after its scoped replacement is verified and reviewed. | Real ContextManager still returns empty context with no diagnostics when both calculations fail. The two generic logger calls do not define caller behavior or establish safe recovery. #808 is under #137, after #743/#746; #137 is blocked by it. No demonstrated security vulnerability is inferred from the PR label. |
| #715 | Split useful graph-query assertions into #809; supersede only after maintained coverage is verified and reviewed. | The corrected real shared-graph/Location probe passes absent-source and0/1/2-hop checks without graph-size changes. The PR's mock-based fallback does not establish NetworkX-free parity. #809 is under #222, after #738/#799. |

Available alternatives for each PR are **retain/rebase**, **split**,
**supersede**, or **defer**. Retaining #701/#704 requires identifying a still unmet
requirement against the current implementation. Retaining #709/#711/#715 as
carriers requires a fresh scoped implementation plan and validation at their
owning issue's activation; a conflict-free old patch is insufficient. Deferral
keeps the contribution open and tracked without treating its requirement as done.

No closure should cite a future replacement as already merged. If a head changes,
recheck it before applying an approved disposition. Credit the original PR and
author in any replacement. Preserve existing assignees and native relationships.

## Executed evidence and its limits

- `evidence/core-summary.json`:72 core tests,3 CLI probes, no failures/errors/skips.
- `evidence/probe-receipts.json`:7 bounded captures, each with a20-second
  diagnostic deadline and25-second parent deadline. Four successful probes:
  cache27 tests, dependency checking, real helper selection, and the corrected
  isolated-location distance probe. Two unresolved runtime-criterion failures:
  helper capability reporting and silent goal/needs failures. The third nonzero
  capture is a retained invalid distance-fixture assumption, not a query defect.
- The original distance probe assumed three controller Characters were unlinked;
  the controller had already connected them. The corrected probe adds three real
  isolated Locations to the existing authoritative WorldState, preserving all
  pre-existing edges. Both captures remain available.
- Existing startup job-catalog diagnostics remain #787. These probes neither fix
  them nor establish gameplay, live inference, native Windows or renderer readiness.

All five source patches are retained in `evidence/pr-*-files.json`. This is a
technical review package, not five submitted GitHub reviews or human approvals.
