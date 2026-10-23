# Approved #737 closeout implementation plan

Approval: D032 in docs/decisions.md. Base: merged #810/main10569c0.

1. Recheck exact candidate heads and merged replacement evidence; reopen #737
   while its native #806 publication blocker remains open.
2. Comment and close #701/#704 unmerged as superseded. Annotate approved split
   work on #709/#711/#715 and #807/#808/#809, preserving existing assignments,
   heads and phase order. Retain before/after receipts.
3. Keep #810's original package unchanged. Create a separately refreshed portable
   package/matrix for all332 issues and the reviewed24 candidate heads. Reject
   substitutions/drift; explicitly exclude this closeout PR from candidate input.
4. Validate scope, ordering, human-gate boundaries, interrupted/idempotent apply,
   and Windows-independent manifest paths. Record actual results and historical
   evidence provenance; obtain independent final technical review.
5. Publish one focused closeout PR and deliver local PowerShell instructions.
   Authenticated read-only prepare must produce a concrete live plan/digest for
   review before apply; require an edit window because there is no remote CAS.
6. After actual publication/readback only: close806, document acceptance and
   close737; prepare/review/publish final readiness and independently read back.
   Gate738 remains Pending; human walkthrough is separate and precedesPhase4.

No runtime API, dependency lock, UI, future-phase code or paid CI changes.
External authorization failures remain explicit blockers, not completion.
