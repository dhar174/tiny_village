# #737 authenticated publication implementation plan

Approval: D033 in docs/decisions.md. Base: merged main5f2bf989 (#811).

1. Preserve historical packages; create a publication package with optional
   read-only inspection --receipt PATH and evidence-backed publication/completed
   validators. Keep runtime receipts/checkpoints outside the frozen package.
2. Test receipt integrity, bindings, incomplete readback, drift, premature closure,
   stage transitions, final gate readiness and existing54 regression contracts.
   Independently review and merge the tooling PR before freezing a fresh source
   snapshot/ZIP against the actual merged main revision.
3. Owner validates locally and runs read-only --prepare. Review the concrete plan
   and digest; preserve human edits, approvals, assignments and unrelated items.
   Apply only that approved delta in a quiet edit window, then inspect in a new
   process and independently recompute the returned receipt against the plan.
4. After verified publication only, complete/close806 then737 with evidence.
   Regenerate all332 rows, review/apply the final readiness delta and independently
   inspect again:806/737Tracking,738Human review with DecisionPending.
5. Publish evidence; after any later merge do a fresh read-only comparison against
   actualmain. Historical verified receipts retain their exact original revisions.
6. Prepare and conduct738's human walkthrough (734–737 evidence, PR ownership,
   runtime gaps, design records, failures and limits). Present Approved / Changes
   requested / Deferred with recommendation; no automatic approval or Phase4 work.

No runtime, dependency lock, UI or paid CI service changes. Failure or missing
owner-authenticated evidence stops only the dependent publication/closure work.
