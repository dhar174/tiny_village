# Phase 2 engineering review for #731

Date: 2026-10-21 (America/New_York). Preparation by Codex; **Approved by
dhar174 under D024** after the PR #795 command-replay correction at `17bed6d`.
D023 approves the preceding design planning/review. #732 remains a separate
phase gate; its approval is not supplied here.

## Recommendation and reviewed revisions

Recommend accepting the Phase 2 engineering foundation within the documented
Python 3.12/Linux x86-64 scope. No required new engineering remediation was
identified by this bounded source/evidence review; no framework rewrite is
needed. The maintainer accepted this scoped recommendation under D024,
authorizing #731 closure; #732 is not approved by that decision.

Reviewed main: `752bc2efaacfadfae2c32d6f3df4ce9241087fe2` (#794).
The #795 preparation head was `6bc999dac7fe289c557805dd79f26eaff9737dd3`.
Runtime, requirements, tests and workflows are unchanged from the verified
#726 rollout at `ac0e25d3674efb03440d2ccce26d7b009d5e82f3`: the git diff between
that revision and reviewed main contains only docs/memory-bank changes.
All fourteen source hashes in the retained #726 local receipt still match,
including the runner, workflows and eight locks. No runtime test cycle or
dependency installation was added for this documentation/evidence review.

## Acceptance mapping

| #731 review requirement | Finding and evidence | Limits / retained downstream work |
| --- | --- | --- |
| Import and renderer boundaries | Controller annotations defer evaluation; Pygame/map imports and display/font ownership sit behind renderer setup. Headless absence is intentional; the bounded runner uses real Characters, graph, actions, events and basic memory. #654's corrected lifecycle evidence and current core/visual CI cover the supported paths. | This is not a guarantee that every legacy import path is lightweight or that the planned UI is implemented. |
| Dependency/version profiles and package integrity | Default requirements select the hashed core lock. Eight explicit Linux/Python 3.12 profiles preserve feature dependencies; the compiler fixes its resolver version/target and constrains optional profiles to core. CI installs fresh environments with hashes/binary wheels and runs pip check/package audits. | Native Windows/macOS, GPU installation and live pretrained inference remain outside this evidence. |
| Configuration and explicit state/error contracts | Minimal mode explicitly selects headless/basic/no-LLM configuration and propagates startup failures. Visual/headless combinations fail before optional imports. Memory rejects unknown modes and basic semantic-index loading; capabilities distinguish disabled, not initialized, ready and unavailable. | The current CLI smoke contract is not the future #724/#658 economic-episode contract. Existing partial save/load does not meet Phase 7 persistence requirements. |
| AI/model-service initialization | Memory loads dependencies/services on demand and raises capability errors with causes. Strategy preserves requested AI separately from readiness and records failed loads. Basic reads/status/restoration do not initialize AI; current tests isolate external model loading instead of fabricating world objects. | Package presence/offline boundary checks do not establish model quality or inference readiness. Final-product AI remains required by D015; live validation belongs in #585. |
| Behavioral validation | Current hosted core/dev/visual reports contain 40/20/68 cases and three actual CLI probes, all PASS with zero skips/failures/errors. Existing #726 NLP/CPU-LLM evidence adds two cases each on the unchanged runtime. | Basic startup/event/memory and real SDL-dummy lifecycle checks execute no economic gameplay acceptance. #751/#658 and later native human checks remain required. |
| CI result reporting and lifecycle | Runner rejects empty, skipped, XFAIL/XPASS, incomplete/missing and malformed probe reports; bounded process capture retains terminal/child evidence and cleans owned sessions. Matrix does not stop at the first failed profile; uploads run on failure; the required aggregate demands success. Actions are pinned and checkout does not retain credentials. | Retention is 30 days; durable summaries/digests must remain in repository evidence. The minor redundant runner assertion documented in #726 is not missing behavior. |
| Required-check enforcement and cost scope | The read-back ruleset is active, requires `Python CI required` from GitHub Actions app 15368 and enforces up-to-date branches. Manual NLP/CPU-LLM workflows remain manual under D017. No new ML run was dispatched for this review. | Existing bypass policy remains unchanged. Project authorization is a separate administrative limitation. |
| Architecture and scoped remediation | Shared graph ownership, existing module interfaces and production AI integrations remain the foundation. D014/D015 preserve the expansive product. Design/persistence/action/episode work remains assigned to the existing ordered issues. | Earlier legacy integration/utility/Door limits remain tracked by #773/#774/#775 under D002; this review does not relabel those as fixed or require a wholesale framework migration. |

## Verified hosted evidence

Downloaded the three archives from
[run 39648188081](https://github.com/dhar174/tiny_village/actions/runs/39648188081)
for #795 and checked every ZIP SHA-256 against GitHub's published artifact
digest. Structured results total **128 cases + three CLI probes**, zero skips,
failures, errors, expected failures, unexpected successes or failed batches.
These are existing hosted executions, not tests newly executed by this review.

The run's PR head is `6bc999d`; its structured reports correctly identify the
executed merge revision `b54bfc6cee5d34192be6fc71c4f274a2444a1a3b`. The API
confirmed parents `752bc2e` and `6bc999d`. Do not substitute the PR head for the
executed revision. The first fetch of that merge object failed when credentials
stopped working. After repository access recovered, the fetch succeeded and
direct comparison with `ac0e25d` confirmed no non-document changes. The failed
attempt remains recorded separately from that successful source verification.

The [#726 rollout receipt](evidence/issue-726-hosted-rollout-results.json)
separately retains **132 cases + three probes** across all five profiles at
`ac0e25d`. Only four cases are from its manual NLP/CPU-LLM run; they are not
counted as re-executed in the current three-profile run. Original failures,
review corrections and the deferred minor remain in the existing validation
reports. Exact current artifact digests, counts, locks and rule read-back are
retained in the [review receipt](evidence/issue-731-engineering-review.json).

## Administrative handoff and human decision

Before authentication failed, prerequisites #725/#654/#726/#722 were read as
closed and #731/#732 as open. The configured Project API returned HTTP 403,
“Resource not accessible by integration.” Later both normal GitHub access and
the configured project credential returned HTTP 401, “Bad credentials.” The
environment's ready binding observation did not establish API authorization.

Repository access subsequently recovered. D023's approval commit was pushed to
#795, and #727/#728/#729/#730 were updated and closed as completed; read-back
verified their states and immutable decision links. Project 10 still returned
403, so its fields remain unsynchronized. No credentials were printed, replaced
globally or added to repository files.

The maintainer explicitly approved the findings within their retained limits
under D024 in docs/decisions.md. Close #731 under that approval; no new required
remediation was requested. #732/#723 stay open pending the separate Phase 2
decision. No product implementation or downstream phase unlock follows from
this engineering review alone.
