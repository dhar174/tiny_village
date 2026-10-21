# Independent review and correction

The independent reviewer examined base `13b8e374d07e7e7ea6f8f562ea6a290fcab967f7`
through `131e5bd3e1607f2e2562b3dde6bd91ccd610ce49`, read-only with fresh context.
It found no Critical/Important issue, verified all 85 artifact-manifest hashes
and probe-source hashes, reproduced the builder outputs entirely in memory,
matched all 307 Markdown acceptance blocks to JSON, and sampled about 50
original issues against their bodies/discussions. It confirmed closure and
failure-evidence boundaries, preserved ownership/native relationships and the
three remediation drafts.

## Acceptance-preservation correction

The reviewer noted that #530's curated acceptance omitted the original
named-constant maintainability improvement. The author treated that as required
scope preservation before publication, rather than leave a satisfiable criterion
that could retain the original magic number. The ledger, CSV, human copy and
exact proposed annotation now explicitly require a named constant or supported
configuration value, documented units/default and boundary behavior.

A scratch assertion against the actual original ledger first fails with the
omitted criterion; the corrected assertion passes for canonical acceptance,
Markdown and proposed annotation. The full catalog validator passes afterward.
No runtime threshold/code changed and no failed runtime probe was retried to
claim a source fix.

## Deferred minor improvements

- Compare all CSV/Markdown fields with canonical JSON in the validator, beyond
  current ID/structural checks. Full artifact-hash validation is now implemented
  in the review correction below.
  Independent/manual checks verify parity and hashes for this exact revision;
  this is a future validation enhancement, not evidence of current corruption.
- Retain a reusable Markdown renderer alongside the builder. The current human
  copy is verified against all canonical acceptance blocks, but the one-shot
  rendering script is not part of the shipped bundle.

Neither minor requires runtime work or changes the review-first publication
boundary. The corrected package is for human review, not phase approval.

## PR review correction — full artifact manifest

[Codex review comment](https://github.com/dhar174/tiny_village/pull/797#discussion_r4342402200)
identified that the documented command checked only two snapshot hashes, leaving
other retained publication artifacts unchecked. Disposable-copy probes confirmed
that ten altered/incomplete bundles still passed the prior validator.

The validator now checks every declared SHA-256 and requires complete file-set
coverage before reading the catalog. A changed document/log, missing artifact,
omitted manifest entry, unlisted file, incorrect digest or absent manifest fails
with a filename or coverage diagnostic. The two root manifest/output files and
bytecode caches are explicitly excluded, as documented in the package README.
The existing catalog checks and retained runtime failures remain separate.

The [retained before/after mutation checks](evidence/manifest-validation-review.json)
contain the exact scratch driver and per-case output. All 11 corrected scenarios
meet their expectations: one pristine bundle passes and ten altered/incomplete
copies fail with the expected artifact/coverage diagnostic. The original ten
false passes remain recorded. The full original catalog validation also passes;
no runtime capture or published-issue disposition was changed.
