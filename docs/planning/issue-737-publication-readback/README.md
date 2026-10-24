# Authenticated Project 10 readback — 2026-10-23 America/New_York

The owner ran the frozen publisher from artifact commit a33f51a297f7864c0ede9e686b0416a5b4fe87af against merged main dc9fcc023b4ee37b3efd579195caaab80993c64a. The complete uploaded files were independently parsed and verified using the merged package's `verify_receipt` function. Their exact uploaded bytes are retained here.

- Inspection observed at 2026-10-24T02:43:39.646747+00:00 (October 23, 10:43 PM America/New_York).
- Project: https://github.com/users/dhar174/projects/10; authenticated node ID `PVT_kwHOAE_6m84Bm2UC`.
- Desired digest: `4d77d37afc2fe8ee547926194a88eba4c36d04ad49535e6933eeb16c24fd2d75`.
- Plan approval digest: `10db84806f23ad0e8d52a5f7fc723aae9835108873c2f64f4971f3a9381db1d3`.
- Plan SHA-256: `db5028bcac1b35b35681caa2a47af4813a20c0ab1baf6475d2ca51f3a40c075d`; 358,987 bytes.
- Inspection SHA-256: `f01b40e9874c68e044a635aa31da158ccdbba5b74eff62d746bdfb246909e9cc`; 389,472 bytes.
- All 332 managed rows, 12 custom fields and six delivery views match. All 333 issue guards, 24 pinned candidate PR heads/states and main match both inspection source reads. A fresh coordinator repository-source recheck also matches.
- Recomputed plan operations: zero. Recomputed remaining differences: zero. Approved-result/unrelated-state preservation: true. GitHub mutations performed by inspection: zero. No apply command or checkpoint was necessary.

Exact command reported by owner:

```powershell
python sync_project.py --inspect --plan ..\publication-evidence\publication-plan.json --receipt ..\publication-evidence\publication-inspect.json
```

Reported terminal result:

```text
{"status": "verified", "source_matches": true, "remaining_operations": 0, "github_mutations_performed": 0}
INSPECT_EXIT_CODE=0
```

The owner supplied these original files and confirmed their live origin. JSON receipts are observations, not signed attestations. This verifies the initial configured board state and satisfies the conditional publication prerequisite under D033. It does not claim new board writes, completed issue closures, final readiness publication or human approval of #738. At inspection, #806/#737/#738 were all OPEN and #738 remained Blocked/Pending. The separately authorized #806 then #737 closeout and final readiness readback follow this evidence.
