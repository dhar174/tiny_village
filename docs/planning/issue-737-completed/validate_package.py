#!/usr/bin/env python3
"""Read-only offline verification of the retained #737 review package."""
import hashlib
import json
from pathlib import Path
import tarfile

from build_preview import assemble
from sync_project import digest, load, validate_value

ROOT = Path(__file__).resolve().parent


def manifest_name(path, root):
    """Keep the shipped manifest independent of the host path separator."""
    return path.relative_to(root).as_posix()


def validate_closeout(snapshot, rows, scope, evidence=None):
    from publication_evidence import verify_prior_publication, timestamp
    actual = {str(p["number"]): p["headRefOid"] for p in snapshot["pull_requests"]}
    if actual != scope["candidate_heads"] or len(actual) != 24 or len(actual) != len(snapshot["pull_requests"]):
        raise ValueError("Reviewed candidate cohort/heads changed; obtain review before rebuilding")
    by_issue = {i["number"]: i for i in snapshot["issues"]}
    for n in (738,741,774,799,800,801,802,803,807,808,809):
        if by_issue[n]["state"] != "OPEN":
            raise ValueError("Open publication/runtime requirement was lost: " + str(n))
    if rows[738]["decision"] != "Pending":
        raise ValueError("Human gate must remain Pending")
    stage = scope["stage"]
    if stage == "publication":
        for n in (737, 806):
            if by_issue[n]["state"] != "OPEN":
                raise ValueError("Open publication requirement was lost: " + str(n))
        if (rows[738]["readiness"] != "Blocked" or rows[737]["readiness"] != "Blocked"
                or 806 not in rows[737]["blocking_issues"]):
            raise ValueError("Human gate/publication blocker is incorrectly unlocked")
    elif stage == "completed":
        inspected_at = verify_prior_publication(evidence, scope, rows)
        for n in (737, 806):
            issue = by_issue[n]
            if issue["state"] != "CLOSED" or issue.get("stateReason") != "COMPLETED":
                raise ValueError("Completed stage requires completed issue #" + str(n))
            if timestamp(issue.get("closedAt")) < inspected_at:
                raise ValueError("Issue closed before the publication receipt: " + str(n))
            if rows[n]["readiness"] != "Tracking":
                raise ValueError("Completed issue must use Tracking: " + str(n))
        if timestamp(by_issue[806]["closedAt"]) > timestamp(by_issue[737]["closedAt"]):
            raise ValueError("Closure order requires #806 before #737")
        if rows[738]["readiness"] != "Human review":
            raise ValueError("Pending human gate must be ready for Human review")
    else:
        raise ValueError("Unsupported completion stage")


def load_stage_evidence(scope):
    if scope["stage"] != "completed":
        return None
    paths = scope.get("publication_evidence", {})
    if set(paths) != {"desired", "plan", "receipt"}:
        raise ValueError("Completed stage requires publication evidence paths")
    evidence = {}
    for key, value in paths.items():
        path = ROOT / value
        if not path.resolve().is_relative_to(ROOT.resolve()):
            raise ValueError("Evidence must be retained inside the frozen package")
        evidence[key] = load(path)
    return evidence


def validate():
    inputs = [load(ROOT / name) for name in ("snapshot.json", "original-roadmap.json", "triage-input.json",
                                            "remediations.json", "approval-records.json", "original-pr-handoffs.json")]
    result = assemble(*inputs)
    preview, report = result["preview"], result["report"]
    if preview != load(ROOT / "project-preview.json"):
        raise ValueError("Retained preview differs from regenerated source inputs")
    if result["matrix"] != load(ROOT / "pr-matrix.json"):
        raise ValueError("Retained PR matrix differs from current retained inputs")
    if list(result["rows_by_number"].values()) != load(ROOT / "issue-matrix.json"):
        raise ValueError("Retained issue matrix differs from current retained inputs")
    if report != load(ROOT / "validation.json"):
        raise ValueError("Retained validation report differs")
    fields = {f["name"]: {"name": f["name"], "type": f["data_type"], "options": dict.fromkeys(f.get("options", []))}
              for f in preview["fields"]}
    if len(fields) != 12 or len(preview["views"]) != 6:
        raise ValueError("Unexpected custom field/view schema")
    seen = set()
    for row in preview["rows"]:
        if set(row["values"]) != set(fields):
            raise ValueError("Incomplete field preview on #" + str(row["number"]))
        for name, value in row["values"].items():
            validate_value(fields[name], value)
        key = row["values"]["Task key"]
        if key in seen:
            raise ValueError("Duplicate task key")
        seen.add(key)
    scope = load(ROOT / "review-scope.json")
    if report["managed_issues"] != scope["managed_issues"] or report["managed_issues"] != 328 + len(inputs[3]):
        raise ValueError("Reviewed issue scope changed")
    validate_closeout(inputs[0], result["rows_by_number"], scope, load_stage_evidence(scope))
    for receipt in load(ROOT / "evidence/probe-receipts.json")["receipts"]:
        source = ROOT / "evidence/probes" / (receipt["probe"] + ".py")
        if hashlib.sha256(source.read_bytes()).hexdigest() != receipt["source_sha256"]:
            raise ValueError("Probe source changed since capture")
        capture = ROOT / receipt["artifact_dir"]
        metadata = load(capture / "run.json")
        expected_outcome = "EXITED_ZERO" if receipt["exit_code"] == 0 else "EXITED_NONZERO"
        if metadata["returncode"] != receipt["exit_code"] or metadata["outcome"] != expected_outcome:
            raise ValueError("Incomplete or mismatched capture")
    members = load(ROOT / "evidence/core-raw-member-hashes.json")
    with tarfile.open(ROOT / "evidence/core-raw.tar.gz", "r:gz") as archive:
        files = archive.getmembers()
        if {f.name for f in files} != set(members) or len(files) != len(members):
            raise ValueError("Incomplete/duplicate core evidence members")
        for member in files:
            if not member.isfile() or hashlib.sha256(archive.extractfile(member).read()).hexdigest() != members[member.name]:
                raise ValueError("Changed core evidence member")
    hashes_path = ROOT / "artifact-hashes.json"
    if not hashes_path.exists():
        raise ValueError("Missing artifact hashes; rebuild a new review package, not an approval")
    if hashes_path.exists():
        hashes = load(hashes_path)
        for name, expected in hashes.items():
            if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != expected:
                raise ValueError("Changed/missing retained artifact: " + name)
        actual = {manifest_name(p, ROOT) for p in ROOT.rglob("*") if p.is_file()
                  and p.name not in ("artifact-hashes.json",) and "__pycache__" not in p.parts
                  and not p.name.startswith(("live-plan", "local-checkpoint")) and p.suffix not in (".lock", ".tmp", ".zip")}
        if actual != set(hashes):
            raise ValueError("Unlisted/missing artifacts: " + str(actual ^ set(hashes)))
    return {"status": "offline_package_verified", "issues": len(preview["rows"]), "prs": len(result["matrix"]),
            "fields": len(fields), "views": len(preview["views"]), "cycles": 0,
            "gate_738": "Pending", "stage": scope["stage"], "project_publication": "prior_publication_verified_final_readback_pending" if scope["stage"] == "completed" else "blocked_not_verified", "preview_digest": digest(preview)}


if __name__ == "__main__":
    print(json.dumps(validate()))
