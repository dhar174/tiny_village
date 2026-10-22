#!/usr/bin/env python3
"""Build the #737 review package from retained read-only inputs; no API writes."""
import heapq
import gzip
import json
from pathlib import Path

from sync_project import digest, issue_signature, unique, load

ROOT = Path(__file__).resolve().parent
URL = "https://github.com/dhar174/tiny_village"


def workstream(row):
    topic = (row or {}).get("canonical_topic", "runtime").lower()
    for words, result in [(('test',), 'Testing'), (('memory',), 'Memory'), (('llm', 'prompt'), 'LLM'),
                          (('econom', 'job', 'item'), 'Economy'), (('persist', 'save'), 'Persistence'),
                          (('aesthet', 'art'), 'Aesthetics'), (('accessib', 'input'), 'Accessibility'),
                          (('ui', 'render', 'panel'), 'UI'), (('doc',), 'Documentation'),
                          (('social', 'relation', 'narrative'), 'Social'), (('tool', 'architecture'), 'Engineering')]:
        if any(word in topic for word in words):
            return result
    return "Runtime"


def assemble(snapshot, roadmap, ledger, remediations, approvals, handoffs):
    issues = unique(snapshot["issues"], "number", "source issue")
    managed = set(snapshot["managed_numbers"])
    if managed - issues.keys():
        raise ValueError("Missing managed issues: " + str(managed - issues.keys()))
    original = unique(roadmap["items"], "number", "roadmap issue")
    by_key = unique(roadmap["items"], "key", "roadmap key")
    triage = unique(ledger, "number", "triage issue")
    extra = unique(remediations, "number", "remediation")
    human = {n for n, i in original.items() if i["kind"] in ("Phase gate", "Human review")}
    approved = approvals["approved"]
    if "738" in approved:
        raise ValueError("#738 approval is outside this #737 package")
    if any(int(n) not in original for n in approved):
        raise ValueError("Approval record references a non-roadmap issue")
    rows = {}
    for n in sorted(managed):
        issue, old, row, remediation = issues[n], original.get(n), triage.get(n), extra.get(n)
        phase = old["phase"] if old else remediation["phase"] if remediation else row["execution_phase"] if row else 4 if 799 <= n <= 803 else None
        if phase is None:
            raise ValueError("Missing phase for #" + str(n))
        native = sorted(i["number"] for i in issue["blockedBy"]["nodes"])
        planned = []
        if old:
            planned += [by_key[k]["number"] for k in old["depends_on"]]
        elif row:
            planned += row.get("proposed_execution_prerequisites", [])
        elif 799 <= n <= 803:
            planned += [738]
        if old and old["kind"] in ("Program", "Phase"):
            planned = []
        rows[n] = {"number": n, "node_id": issue["id"], "title": issue["title"], "url": issue["url"],
                   "phase": phase, "kind": old["kind"] if old else "Task", "task_key": old["key"] if old else "ISSUE-" + str(n),
                   "scope": old["scope"] if old else remediation["scope"] if remediation else "Post-MVP" if phase == 9 else "MVP",
                   "priority": old["priority"] if old else remediation["priority"] if remediation else "P2",
                   "workstream": old["workstream"] if old else remediation["workstream"] if remediation else workstream(row),
                   "native_parent": (issue["parent"] or {}).get("number"), "native_prerequisites": native,
                   "planned_activation": sorted(set(planned) - set(native)),
                   "actual_assignees": [a["login"] for a in issue["assignees"]["nodes"]],
                   "accountable_owner": (row or {}).get("accountable_owner", "dhar174"),
                   "technical_reviewer": "Codex", "human_acceptance": "dhar174",
                   "decision": approved.get(str(n), {}).get("decision", "Pending" if n in human else "Not applicable"),
                   "candidate_prs": [], "acceptance": (row or {}).get("acceptance", (remediation or {}).get("body", "Preserve original issue acceptance."))}
    dependencies = {n: set(i["number"] for i in issue["blockedBy"]["nodes"]) for n, issue in issues.items()}
    for n, row in rows.items():
        dependencies[n].update(row["planned_activation"])
    missing = set().union(*dependencies.values()) - issues.keys() if dependencies else set()
    if missing:
        raise ValueError("Missing prerequisite issues: " + str(missing))
    reverse = {n: [] for n in issues}
    for n, deps in dependencies.items():
        for dep in deps:
            reverse[dep].append(n)
    remaining = {n: len(deps) for n, deps in dependencies.items()}
    def sort_key(n):
        row = rows.get(n)
        if not row:
            return (-1, n, n)
        position = original[n]["sequence"] if n in original else row["phase"] * 1000 + 500
        return row["phase"], position, n
    ready = [(sort_key(n), n) for n, count in remaining.items() if count == 0]
    heapq.heapify(ready)
    order = []
    while ready:
        _, n = heapq.heappop(ready)
        order.append(n)
        for successor in reverse[n]:
            remaining[successor] -= 1
            if remaining[successor] == 0:
                heapq.heappush(ready, (sort_key(successor), successor))
    if len(order) != len(issues):
        raise ValueError("Dependency cycle involving " + str([n for n, count in remaining.items() if count]))
    counters = {}
    for n in order:
        if n not in rows:
            continue
        row = rows[n]
        phase = row["phase"]
        if row["kind"] in ("Program", "Phase"):
            sequence = phase * 1000
        elif row["kind"] == "Phase gate":
            sequence = phase * 1000 + 900
        else:
            counters[phase] = counters.get(phase, 0) + 1
            if counters[phase] >= 900:
                raise ValueError("Phase sequence capacity exceeded")
            sequence = phase * 1000 + counters[phase]
        row["sequence"] = sequence
        deps = dependencies[n]
        blocked = [d for d in deps if issues[d]["state"] != "CLOSED" or (d in human and approved.get(str(d), {}).get("decision") != "Approved")]
        if row["kind"] in ("Program", "Phase") or (issues[n]["state"] == "CLOSED" and (n not in human or row["decision"] == "Approved")):
            readiness = "Tracking"
        elif row["scope"] in ("Optional", "Post-MVP"):
            readiness = "Deferred"
        elif blocked:
            readiness = "Blocked"
        elif n in human:
            readiness = "Human review"
        else:
            readiness = "Ready"
        row["readiness"], row["blocking_issues"] = readiness, sorted(blocked)
    ordering_conflicts = []
    for n, row in rows.items():
        for dep in dependencies[n]:
            if dep in rows and rows[dep]["sequence"] >= row["sequence"] and row["kind"] not in ("Program", "Phase"):
                ordering_conflicts.append({"issue": n, "prerequisite": dep, "reason": "phase/sequence contradicts dependency"})
    if ordering_conflicts:
        raise ValueError("Ordering conflicts require review: " + json.dumps(ordering_conflicts))
    handoff = {p["number"]: p["handoff_issue"] for p in handoffs}
    direct_owners = {701: 737, 704: 737}
    for r in remediations:
        if r["key"].startswith("R737-") and r["key"].split("-")[-1].isdigit():
            direct_owners[int(r["key"].split("-")[-1])] = r["number"]
    matrix = []
    for pr in snapshot["pull_requests"]:
        number = pr["number"]
        owner = direct_owners.get(number, handoff.get(number))
        if owner not in rows:
            raise ValueError("Missing managed PR owner: " + str(number) + " → " + str(owner))
        rows[owner]["candidate_prs"].append(number)
        rollup = pr["commits"]["nodes"][0]["commit"]["statusCheckRollup"]
        checks = rollup["contexts"]["nodes"] if rollup else []
        required = [c for c in checks if c.get("name", c.get("context")) == "Python CI required"]
        row = rows[owner]
        matrix.append({"number": number, "url": pr["url"], "title": pr["title"], "head_sha": pr["headRefOid"],
                       "head_ref": pr["headRefName"], "base_ref": pr["baseRefName"], "base_sha_reported": pr["baseRefOid"],
                       "snapshot_main_sha": snapshot["main_sha"], "is_draft": pr["isDraft"],
                       "mergeable": pr["mergeable"], "merge_state": pr["mergeStateStatus"],
                       "actual_assignees": [a["login"] for a in pr["assignees"]["nodes"]],
                       "actual_review_requests": pr["reviewRequests"]["nodes"], "reviews": pr["reviews"]["nodes"],
                       "closing_issue_references": [i["number"] for i in pr["closingIssuesReferences"]["nodes"]],
                       "owner_issue": owner, "original_handoff": handoff.get(number), "phase": row["phase"],
                       "sequence": row["sequence"], "accountable_owner": "dhar174", "technical_reviewer": "Codex",
                       "human_acceptance": "dhar174", "native_prerequisites": row["native_prerequisites"],
                       "planned_activation": row["planned_activation"], "blocking_issues": row["blocking_issues"],
                       "checks": checks, "required_check": required,
                       "validation_status": "historical head checks only; current-main integration not executed",
                       "refresh_policy": "After owning-issue activation and prerequisite merges, refresh the retained carrier against latest main; rerun applicable supported profiles and exact issue acceptance, then re-review the new head. No rewrite/merge approval from this matrix.",
                       "required_validation": row["acceptance"], "files": pr["files"]["nodes"]})
    preview_rows = []
    for n in sorted(rows, key=lambda n: (rows[n]["sequence"], n)):
        row = rows[n]
        native = "\n".join(URL + "/issues/" + str(d) for d in row["native_prerequisites"])
        activation = "\n".join(URL + "/issues/" + str(d) for d in row["planned_activation"])
        depends = ("Native prerequisites:\n" + native if native else "")
        if activation:
            depends += ("\n" if depends else "") + "Planned activation (not added native edges):\n" + activation
        values = {"Phase": "Program" if row["phase"] == 0 else f"{row['phase']:02d} " + roadmap["phases"][str(row["phase"])]["name"],
                  "Sequence": row["sequence"], "Priority": row["priority"], "Kind": row["kind"], "Workstream": row["workstream"],
                  "Scope": row["scope"], "Readiness": row["readiness"], "Decision": row["decision"],
                  "Depends on": depends or None, "Review owner": "dhar174", "Candidate PRs": "\n".join(URL + "/pull/" + str(p) for p in sorted(row["candidate_prs"])) or None,
                  "Task key": row["task_key"]}
        preview_rows.append({"number": n, "node_id": row["node_id"], "issue_state": issues[n]["state"], "values": values})
    guards = set(managed)
    todo = list(managed)
    while todo:
        for dep in dependencies[todo.pop()]:
            if dep not in guards:
                guards.add(dep)
                todo.append(dep)
    views = [{"name": v["name"], "layout": v["layout"].upper() + "_LAYOUT", "filter": v["filter"],
              "group_by": v.get("group_by", []), "vertical_group_by": v.get("vertical_group_by", []),
              "sort_by": [[name, direction.upper()] for name, direction in v["sort_by"]]} for v in roadmap["project"]["views"]]
    preview = {"schema_version": 1, "project_url": "https://github.com/users/dhar174/projects/10", "fields": roadmap["project"]["fields"],
               "views": views, "rows": preview_rows,
               "source_guard": {"main": snapshot["main_sha"], "issues": {str(n): issue_signature(issues[n]) for n in sorted(guards)},
                                "prs": {str(p["number"]): {"head": p["headRefOid"], "state": "OPEN"} for p in snapshot["pull_requests"]}}}
    report = {"managed_issues": len(rows), "inventoried_prs": len(matrix), "direct_handoff_prs": sorted({701,704,709,711,715} & {p['number'] for p in matrix}), "source_issues": len(issues),
              "native_edge_count": sum(len(r["native_prerequisites"]) for r in rows.values()),
              "planned_activation_edge_count": sum(len(r["planned_activation"]) for r in rows.values()),
              "cycles": 0, "ordering_conflicts": ordering_conflicts, "source_guard_issues": len(guards),
              "project_readback": "not_verified", "github_mutations_performed_by_builder": 0,
              "preview_digest": digest(preview), "gate_738": rows.get(738, {}).get("decision", "not in fixture")}
    return {"preview": preview, "matrix": matrix, "report": report, "rows_by_number": rows}


def main():
    read = lambda name: load(ROOT / name)
    result = assemble(read("snapshot.json"), read("original-roadmap.json"), read("triage-input.json"),
                      read("remediations.json"), read("approval-records.json"), read("original-pr-handoffs.json"))
    for name, value in (("project-preview.json", result["preview"]), ("pr-matrix.json", result["matrix"]),
                        ("issue-matrix.json.gz", list(result["rows_by_number"].values())), ("validation.json", result["report"])):
        if name.endswith(".gz"):
            with gzip.open(ROOT / name, "wt", encoding="utf-8") as stream:
                json.dump(value, stream, indent=2)
        else:
            (ROOT / name).write_text(json.dumps(value, indent=2) + "\n")
    lines = ["# Remaining PR ownership and execution order", "", "Snapshot main: `" + read("snapshot.json")["main_sha"] + "`.", "",
             "Accountable owner: dhar174; technical reviewer: Codex; human acceptance: dhar174. Existing GitHub assignments/review requests are preserved. A role here is not a submitted GitHub review or a merge approval.", "",
             "| PR | Owning issue | Phase / sequence | Head | Merge state | Required check at captured head |", "|---|---|---|---|---|---|"]
    for row in sorted(result["matrix"], key=lambda r: (r["sequence"], r["number"])):
        check = ", ".join(c.get("conclusion", c.get("state", "unknown")) for c in row["required_check"]) or "Absent"
        lines.append(f"| [#{row['number']}]({row['url']}) | #{row['owner_issue']} | {row['phase']} / {row['sequence']} | `{row['head_sha'][:12]}` | {row['merge_state']} | {check}; integration with current main unverified |")
    lines += ["", "Full JSON includes files, actual assignees/review requests, reviews and check URLs/times, original handoffs, closing references, native blockers, planned activation, validation acceptance and the refresh/re-review policy.", "",
              "Refresh retained carriers only when their owning issue activates and prerequisites have merged. Conflict-free/green historical heads are not release evidence. Draft status, existing native parents and closing references are not changed by this package."]
    (ROOT / "pr-matrix.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(result["report"]))


if __name__ == "__main__":
    main()
