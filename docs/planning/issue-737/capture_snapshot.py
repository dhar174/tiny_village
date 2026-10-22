#!/usr/bin/env python3
"""Capture read-only GitHub inputs for #737; never mutate remote state."""
import datetime as dt
import gzip
import json
import os
from pathlib import Path
import subprocess

from sync_project import load

ROOT = Path(__file__).resolve().parent
REPO = "dhar174/tiny_village"


def request(endpoint, query=None, project=False):
    env = dict(os.environ)
    env.pop("GH_DEBUG", None)
    if project and env.get("GH_PROJECT_TOKEN"):
        env["GH_TOKEN"] = env["GH_PROJECT_TOKEN"]
    cmd = ["gh", "api", endpoint]
    if query:
        cmd += ["-f", "query=" + query]
    result = subprocess.run(cmd, text=True, encoding="utf-8", capture_output=True, env=env, timeout=60)
    if result.returncode:
        raise RuntimeError((result.stderr.strip() + " " + result.stdout.strip())[:1600])
    value = json.loads(result.stdout)
    if isinstance(value, dict) and value.get("errors"):
        raise RuntimeError(json.dumps(value["errors"]))
    return value


def collect_connection(field, node_fields, extra=""):
    result, cursor = [], None
    while True:
        after = ",after:" + json.dumps(cursor) if cursor else ""
        query = ('{repository(owner:"dhar174",name:"tiny_village"){' + field +
                 '(first:50' + extra + after + '){totalCount pageInfo{hasNextPage endCursor} nodes{' +
                 node_fields + '}}}}')
        data = request("graphql", query)["data"]["repository"][field]
        result.extend(data["nodes"])
        print(field, len(result), "/", data["totalCount"], flush=True)
        if not data["pageInfo"]["hasNextPage"]:
            if len(result) != data["totalCount"]:
                raise RuntimeError("Incomplete " + field)
            return result
        cursor = data["pageInfo"]["endCursor"]


def main():
    started = dt.datetime.now(dt.timezone.utc).isoformat()
    issue_fields = '''number id url title body state stateReason updatedAt closedAt
        assignees(first:100){nodes{login} pageInfo{hasNextPage}}
        parent{number} blockedBy(first:100){nodes{number state} pageInfo{hasNextPage}}
        labels(first:100){nodes{name} pageInfo{hasNextPage}}'''
    all_issues = collect_connection("issues", issue_fields)
    by_number = {i["number"]: i for i in all_issues}
    roadmap = load(ROOT / "original-roadmap.json")
    ledger = load(ROOT / "triage-input.json")
    remediations = load(ROOT / "remediations.json") if (ROOT / "remediations.json").exists() else []
    managed = {i["number"] for i in roadmap["items"]} | {i["number"] for i in ledger} | set(range(799, 804)) | {i['number'] for i in remediations}
    if len(managed) != 328 + len(remediations) or managed - by_number.keys():
        raise RuntimeError("Managed cohort changed/missing: " + str(managed - by_number.keys()))
    for issue in all_issues:
        for key in ("assignees", "blockedBy", "labels"):
            if issue[key]["pageInfo"]["hasNextPage"]:
                raise RuntimeError(f"Issue {issue['number']} {key} overflow")
    pr_fields = '''number id url title body updatedAt headRefName headRefOid baseRefName baseRefOid
        isDraft mergeable mergeStateStatus reviewDecision author{login}
        assignees(first:100){nodes{login} pageInfo{hasNextPage}}
        reviewRequests(first:100){nodes{requestedReviewer{... on User{login} ... on Team{name}}} pageInfo{hasNextPage}}
        closingIssuesReferences(first:100){nodes{number} pageInfo{hasNextPage}}
        files(first:100){nodes{path additions deletions} pageInfo{hasNextPage}}
        reviews(last:100){nodes{url state body submittedAt author{login} commit{oid}} pageInfo{hasPreviousPage}}
        commits(last:1){nodes{commit{oid statusCheckRollup{state contexts(first:100){nodes{
          __typename ... on CheckRun{name status conclusion completedAt detailsUrl}
          ... on StatusContext{context state targetUrl}} pageInfo{hasNextPage}}}}}}'''
    prs = collect_connection("pullRequests", pr_fields, ",states:OPEN")
    excluded_coordination = [p for p in prs if p["headRefName"] == "codex/issue-737-coordination"]
    prs = [p for p in prs if p not in excluded_coordination]
    for pr in prs:
        for key in ("assignees", "reviewRequests", "closingIssuesReferences", "files", "reviews"):
            if any(v for k, v in pr[key]["pageInfo"].items() if k.startswith("has")):
                raise RuntimeError(f"PR {pr['number']} {key} overflow")
        rollup = pr["commits"]["nodes"][0]["commit"]["statusCheckRollup"]
        if rollup and rollup["contexts"]["pageInfo"]["hasNextPage"]:
            raise RuntimeError(f"PR {pr['number']} checks overflow")
    main_sha = request("repos/" + REPO + "/commits/main")["sha"]
    for number in (701, 704, 709, 711, 715):
        patches = request(f"repos/{REPO}/pulls/{number}/files?per_page=100")
        (ROOT / f"evidence/pr-{number}-files.json").write_text(json.dumps(patches, indent=2) + "\n")
    project = {"checked_at": dt.datetime.now(dt.timezone.utc).isoformat(), "github_mutations_performed": 0}
    try:
        project["response"] = request("users/dhar174/projectsV2/10", project=True)
        project["status"] = "readable"
    except RuntimeError as error:
        project.update(status="blocked", error=str(error))
    (ROOT / "evidence/project-access.json").write_text(json.dumps(project, indent=2) + "\n")
    rules = request(f"repos/{REPO}/rulesets?per_page=100")
    rules = [request(f"repos/{REPO}/rulesets/{r['id']}") for r in rules]
    (ROOT / "evidence/rulesets.json").write_text(json.dumps(rules, indent=2) + "\n")
    try:
        local = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True, encoding="utf-8", capture_output=True, timeout=10)
        collector_worktree_sha = local.stdout.strip() if local.returncode == 0 else None
    except FileNotFoundError:
        collector_worktree_sha = None
    snapshot = {"started_at": started, "completed_at": dt.datetime.now(dt.timezone.utc).isoformat(),
                "collector_worktree_sha": collector_worktree_sha,
                "main_sha": main_sha, "repository": REPO, "github_mutations_performed": 0,
                "managed_numbers": sorted(managed), "issues": all_issues, "pull_requests": prs,
                "excluded_coordination_prs": excluded_coordination}
    with gzip.open(ROOT / "snapshot.json.gz", "wt", encoding="utf-8") as stream:
        json.dump(snapshot, stream, indent=2)
    print(json.dumps({"issues": len(all_issues), "managed": len(managed), "prs": len(prs),
                      "main": main_sha, "project_access": project["status"]}))


if __name__ == "__main__":
    main()
