#!/usr/bin/env python3
"""Scoped Project 10 updater. Preparation/inspection are read-only on GitHub.

Never creates issues, fields or views, changes issue state/relationships, requests
reviews, merges PRs, or approves a human gate. Apply needs the digest of a concrete
live before/after plan. Unacknowledged writes stop for inspection and reprepare.
"""
import argparse
from contextlib import contextmanager
import copy
import gzip
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
REPO = "dhar174/tiny_village"
VIEW_KEYS = ("name", "layout", "filter", "group_by", "vertical_group_by", "sort_by")


def excluded_coordinator(pr):
    # Exact reviewed exclusions; unrelated future PRs still invalidate the plan.
    return ((pr.get("headRepository") or {}).get("nameWithOwner") == REPO
            and pr["headRefName"] in load(ROOT / "review-scope.json")["excluded_coordinator_branches"])


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def load(path):
    path = Path(path)
    if not path.exists() and path.with_suffix(path.suffix + ".gz").exists():
        path = path.with_suffix(path.suffix + ".gz")
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as stream:
            return json.load(stream)
    return json.loads(path.read_text(encoding="utf-8"))


def connection(fetch):
    nodes, cursor, seen = [], None, set()
    while True:
        page = fetch(cursor)
        nodes.extend(page["nodes"])
        if not page["pageInfo"]["hasNextPage"]:
            return nodes
        cursor = page["pageInfo"].get("endCursor")
        if not cursor or cursor in seen:
            raise ValueError("Missing or repeated pagination cursor")
        seen.add(cursor)


def unique(values, key, label):
    result = {}
    for value in values:
        identity = value[key]
        if identity in result:
            raise ValueError("Duplicate " + label + ": " + str(identity))
        result[identity] = value
    return result


def managed_items(project):
    items = [i for i in project["items"] if i.get("repository") == REPO]
    return unique(items, "number", "Project issue")


def validate_value(field, value):
    if value is None:
        return
    kind = field["type"]
    if kind == "single_select" and value not in field["options"]:
        raise ValueError(f"Missing option {value!r} in field {field['name']}")
    if kind == "number" and (not isinstance(value, (int, float)) or isinstance(value, bool)):
        raise ValueError("Invalid number in field " + field["name"])
    if kind == "text" and not isinstance(value, str):
        raise ValueError("Invalid text in field " + field["name"])
    if kind not in ("text", "number", "single_select"):
        raise ValueError("Unsupported field type: " + kind)


def prepare(desired, project):
    fields = unique(project["fields"], "name", "field")
    for spec in desired.get("fields", []):
        field = fields.get(spec["name"])
        if not field or field["type"] != spec["data_type"]:
            raise ValueError("Missing field or conflicting type: " + spec["name"])
        if set(spec.get("options", [])) - field["options"].keys():
            raise ValueError("Missing schema option in field " + spec["name"])
    rows = unique(desired["rows"], "number", "desired issue")
    current = managed_items(project)
    views = unique(project["views"], "name", "view")
    operations = []
    for number, row in rows.items():
        before = current.get(number)
        if before and before.get("archived"):
            raise ValueError(f"Managed issue #{number} is archived; review before changing")
        if before and before["node_id"] != row["node_id"]:
            raise ValueError("Issue identity mismatch")
        if before and row.get("issue_state") == "OPEN" and str(before["values"].get("Status", "")).strip().casefold() in ("done", "complete", "completed", "closed"):
            raise ValueError(f"Status conflict on open issue #{number}; review the preserved Project Status")
        values = copy.deepcopy(row["values"])
        old_values = before["values"] if before else {}
        for name in ("Scope", "Review owner"):
            if old_values.get(name):
                values[name] = old_values[name]
        if values.get("Scope") in ("Optional", "Post-MVP") and values.get("Readiness") != "Tracking":
            values["Readiness"] = "Deferred"
        old_decision = old_values.get("Decision")
        if old_decision and old_decision != values.get("Decision") and old_decision not in ("Pending", "Not applicable"):
            raise ValueError(f"Decision conflict on #{number}; reconcile the human record first")
        if old_values.get("Task key") and old_values["Task key"] != values.get("Task key"):
            raise ValueError(f"Task key conflict on #{number}")
        for name, value in values.items():
            if name not in fields:
                raise ValueError("Missing managed field: " + name)
            validate_value(fields[name], value)
        after = copy.deepcopy(before) if before else {
            "id": None, "number": number, "node_id": row["node_id"],
            "repository": REPO, "archived": False, "values": {}}
        for name, value in values.items():
            if value is None or value == "":
                after["values"].pop(name, None)
            else:
                after["values"][name] = value
        if before != after:
            operations.append({"kind": "item", "number": number, "node_id": row["node_id"],
                               "before": copy.deepcopy(before), "after": after, "managed_fields": sorted(values)})
    for spec in desired["views"]:
        before = views.get(spec["name"])
        if not before:
            raise ValueError("Missing saved view: " + spec["name"] + "; explicit schema review required")
        after = copy.deepcopy(before)
        after.update(spec)
        for name in spec["group_by"] + spec.get("vertical_group_by", []) + [s[0] for s in spec["sort_by"]]:
            if name not in fields:
                raise ValueError("Missing view field: " + name)
        if before != after:
            operations.append({"kind": "view", "name": spec["name"], "before": copy.deepcopy(before), "after": after})
    return {"schema_version": 1, "project_url": project["url"], "desired_digest": digest(desired),
            "baseline": copy.deepcopy(project), "source_guard": copy.deepcopy(desired["source_guard"]),
            "operations": operations}


def canonical(project):
    value = copy.deepcopy(project)
    value["items"].sort(key=lambda i: i["id"] or "")
    value["fields"].sort(key=lambda i: i["name"])
    value["views"].sort(key=lambda i: i["name"])
    return value


def replace_expected(project, operation, actual):
    key = "items" if operation["kind"] == "item" else "views"
    identity = "node_id" if key == "items" else "name"
    target = operation[identity] if key == "items" else operation["name"]
    indexes = [i for i, value in enumerate(project[key]) if value.get(identity) == target]
    if indexes:
        project[key][indexes[0]] = copy.deepcopy(actual)
    else:
        project[key].append(copy.deepcopy(actual))


def operation_value(project, operation):
    if operation["kind"] == "item":
        return managed_items(project).get(operation["number"])
    return unique(project["views"], "name", "view").get(operation["name"])


def expected_readback(operation, actual):
    wanted = copy.deepcopy(operation["after"])
    if operation["kind"] == "item" and operation["before"] is None and actual:
        wanted["id"] = actual["id"]
        # Preserve automatic Status/other fields on a newly added row. No
        # managed field can be accepted merely because the service returned it.
        wanted["values"].update({k: v for k, v in actual["values"].items() if k not in operation["managed_fields"]})
    return wanted


def apply_plan(plan, desired, service, state, persist, approval):
    plan_digest = digest(plan)
    if approval != plan_digest:
        raise ValueError("Exact reviewed approval digest is required")
    if plan["desired_digest"] != digest(desired):
        raise ValueError("Desired preview changed since approval")
    if state.get("plan_digest", plan_digest) != plan_digest:
        raise ValueError("Checkpoint belongs to a different plan")
    if state.get("pending") is not None:
        raise ValueError("Unacknowledged write: inspect, reconcile and prepare a fresh plan/checkpoint")
    if service.sources(desired) != plan["source_guard"]:
        raise ValueError("Stale source inputs: refresh and review before publication")
    expected = copy.deepcopy(plan["baseline"])
    completed = state.setdefault("completed", [])
    if len(completed) > len(plan["operations"]) or any(r.get("index") != i for i, r in enumerate(completed)):
        raise ValueError("Checkpoint receipts are not a valid confirmed prefix")
    for receipt in completed:
        wanted = expected_readback(plan["operations"][receipt["index"]], receipt["actual"])
        if not wanted.get("id") or receipt["actual"] != wanted:
            raise ValueError("Checkpoint receipt does not match its approved operation")
        replace_expected(expected, plan["operations"][receipt["index"]], receipt["actual"])
    if canonical(service.project()) != canonical(expected):
        raise ValueError("Project state/schema changed since the reviewed plan/checkpoint")
    state.update(plan_digest=plan_digest, status="applying")
    persist()
    for index in range(len(completed), len(plan["operations"])):
        operation = plan["operations"][index]
        # Recheck the target immediately before its transaction. Full source and
        # Project readbacks bracket the run; avoid thousands of whole-project
        # scans in a 328-row publication. Never write unrelated rows/fields.
        current = service.current(operation) if hasattr(service, "current") else operation_value(service.project(), operation)
        if current != operation["before"]:
            raise ValueError("Project changed during publication; preserve edits and reprepare")
        if hasattr(service, "check_target_source"):
            service.check_target_source(desired, operation)
        state["pending"] = index
        persist()
        service.execute(operation, expected)
        actual = service.current(operation) if hasattr(service, "current") else operation_value(service.project(), operation)
        wanted = expected_readback(operation, actual)
        if actual != wanted:
            raise ValueError("Mutation readback differs; inspect unacknowledged publication")
        replace_expected(expected, operation, actual)
        completed.append({"index": index, "actual": actual})
        state.pop("pending", None)
        persist()
    if service.sources(desired) != plan["source_guard"] or canonical(service.project()) != canonical(expected):
        raise ValueError("Final readback changed; publication is not verified")
    state["status"] = "verified"
    state["operations_confirmed"] = len(completed)
    persist()


def issue_signature(issue):
    value = copy.deepcopy(issue)
    for key, identity in (("assignees", "login"), ("blockedBy", "number"), ("labels", "name")):
        value[key]["nodes"].sort(key=lambda n: n[identity])
    return digest(value)


FIELD_FRAGMENT = '''... on ProjectV2Field{id name dataType databaseId}
 ... on ProjectV2SingleSelectField{id name dataType databaseId options{id name}}
 ... on ProjectV2IterationField{id name dataType databaseId}'''
FIELD_NAME = '''... on ProjectV2Field{name} ... on ProjectV2SingleSelectField{name}
 ... on ProjectV2IterationField{name}'''
ITEM_SELECTION = '''id isArchived content{... on Issue{id number repository{nameWithOwner}}}
 fieldValues(first:100){pageInfo{hasNextPage} nodes{
 ... on ProjectV2ItemFieldTextValue{text field{''' + FIELD_NAME + '''}}
 ... on ProjectV2ItemFieldNumberValue{number field{''' + FIELD_NAME + '''}}
 ... on ProjectV2ItemFieldSingleSelectValue{name field{''' + FIELD_NAME + '''}}
 ... on ProjectV2ItemFieldDateValue{date field{''' + FIELD_NAME + '''}}}}'''
VIEW_SELECTION = '''id number name layout filter
 groupByFields(first:100){nodes{''' + FIELD_NAME + '''} pageInfo{hasNextPage}}
 verticalGroupByFields(first:100){nodes{''' + FIELD_NAME + '''} pageInfo{hasNextPage}}
 sortByFields(first:100){nodes{direction field{''' + FIELD_NAME + '''}} pageInfo{hasNextPage}}'''


def normalize_item(item):
    if item["fieldValues"]["pageInfo"]["hasNextPage"]:
        raise ValueError("Item field overflow")
    values = {}
    for field in item["fieldValues"]["nodes"]:
        if "field" in field:
            value = next(field[k] for k in ("text", "number", "name", "date") if k in field)
            if value is not None and value != "":
                values[field["field"]["name"]] = value
    content = item.get("content") or {}
    return {"id": item["id"], "number": content.get("number"), "node_id": content.get("id", item["id"]),
            "repository": content.get("repository", {}).get("nameWithOwner"),
            "archived": item["isArchived"], "values": values}


def normalize_view(view):
    for key in ("groupByFields", "verticalGroupByFields", "sortByFields"):
        if view[key]["pageInfo"]["hasNextPage"]:
            raise ValueError("View configuration overflow")
    return {"id": view["id"], "number": view["number"], "name": view["name"],
            "layout": view["layout"], "filter": view["filter"] or "",
            "group_by": [f["name"] for f in view["groupByFields"]["nodes"]],
            "vertical_group_by": [f["name"] for f in view["verticalGroupByFields"]["nodes"]],
            "sort_by": [[s["field"]["name"], s["direction"]] for s in view["sortByFields"]["nodes"]]}


class GitHub:
    def __init__(self):
        self.last_write = 0
        self.added_items = {}

    def api(self, endpoint, payload=None, method=None, project=True):
        env = dict(os.environ)
        env.pop("GH_DEBUG", None)
        if project and env.get("GH_PROJECT_TOKEN"):
            env["GH_TOKEN"] = env["GH_PROJECT_TOKEN"]
        cmd = ["gh", "api", endpoint]
        if method:
            cmd += ["--method", method]
        if payload is not None:
            cmd += ["--input", "-"]
        result = subprocess.run(cmd, input=json.dumps(payload) if payload is not None else None,
                                text=True, encoding="utf-8", capture_output=True, env=env, timeout=60)
        if result.returncode:
            raise RuntimeError((result.stderr.strip() + " " + result.stdout.strip())[:1800])
        value = json.loads(result.stdout) if result.stdout.strip() else None
        if isinstance(value, dict) and value.get("errors"):
            raise RuntimeError(json.dumps(value["errors"]))
        return value

    def gql(self, query, variables=None, project=True):
        return self.api("graphql", {"query": query, "variables": variables or {}}, project=project)["data"]

    def sources(self, desired):
        # Fetch the reviewed source set without changing issue/PR state. The
        # current coordinator review PR is deliberately outside the input set.
        issues = {}
        numbers = sorted(map(int, desired["source_guard"]["issues"]))
        issue_fields = '''number id url title body state stateReason updatedAt closedAt
          assignees(first:100){nodes{login} pageInfo{hasNextPage}} parent{number}
          blockedBy(first:100){nodes{number state} pageInfo{hasNextPage}}
          labels(first:100){nodes{name} pageInfo{hasNextPage}}'''
        for start in range(0, len(numbers), 30):
            selections = " ".join(f"i{n}:issue(number:{n}){{{issue_fields}}}" for n in numbers[start:start + 30])
            values = self.gql('{repository(owner:"dhar174",name:"tiny_village"){' + selections + '}}', project=False)["repository"]
            for issue in values.values():
                if not issue or any(issue[k]["pageInfo"]["hasNextPage"] for k in ("assignees", "blockedBy", "labels")):
                    raise ValueError("Missing/overflowed source issue")
                issues[str(issue["number"])] = issue_signature(issue)
        numbers = sorted(map(int, desired["source_guard"]["prs"]))
        selections = " ".join(f"p{n}:pullRequest(number:{n}){{number headRefOid state}}" for n in numbers)
        if numbers:
            selections += ' openInventory:pullRequests(first:100,states:OPEN){nodes{number headRefOid state headRefName headRepository{nameWithOwner}} pageInfo{hasNextPage}}'
        data = self.gql('{repository(owner:"dhar174",name:"tiny_village"){defaultBranchRef{target{oid}} ' + selections + '}}', project=False)["repository"]
        main = data.pop("defaultBranchRef")["target"]["oid"]
        inventory = data.pop("openInventory", None)
        if inventory and inventory["pageInfo"]["hasNextPage"]:
            raise ValueError("Open PR inventory overflow")
        prs = {str(pr["number"]): {"head": pr["headRefOid"], "state": pr["state"]} for pr in data.values() if pr}
        if inventory:
            for pr in inventory["nodes"]:
                if not excluded_coordinator(pr):
                    prs[str(pr["number"])] = {"head": pr["headRefOid"], "state": pr["state"]}
        return {"main": main, "issues": issues, "prs": prs}

    def project(self):
        config = self.gql('''{user(login:"dhar174"){projectV2(number:10){id url
          fields(first:100){nodes{''' + FIELD_FRAGMENT + '''} pageInfo{hasNextPage}}
          views(first:100){nodes{id number name layout filter
            groupByFields(first:100){nodes{''' + FIELD_NAME + '''} pageInfo{hasNextPage}}
            verticalGroupByFields(first:100){nodes{''' + FIELD_NAME + '''} pageInfo{hasNextPage}}
            sortByFields(first:100){nodes{direction field{''' + FIELD_NAME + '''}} pageInfo{hasNextPage}}
          } pageInfo{hasNextPage}}}}}''')["user"]["projectV2"]
        if not config:
            raise ValueError("Project 10 is inaccessible")
        if config["fields"]["pageInfo"]["hasNextPage"] or config["views"]["pageInfo"]["hasNextPage"]:
            raise ValueError("Schema/view overflow; stop rather than truncate")
        fields = [{"id": f["id"], "rest_id": f["databaseId"], "name": f["name"], "type": f["dataType"].lower(),
                   "options": {o["name"]: o["id"] for o in f.get("options", [])}} for f in config["fields"]["nodes"]]
        views = [normalize_view(v) for v in config["views"]["nodes"]]
        def page(cursor):
            query = '''query($id:ID!,$after:String){node(id:$id){... on ProjectV2{
              items(first:100,after:$after){pageInfo{hasNextPage endCursor} nodes{id isArchived
              content{... on Issue{id number repository{nameWithOwner}}}
              fieldValues(first:100){pageInfo{hasNextPage} nodes{
                ... on ProjectV2ItemFieldTextValue{text field{''' + FIELD_NAME + '''}}
                ... on ProjectV2ItemFieldNumberValue{number field{''' + FIELD_NAME + '''}}
                ... on ProjectV2ItemFieldSingleSelectValue{name field{''' + FIELD_NAME + '''}}
                ... on ProjectV2ItemFieldDateValue{date field{''' + FIELD_NAME + '''}}
              }}}}}}}'''
            return self.gql(query, {"id": config["id"], "after": cursor})["node"]["items"]
        items = [normalize_item(i) for i in connection(page)]
        return {"id": config["id"], "url": config["url"], "fields": fields, "views": views, "items": items}

    def current(self, operation):
        if operation["kind"] == "view":
            data = self.gql('query($id:ID!){node(id:$id){... on ProjectV2View{' + VIEW_SELECTION + '}}}',
                            {"id": operation["before"]["id"]})["node"]
            return normalize_view(data) if data else None
        item_id = operation["before"]["id"] if operation["before"] else self.added_items.get(operation["node_id"])
        if item_id:
            data = self.gql('query($id:ID!){node(id:$id){... on ProjectV2Item{' + ITEM_SELECTION + '}}}',
                            {"id": item_id})["node"]
            return normalize_item(data) if data else None
        # A public issue can belong to other private Projects. Enumerating
        # Issue.projectItems would demand unrelated permissions; stay scoped.
        return managed_items(self.project()).get(operation["number"])

    def check_target_source(self, desired, operation):
        if operation["kind"] == "item":
            subset = {"source_guard": {"issues": {str(operation["number"]): desired["source_guard"]["issues"][str(operation["number"])]}, "prs": {}}}
            actual = self.sources(subset)
            if actual["main"] != desired["source_guard"]["main"] or actual["issues"] != subset["source_guard"]["issues"]:
                raise ValueError("Stale source input before row write")

    def execute(self, operation, project):
        time.sleep(max(0, 1.05 - (time.monotonic() - self.last_write)))
        if operation["kind"] == "view":
            view = operation["after"]
            fields = {f["name"]: f["rest_id"] for f in project["fields"]}
            payload = {"filter": view["filter"], "layout": view["layout"].removesuffix("_LAYOUT").lower(),
                       "group_by": [fields[n] for n in view["group_by"]],
                       "vertical_group_by": [fields[n] for n in view["vertical_group_by"]],
                       "sort_by": [[fields[n], d.lower()] for n, d in view["sort_by"]]}
            self.api(f"users/dhar174/projectsV2/10/views/{view['number']}", payload, "PATCH")
        else:
            before = operation["before"]
            item_id = before["id"] if before else None
            if not item_id:
                data = self.gql('''mutation($input:AddProjectV2ItemByIdInput!){addProjectV2ItemById(input:$input){item{id}}}''',
                                {"input": {"projectId": project["id"], "contentId": operation["node_id"]}})
                item_id = data["addProjectV2ItemById"]["item"]["id"]
                self.added_items[operation["node_id"]] = item_id
                initial = self.current(operation)
                if not initial or initial["id"] != item_id or initial["archived"]:
                    raise ValueError("New item identity/availability was not confirmed")
                for name in operation["managed_fields"]:
                    value = initial["values"].get(name)
                    if value is not None and value != operation["after"]["values"].get(name):
                        raise ValueError("New item has managed defaults; inspect and reprepare rather than overwrite")
                before = initial
            fields = {f["name"]: f for f in project["fields"]}
            declarations, selections, variables = [], [], {}
            old_values = before["values"] if before else {}
            for name in operation["managed_fields"]:
                value = operation["after"]["values"].get(name)
                if old_values.get(name) == value:
                    continue
                field = fields[name]
                alias = "f" + str(len(variables))
                variables[alias] = {"projectId": project["id"], "itemId": item_id, "fieldId": field["id"]}
                if value is None:
                    input_type, method = "ClearProjectV2ItemFieldValueInput", "clearProjectV2ItemFieldValue"
                else:
                    input_type, method = "UpdateProjectV2ItemFieldValueInput", "updateProjectV2ItemFieldValue"
                    key = {"text": "text", "number": "number", "single_select": "singleSelectOptionId"}[field["type"]]
                    variables[alias]["value"] = {key: field["options"][value] if field["type"] == "single_select" else value}
                declarations.append("$" + alias + ":" + input_type + "!")
                selections.append(alias + ":" + method + "(input:$" + alias + "){projectV2Item{id}}")
            if variables:
                result = self.gql("mutation(" + ",".join(declarations) + "){" + " ".join(selections) + "}", variables)
                if any((r or {}).get("projectV2Item", {}).get("id") != item_id for r in result.values()) or len(result) != len(variables):
                    raise ValueError("Field mutation acknowledgement missing")
        self.last_write = time.monotonic()


def save(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


@contextmanager
def publication_lock(checkpoint):
    """OS-released, nonblocking local lock on Windows and Unix."""
    with checkpoint.with_suffix(".lock").open("a+b") as stream:
        if os.name == "nt":
            import msvcrt
            stream.seek(0, os.SEEK_END)
            if stream.tell() == 0:
                stream.write(b"0")
                stream.flush()
            stream.seek(0)
            try:
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError as error:
                raise ValueError("Another publisher holds this checkpoint lock") from error
            try:
                yield
            finally:
                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl
            try:
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise ValueError("Another publisher holds this checkpoint lock") from error
            try:
                yield
            finally:
                fcntl.flock(stream, fcntl.LOCK_UN)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    for name in ("prepare", "apply", "inspect"):
        mode.add_argument("--" + name, action="store_true")
    parser.add_argument("--desired", type=Path, default=ROOT / "project-preview.json")
    parser.add_argument("--plan", type=Path, default=ROOT / "live-plan.json")
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "local-checkpoint.json")
    parser.add_argument("--approval")
    parser.add_argument("--receipt", type=Path, help="Retain read-only inspection evidence outside the frozen package")
    parser.add_argument("--edit-window-confirmed", action="store_true",
                        help="Confirm no other Project writers during apply; the API has no compare-and-swap")
    args = parser.parse_args()
    if args.receipt and not args.inspect:
        parser.error("--receipt requires --inspect")
    if args.receipt:
        receipt_paths = {args.receipt.resolve(), args.receipt.with_suffix(args.receipt.suffix + ".tmp").resolve()}
        if any(path.is_relative_to(ROOT.resolve()) for path in receipt_paths):
            parser.error("--receipt and its temporary file must be outside the frozen package")
        if receipt_paths & {p.resolve() for p in (args.plan, args.desired, args.checkpoint)}:
            parser.error("--receipt and its temporary file must be distinct from the plan, desired preview and checkpoint")
    if args.apply and not args.edit_window_confirmed:
        parser.error("--apply requires --edit-window-confirmed: coordinate an edit window because the remote field API has no conditional/version writes")
    desired = load(args.desired)
    service = GitHub()
    try:
        if args.prepare:
            if service.sources(desired) != desired["source_guard"]:
                raise ValueError("Stale source inputs; refresh the reviewed snapshot first")
            plan = prepare(desired, service.project())
            save(args.plan, plan)
            print(json.dumps({"status": "prepared_not_applied", "operations": len(plan["operations"]),
                              "approval_digest": digest(plan), "plan": str(args.plan), "github_mutations_performed": 0}))
        elif args.inspect and args.receipt:
            from publication_evidence import inspect_publication
            receipt = inspect_publication(desired, load(args.plan), service)
            args.receipt.parent.mkdir(parents=True, exist_ok=True)
            save(args.receipt, receipt)
            print(json.dumps({key: receipt[key] for key in ("status", "source_matches", "remaining_operations", "github_mutations_performed")}))
            return 0 if receipt["status"] == "verified" else 2
        elif args.inspect:
            source_matches = service.sources(desired) == desired["source_guard"]
            delta = prepare(desired, service.project())
            print(json.dumps({"status": "verified" if source_matches and not delta["operations"] else "incomplete",
                              "source_matches": source_matches, "remaining_operations": len(delta["operations"]),
                              "github_mutations_performed": 0}))
            return 0 if source_matches and not delta["operations"] else 2
        else:
            plan = load(args.plan)
            with publication_lock(args.checkpoint):
                # Load inside the lock so another local process cannot replace
                # a confirmed checkpoint between reading and acquiring it.
                state = load(args.checkpoint) if args.checkpoint.exists() else {}
                apply_plan(plan, desired, service, state, lambda: save(args.checkpoint, state), args.approval)
            print(json.dumps({"status": state["status"], "operations_confirmed": state["operations_confirmed"],
                              "checkpoint": str(args.checkpoint)}))
    except (ValueError, RuntimeError, KeyError, OSError, subprocess.TimeoutExpired) as error:
        if args.receipt:
            args.receipt.parent.mkdir(parents=True, exist_ok=True)
            save(args.receipt, {"schema_version": 1, "status": "blocked_or_partial", "error": str(error),
                                "github_mutations_performed": 0})
        print(json.dumps({"status": "blocked_or_partial", "error": str(error), "checkpoint": str(args.checkpoint)}), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
