"""Offline service-boundary regressions; these do not prove GitHub publication."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import sys
import io
from contextlib import redirect_stderr

ROOT = Path(__file__).resolve().parent


def load_sync():
    path = ROOT / "sync_project.py"
    if not path.exists():
        raise AssertionError("Scoped Project synchronizer has not been implemented")
    spec = importlib.util.spec_from_file_location("issue737_sync", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fixture():
    fields = [{"id": "f1", "name": "Phase", "type": "single_select", "options": {"04": "o4"}},
              {"id": "f2", "name": "Sequence", "type": "number", "options": {}},
              {"id": "f3", "name": "Readiness", "type": "single_select", "options": {"Blocked": "ob", "Ready": "or", "Deferred": "od"}},
              {"id": "f4", "name": "Decision", "type": "single_select", "options": {"Pending": "op", "Approved": "oa"}},
              {"id": "f5", "name": "Scope", "type": "text", "options": {}},
              {"id": "f6", "name": "Review owner", "type": "text", "options": {}}]
    source = {"main": "main-sha", "issues": {"799": "signature"}, "prs": {"701": "head-sha"}}
    desired = {"source_guard": source, "rows": [{"number": 799, "node_id": "issue799",
               "values": {"Phase": "04", "Sequence": 4001, "Readiness": "Blocked", "Decision": "Pending",
                          "Scope": "MVP", "Review owner": "dhar174"}}],
               "views": [{"name": "01 Ordered delivery", "layout": "TABLE_LAYOUT", "filter": "", "group_by": ["Phase"], "vertical_group_by": [], "sort_by": [["Sequence", "ASC"]]}]}
    project = {"id": "P10", "url": "https://github.com/users/dhar174/projects/10", "fields": fields,
               "items": [{"id": "item799", "number": 799, "node_id": "issue799", "repository": "dhar174/tiny_village", "archived": False,
                          "values": {"Phase": "04", "Sequence": 3000, "Status": "In progress", "Scope": "MVP"}},
                         {"id": "other", "number": 12, "node_id": "other12", "repository": "another/repo", "archived": False, "values": {"Notes": "Keep me"}}],
               "views": [dict(desired["views"][0], id="v2", number=2), {"name": "Personal", "id": "v1", "number": 1, "filter": "custom"}]}
    return desired, project


class MemoryService:
    """Controlled external API; real planning/checkpoint code is under test."""
    def __init__(self, desired, project):
        self.state = copy.deepcopy(project)
        self.source = copy.deepcopy(desired["source_guard"])
        self.writes = []
        self.fail_after_write = False

    def project(self):
        return copy.deepcopy(self.state)

    def sources(self, desired):
        return copy.deepcopy(self.source)

    def execute(self, operation, project):
        self.writes.append(copy.deepcopy(operation))
        if operation["kind"] == "item":
            matches = [i for i in self.state["items"] if i["node_id"] == operation["node_id"]]
            if matches:
                matches[0]["values"] = copy.deepcopy(operation["after"]["values"])
            else:
                item = copy.deepcopy(operation["after"])
                item["id"] = "new-item"
                self.state["items"].append(item)
        else:
            view = next(v for v in self.state["views"] if v["name"] == operation["name"])
            view.update(copy.deepcopy(operation["after"]))
        if self.fail_after_write:
            raise RuntimeError("connection lost after write")


class AvailabilityTests(unittest.TestCase):
    def test_scoped_synchronizer_exists(self):
        self.assertTrue((ROOT / "sync_project.py").is_file(), "Scoped synchronizer is missing")


class SyncTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sync = load_sync()

    def setUp(self):
        self.desired, self.project = fixture()

    def prepare(self):
        return self.sync.prepare(self.desired, self.project)

    def apply(self, service, plan=None, state=None):
        plan = plan or self.prepare()
        state = state if state is not None else {}
        self.sync.apply_plan(plan, self.desired, service, state, lambda: None, self.sync.digest(plan))
        return state

    def test_preview_has_before_after_and_preserves_status_and_foreign_item(self):
        plan = self.prepare()
        op = plan["operations"][0]
        self.assertEqual(op["before"]["values"]["Sequence"], 3000)
        self.assertEqual(op["after"]["values"]["Sequence"], 4001)
        self.assertEqual(op["after"]["values"]["Status"], "In progress")
        self.assertEqual(len(plan["operations"]), 1)

    def test_duplicate_same_repository_issue_is_rejected(self):
        self.project["items"].append(copy.deepcopy(self.project["items"][0]))
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            self.prepare()

    def test_foreign_issue_same_number_is_not_selected(self):
        self.project["items"][1]["number"] = 799
        self.assertEqual(len(self.prepare()["operations"]), 1)

    def test_archived_managed_item_requires_review(self):
        self.project["items"][0]["archived"] = True
        with self.assertRaisesRegex(ValueError, "archived"):
            self.prepare()

    def test_missing_field_is_rejected_without_schema_creation(self):
        self.project["fields"] = self.project["fields"][1:]
        with self.assertRaisesRegex(ValueError, "field"):
            self.prepare()

    def test_unknown_option_is_rejected(self):
        self.desired["rows"][0]["values"]["Readiness"] = "Imaginary"
        with self.assertRaisesRegex(ValueError, "option"):
            self.prepare()

    def test_duplicate_field_names_are_rejected(self):
        self.project["fields"].append(self.project["fields"][0])
        with self.assertRaisesRegex(ValueError, "field"):
            self.prepare()

    def test_unrecorded_gate_approval_is_not_overwritten(self):
        self.project["items"][0]["values"]["Decision"] = "Approved"
        with self.assertRaisesRegex(ValueError, "Decision"):
            self.prepare()

    def test_existing_scope_and_review_owner_are_preserved(self):
        self.project["items"][0]["values"].update({"Scope": "Post-MVP", "Review owner": "maintainer"})
        values = self.prepare()["operations"][0]["after"]["values"]
        self.assertEqual(values["Scope"], "Post-MVP")
        self.assertEqual(values["Readiness"], "Deferred")
        self.assertEqual(values["Review owner"], "maintainer")

    def test_missing_saved_view_requires_explicit_schema_review(self):
        self.project["views"] = self.project["views"][1:]
        with self.assertRaisesRegex(ValueError, "view"):
            self.prepare()

    def test_view_filter_and_sort_drift_is_previewed(self):
        self.project["views"][0]["filter"] = "kind:Task"
        self.project["views"][0]["sort_by"] = [["Sequence", "DESC"]]
        op = next(o for o in self.prepare()["operations"] if o["kind"] == "view")
        self.assertEqual(op["after"]["filter"], "")
        self.assertEqual(op["after"]["sort_by"], [["Sequence", "ASC"]])

    def test_apply_requires_exact_reviewed_digest(self):
        service = MemoryService(self.desired, self.project)
        with self.assertRaisesRegex(ValueError, "approval"):
            self.sync.apply_plan(self.prepare(), self.desired, service, {}, lambda: None, "wrong")
        self.assertEqual(service.writes, [])

    def test_stale_issue_input_stops_before_any_write(self):
        service = MemoryService(self.desired, self.project)
        service.source["issues"]["799"] = "changed"
        with self.assertRaisesRegex(ValueError, "source"):
            self.apply(service)
        self.assertEqual(service.writes, [])

    def test_stale_pr_head_stops_before_any_write(self):
        service = MemoryService(self.desired, self.project)
        service.source["prs"]["701"] = "new-head"
        with self.assertRaisesRegex(ValueError, "source"):
            self.apply(service)
        self.assertEqual(service.writes, [])

    def test_human_field_edit_stops_before_any_write(self):
        service = MemoryService(self.desired, self.project)
        service.state["items"][0]["values"]["Sequence"] = 3333
        with self.assertRaisesRegex(ValueError, "Project"):
            self.apply(service)
        self.assertEqual(service.writes, [])

    def test_schema_drift_stops_before_any_write(self):
        service = MemoryService(self.desired, self.project)
        service.state["fields"][0]["id"] = "new-field"
        with self.assertRaisesRegex(ValueError, "Project"):
            self.apply(service)
        self.assertEqual(service.writes, [])

    def test_successful_apply_and_repeat_are_idempotent(self):
        service = MemoryService(self.desired, self.project)
        state = self.apply(service)
        self.assertEqual(state["status"], "verified")
        self.apply(service, state=state)
        self.assertEqual(len(service.writes), 1)
        self.assertEqual(self.sync.prepare(self.desired, service.project())["operations"], [])

    def test_confirmed_checkpoint_resumes_after_interruption(self):
        service = MemoryService(self.desired, self.project)
        state = {}
        def persist():
            if state.get("completed") and state.get("status") != "verified":
                raise RuntimeError("process interrupted after confirmed checkpoint")
        with self.assertRaisesRegex(RuntimeError, "interrupted"):
            self.sync.apply_plan(self.prepare(), self.desired, service, state, persist, self.sync.digest(self.prepare()))
        self.apply(service, state=state)
        self.assertEqual(len(service.writes), 1)

    def test_unacknowledged_write_requires_inspection_and_fresh_prepare(self):
        service = MemoryService(self.desired, self.project)
        service.fail_after_write = True
        state = {}
        with self.assertRaisesRegex(RuntimeError, "connection lost"):
            self.apply(service, state=state)
        service.fail_after_write = False
        with self.assertRaisesRegex(ValueError, "Unacknowledged"):
            self.apply(service, state=state)
        self.assertEqual(len(service.writes), 1)
        new_plan = self.sync.prepare(self.desired, service.project())
        self.apply(service, plan=new_plan)
        self.assertEqual(len(service.writes), 1)

    def test_missing_item_is_added_once(self):
        self.project["items"] = self.project["items"][1:]
        service = MemoryService(self.desired, self.project)
        state = self.apply(service)
        self.apply(service, state=state)
        self.assertEqual(len(service.writes), 1)
        self.assertEqual(len(service.state["items"]), 2)

    def test_empty_optional_fields_do_not_invent_readback_values(self):
        self.desired["rows"][0]["values"]["Review owner"] = None
        self.project["items"] = self.project["items"][1:]
        service = MemoryService(self.desired, self.project)
        self.apply(service)
        item = next(i for i in service.state["items"] if i["node_id"] == "issue799")
        self.assertNotIn("Review owner", item["values"])

    def test_clear_text_is_explicitly_previewed_as_removal(self):
        self.desired["rows"][0]["values"]["Extra"] = None
        self.project["fields"].append({"id": "f7", "name": "Extra", "type": "text", "options": {}})
        self.project["items"][0]["values"]["Extra"] = "old"
        plan = self.prepare()
        self.assertNotIn("Extra", plan["operations"][0]["after"]["values"])

    def test_corrupt_checkpoint_is_rejected_before_writes(self):
        service = MemoryService(self.desired, self.project)
        with self.assertRaisesRegex(ValueError, "Checkpoint"):
            self.apply(service, state={"completed": [{"index": 9, "actual": {}}]})
        self.assertEqual(service.writes, [])

    def test_new_item_preserves_automatic_unmanaged_status(self):
        self.project["items"] = self.project["items"][1:]
        class DefaultsService(MemoryService):
            def execute(self, operation, project):
                super().execute(operation, project)
                next(i for i in self.state["items"] if i["node_id"] == "issue799")["values"]["Status"] = "Todo"
        service = DefaultsService(self.desired, self.project)
        state = self.apply(service)
        self.assertEqual(state["status"], "verified")
        self.assertEqual(state["completed"][0]["actual"]["values"]["Status"], "Todo")

    def test_schema_type_is_checked_even_for_string_shaped_values(self):
        self.desired["fields"] = [{"name": "Phase", "data_type": "single_select", "options": ["04"]}]
        self.project["fields"][0]["type"] = "text"
        with self.assertRaisesRegex(ValueError, "type"):
            self.prepare()

    def test_second_local_publisher_cannot_acquire_lock(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "checkpoint.json"
            with self.sync.publication_lock(checkpoint):
                with self.assertRaisesRegex(ValueError, "publisher"):
                    with self.sync.publication_lock(checkpoint):
                        self.fail("Second publisher acquired lock")

    def test_apply_requires_edit_window_before_reading_credentials_or_files(self):
        with patch.object(sys, "argv", ["sync_project.py", "--apply", "--desired", "missing.json"]), redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as caught:
                self.sync.main()
        self.assertEqual(caught.exception.code, 2)

    def test_open_issue_with_done_status_requires_review(self):
        self.desired["rows"][0]["issue_state"] = "OPEN"
        self.project["items"][0]["values"]["Status"] = "Done"
        with self.assertRaisesRegex(ValueError, "Status"):
            self.prepare()

    def test_documents_unavoidable_remote_same_field_race(self):
        class RacingService(MemoryService):
            def execute(self, operation, project):
                self.state["items"][0]["values"]["Sequence"] = 9999
                self.human_value = 9999
                super().execute(operation, project)
        service = RacingService(self.desired, self.project)
        state = self.apply(service)
        self.assertEqual(service.human_value, 9999)
        self.assertEqual(service.state["items"][0]["values"]["Sequence"], 4001)
        self.assertEqual(state["status"], "verified")
        # This demonstrates the remote API boundary; it is NOT evidence that
        # concurrent human edits are protected. CLI requires a quiet edit window.

    def test_new_item_lookup_does_not_require_other_private_projects(self):
        desired, project = fixture()
        project["items"] = project["items"][1:]
        operation = self.sync.prepare(desired, project)["operations"][0]
        class ProjectOnly(self.sync.GitHub):
            def project(self):
                return copy.deepcopy(project)
            def gql(self, query, variables=None, project=True):
                raise AssertionError("New item lookup must stay within Project 10, not Issue.projectItems")
        self.assertIsNone(ProjectOnly().current(operation))

    def test_pagination_reads_later_pages_and_rejects_missing_cursor(self):
        pages = {None: {"nodes": [1], "pageInfo": {"hasNextPage": True, "endCursor": "next"}},
                 "next": {"nodes": [2], "pageInfo": {"hasNextPage": False, "endCursor": None}}}
        self.assertEqual(self.sync.connection(lambda cursor: pages[cursor]), [1, 2])
        pages[None]["pageInfo"]["endCursor"] = None
        with self.assertRaisesRegex(ValueError, "cursor"):
            self.sync.connection(lambda cursor: pages[cursor])


if __name__ == "__main__":
    unittest.main(verbosity=2)
