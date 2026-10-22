"""Validate issue ordering and human-gate rules without GitHub writes."""
import importlib.util
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


def issue(number, state="OPEN", blockers=(), parent=None):
    return {"number": number, "id": "i" + str(number), "url": "issue/" + str(number), "title": str(number),
            "body": "", "state": state, "stateReason": None, "updatedAt": "time", "closedAt": None,
            "assignees": {"nodes": [], "pageInfo": {"hasNextPage": False}},
            "labels": {"nodes": [], "pageInfo": {"hasNextPage": False}},
            "blockedBy": {"nodes": [{"number": n, "state": "OPEN"} for n in blockers], "pageInfo": {"hasNextPage": False}},
            "parent": {"number": parent} if parent else None}


def item(number, phase, key, kind="Task", depends=()):
    return {"number": number, "phase": phase, "key": key, "kind": kind,
            "sequence": phase * 1000 + number, "depends_on": list(depends), "priority": "P1",
            "workstream": "Planning", "scope": "MVP", "review_owner": "dhar174", "decision": "Pending" if kind == "Phase gate" else "Not applicable"}


class PreviewAvailabilityTests(unittest.TestCase):
    def test_builder_exists(self):
        self.assertTrue((ROOT / "build_preview.py").exists(), "Ordering/preview builder is missing")


class PreviewTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        spec = importlib.util.spec_from_file_location("issue737_builder", ROOT / "build_preview.py")
        cls.builder = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.builder)

    def setUp(self):
        self.snapshot = {"main_sha": "main", "managed_numbers": [10, 20, 30],
                         "issues": [issue(10, "CLOSED"), issue(20, blockers=(10,)), issue(30, blockers=(20,))],
                         "pull_requests": []}
        self.roadmap = {"items": [item(10, 2, "H2", "Phase gate"), item(20, 3, "P3"), item(30, 3, "H3", "Phase gate")],
                        "project": {"fields": [], "views": []},
                        "phases": {"2": {"name": "Two"}, "3": {"name": "Three"}}}

    def build(self, approvals=None):
        return self.builder.assemble(self.snapshot, self.roadmap, [], [], {"approved": approvals or {}}, [])

    def test_closed_gate_without_human_approval_keeps_dependents_blocked(self):
        rows = {r["number"]: r for r in self.build()["preview"]["rows"]}
        self.assertEqual(rows[20]["values"]["Readiness"], "Blocked")
        self.assertEqual(rows[10]["values"]["Decision"], "Pending")

    def test_recorded_human_approval_unlocks_only_immediate_ready_task(self):
        rows = {r["number"]: r for r in self.build({"10": {"decision": "Approved"}})["preview"]["rows"]}
        self.assertEqual(rows[20]["values"]["Readiness"], "Ready")
        self.assertEqual(rows[30]["values"]["Readiness"], "Blocked")

    def test_open_dependency_blocks_even_if_decision_says_approved(self):
        self.snapshot["issues"][0]["state"] = "OPEN"
        rows = {r["number"]: r for r in self.build({"10": {"decision": "Approved"}})["preview"]["rows"]}
        self.assertEqual(rows[20]["values"]["Readiness"], "Blocked")

    def test_native_dependency_cycle_is_rejected(self):
        self.snapshot["issues"][1]["blockedBy"]["nodes"] = [{"number": 30, "state": "OPEN"}]
        with self.assertRaisesRegex(ValueError, "cycle"):
            self.build()

    def test_ordered_sequences_follow_prerequisites(self):
        rows = {r["number"]: r for r in self.build()["preview"]["rows"]}
        self.assertLess(rows[20]["values"]["Sequence"], rows[30]["values"]["Sequence"])

    def test_phase_parent_does_not_become_a_dependency(self):
        self.snapshot["issues"][1]["parent"] = {"number": 30}
        result = self.build()
        self.assertNotIn(30, result["rows_by_number"][20]["native_prerequisites"])

    def test_missing_managed_issue_is_rejected(self):
        self.snapshot["managed_numbers"].append(404)
        with self.assertRaisesRegex(ValueError, "Missing"):
            self.build()

    def test_missing_prerequisite_is_rejected(self):
        self.snapshot["issues"][1]["blockedBy"]["nodes"].append({"number": 404, "state": "OPEN"})
        with self.assertRaisesRegex(ValueError, "Missing"):
            self.build()

    def test_closed_task_is_tracking_and_not_reported_ready(self):
        self.snapshot["issues"][1]["state"] = "CLOSED"
        rows = {r["number"]: r for r in self.build()["preview"]["rows"]}
        self.assertEqual(rows[20]["values"]["Readiness"], "Tracking")

    def test_738_approval_cannot_be_invented_from_coordinator_approval(self):
        self.snapshot["issues"][-1] = issue(738, blockers=(20,))
        self.snapshot["managed_numbers"][-1] = 738
        self.roadmap["items"][-1]["number"] = 738
        with self.assertRaisesRegex(ValueError, "738"):
            self.build({"738": {"decision": "Approved"}})


if __name__ == "__main__":
    unittest.main(verbosity=2)
