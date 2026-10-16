"""Durable capture checks use real, bounded interpreters, without game imports."""

import ctypes
import hashlib
import json
import os
from pathlib import Path
import signal
import stat
import subprocess
import sys
import tempfile
import textwrap
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from tests import subprocess_capture as capture


class TestSubprocessCapture(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory(prefix="tiny-capture-check-")
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.env = os.environ.copy()
        self.env.pop("TINY_TEST_ARTIFACT_DIR", None)

    def run_child(self, source, **limits):
        return capture.capture_python(
            textwrap.dedent(source), cwd=self.root, env=self.env,
            test_id=self.id(), fatal_timeout=limits.get("fatal_timeout", 3),
            timeout=limits.get("timeout", 5),
        )

    def metadata(self, result):
        return json.loads((result.artifact_dir / "run.json").read_text())

    def events(self, directory):
        return [json.loads(line) for line in (directory / "events.jsonl").read_text().splitlines()]

    def test_success_preserves_main_context_and_future_imports(self):
        source = "from __future__ import annotations\nimport __main__\nvalue = 42\nassert __main__.value == value\nprint('héllo')\n"
        result = self.run_child(source)
        self.assertEqual(result.completed.returncode, 0, result.completed.stderr)
        self.assertEqual(result.completed.stdout, "héllo\n")
        self.assertEqual(result.completed.stderr, "")
        metadata = self.metadata(result)
        self.assertEqual(metadata["outcome"], "EXITED_ZERO")
        self.assertEqual(metadata["source_sha256"], hashlib.sha256(source.encode()).hexdigest())
        self.assertEqual(metadata["test_id"], self.id())
        self.assertEqual([e["phase"] for e in self.events(result.artifact_dir)],
                         ["child_ready", "source_complete", "atexit_marker"])
        self.assertEqual(result.artifact_dir.parent, self.root / ".test-artifacts/subprocess")
        if os.name == "posix":
            self.assertEqual(stat.S_IMODE(result.artifact_dir.stat().st_mode), 0o700)
            for path in result.artifact_dir.iterdir():
                self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o600, str(path))

    def test_nonzero_exit_retains_full_large_and_invalid_byte_output(self):
        source = "import os\nos.write(1, b'x'*100000 + b'\\xff')\nos.write(2, b'y'*120000)\nraise RuntimeError('controlled failure')\n"
        result = self.run_child(source)
        self.assertNotEqual(result.completed.returncode, 0)
        self.assertEqual((result.artifact_dir / "stdout.log").read_bytes(), b"x" * 100000 + b"\xff")
        self.assertTrue((result.artifact_dir / "stderr.log").read_bytes().startswith(b"y" * 120000))
        self.assertTrue(result.completed.stdout.endswith("\ufffd"))
        self.assertIn("controlled failure", result.completed.stderr)
        self.assertEqual(self.metadata(result)["outcome"], "EXITED_NONZERO")
        self.assertNotIn("source_complete", [e["phase"] for e in self.events(result.artifact_dir)])

    def test_repeated_test_id_does_not_overwrite_artifacts(self):
        first = self.run_child("print('first')")
        second = self.run_child("print('second')")
        self.assertNotEqual(first.artifact_dir, second.artifact_dir)
        self.assertEqual((first.artifact_dir / "stdout.log").read_text(), "first\n")
        self.assertEqual((second.artifact_dir / "stdout.log").read_text(), "second\n")

    def test_relative_override_survives_temporary_cache_cleanup(self):
        self.env["TINY_TEST_ARTIFACT_DIR"] = "retained"
        self.env["CONTROLLED_SECRET_SENTINEL"] = "sentinel-never-copy-to-metadata"
        with tempfile.TemporaryDirectory(dir=self.root) as cache:
            self.env["TMPDIR"] = cache
            result = self.run_child("print('survives')")
        self.assertEqual(result.artifact_dir.parent, self.root / "retained")
        self.assertEqual((result.artifact_dir / "stdout.log").read_text(), "survives\n")
        self.assertNotIn("sentinel-never-copy-to-metadata", (result.artifact_dir / "run.json").read_text())

    def test_fatal_watchdog_dump_is_retained(self):
        result = self.run_child("import time\nprint('before hang', flush=True)\ntime.sleep(10)",
                                fatal_timeout=0.7, timeout=3)
        self.assertNotEqual(result.completed.returncode, 0)
        self.assertIn("Timeout (", result.completed.stderr)
        self.assertIn("<string>", result.completed.stderr)
        self.assertEqual(result.completed.stdout, "before hang\n")
        metadata = self.metadata(result)
        self.assertEqual(metadata["fatal_timeout"], 0.7)
        self.assertTrue(metadata["watchdog_dump_observed"])

    def test_parent_timeout_retains_output_and_reaps_direct_child(self):
        with self.assertRaises(capture.CapturedTimeout) as raised:
            self.run_child("import faulthandler, time\nfaulthandler.cancel_dump_traceback_later()\nprint('before parent timeout', flush=True)\ntime.sleep(10)",
                           fatal_timeout=0.7, timeout=1.4)
        error = raised.exception
        self.assertIsInstance(error, subprocess.TimeoutExpired)
        self.assertEqual(error.timeout, 1.4)
        self.assertIn("before parent timeout", error.stdout)
        self.assertIn(str(error.artifact_dir), str(error))
        metadata = json.loads((error.artifact_dir / "run.json").read_text())
        self.assertEqual(metadata["outcome"], "PARENT_TIMEOUT")
        with self.assertRaises(ChildProcessError):
            os.waitpid(metadata["pid"], os.WNOHANG)

    def _run_with_setup_delay(self, boundary):
        # Advance only the launcher's clock: real child execution stays fast,
        # and the regression does not depend on a slow machine or tight timing.
        elapsed_setup = [0]
        clock = SimpleNamespace(monotonic=lambda: time.monotonic() + elapsed_setup[0])
        real_popen = subprocess.Popen
        real_write_json = capture._write_json

        def launch(*args, **kwargs):
            if boundary == "launch":
                elapsed_setup[0] += 10
            return real_popen(*args, **kwargs)

        def write_metadata(path, values):
            real_write_json(path, values)
            if boundary == "metadata" and values.get("outcome") == "STARTED":
                elapsed_setup[0] += 10

        with patch.object(capture, "time", clock), \
                patch.object(capture.subprocess, "Popen", launch), \
                patch.object(capture, "_write_json", write_metadata):
            try:
                result = self.run_child("import time\ntime.sleep(0.25)\nprint('completed')")
            except capture.CapturedTimeout as error:
                self.fail(f"{boundary} setup consumed the child's timeout budget: {error}")
        self.assertEqual(result.completed.returncode, 0, result.completed.stderr)
        self.assertEqual(result.completed.stdout, "completed\n")
        metadata = self.metadata(result)
        self.assertEqual(metadata["outcome"], "EXITED_ZERO")
        self.assertEqual((metadata["fatal_timeout"], metadata["timeout"]), (3, 5))

    def test_launch_time_does_not_consume_parent_timeout(self):
        self._run_with_setup_delay("launch")

    def test_started_metadata_time_does_not_consume_parent_timeout(self):
        self._run_with_setup_delay("metadata")

    def test_invalid_deadlines_do_not_launch_or_create_artifacts(self):
        for fatal, parent in [(0, 1), (2, 1), (1, 1), (float("nan"), 2), (1, float("inf"))]:
            with self.subTest(fatal=fatal, parent=parent), self.assertRaises(ValueError):
                self.run_child("raise AssertionError('must not execute')", fatal_timeout=fatal, timeout=parent)
        self.assertFalse((self.root / ".test-artifacts").exists())

    def test_unusable_artifact_root_is_an_actionable_error(self):
        blocked = self.root / "blocked"
        blocked.write_text("owned file")
        self.env["TINY_TEST_ARTIFACT_DIR"] = str(blocked / "child")
        with self.assertRaisesRegex(OSError, "artifact"):
            self.run_child("print('must not silently run')")
        self.assertEqual(blocked.read_text(), "owned file")

    def test_spawn_failure_is_recorded_without_false_success(self):
        with patch.object(sys, "executable", str(self.root / "missing-python")):
            with self.assertRaises(OSError):
                self.run_child("print('cannot run')")
        records = list((self.root / ".test-artifacts/subprocess").glob("*/run.json"))
        self.assertEqual(len(records), 1)
        metadata = json.loads(records[0].read_text())
        self.assertEqual(metadata["outcome"], "SPAWN_ERROR")
        self.assertIsNone(metadata["returncode"])

    def test_missing_resource_observations_are_unknown(self):
        snapshot = capture._resource_snapshot(-1)
        self.assertIsNone(snapshot["process"]["status"])
        self.assertTrue(snapshot["unavailable"])
        self.assertNotIn("environ", json.dumps(snapshot))

    @unittest.skipUnless(sys.platform == "linux", "owned orphan reaping uses Linux subreaper")
    def test_abrupt_parent_exit_retains_started_evidence(self):
        # This test supervisor adopts/reaps only the group it creates.
        libc = ctypes.CDLL(None, use_errno=True)
        previous = ctypes.c_int()
        self.assertEqual(libc.prctl(37, ctypes.byref(previous), 0, 0, 0), 0)
        self.assertEqual(libc.prctl(36, 1, 0, 0, 0), 0)
        repo = Path(__file__).resolve().parents[1]
        artifacts = self.root / "abrupt"
        child = "import time\nprint('durable before abrupt exit', flush=True)\ntime.sleep(20)"
        parent_source = f"""
import os, threading, time
from pathlib import Path
from tests.subprocess_capture import capture_python
environment = os.environ.copy()
environment['TINY_TEST_ARTIFACT_DIR'] = {str(artifacts)!r}
def launch():
    capture_python({child!r}, cwd=Path({str(repo)!r}), env=environment,
                   test_id='abrupt-parent', fatal_timeout=5, timeout=7)
threading.Thread(target=launch, daemon=True).start()
deadline = time.monotonic() + 4
while time.monotonic() < deadline:
    logs = list(Path({str(artifacts)!r}).glob('*/stdout.log'))
    if logs and b'durable before abrupt exit' in logs[0].read_bytes():
        os._exit(7)
    time.sleep(0.02)
os._exit(8)
"""
        process = None
        try:
            process = subprocess.Popen([sys.executable, "-c", parent_source], cwd=repo,
                                       stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                                       start_new_session=True)
            _, stderr = process.communicate(timeout=8)
            self.assertEqual(process.returncode, 7, stderr)
            records = list(artifacts.glob("*/run.json"))
            self.assertEqual(len(records), 1)
            metadata = json.loads(records[0].read_text())
            self.assertEqual(metadata["outcome"], "STARTED")
            self.assertIsNone(metadata["returncode"])
            self.assertIn("durable before abrupt exit", (records[0].parent / "stdout.log").read_text())
        finally:
            if process is not None:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait(timeout=3)
                while True:
                    try:
                        os.waitpid(-process.pid, 0)
                    except ChildProcessError:
                        break
            libc.prctl(36, previous.value, 0, 0, 0)


if __name__ == "__main__":
    unittest.main()
