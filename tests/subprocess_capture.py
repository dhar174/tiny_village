"""Standard-library evidence capture for bounded fresh-interpreter tests."""

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import threading
import time
from typing import Mapping


@dataclass(frozen=True)
class PythonCapture:
    completed: subprocess.CompletedProcess[str]
    artifact_dir: Path


class CapturedTimeout(subprocess.TimeoutExpired):
    def __init__(self, cmd, timeout, *, artifact_dir, output, stderr):
        super().__init__(cmd, timeout, output=output, stderr=stderr)
        self.artifact_dir = artifact_dir

    def __str__(self):
        return f"Child exceeded {self.timeout}s; artifacts: {self.artifact_dir}"


def _write_json(path, values):
    descriptor, temporary = tempfile.mkstemp(prefix=".capture-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(values, stream, indent=2)
            stream.write("\n")
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def _resource_snapshot(pid):
    """Read only bounded, known process/cgroup files; unknown is never zero."""
    snapshot = {"monotonic": time.monotonic(), "process": {}, "cgroup": {}, "unavailable": {}}

    def read(path, label):
        try:
            descriptor = os.open(path, os.O_RDONLY | os.O_NONBLOCK)
            try:
                return os.read(descriptor, 65536).decode("utf-8", errors="replace")
            finally:
                os.close(descriptor)
        except OSError as error:
            snapshot["unavailable"][label] = type(error).__name__
            return None

    if not sys.platform.startswith("linux"):
        snapshot["process"]["status"] = None
        snapshot["unavailable"]["platform"] = "Linux /proc unavailable"
        return snapshot
    for name in ("status", "stat", "wchan", "schedstat"):
        snapshot["process"][name] = read(Path(f"/proc/{pid}/{name}"), "process/" + name)
    for name in ("cpu.stat", "memory.events", "cpu.pressure", "io.pressure"):
        snapshot["cgroup"][name] = read(Path("/sys/fs/cgroup") / name, "cgroup/" + name)
    return snapshot


def _snapshot_to_file(pid, path):
    # A daemon observer cannot delay the parent's original process deadline.
    # If the owning test removes its artifact root, there is nowhere to save it.
    try:
        _write_json(path, _resource_snapshot(pid))
    except OSError:
        pass


def _child_program(source, events, fatal_timeout):
    # Execute the original source in __main__, including legal future imports.
    # Locals retain the event writer even if source changes its global namespace.
    return f"""import faulthandler
faulthandler.dump_traceback_later({fatal_timeout!r}, exit=True)
def _tiny_capture_run():
    import atexit, json, sys, time
    from pathlib import Path
    event_path = Path({str(events)!r})
    def emit(phase):
        with event_path.open('a', encoding='utf-8') as stream:
            stream.write(json.dumps({{'phase': phase, 'monotonic': time.monotonic()}}) + '\\n')
            stream.flush()
    atexit.register(emit, 'atexit_marker')
    emit('child_ready')
    exec(compile({source!r}, '<string>', 'exec'), vars(sys.modules['__main__']))
    emit('source_complete')
_tiny_capture_run()
"""


def capture_python(
    source: str, *, cwd: Path, env: Mapping[str, str], test_id: str,
    fatal_timeout: float, timeout: float,
) -> PythonCapture:
    """Capture emitted bytes durably; preserve the direct child's time limits.

    Children inherit the caller's process group. A verification supervisor must
    own descendant cleanup, especially when an aggregate timer kills the caller.
    A STARTING/STARTED run without a terminal outcome is incomplete, not passing.
    """
    if not (math.isfinite(fatal_timeout) and math.isfinite(timeout)
            and 0 < fatal_timeout < timeout):
        raise ValueError("Expected finite deadlines: 0 < fatal_timeout < timeout")
    cwd = Path(cwd).resolve()
    root = Path(env.get("TINY_TEST_ARTIFACT_DIR", str(cwd / ".test-artifacts/subprocess")))
    if not root.is_absolute():
        root = cwd / root
    try:
        root.mkdir(mode=0o700, parents=True, exist_ok=True)
        artifact_dir = Path(tempfile.mkdtemp(prefix="child-", dir=root))
    except OSError as error:
        raise OSError(f"Cannot create subprocess artifact directory under {root}: {error}") from error

    metadata = {
        "schema_version": 1, "test_id": test_id,
        "source_sha256": hashlib.sha256(source.encode("utf-8")).hexdigest(),
        "interpreter": sys.executable, "cwd": str(cwd), "pid": None,
        "fatal_timeout": fatal_timeout, "timeout": timeout,
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "started_monotonic": time.monotonic(), "returncode": None,
        "outcome": "STARTING", "watchdog_dump_observed": False,
    }
    metadata_path = artifact_dir / "run.json"
    _write_json(metadata_path, metadata)
    paths = {name: artifact_dir / name for name in ("stdout.log", "stderr.log", "events.jsonl")}
    for path in paths.values():
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        os.close(descriptor)
    command = [sys.executable, "-c", _child_program(source, paths["events.jsonl"], fatal_timeout)]
    process = None
    timed_out = False
    try:
        with paths["stdout.log"].open("wb") as stdout, paths["stderr.log"].open("wb") as stderr:
            started = time.monotonic()
            deadline = started + timeout
            try:
                process = subprocess.Popen(command, cwd=cwd, env=env, stdout=stdout, stderr=stderr)
            except OSError as error:
                metadata.update(outcome="SPAWN_ERROR", spawn_error=type(error).__name__)
                _write_json(metadata_path, metadata)
                raise
            metadata.update(pid=process.pid, outcome="STARTED", launched_monotonic=started)
            _write_json(metadata_path, metadata)
            try:
                sample_at = started + max(0, fatal_timeout - 2)
                try:
                    process.wait(timeout=max(0, min(sample_at, deadline) - time.monotonic()))
                except subprocess.TimeoutExpired:
                    if process.poll() is None:
                        threading.Thread(target=_snapshot_to_file,
                                         args=(process.pid, artifact_dir / "resources.json"),
                                         daemon=True).start()
                process.wait(timeout=max(0, deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                timed_out = True
                process.kill()
                process.wait()
            finally:
                if process.poll() is None:
                    process.kill()
                    process.wait()
    finally:
        # Includes a metadata/output exception after Popen, not only wait errors.
        if process is not None and process.poll() is None:
            process.kill()
            process.wait()

    output = paths["stdout.log"].read_bytes().decode("utf-8", errors="replace")
    errors = paths["stderr.log"].read_bytes().decode("utf-8", errors="replace")
    metadata.update(
        returncode=process.returncode,
        outcome="PARENT_TIMEOUT" if timed_out else ("EXITED_ZERO" if process.returncode == 0 else "EXITED_NONZERO"),
        finished_utc=datetime.now(timezone.utc).isoformat(), finished_monotonic=time.monotonic(),
        watchdog_dump_observed=bool(re.search(r"(?m)^Timeout \([^\r\n)]+\)!\r?$", errors)),
    )
    _write_json(metadata_path, metadata)
    if timed_out:
        raise CapturedTimeout(command, timeout, artifact_dir=artifact_dir, output=output, stderr=errors)
    return PythonCapture(subprocess.CompletedProcess(command, process.returncode, output, errors), artifact_dir)
