"""Exercise the public CLI and inspect its bounded startup/memory claims."""

import os
from pathlib import Path

import pytest
from tests.subprocess_capture import capture_python


REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(('arguments', 'count'), [
    (['--mode', 'minimal'], 5),
    (['--mode', 'minimal', '--headless', '--no-llm', '--characters', '2'], 2),
])
def test_minimal_demo_runs_and_reports_progress(arguments, count):
    environment = os.environ.copy()
    environment['PYTHONDONTWRITEBYTECODE'] = '1'
    captured = capture_python(
        f'import sys\nsys.argv = {["main.py", *arguments]!r}\n'
        'import main\nstatus = main.main()\nassert status == 0, status\n',
        cwd=REPO_ROOT, env=environment, test_id=f'minimal-cli-{count}',
        fatal_timeout=115, timeout=120,
    )
    result = captured.completed
    output = result.stdout + result.stderr
    assert result.returncode == 0, f'Artifacts: {captured.artifact_dir}\n{output}'
    assert f'Startup checks passed: {count} real Characters' in output
    assert 'Event registration checks passed: Minimal Startup Check' in output
    assert f'Basic memory checks passed: {count} records stored and recalled' in output
    assert 'Gameplay turns were not exercised; actions executed: 0' in output
    assert 'Minimal startup/event/memory checks passed' in output
    assert 'Turn processed successfully' not in output
    assert 'Traceback' not in output
