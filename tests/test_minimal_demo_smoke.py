import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_minimal_demo_runs_and_reports_progress():
    result = subprocess.run(
        [sys.executable, "main.py", "--mode", "minimal", "--headless"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=120,
    )

    output = result.stdout + result.stderr

    assert result.returncode == 0, output
    assert "Created event: Morning Gathering" in output
    assert output.count("Processing turn for ") >= 2
    assert output.count("Turn processed successfully") >= 2
    assert "Added event notification:" in output
    assert "Action resolved: Rest" in output
    assert "Total actions tracked:" in output
    assert "Minimal demo completed successfully" in output
    assert "Traceback" not in output
