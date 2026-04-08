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
    assert "Created event: Morning Gathering" in output, "Missing 'Created event' log in output."
    assert output.count("Processing turn for ") >= 2, f"Expected at least 2 turns, but found {output.count('Processing turn for ')}."
    assert output.count("Turn processed successfully") >= 2, f"Expected at least 2 successful turns, but found {output.count('Turn processed successfully')}."
    assert "Added event notification:" in output, "Missing 'event notification' log in output."
    assert "Action resolved: Rest" in output, "Missing 'Action resolved' log in output."
    assert "Total actions tracked:" in output, "Missing 'Total actions tracked' log in output."
    assert "Minimal demo completed successfully" in output, "Missing 'completed successfully' log in output."
    assert "Traceback" not in output, f"Found a Traceback in the output:\n{output}"
