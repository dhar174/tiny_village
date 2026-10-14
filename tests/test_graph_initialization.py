"""Graph lifecycle regressions run in bounded, fresh interpreters."""

import os
from pathlib import Path
import subprocess
import sys
import textwrap
import unittest


REPO_ROOT = Path(__file__).resolve().parents[1]


class FreshPythonTestCase(unittest.TestCase):
    def run_python(self, source, timeout=25):
        env = os.environ.copy()
        env.update(
            SDL_VIDEODRIVER="dummy",
            SDL_AUDIODRIVER="dummy",
            PYGAME_HIDE_SUPPORT_PROMPT="1",
            PYTHONDONTWRITEBYTECODE="1",
            OMP_NUM_THREADS="2",
            OPENBLAS_NUM_THREADS="2",
        )
        prelude = (
            "import faulthandler, logging\n"
            f"faulthandler.dump_traceback_later({timeout - 5}, exit=True)\n"
            "logging.basicConfig(level=logging.ERROR)\n"
        )
        try:
            result = subprocess.run(
                [sys.executable, "-c", prelude + textwrap.dedent(source)],
                cwd=REPO_ROOT,
                env=env,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired as error:
            self.fail(f"Fresh process exceeded {timeout}s: {error.stdout!r}\n{error.stderr!r}")
        self.assertEqual(
            result.returncode, 0,
            f"Child exited {result.returncode}:\n{result.stdout[-3000:]}\n{result.stderr[-6000:]}",
        )
        return result


class TestGraphImports(FreshPythonTestCase):
    def test_imports_do_not_construct_world(self):
        orders = [
            ["tiny_globals", "tiny_characters", "tiny_graph_manager"],
            ["tiny_graph_manager", "tiny_characters"],
            ["tiny_characters", "tiny_graph_manager"],
            ["tiny_utility_functions", "tiny_graph_manager", "tiny_characters"],
            ["tiny_goap_system", "tiny_characters", "tiny_graph_manager"],
            ["actions", "tiny_locations", "tiny_characters", "tiny_graph_manager"],
            ["tiny_items", "tiny_event_handler", "tiny_characters", "tiny_graph_manager"],
            ["tiny_strategy_manager", "tiny_gameplay_controller", "tiny_graph_manager"],
        ]
        for order in orders:
            with self.subTest(order=order):
                self.run_python(f"""
                    import importlib
                    for module_name in {order!r}:
                        importlib.import_module(module_name)
                    from tiny_globals import has_global_graph_manager
                    from tiny_graph_manager import GraphManager
                    assert not has_global_graph_manager(), 'Import constructed a global world'
                    assert GraphManager._instance is None, 'Import constructed a class singleton'
                """)


if __name__ == "__main__":
    unittest.main()
