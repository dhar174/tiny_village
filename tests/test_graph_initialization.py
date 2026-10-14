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


class TestGraphLifecycle(FreshPythonTestCase):
    def test_accessors_share_a_ready_graph(self):
        self.run_python("""
            from tiny_globals import get_global_graph_manager, initialize_global_graph_manager
            graph = get_global_graph_manager()
            from tiny_graph_manager import GraphManager
            assert graph is initialize_global_graph_manager() is GraphManager()
            assert graph.G is graph.world_state.graph
            assert graph.graph_analytics.world_state is graph.world_state
            assert graph.social_model.world_state is graph
        """)

    def test_cold_reset_does_not_import_graph(self):
        self.run_python("""
            import sys
            from tiny_globals import reset_global_graph_manager, has_global_graph_manager
            assert 'tiny_graph_manager' not in sys.modules
            reset_global_graph_manager()
            assert 'tiny_graph_manager' not in sys.modules, 'Reset imported the graph module'
            assert not has_global_graph_manager()
        """)

    def test_reset_recreates_both_singleton_references(self):
        self.run_python("""
            from tiny_globals import get_global_graph_manager, reset_global_graph_manager
            from tiny_graph_manager import GraphManager
            old = get_global_graph_manager()
            reset_global_graph_manager()
            assert GraphManager._instance is None
            new = get_global_graph_manager()
            assert new is GraphManager() and new is not old
            assert new.unique_graph_id != old.unique_graph_id
            assert new.G is new.world_state.graph
        """)

    def test_setter_aligns_the_real_class_singleton(self):
        self.run_python("""
            from tiny_globals import get_global_graph_manager, reset_global_graph_manager, set_global_graph_manager
            from tiny_graph_manager import GraphManager
            graph = get_global_graph_manager()
            reset_global_graph_manager()
            set_global_graph_manager(graph)
            assert get_global_graph_manager() is GraphManager() is graph
            set_global_graph_manager(None)
            assert get_global_graph_manager() is graph
            supplied = object()
            set_global_graph_manager(supplied)
            assert get_global_graph_manager() is supplied
        """)

    def test_concurrent_mixed_first_access(self):
        self.run_python("""
            from concurrent.futures import ThreadPoolExecutor
            import threading, time
            from unittest.mock import patch
            import tiny_graph_manager as module
            from tiny_globals import get_global_graph_manager, initialize_global_graph_manager, reset_global_graph_manager
            original_world = module.WorldState
            constructors = [get_global_graph_manager, initialize_global_graph_manager, module.GraphManager]
            for round_index in range(10):
                reset_global_graph_manager()
                barrier = threading.Barrier(8)
                calls = []
                def build_world():
                    calls.append(1)
                    time.sleep(0.01)
                    return original_world()
                def access(index):
                    barrier.wait(timeout=5)
                    graph = constructors[index % 3]()
                    assert graph.G is graph.world_state.graph
                    assert graph.graph_analytics.world_state is graph.world_state
                    assert graph.social_model.world_state is graph
                    return graph
                with patch.object(module, 'WorldState', side_effect=build_world):
                    with ThreadPoolExecutor(max_workers=8) as pool:
                        futures = [pool.submit(access, index) for index in range(8)]
                        graphs = [future.result(timeout=10) for future in futures]
                assert len(calls) == 1, (round_index, 'duplicate WorldState construction', len(calls))
                assert all(graph is graphs[0] for graph in graphs)
                assert len({graph.unique_graph_id for graph in graphs}) == 1
        """)

    def test_recursion_is_prompt_and_retryable(self):
        for entry in ("get_global_graph_manager", "GraphManager"):
            with self.subTest(entry=entry):
                self.run_python(f"""
                    from unittest.mock import patch
                    import tiny_graph_manager as module
                    from tiny_graph_manager import GraphManager
                    from tiny_globals import get_global_graph_manager, has_global_graph_manager, reset_global_graph_manager
                    with patch.object(module, 'WorldState', side_effect=get_global_graph_manager):
                        try:
                            {entry}()
                        except RuntimeError as error:
                            assert 'Recursive GraphManager initialization' in str(error)
                        else:
                            raise AssertionError('Recursive initialization succeeded')
                    assert not has_global_graph_manager()
                    assert GraphManager._instance is None
                    graph = get_global_graph_manager()
                    assert graph is GraphManager() and graph.G is graph.world_state.graph
                """)

    def test_failed_initialization_is_retryable(self):
        for entry in ("get_global_graph_manager", "GraphManager"):
            with self.subTest(entry=entry):
                self.run_python(f"""
                    from unittest.mock import patch
                    import tiny_graph_manager as module
                    from tiny_graph_manager import GraphManager
                    from tiny_globals import get_global_graph_manager, has_global_graph_manager
                    failure = ValueError('controlled WorldState failure')
                    with patch.object(module, 'WorldState', side_effect=failure):
                        try:
                            {entry}()
                        except ValueError as error:
                            assert error is failure
                        else:
                            raise AssertionError('Construction failure was suppressed')
                    assert not has_global_graph_manager()
                    assert GraphManager._instance is None, 'Failed class singleton survived'
                    graph = get_global_graph_manager()
                    assert graph is GraphManager() and graph.G is graph.world_state.graph
                """)

    def test_constructing_thread_can_query_readiness(self):
        self.run_python("""
            from unittest.mock import patch
            import tiny_graph_manager as module
            from tiny_globals import get_global_graph_manager, has_global_graph_manager
            original_world = module.WorldState
            def build_world():
                assert not has_global_graph_manager(), 'Graph was published before construction finished'
                return original_world()
            with patch.object(module, 'WorldState', side_effect=build_world):
                graph = get_global_graph_manager()
            assert has_global_graph_manager()
            assert graph.G is graph.world_state.graph
        """)

    def test_recursive_lifecycle_mutation_is_rejected(self):
        for mutation in ("reset_global_graph_manager()", "set_global_graph_manager(None)"):
            with self.subTest(mutation=mutation):
                self.run_python(f"""
                    from unittest.mock import patch
                    import tiny_graph_manager as module
                    from tiny_globals import get_global_graph_manager, reset_global_graph_manager, set_global_graph_manager
                    with patch.object(module, 'WorldState', side_effect=lambda: {mutation}):
                        try:
                            get_global_graph_manager()
                        except RuntimeError as error:
                            assert 'Recursive GraphManager initialization' in str(error)
                        else:
                            raise AssertionError('Recursive lifecycle mutation succeeded')
                    assert module.GraphManager._instance is None
                    assert get_global_graph_manager().G is not None
                """)

    def test_waiting_callers_progress_after_a_construction_failure(self):
        self.run_python("""
            from concurrent.futures import ThreadPoolExecutor
            import threading, time
            from unittest.mock import patch
            import tiny_graph_manager as module
            from tiny_globals import get_global_graph_manager, initialize_global_graph_manager
            original_world = module.WorldState
            barrier = threading.Barrier(8)
            calls = []
            counter_lock = threading.Lock()
            failure = ValueError('first construction failed')
            def build_world():
                with counter_lock:
                    calls.append(1)
                    attempt = len(calls)
                time.sleep(0.01)
                if attempt == 1:
                    raise failure
                return original_world()
            entries = [get_global_graph_manager, initialize_global_graph_manager, module.GraphManager]
            def access(index):
                barrier.wait(timeout=5)
                try:
                    return entries[index % 3]()
                except ValueError as error:
                    assert error is failure
                    return None
            with patch.object(module, 'WorldState', side_effect=build_world):
                with ThreadPoolExecutor(max_workers=8) as pool:
                    futures = [pool.submit(access, index) for index in range(8)]
                    results = [future.result(timeout=10) for future in futures]
            assert results.count(None) == 1
            graphs = [graph for graph in results if graph is not None]
            assert len(calls) == 2, ('failure/retry initialized extra worlds', len(calls))
            assert all(graph is graphs[0] for graph in graphs)
            assert get_global_graph_manager() is module.GraphManager() is graphs[0]
            assert graphs[0].G is graphs[0].world_state.graph
        """)

    def test_direct_constructor_recursion_is_rejected(self):
        self.run_python("""
            from unittest.mock import patch
            import tiny_graph_manager as module
            with patch.object(module, 'WorldState', side_effect=module.GraphManager):
                try:
                    module.GraphManager()
                except RuntimeError as error:
                    assert 'Recursive GraphManager initialization' in str(error)
                else:
                    raise AssertionError('Direct recursive construction succeeded')
            assert module.GraphManager._instance is None
            assert module.GraphManager().G is not None
        """)

    def test_failed_import_keeps_its_cause_and_allows_retry(self):
        self.run_python("""
            import importlib.abc, sys
            from tiny_globals import initialize_global_graph_manager, has_global_graph_manager
            failure = ImportError('controlled graph import failure')
            class UnavailableGraph(importlib.abc.MetaPathFinder):
                def find_spec(self, fullname, path=None, target=None):
                    if fullname == 'tiny_graph_manager':
                        raise failure
            finder = UnavailableGraph()
            sys.meta_path.insert(0, finder)
            try:
                initialize_global_graph_manager()
            except ImportError as error:
                assert error.__cause__ is failure
            else:
                raise AssertionError('Import failure was suppressed')
            finally:
                sys.meta_path.remove(finder)
            assert not has_global_graph_manager()
            assert initialize_global_graph_manager().G is not None
        """)


if __name__ == "__main__":
    unittest.main()
