"""Headless runtime and CLI behavior using real domain components."""

from tests.test_graph_initialization import FreshPythonTestCase
from tests.test_memory_runtime import NO_AI_IMPORTS


class TestHeadlessRuntime(FreshPythonTestCase):
    def test_invalid_character_input_raises_instead_of_exiting_process(self):
        self.run_python(NO_AI_IMPORTS + '''
from tiny_gameplay_controller import GameplayController
from tiny_graph_manager import GraphManager
try:
    GameplayController(GraphManager(), {'headless':True, 'memory':{'mode':'basic'},
        'characters':{'count':1, 'use_llm':False, 'use_custom':True,
            'character_data':[{'name':'Invalid job', 'job':'No such job 654'}]},
        'checkpoint':{'auto_enabled':False}})
except RuntimeError as error:
    assert 'character' in str(error).lower()
else:
    raise AssertionError('Invalid required Character input was accepted')
''')

    def test_controller_creates_real_characters_without_rendering(self):
        self.run_python(NO_AI_IMPORTS + '''
from tiny_gameplay_controller import GameplayController
from tiny_graph_manager import GraphManager
from tiny_characters import Character
graph = GraphManager()
controller = GameplayController(graph, {'headless': True,
    'memory': {'mode':'basic'}, 'characters': {'count':2, 'use_llm':False},
    'checkpoint': {'auto_enabled':False}})
try:
    assert len(controller.characters) == 2
    assert all(isinstance(c, Character) and c._initialized for c in controller.characters.values())
    assert all(c.graph_manager is graph for c in controller.characters.values())
    assert controller.screen is None and controller.clock is None
    assert controller.map_controller is None
    assert not controller.ui_fonts and not controller.ui_panels
    assert not controller.initialization_errors, controller.initialization_errors
    assert controller.recovery_manager.get_system_status()['map_controller'] == 'disabled'
    assert not controller.recovery_manager._recover_map_controller()
    controller.add_event_notification('Real console feedback')
    controller.render()
    controller.handle_events()
    assert not (blocked & sys.modules.keys())
finally:
    controller.close()
    controller.close()
''')

    def test_required_character_failure_is_not_replaced_by_a_fake(self):
        self.run_python(NO_AI_IMPORTS + '''
from unittest.mock import patch
from tiny_gameplay_controller import GameplayController
from tiny_characters import Character
from tiny_graph_manager import GraphManager
with patch.object(Character, '__init__', side_effect=RuntimeError('Controlled real constructor failure')):
    try:
        GameplayController(GraphManager(), {'headless':True, 'memory':{'mode':'basic'},
            'characters':{'count':1,'use_llm':False}, 'checkpoint':{'auto_enabled':False}})
    except RuntimeError as error:
        assert 'character' in str(error).lower()
    else:
        raise AssertionError('Failed required Character replaced or ignored')
assert not (blocked & sys.modules.keys())
''')

    def test_headless_checkpoint_clock_does_not_start_already_due(self):
        self.run_python(NO_AI_IMPORTS + '''
from tiny_gameplay_controller import GameplayController, _ticks_ms
from tiny_graph_manager import GraphManager
controller = GameplayController(GraphManager(), {'headless':True,
    'memory':{'mode':'basic'}, 'characters':{'count':0,'use_llm':False}})
try:
    assert not controller.checkpoint_manager.should_checkpoint(_ticks_ms())
finally:
    controller.close()
assert not (blocked & sys.modules.keys())
''')


class TestMinimalRunner(FreshPythonTestCase):
    def test_demo_returns_real_registered_state_and_memories(self):
        self.run_python(NO_AI_IMPORTS + '''
from demo_minimal_integration import demonstrate_minimal_integration
from tiny_characters import Character
controller = demonstrate_minimal_integration({'characters':{'count':7,'use_llm':True}})
try:
    assert len(controller.characters) == 7
    assert all(isinstance(c, Character) for c in controller.characters.values())
    assert controller.headless and not controller.strategy_manager.use_llm
    event = controller.event_handler.get_event('Minimal Startup Check')
    assert event is not None
    assert controller.graph_manager.G.has_node(event)
    for char in controller.characters.values():
        recent = char.recall_recent_memories()
        assert len(recent) == 1
        assert recent[0].description == f'{char.name} observed Minimal Startup Check'
        assert recent[0].importance_score == 1
        assert char.memory_manager.mode == 'basic'
    assert controller.game_statistics['actions_executed'] == 0
    assert not controller.checkpoint_manager.auto_checkpoint_enabled
    assert controller._temporary_checkpoint_directory is None
    assert not (blocked & sys.modules.keys())
finally:
    controller.close()
''')

    def test_invalid_visual_headless_and_empty_minimal_fail_before_imports(self):
        self.run_python(NO_AI_IMPORTS + '''
import main
from unittest.mock import patch
for args in (['--headless'], ['--mode','visual','--headless'],
             ['--mode','minimal','--characters','0'],
             ['--mode','minimal','--characters','-1']):
    with patch('sys.argv',['main.py', *args]), patch.object(main,'check_dependencies',
            side_effect=AssertionError('Dependency check happened before CLI validation')):
        try:
            main.main()
        except SystemExit as error:
            assert error.code == 2
        else:
            raise AssertionError('Invalid CLI arguments accepted')
assert not (blocked & sys.modules.keys())
''')

    def test_required_demo_failure_produces_nonzero_cli_result(self):
        self.run_python(NO_AI_IMPORTS + '''
from unittest.mock import patch
import main
with patch('sys.argv',['main.py','--mode','minimal','--headless']), \
        patch('demo_minimal_integration.demonstrate_minimal_integration',
              side_effect=RuntimeError('Controlled required demo failure')):
    assert main.main() == 1
''')


class TestVisualLifecycle(FreshPythonTestCase):
    def test_real_renderer_and_owned_cleanup(self):
        self.run_python(NO_AI_IMPORTS.replace("'pygame', ", '') + '''
import os, tempfile
from pathlib import Path
import pygame
from tiny_gameplay_controller import GameplayController
from tiny_graph_manager import GraphManager
root = Path.cwd()
sys.path.insert(0, str(root))
config = {'memory':{'mode':'basic'}, 'characters':{'count':0,'use_llm':False},
    'checkpoint':{'auto_enabled':False},
    'map':{'image_path':str(root/'assets/default_map.png')}}
with tempfile.TemporaryDirectory() as directory:
    os.chdir(directory)
    pygame.quit()
    controller = GameplayController(GraphManager(), config)
    assert pygame.get_init()
    assert controller.screen is not None and controller.map_controller is not None
    assert controller.ui_fonts and controller.ui_panels
    controller.render()
    controller.close()
    controller.close()
    assert config["characters"]["count"] == 0
    assert not pygame.get_init()
    pygame.init()
    foreign_display = pygame.display.set_mode((100,100))
    borrower = GameplayController(GraphManager(), config)
    borrower.close()
    assert pygame.get_init()
    assert pygame.display.get_surface() is foreign_display
    pygame.quit()
assert not (blocked & sys.modules.keys())
''')

