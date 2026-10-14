"""Real runtime objects retain explicit world ownership during composition."""

from tests.test_graph_initialization import FreshPythonTestCase
import unittest


class TestGraphOwnership(FreshPythonTestCase):
    def test_strategy_generated_actions_keep_the_explicit_world(self):
        self.run_python("""
            from unittest.mock import patch
            from tiny_graph_manager import GraphManager
            from tiny_globals import set_global_graph_manager
            from tiny_strategy_manager import StrategyManager, EatAction, SleepAction, WorkAction, NoOpAction, WanderAction
            owner = GraphManager()
            other = object()
            set_global_graph_manager(other)
            strategy = StrategyManager(graph_manager=owner)
            actions = strategy.get_daily_actions({'name': 'Ada', 'energy': 5, 'hunger_level': 5})
            assert actions, 'Strategy generated no actions'
            assert all(action.graph_manager is owner for action in actions), 'Generated action resolved a different global world'
            with patch('tiny_globals.get_global_graph_manager', side_effect=AssertionError('Unexpected global access')):
                actions = strategy.get_daily_actions({'name': 'Ada', 'energy': 5, 'hunger_level': 5})
                helpers = [kind(graph_manager=owner) for kind in (EatAction, SleepAction, WorkAction, NoOpAction, WanderAction)]
            assert all(action.graph_manager is owner for action in actions + helpers)
            assert strategy.goap_planner.graph_manager is owner
        """)

    def test_strategy_uses_an_explicit_owner_and_preserves_positional_arguments(self):
        self.run_python("""
            from unittest.mock import patch
            from tiny_graph_manager import GraphManager
            from tiny_strategy_manager import StrategyManager
            from tiny_goap_system import GOAPPlanner
            graph = GraphManager()
            with patch('tiny_globals.get_global_graph_manager', side_effect=AssertionError('Unexpected global access')):
                for strategy in (StrategyManager(graph_manager=graph), StrategyManager(False, None, graph)):
                    assert strategy.graph_manager is graph
                    assert isinstance(strategy.goap_planner, GOAPPlanner)
                    assert strategy.goap_planner.graph_manager is graph
                    assert not strategy.use_llm
        """)

    def test_strategy_preserves_legacy_global_access(self):
        self.run_python("""
            from tiny_globals import get_global_graph_manager
            from tiny_strategy_manager import StrategyManager
            graph = get_global_graph_manager()
            strategy = StrategyManager(False, None)
            assert strategy.graph_manager is graph
            assert strategy.goap_planner.graph_manager is graph
        """)

    def test_controller_composes_and_recovers_with_the_explicit_owner(self):
        self.run_python("""
            import tempfile
            from unittest.mock import patch
            from tiny_graph_manager import GraphManager
            from tiny_gameplay_controller import GameplayController
            from tiny_globals import has_global_graph_manager
            import pygame
            graph = GraphManager()
            with tempfile.TemporaryDirectory() as checkpoint_directory:
                config = {'headless': True, 'characters': {'use_llm': False, 'count': 0},
                    'checkpoint': {'directory': checkpoint_directory, 'auto_enabled': False}}
                with patch('tiny_globals.get_global_graph_manager', side_effect=AssertionError('Unexpected global access')):
                    try:
                        controller = GameplayController(graph_manager=graph, config=config)
                        assert controller.graph_manager is graph
                        assert controller.strategy_manager.graph_manager is graph
                        assert controller.strategy_manager.goap_planner.graph_manager is graph
                        assert controller.action_system.action_generator.graph_manager is graph
                        assert controller.action_resolver.graph_manager is graph
                        assert controller.event_handler.graph_manager is graph
                        assert not has_global_graph_manager(), 'Explicit composition initialized global provider'
                        controller.strategy_manager = None
                        assert controller.recovery_manager._recover_strategy_manager()
                        assert controller.strategy_manager.graph_manager is graph
                        assert controller.strategy_manager.goap_planner.graph_manager is graph
                    finally:
                        pygame.quit()
        """)

    def test_location_actions_retain_the_supplied_graph(self):
        self.run_python("""
            from unittest.mock import patch
            from tiny_graph_manager import GraphManager
            from actions import ActionSystem
            from tiny_locations import Location
            graph = GraphManager()
            system = ActionSystem(graph_manager=graph)
            with patch('tiny_globals.get_global_graph_manager', side_effect=AssertionError('Unexpected global access')):
                location = Location('Square', 0, 0, 10, 10, system)
            assert location.possible_interactions
            assert all(action.graph_manager is graph for action in location.possible_interactions)
        """)

    def test_items_with_configured_effects_preserve_the_supplied_action_system(self):
        for constructor in (
            "ItemObject('Tool', 'A tool', 1, 1, 1, action_system=system)",
            "FoodItem('Apple', 'A fruit', 1, True, 0.5, 1, system, calories=2)",
            "Door('Door', 'A door', 1, 1, 1, system)",
        ):
            with self.subTest(constructor=constructor):
                self.run_python(f"""
                    from unittest.mock import patch
                    from tiny_graph_manager import GraphManager
                    from actions import ActionSystem
                    from tiny_items import ItemObject, FoodItem, Door
                    import tiny_items
                    graph = GraphManager()
                    system = ActionSystem(graph_manager=graph)
                    with patch('tiny_globals.get_global_graph_manager', side_effect=AssertionError('Unexpected global access')):
                        # Door's default Open effects are absent in current main.
                        # Supply action-definition input to test ownership only;
                        # repairing the production definition is a separate issue.
                        with patch.dict(tiny_items.effect_dict, {{'Open': [{{
                            'targets': ['target'], 'attribute': 'status', 'change_value': 'set:open'
                        }}]}}):
                            item = {constructor}
                    assert item.action_system is system
                    assert all(action.graph_manager is graph for action in item.location.possible_interactions)
                    assert all(action.graph_manager is graph for action in item.possible_interactions)
                """)

    def test_event_preserves_a_falsy_supplied_action_system(self):
        self.run_python("""
            from unittest.mock import patch
            from tiny_graph_manager import GraphManager
            from actions import ActionSystem
            from tiny_event_handler import Event
            class FalsyActionSystem(ActionSystem):
                def __bool__(self):
                    return False
            graph = GraphManager()
            system = FalsyActionSystem(graph_manager=graph)
            with patch('tiny_globals.get_global_graph_manager', side_effect=AssertionError('Unexpected global access')):
                event = Event('Festival', '2022-07-04', 'annual', 0.8, 0.8, action_system=system)
            assert event.action_system is system
            assert event.action_system.action_generator.graph_manager is graph
        """)

    def test_example_factory_builds_fresh_objects_in_the_supplied_world(self):
        self.run_python("""
            from unittest.mock import patch
            from tiny_graph_manager import GraphManager
            from tiny_characters import build_example_world_criteria
            graph = GraphManager()
            with patch('tiny_globals.get_global_graph_manager', side_effect=AssertionError('Unexpected global access')):
                first = build_example_world_criteria(graph)
                second = build_example_world_criteria(graph)
            event = first['example_criteria_i'][0]['event_participation']
            inventory = first['example_criteria_j'][0]['offer_item_trade']
            apple = inventory.food_items[0]
            assert event.name == 'Festival' and apple.name == 'Apple'
            assert event is not second['example_criteria_i'][0]['event_participation']
            assert apple is not second['example_criteria_j'][0]['offer_item_trade'].food_items[0]
            assert event.action_system.action_generator.graph_manager is graph
            assert apple.action_system is event.action_system
            assert all(action.graph_manager is graph for action in apple.possible_interactions)
            assert all(action.graph_manager is graph for action in apple.location.possible_interactions)
            try:
                build_example_world_criteria(None)
            except ValueError:
                pass
            else:
                raise AssertionError('Examples accepted an unspecified owner')
        """)

    def test_character_binds_ownership_before_optional_prompt_initialization(self):
        # Stop at a documented integration boundary, after the real constructor's
        # graph, action-system, actions and GOAP setup. This does not claim that
        # downstream character statistics/memory initialization succeeds.
        for supplied in ("None", "system"):
            with self.subTest(action_system=supplied):
                self.run_python(f"""
                    from unittest.mock import patch
                    from tiny_graph_manager import GraphManager
                    from tiny_characters import Character
                    from actions import ActionSystem, Action
                    graph = GraphManager()
                    system = ActionSystem(graph_manager=graph)
                    character = Character.__new__(Character)
                    class StopAtPrompt(RuntimeError):
                        pass
                    with patch('tiny_globals.get_global_graph_manager', side_effect=AssertionError('Unexpected global access')):
                        with patch('tiny_prompt_builder.PromptBuilder', side_effect=StopAtPrompt):
                            try:
                                character.__init__('TestSubject', 30, graph_manager=graph, action_system={supplied})
                            except StopAtPrompt:
                                pass
                            else:
                                raise AssertionError('Prompt boundary was not reached')
                    assert character.graph_manager is graph
                    assert character.action_system.action_generator.graph_manager is graph
                    assert character.goap_planner.graph_manager is graph
                    assert len(character.character_actions) == 12
                    assert all(isinstance(action, Action) and action.graph_manager is graph for action in character.character_actions)
                    if {supplied!r} == 'system':
                        assert character.action_system is system
                """)

    def test_real_graph_records_the_named_target_effect(self):
        self.run_python("""
            from tiny_graph_manager import GraphManager
            from actions import Action, ActionSystem
            from tiny_locations import Location
            from demo_character_factory import Character
            graph = GraphManager()
            system = ActionSystem(graph_manager=graph)
            location = Location('Square', 0, 0, 10, 10, system)
            initiator = Character(name='Initiator', social_wellbeing=5)
            target = Character(name='Target', social_wellbeing=3)
            bystander = Character(name='Bystander', social_wellbeing=9)
            for character in (initiator, target, bystander):
                character.uuid = character.name.lower() + '-uuid'
                character.location = location
                graph.characters[character.uuid] = character
                graph.G.add_node(character.uuid, type='character', social_wellbeing=character.social_wellbeing)
            location.current_visitors.extend([initiator, target, bystander, target])
            action = Action('Encourage', [], [{'targets': ['target_character_in_location'],
                'attribute': 'social_wellbeing', 'change_value': 2}],
                initiator=initiator, target=target.name, graph_manager=graph)
            assert action.execute()
            assert target.social_wellbeing == 5
            assert initiator.social_wellbeing == 5 and bystander.social_wellbeing == 9
            assert graph.G.nodes[target.uuid]['social_wellbeing'] == 5
            assert graph.G.nodes[bystander.uuid]['social_wellbeing'] == 9
            assert graph.G is graph.world_state.graph
        """)


if __name__ == '__main__':
    unittest.main()
