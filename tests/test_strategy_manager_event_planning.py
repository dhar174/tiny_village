"""Real event planning outcomes; immediate LLM choices remain a separate API."""

from tests import test_strategy_scale_contracts
from tests.test_graph_initialization import FreshPythonTestCase


class TestStrategyManagerEventPlanning(FreshPythonTestCase):
    run_world = test_strategy_scale_contracts.TestStrategyScaleContracts.run_world

    def test_get_character_state_dict_aliases_wealth_money_for_dict_input(self):
        self.run_world('''
from tiny_strategy_manager import StrategyManager
strategy = StrategyManager(graph_manager=graph)
state = strategy.get_character_state_dict({'name': 'Alice', 'wealth_money': 75})
assert state['money'] == 75.0 and state['wealth_money'] == 75
''')

    def test_get_character_state_dict_only_falls_back_to_satisfaction_when_happiness_missing(self):
        self.run_world('''
from tiny_strategy_manager import StrategyManager
strategy = StrategyManager(graph_manager=graph)
graph.G.nodes[character].update(satisfaction=80, current_satisfaction=80, happiness='not-a-number')
character.current_satisfaction, character.happiness = 80, 'not-a-number'
state = strategy.get_character_state_dict(character)
assert state['satisfaction'] == .8 and state['happiness'] == .5
''')

    def test_get_character_state_dict_uses_satisfaction_when_happiness_attribute_is_missing(self):
        self.run_world('''
from tiny_strategy_manager import StrategyManager
strategy = StrategyManager(graph_manager=graph)
graph.G.nodes[character].update(satisfaction=80, current_satisfaction=80)
character.current_satisfaction = 80
old_happiness = character.happiness
del character.happiness
state = strategy.get_character_state_dict(character)
character.happiness = old_happiness
assert state['satisfaction'] == .8 and state['happiness'] == .8
''')

    def test_events_keep_separate_real_actor_bindings(self):
        self.run_world('''
from types import SimpleNamespace
from tiny_strategy_manager import StrategyManager
strategy = StrategyManager(graph_manager=graph)
partner = Character(name='EventPartner', age=30, job='unemployed', graph_manager=graph,
    personality_traits=character.personality_traits, gametime_manager=character.gametime_manager, memory_mode='basic')
graph.add_character_node(partner)
for actor in (character, partner):
    graph.G.nodes[actor].update(energy=50, current_satisfaction=70, satisfaction=70)
    actor.energy, actor.current_satisfaction = 50, 70
events = [SimpleNamespace(type='unknown', participants=[actor.name]) for actor in (character, partner)]
plans = strategy.update_strategy(events)
assert set(plans) == {character.name.lower(), partner.name.lower()}
first, second = plans[character.name.lower()], plans[partner.name.lower()]
assert first and second and first[0] is not second[0]
for actor, plan in ((character, first), (partner, second)):
    for action in plan:
        assert controller._execute_single_action(actor, action)
        assert action.initiator is actor
    assert graph.G.nodes[actor]['energy'] == actor.energy >= 65
    assert graph.G.nodes[actor]['current_satisfaction'] == actor.current_satisfaction >= 70
assert strategy.goap_planner.plans == {}
''')

    def test_no_event_plan_is_not_replaced_by_daily_fallback(self):
        self.run_world('''
from types import SimpleNamespace
from unittest.mock import patch
from tiny_strategy_manager import StrategyManager
strategy = StrategyManager(graph_manager=graph)
node = graph.G.nodes[character]
node.update(social_wellbeing=10, happiness=.8)
character.social_wellbeing, character.happiness = 10, .8
event = SimpleNamespace(type='social', participants=[character])
with patch.object(strategy, 'plan_daily_activities', side_effect=AssertionError('Goal was replaced')):
    assert strategy.update_strategy([event])[character.name.lower()] is None
social = Action('Meet a friend', [], [
    {'targets': ['initiator'], 'attribute': 'social_wellbeing', 'change_value': 70}], graph_manager=graph)
goal = strategy._goal_for_event_type('social')
plan = strategy._plan_with_goal_and_actions(character, goal, [social])
assert plan == [social]
assert controller._execute_single_action(character, plan[0])
assert node['social_wellbeing'] == character.social_wellbeing == 80
''')

    def test_satisfied_daily_and_optional_llm_paths_preserve_outcomes(self):
        self.run_world('''
from unittest.mock import patch
from tiny_strategy_manager import StrategyManager
strategy = StrategyManager(graph_manager=graph)
node = graph.G.nodes[character]
node.update(energy=80, current_satisfaction=90, satisfaction=90)
character.energy, character.current_satisfaction = 80, 90
with patch.object(strategy, '_ensure_llm', side_effect=AssertionError('GOAP initialized a provider')):
    assert strategy.plan_daily_activities(character) == []
    assert strategy._handle_new_day_strategy(character, False) == []
    assert strategy.decide_action_with_llm(character)  # disabled: immediate choices
strategy.use_llm = True
with patch.object(strategy, '_ensure_llm', return_value=False):
    assert strategy._handle_new_day_strategy(character, True) == []
    node.update(energy=20, current_satisfaction=0, satisfaction=0)
    strategy.goap_planner.MAX_PLANNING_EXPANSIONS = 0
    assert strategy._handle_new_day_strategy(character, True) is None
selected = Action('Provider choice', [], [], graph_manager=graph)
with patch.object(strategy, '_ensure_llm', return_value=True), patch.object(
        strategy, 'decide_action_with_llm', return_value=[selected]):
    result = strategy._handle_new_day_strategy(character, True)
    assert result == [selected] and result[0] is selected
''')

    def test_graph_records_preserve_real_action_identity_and_reject_wrappers(self):
        self.run_world('''
from tiny_goap_system import ActionWrapper
from tiny_strategy_manager import StrategyManager
strategy = StrategyManager(graph_manager=graph)
real = Action('Graph action', [], [], graph_manager=graph)
result = strategy._convert_actions_to_objects([
    {'action': real, 'name': real.name}, real,
    {'name': 'Descriptor only', 'effects': [{'attribute': 'energy', 'change_value': 100}]},
    ActionWrapper('No-op wrapper')])
assert len(result) == 2 and result[0] is real and result[1] is real
''')
