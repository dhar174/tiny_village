"""#218: returned strategy sequences must work through real execution."""

from tests import test_strategy_scale_contracts
from tests.test_graph_initialization import FreshPythonTestCase


class TestStrategyGOAPIntegration(FreshPythonTestCase):
    run_world = test_strategy_scale_contracts.TestStrategyScaleContracts.run_world

    def test_generated_prerequisite_sequence_reaches_production_goal(self):
        self.run_world('''
from unittest.mock import patch
from tiny_strategy_manager import StrategyManager
strategy = StrategyManager(graph_manager=graph)
node = graph.G.nodes[character]
node.update(energy=20, wealth_money=5)
character.energy, character.wealth_money, character.job = 90, 5, 'Baker'
goal = Goal('Earn credits', character, character, 1, 'earn',
    {False: [Condition('earn', 'wealth_money', character, 25)]},
    None, None, None, None, None, None, [], graph, 'Economic', {'wealth_money': 25})
assert not any(a.name.startswith('Work as') for a in strategy.get_daily_actions(character))
candidates = [a for a in strategy._get_planning_actions(character)
              if a.name == 'Rest' or a.name.startswith('Work as')]
before = strategy.get_character_planning_state(character)
plan = strategy.plan_for_goal(character, goal, actions=candidates)
assert [a.name for a in plan] == ['Rest', 'Work as Baker'], plan
from tiny_utility_functions import UtilityGoal
assert strategy.plan_for_goal(character, UtilityGoal('earn', {'wealth_money': 25}), candidates) == plan
assert strategy.get_character_planning_state(character) == before
assert character.energy == 90 and node['energy'] == 20
trace = []
count = controller.game_statistics.get('actions_executed', 0)
with patch.object(graph, 'update_node_attribute', wraps=graph.update_node_attribute) as writes:
    for action in plan:
        assert controller._execute_single_action(character, action)
        assert character.energy == node['energy']
        assert character.wealth_money == node['wealth_money']
        trace.append((node['energy'], node['wealth_money']))
assert trace == [(35, 5), (5, 25)], trace
assert writes.call_count == 4, writes.call_args_list
assert controller.game_statistics['actions_executed'] == count + 2
assert goal.check_completion() and node['wealth_money'] >= 25
assert strategy.plan_for_goal(character, goal, actions=[]) == []
print({'plan': [a.name for a in plan], 'trace': trace, 'goal_complete': True})
''')

    def test_outcomes_validation_and_failure_handoff(self):
        self.run_world('''
from tiny_strategy_manager import StrategyManager
strategy = StrategyManager(graph_manager=graph)
node = graph.G.nodes[character]
goal = Goal('Recover', character, character, 1, 'recover',
    {False: [Condition('rested', 'energy', character, 40)]},
    None, None, None, None, None, None, [], graph, 'Basic', {'energy': 40})
def recovery(name, gain):
    return Action(name, [], [{'targets': ['initiator'], 'attribute': 'energy', 'change_value': gain}],
                  initiator=character, graph_manager=graph)
failed, alternative = recovery('Failed', 20), recovery('Alternative', 10)
assert strategy.plan_for_goal(character, goal, actions=[]) is None
assert strategy.plan_for_goal(character, goal, actions=[failed]) == [failed]
assert not strategy.goap_planner.validate_plan(character, goal, [])
assert not strategy.goap_planner.validate_plan(character, goal, [alternative])
replacement = strategy.handle_plan_execution_failure(character, goal, failed, [failed, alternative])
assert replacement == [alternative, alternative], replacement
assert strategy.goap_planner.validate_plan(character, goal, replacement)
assert strategy.handle_plan_execution_failure(character, goal, failed, [failed]) is None
for action in replacement:
    assert controller._execute_single_action(character, action)
assert goal.check_completion() and node['energy'] == character.energy == 40
assert strategy.handle_plan_execution_failure(character, goal, failed, []) == []
print({'replacement': [a.name for a in replacement], 'energy': node['energy']})
''')

    def test_daily_and_generic_event_plans_execute_to_native_targets(self):
        self.run_world('''
from types import SimpleNamespace
from tiny_strategy_manager import StrategyManager
strategy = StrategyManager(graph_manager=graph)
node = graph.G.nodes[character]
def reset(satisfaction, energy):
    node.update(current_satisfaction=satisfaction, satisfaction=satisfaction, energy=energy)
    character.current_satisfaction, character.energy = satisfaction, energy
reset(70, 50)
plan = strategy._handle_generic_event(None, character)
assert plan and len(plan) >= 1
for action in plan:
    assert controller._execute_single_action(character, action)
assert character.energy == node['energy'] >= 65
assert character.current_satisfaction == node['current_satisfaction'] >= 70
assert strategy._handle_generic_event(None, character) == []
reset(68, 50)
plan = strategy.plan_daily_activities(character.name)
assert plan and len(plan) >= 2
for action in plan:
    assert controller._execute_single_action(character, action)
assert character.energy == node['energy'] >= 65
assert character.current_satisfaction == node['current_satisfaction'] >= 80
event = SimpleNamespace(type='new_day', participants=[character])
assert strategy.update_strategy([event])[character.name.lower()] == []
assert strategy.update_strategy([SimpleNamespace(type='unknown', participants=[character])])[character.name.lower()] == []
strategy.goap_planner.MAX_PLANNING_EXPANSIONS = 0
reset(0, 20)
assert strategy.update_strategy([event])[character.name.lower()] is None
print({'daily_targets': [80, 65], 'generic_targets': [70, 65], 'bounded_failure': True})
''')

    def test_default_daily_state_reaches_wellbeing_within_original_bounds(self):
        self.run_world('''
from tiny_strategy_manager import StrategyManager
strategy = StrategyManager(graph_manager=graph)
node = graph.G.nodes[character]
assert node['energy'] == 20 and node['current_satisfaction'] == 0
assert strategy.goap_planner.MAX_PLANNING_EXPANSIONS == 1000
plan = strategy.plan_daily_activities(character)
assert plan and len(plan) <= 10, plan
for action in plan:
    assert controller._execute_single_action(character, action)
assert node['energy'] == character.energy >= 65
assert node['current_satisfaction'] == character.current_satisfaction >= 80
print({'default_daily_plan': [a.name for a in plan], 'energy': node['energy'],
       'satisfaction': node['current_satisfaction']})
''')

    def test_positive_and_negative_weather_execute_fractional_happiness(self):
        self.run_world('''
from types import SimpleNamespace
from tiny_strategy_manager import StrategyManager
strategy = StrategyManager(graph_manager=graph)
node = graph.G.nodes[character]
node.update(energy=70, happiness=.65)
character.energy, character.happiness = 70, .65
positive = SimpleNamespace(type='weather', impact=5, importance=9, participants=[character])
goal = strategy._goal_for_event_type('weather', situation_context={'impact': 5, 'importance': 9})
assert abs(goal.target_effects['happiness'] - .85) < 1e-9 and goal.priority == .9
plan = strategy.update_strategy([positive])[character.name.lower()]
assert plan and strategy.goap_planner.validate_plan(character, goal, plan)
for action in plan:
    assert controller._execute_single_action(character, action)
assert node['happiness'] == character.happiness >= .85 - 1e-9
assert node['energy'] == character.energy >= 75
node.update(energy=50, safety=65)
character.energy, character.safety = 50, 65
negative = SimpleNamespace(type='weather', impact=-5, participants=[character])
plan = strategy.update_strategy([negative])[character.name.lower()]
assert plan
for action in plan:
    assert controller._execute_single_action(character, action)
assert node['energy'] == character.energy >= 70
assert node['safety'] == character.safety >= 85
print({'happiness': node['happiness'], 'safety': node['safety'], 'energy': node['energy']})
''')

    def test_work_event_keeps_enabling_actions_and_original_goal(self):
        self.run_world('''
from types import SimpleNamespace
from tiny_strategy_manager import StrategyManager
strategy = StrategyManager(graph_manager=graph)
node = graph.G.nodes[character]
node.update(energy=20, job_performance=75, current_satisfaction=57, satisfaction=57)
character.energy, character.job_performance, character.current_satisfaction = 20, 75, 57
plan = strategy.update_strategy([SimpleNamespace(type='work', participants=[character])])[character.name.lower()]
assert plan and any(a.name == 'Rest' for a in plan), plan
for action in plan:
    assert controller._execute_single_action(character, action)
assert node['job_performance'] == character.job_performance >= 85
assert node['current_satisfaction'] == character.current_satisfaction >= 65
print({'work_plan': [a.name for a in plan]})
''')

    def test_target_resolution_isolation_and_unsupported_actions(self):
        self.run_world('''
from tiny_strategy_manager import StrategyManager
from tiny_utility_functions import UtilityGoal
strategy = StrategyManager(graph_manager=graph)
planner = strategy.goap_planner
node = graph.G.nodes[character]
partner = Character(name='PlanPartner', age=30, job='unemployed', graph_manager=graph,
    personality_traits=character.personality_traits, gametime_manager=character.gametime_manager, memory_mode='basic')
graph.add_character_node(partner)
goal = UtilityGoal('recover', {'energy': 40})
def action(name, effects, **kwargs):
    return Action(name, [], effects, graph_manager=graph, **kwargs)
partner_effect = action('Partner only', [{'targets': ['target'], 'attribute': 'energy', 'change_value': 50}], target=partner)
missing = action('No targets', [{'attribute': 'energy', 'change_value': 50}])
mixed = action('Mixed', [{'targets': ['initiator'], 'attribute': 'energy', 'change_value': 50},
                         {'targets': ['target'], 'attribute': 'energy', 'change_value': 10}], target=partner)
dynamic = action('Callback', [{'targets': ['initiator'], 'attribute': 'energy', 'change_value': lambda value: value + 50}])
custom = action('Custom execute', [{'targets': ['initiator'], 'attribute': 'energy', 'change_value': 50}])
custom.execute = lambda **kwargs: (_ for _ in ()).throw(AssertionError('Planning executed custom code'))
for unsupported in (partner_effect, missing, mixed, dynamic, custom):
    assert strategy.plan_for_goal(character, goal, [unsupported]) is None
malformed = action('Malformed descriptor', [None])
assert strategy.plan_for_goal(character, goal, [malformed]) is None
invalid_inventory = action('Numeric inventory', [{'targets': ['initiator'], 'attribute': 'inventory', 'change_value': 1}])
invalid_location = action('Numeric location', [{'targets': ['initiator'], 'attribute': 'location', 'change_value': 1}])
recovery = action('Valid recovery', [{'targets': ['initiator'], 'attribute': 'energy', 'change_value': 20}])
assert strategy.plan_for_goal(character, goal, [invalid_inventory, invalid_location, recovery]) == [recovery]
assert node['energy'] == character.energy == 20
assert controller._execute_single_action(character, missing)
assert node['energy'] == character.energy == 20
implicit = action('Implicit self', [{'targets': ['initiator', 'target'], 'attribute': 'energy', 'change_value': 20}],
                  initiator=partner, default_target_is_initiator=True)
bindings = implicit.initiator, implicit.target, implicit.graph_manager
plan = strategy.plan_for_goal(character, goal, [implicit])
assert plan == [implicit]
assert implicit.initiator is bindings[0] and implicit.target is bindings[1]
assert implicit.graph_manager is bindings[2]
assert controller._execute_single_action(character, implicit)
assert node['energy'] == character.energy == 40  # actor/target duplicate applied once
assert graph.G.nodes[partner]['energy'] == partner.energy
assert planner.plans == {}
print({'unsupported_rejected': 5, 'deduplicated_energy': node['energy']})
''')

    def test_unbound_actor_conditions_and_nested_snapshot_isolation(self):
        self.run_world('''
from copy import deepcopy
from tiny_strategy_manager import StrategyManager
from tiny_utility_functions import UtilityGoal
strategy = StrategyManager(graph_manager=graph)
planner = strategy.goap_planner
state = State({'energy': 20, 'hunger_level': 90, 'nested': {'steps': [1, {'x': 2}]}})
before = deepcopy(state.dict_or_obj)
rest = Action('Rest first', [], [{'targets': ['initiator'], 'attribute': 'energy', 'change_value': 30}], graph_manager=graph)
cook = Action('Cook next', [Condition('actor object', 'energy', character, 45)],
    [{'targets': ['initiator'], 'attribute': 'hunger', 'change_value': -40}], graph_manager=graph)
goal = UtilityGoal('reduce hunger', {'hunger_level': -60})
plan = strategy.plan_for_goal(character, goal, [cook, rest], state)
assert plan == [rest, cook]
assert rest.initiator is None and cook.initiator is None
assert state.dict_or_obj == before
projected = planner._apply_action_effects(rest, state)
projected.dict_or_obj['nested']['steps'][1]['x'] = 99
assert state.dict_or_obj == before
assert planner._hash_state(State({'happiness': .75})) != planner._hash_state(State({'happiness': .750000001}))
assert planner._hash_state(State({'a': [1]})) != planner._hash_state(State({'a': (1,)}))
assert character.energy == graph.G.nodes[character]['energy'] == 20
assert strategy.plan_for_goal(character, UtilityGoal('fraction', {'happiness': .75}), [],
                              {'happiness': .65}) is None
print({'unbound_order': [a.name for a in plan], 'isolated': True})
''')

    def test_depth_sensitive_cost_search_and_expansion_bounds(self):
        self.run_world('''
from tiny_strategy_manager import StrategyManager
from tiny_utility_functions import UtilityGoal
strategy = StrategyManager(graph_manager=graph)
planner = strategy.goap_planner
node = graph.G.nodes[character]
node.update(route=0, done=0)
character.route, character.done = 0, 0
def step(name, origin, destination, cost):
    return Action(name, [Condition('route', 'route', character, origin, '==')],
        [{'targets': ['initiator'], 'attribute': 'route', 'change_value': 'set:' + str(destination)}],
        cost=cost, graph_manager=graph)
a, b = step('Cheap first', 0, 1, .1), step('Cheap second', 1, 2, .1)
shortcut = step('Short route', 0, 2, 10)
finish = Action('Finish', [Condition('ready', 'route', character, 2, '==')],
    [{'targets': ['initiator'], 'attribute': 'done', 'change_value': 1}], cost=.1, graph_manager=graph)
goal = UtilityGoal('finish', {'done': 1})
candidates = [a, b, shortcut, finish]
planner.MAX_PLAN_LENGTH = 2
assert strategy.plan_for_goal(character, goal, candidates) == [shortcut, finish]
planner.MAX_PLAN_LENGTH = 3
assert strategy.plan_for_goal(character, goal, list(reversed(candidates))) == [a, b, finish]
planner.MAX_PLAN_LENGTH = 1
assert strategy.plan_for_goal(character, goal, candidates) is None
planner.MAX_PLAN_LENGTH = 3
planner.MAX_PLANNING_EXPANSIONS = 1
assert strategy.plan_for_goal(character, goal, candidates) is None
planner.MAX_PLANNING_EXPANSIONS = 1000
plan = strategy.plan_for_goal(character, goal, candidates)
for action in plan:
    assert controller._execute_single_action(character, action)
assert node['done'] == character.done == 1
print({'cost_route': [a.name for a in plan], 'done': node['done']})
''')

    def test_malformed_required_state_and_goal_errors_propagate(self):
        self.run_world('''
from tiny_strategy_manager import StrategyManager
from tiny_utility_functions import UtilityGoal
strategy = StrategyManager(graph_manager=graph)
goal = Goal('Recover', character, character, 1, 'recover',
    {False: [Condition('required energy', 'energy', character, 40)]},
    None, None, None, None, None, None, [], graph, 'Basic', {'energy': 40})
for invoke in (lambda: strategy.plan_for_goal(character, goal, [], {}),
               lambda: strategy.plan_for_goal(character, goal, [], {'energy': float('nan')}),
               lambda: strategy.plan_for_goal('MissingCharacter', goal, []),
               lambda: strategy.plan_for_goal(character, UtilityGoal('bad target', {'energy': float('inf')}), [])):
    try:
        invoke()
    except (ValueError, KeyError):
        pass
    else:
        raise AssertionError('Invalid planning contract was swallowed')
assert strategy.plan_for_goal(character, goal, [], {'energy': 40}) == []
strategy.goap_planner = None
assert strategy.plan_daily_activities(character) is None
''')

    def test_partial_execution_failure_replans_from_fresh_graph_state(self):
        self.run_world('''
from unittest.mock import patch
from tiny_strategy_manager import StrategyManager
strategy = StrategyManager(graph_manager=graph)
node = graph.G.nodes[character]
goal = Goal('Recover', character, character, 1, 'recover',
    {False: [Condition('rested', 'energy', character, 60)]},
    None, None, None, None, None, None, [], graph, 'Basic', {'energy': 60})
failed = Action('Partially failing recovery', [], [
    {'targets': ['initiator'], 'attribute': 'energy', 'change_value': 10},
    {'targets': ['initiator'], 'attribute': 'money', 'change_value': 5}], graph_manager=graph)
alternative = Action('Alternative recovery', [], [
    {'targets': ['initiator'], 'attribute': 'energy', 'change_value': 15}], graph_manager=graph)
assert strategy.plan_for_goal(character, goal, [failed])
original_write = graph.update_node_attribute
def reject_money(target, attribute, value):
    if attribute == 'wealth_money':
        raise RuntimeError('Injected synchronization failure')
    return original_write(target, attribute, value)
with patch.object(graph, 'update_node_attribute', side_effect=reject_money):
    assert not controller._execute_single_action(character, failed)
assert node['energy'] == character.energy == 30
assert not goal.check_completion()
replacement = strategy.handle_plan_execution_failure(character, goal, failed, [failed, alternative])
assert replacement == [alternative, alternative], replacement
for action in replacement:
    assert controller._execute_single_action(character, action)
assert goal.check_completion() and node['energy'] == character.energy == 60
print({'partial_energy': 30, 'replacement_steps': len(replacement), 'final_energy': node['energy']})
''')

    def test_actor_objects_and_uuid_identifiers_keep_graph_identity(self):
        self.run_world('''
from tiny_strategy_manager import StrategyManager
from tiny_utility_functions import UtilityGoal
strategy = StrategyManager(graph_manager=graph)
partner = Character(name='IdentityPartner', age=30, job='unemployed', graph_manager=graph,
    personality_traits=character.personality_traits, gametime_manager=character.gametime_manager, memory_mode='basic')
graph.add_character_node(partner)
graph.G.nodes[character]['energy'], graph.G.nodes[partner]['energy'] = 20, 80
character.energy, partner.energy = 90, 10
assert strategy.get_character_planning_state(character)['energy'] == 20
assert strategy.get_character_planning_state(partner)['energy'] == 80
assert strategy.get_character_planning_state(str(character.uuid))['energy'] == 20
recover = Action('Recover', [], [{'targets': ['initiator'], 'attribute': 'energy', 'change_value': 10}], graph_manager=graph)
goal = UtilityGoal('recover', {'energy': 30})
assert strategy.plan_for_goal(str(partner.uuid), goal, [recover]) == []
plan = strategy.plan_for_goal(str(character.uuid), goal, [recover])
assert plan == [recover]
assert controller._execute_single_action(character, plan[0])
assert character.energy == graph.G.nodes[character]['energy'] == 30
assert partner.energy == 10 and graph.G.nodes[partner]['energy'] == 80
full_goal = Goal('Actor owned', character, character, 1, 'recover',
    {False: [Condition('actor energy', 'energy', character, 30)]},
    None, None, None, None, None, None, [], graph, 'Basic', {'energy': 30})
for invoke in (lambda: strategy.goap_planner.plan_actions(partner, full_goal, {'energy': 100}, []),
               lambda: strategy.goap_planner.validate_plan(partner, full_goal, [], {'energy': 100})):
    try:
        invoke()
    except ValueError as error:
        assert 'actor' in str(error)
    else:
        raise AssertionError('Planner accepted another actor Goal')
standalone = GOAPPlanner(None)
assert standalone.plan_actions(character, goal, actions=[]) == []
''')
