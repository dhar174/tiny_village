"""Action execution and bookkeeping against real graph-backed characters."""

from tests import test_goap_graph_contracts
from tests.test_graph_initialization import FreshPythonTestCase


class TestActionExecutionIntegration(FreshPythonTestCase):
    run_world = test_goap_graph_contracts.TestSupportedPlannerPrimitives.run_world

    def test_graph_conditions_and_native_scalar_aliases(self):
        self.run_world("""
from unittest.mock import patch
node = graph.G.nodes[character]
character.energy = 0
character.social_wellbeing = 7
node.update(energy=20, health_status=80, hunger_level=30, wealth_money=15,
            job_performance=10, happiness=4, social_wellbeing=7)
action = Action('Aliases', {'energy': Condition('Ready', 'energy', None, 20),
    'health': Condition('Healthy', 'health', None, 80)}, [
    {'targets': ['initiator'], 'attribute': 'health', 'change_value': 2},
    {'targets': ['initiator'], 'attribute': 'hunger', 'change_value': 'add:-3'},
    {'targets': ['initiator'], 'attribute': 'money', 'change_value': 'set:25'},
    {'targets': ['initiator'], 'attribute': 'wealth', 'change_value': 5},
    {'targets': ['initiator'], 'attribute': 'job_performance', 'change_value': 1},
    {'targets': ['initiator'], 'attribute': 'happiness', 'change_value': 2},
], graph_manager=graph)
with patch.object(graph, 'update_node_attribute', wraps=graph.update_node_attribute) as writes:
    assert action.execute(str(character.uuid))
    assert writes.call_count == 6
for field, value in dict(health_status=82, hunger_level=27, wealth_money=30,
                         job_performance=11, happiness=6, social_wellbeing=7).items():
    assert getattr(character, field) == node[field] == value, (field, node[field])
assert not {'health', 'hunger', 'money', 'wealth'}.intersection(node)
print({'health_status': node['health_status'], 'hunger_level': node['hunger_level'],
       'wealth_money': node['wealth_money'], 'job_performance': node['job_performance'],
       'happiness': node['happiness'], 'social_wellbeing': node['social_wellbeing']})
""")

    def test_required_targets_and_invalid_inputs_are_checked_before_writes(self):
        self.run_world("""
from unittest.mock import patch
node = graph.G.nodes[character]
delta = {'targets': ['initiator'], 'attribute': 'energy', 'change_value': 10}
action = Action('Validate', [], [delta], graph_manager=graph)
for identifier in (character, character.uuid, str(character.uuid), character.name):
    assert action.execute(identifier)
assert node['energy'] == character.energy == 60
graph.G.add_node('legacy-one', name='Ambiguous')
graph.G.add_node('legacy-two', name='Ambiguous')
invalid = [
    ([], [delta, dict(delta, targets=['target'])], None, character),
    ([], [delta, dict(delta, targets=['target'])], 'missing', character),
    ([], [delta, dict(delta, targets=['target'])], 'Ambiguous', character),
    ([], [delta], None, 'missing'),
    ([Condition('Missing actor', 'energy', 'missing', 0)], [delta], None, character),
    ([Condition('Too little', 'energy', None, 100)], [delta], None, character),
    ([Condition('Unknown field', 'typo_energy', None, 0, '==')], [delta], None, character),
    ({'energy': 0}, [delta], None, character),
    ([], [delta, dict(delta, change_value='add:bad')], None, character),
    ([], [delta, dict(delta, change_value='set:inf')], None, character),
    ([], [delta, dict(delta, attribute='location', change_value='set:Somewhere')], None, character),
    ([], [delta, dict(delta, attribute='energy', change_value='set:bad')], None, character),
    ([], [delta, dict(delta, attribute='name', change_value=1)], None, character),
    ([], [delta, {'targets': ['initiator'], 'attribute': 'energy'}], None, character),
]
with patch.object(graph, 'update_node_attribute', wraps=graph.update_node_attribute) as writes:
    for conditions, effects, target, actor in invalid:
        invalid_action = Action('Reject', conditions, effects, target=target, graph_manager=graph)
        assert invalid_action.execute(actor) is False, (conditions, effects, target, actor)
    writes.assert_not_called()
assert node['energy'] == character.energy == 60
""")

    def test_required_targets_resolve_to_the_same_real_nearby_entity(self):
        self.run_python('''
import runpy
from unittest.mock import patch
from actions import Action
from tiny_characters import Character
from tiny_locations import Location

probe = runpy.run_path('docs/planning/issue-734/evidence/audit_probes.py')
with probe['controller'](count=2) as controller:
    actor, target = controller.characters.values()
    graph = controller.graph_manager
    assert isinstance(actor, Character) and isinstance(target, Character)
    assert isinstance(actor.location, Location)
    target.location = actor.location
    actor.location.current_visitors[:] = [actor, target, target]
    graph.G.nodes[target]['energy'] = target.energy = 20
    direct = {'targets': ['target'], 'attribute': 'energy', 'change_value': 2}
    with patch.object(graph, 'update_node_attribute', wraps=graph.update_node_attribute) as writes:
        for identifier in (target, target.uuid, str(target.uuid), target.name):
            assert Action('Direct', [], [direct], initiator=actor,
                          target=identifier, graph_manager=graph).execute()
        assert Action('Nearby', [], [dict(direct, targets=['target',
            'target_character_in_location'])], initiator=actor.name, target=target.uuid,
            graph_manager=graph).execute()
        assert Action('Nobody nearby', [], [dict(direct, targets=[
            'target_character_in_location'])], initiator=actor.uuid, target='missing',
            graph_manager=graph).execute()
        assert [call.args for call in writes.call_args_list] == [
            (target, 'energy', energy) for energy in (22, 24, 26, 28, 30)
        ]
    assert target.energy == graph.G.nodes[target]['energy'] == 30
''')

    def test_catalog_defaults_and_overrides(self):
        self.run_world("""
from actions import EatAction, SleepAction, VisitDoctorAction, WorkAction
node = graph.G.nodes[character]
node.update(energy=0, hunger_level=10, health_status=80, wealth_money=9, job_performance=10)
eat = EatAction('food', character.name, graph_manager=graph)
work = WorkAction(initiator_id=character.uuid, graph_manager=graph)
doctor = VisitDoctorAction(initiator_id=str(character.uuid), graph_manager=graph)
sleep = SleepAction(initiator_id=character, graph_manager=graph)
assert eat.execute() is False
assert work.execute() is False
assert doctor.execute() is False
assert sleep.execute()
for catalog_action in (eat, work, doctor, sleep):
    assert catalog_action.graph_manager is graph
assert node['energy'] == character.energy == 5 and node['health_status'] == 81
assert eat.execute(graph_manager=graph)
assert node['hunger_level'] == character.hunger_level == 8
assert work.execute(graph_manager=graph)
assert node['energy'] == 4 and node['job_performance'] == 11 and node['wealth_money'] == 29
assert doctor.execute(graph_manager=graph)
assert node['health_status'] == 84 and node['wealth_money'] == 19
node['energy'] = 0
assert WorkAction(preconditions=[], effects=[]).execute(character, graph)
assert node['energy'] == 0
""")

    def test_daily_single_and_strategic_paths_credit_once(self):
        self.run_world("""
from unittest.mock import patch
node = graph.G.nodes[character]
action = Action('Work', [], [{'targets': ['initiator'], 'attribute': 'money',
    'change_value': 20}], initiator=character, graph_manager=graph)
node['wealth_money'] = 30
quest = {'name': 'Practice', 'type': 'skill', 'progress': 0}
controller.quest_system = {'active_quests': {character.uuid: [quest]}}
with patch.object(controller.strategy_manager, 'get_daily_actions', return_value=[action]), (
    patch.object(action, 'execute', wraps=action.execute)
) as execute, patch.object(controller, '_update_character_state_after_action',
                            side_effect=RuntimeError('bookkeeping failure')) as memory:
    assert controller._execute_character_actions(character)
    execute.assert_called_once_with(character=character, graph_manager=graph)
    memory.assert_called_once_with(character, action)
assert node['wealth_money'] == character.wealth_money == 50
assert quest['progress'] == 5
history = controller.action_resolver.execution_history
assert len(history) == 1 and history[0]['success']
action.preconditions = [Condition('Needs wealth', 'money', None, 50)]
with patch.object(action, 'execute', wraps=action.execute) as execute:
    assert controller._execute_single_action(character, action)
    execute.assert_called_once_with(character=character, graph_manager=graph)
assert node['wealth_money'] == character.wealth_money == 70
assert quest['progress'] == 10
decision = {'type': 'character_action', 'character_id': next(iter(controller.characters)),
            'action': action}
with patch.object(action, 'execute', wraps=action.execute) as execute:
    controller.apply_decision(decision, None)
    execute.assert_called_once_with(character=character, graph_manager=graph)
assert node['wealth_money'] == character.wealth_money == 90
assert quest['progress'] == 15
assert len(history) == 3 and all(record['success'] for record in history)
assert controller.game_statistics['actions_executed'] == 3
action.preconditions = [Condition('Too little wealth', 'money', None, 91)]
with patch.object(graph, 'update_node_attribute', wraps=graph.update_node_attribute) as writes:
    assert controller._execute_single_action(character, action) is False
    writes.assert_not_called()
assert quest['progress'] == 15 and controller.game_statistics['actions_executed'] == 3
assert len(history) == 4 and history[-1]['success'] is False
class BrokenAction(Action):
    def execute(self, character=None, graph_manager=None):
        raise TypeError('action body failed')

broken = BrokenAction('Broken', [], [])
with patch.object(controller.strategy_manager, 'get_daily_actions', return_value=[broken]), (
    patch.object(broken, 'execute', wraps=broken.execute)
) as execute, patch.object(controller.action_resolver, 'get_fallback_action') as fallback:
    assert controller._execute_character_actions(character) is False
    execute.assert_called_once()
    fallback.assert_not_called()
assert len(history) == 5 and history[-1]['success'] is False
assert quest['progress'] == 15 and controller.game_statistics['actions_executed'] == 3
""")

    def test_plan_neither_replays_partial_writes_nor_retries_type_error(self):
        self.run_world("""
from types import SimpleNamespace
from unittest.mock import patch
node = graph.G.nodes[character]
delta = {'targets': ['initiator'], 'attribute': 'energy', 'change_value': 10}
partial = Action('Partial', [], [delta, dict(delta, attribute='temp_220')],
                 initiator=character, graph_manager=graph)
original = graph.update_node_attribute
def fail_after_second(node, attribute, value):
    original(node, attribute, value)
    if attribute == 'temp_220':
        raise RuntimeError('second write failed')

goal = SimpleNamespace(name='Recover', check_completion=lambda: node['energy'] >= 99)
plan = Plan('partial', [goal], graph_manager=graph)
plan.add_action(partial)
with patch.object(graph, 'update_node_attribute', side_effect=fail_after_second) as writes:
    assert plan.execute() is False
    assert writes.call_count == 2
assert node['energy'] == character.energy == 30
assert 'temp_220' not in node and not hasattr(character, 'temp_220')
assert not plan.completed_actions

class BrokenAction(Action):
    calls = 0
    def execute(self, character=None, graph_manager=None):
        self.calls += 1
        super().execute(character, graph_manager)
        raise TypeError('internal execution failure')

broken = BrokenAction('Broken', [], [delta], initiator=character)
plan = Plan('replanned', [goal], graph_manager=graph)
def populate():
    plan.add_action(broken)
    return True
with patch.object(plan, 'replan', side_effect=populate) as replan:
    assert plan.execute() is False
    replan.assert_called_once()
assert broken.calls == 1
assert node['energy'] == character.energy == 40 and not plan.completed_actions
plan = Plan('success', [SimpleNamespace(name='Success',
    check_completion=lambda: node['energy'] == 50)], graph_manager=graph)
action = Action('Recovery', [], [delta], initiator=character)
plan.add_action(action)
with patch.object(action, 'execute', wraps=action.execute) as execute:
    assert plan.execute()
    execute.assert_called_once_with(character=character, graph_manager=graph)
assert plan.completed_actions == {'Recovery'}
assert node['energy'] == character.energy == 50
""")

    def test_legacy_method_uses_graph_field_and_rolls_back_both_values(self):
        self.run_world("""
from unittest.mock import patch
node = graph.G.nodes[character]
character.energy = 40
node['energy'] = 20
def recover():
    character.energy += 5
character.recover_test_energy = recover
action = Action('MethodRecovery', [], [{'targets': ['initiator'], 'attribute': 'energy',
    'change_value': 'recover_test_energy'}], initiator=character, graph_manager=graph)
original = graph.update_node_attribute
def fail_after_write(*args):
    original(*args)
    raise RuntimeError('legacy method graph write failed')
with patch.object(graph, 'update_node_attribute', side_effect=fail_after_write) as writes:
    assert action.execute() is False
    writes.assert_called_once_with(character, 'energy', 25)
assert character.energy == 40 and node['energy'] == 20
with patch.object(graph, 'update_node_attribute', wraps=original) as writes:
    assert action.execute()
    writes.assert_called_once_with(character, 'energy', 25)
assert character.energy == node['energy'] == 25
""")

    def test_weather_and_following_action_use_same_authoritative_energy(self):
        self.run_world("""
from unittest.mock import patch
node = graph.G.nodes[character]
character.energy = 40
node['energy'] = 20
controller.weather_system = {'current_weather': 'rainy'}
action = Action('Recovery', [], [{'targets': ['initiator'], 'attribute': 'energy',
    'change_value': 10}], initiator=character, graph_manager=graph)
with patch.object(controller.strategy_manager, 'get_daily_actions', return_value=[]), (
    patch.object(graph, 'update_node_attribute', wraps=graph.update_node_attribute)
) as writes:
    controller._update_character(character, 10)
    assert node['energy'] == character.energy == 15
    assert action.execute()
    assert node['energy'] == character.energy == 25
    assert [call.args[1:] for call in writes.call_args_list] == [('energy', 15), ('energy', 25)]
print({'graph_and_character_energy': [20, 15, 25], 'updates': 2})
original = graph.update_node_attribute
def fail_after_write(*args):
    original(*args)
    raise RuntimeError('weather graph write failed')
with patch.object(controller.strategy_manager, 'get_daily_actions', return_value=[]), (
    patch.object(graph, 'update_node_attribute', side_effect=fail_after_write)
) as writes:
    assert controller._update_character(character, 10) is False
    writes.assert_called_once_with(character, 'energy', 20)
assert node['energy'] == character.energy == 25
graph.world_state.remove_node(character)
with patch.object(controller.strategy_manager, 'get_daily_actions', return_value=[]), (
    patch.object(graph, 'update_node_attribute', wraps=graph.update_node_attribute)
) as writes:
    assert controller._update_character(character, 10) is False
    writes.assert_not_called()
assert character not in graph.G and character.energy == 25
""")
