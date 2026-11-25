"""Action execution and bookkeeping against real graph-backed characters."""

from tests import test_goap_graph_contracts
from tests.test_graph_initialization import FreshPythonTestCase


class TestActionExecutionIntegration(FreshPythonTestCase):
    run_world = test_goap_graph_contracts.TestSupportedPlannerPrimitives.run_world

    def test_sequential_drain_and_recovery_share_authoritative_energy(self):
        self.run_world("""
from unittest.mock import patch
from tiny_utility_functions import native_character_state
node = graph.G.nodes[character]
for object_energy in (20, 40):
    character.energy = object_energy
    node['energy'] = 20
    with patch.object(graph, 'update_node_attribute', wraps=graph.update_node_attribute) as writes:
        for name, delta, expected in [('Drain', -5, 15), ('Recovery', 10, 25)]:
            action = Action(name, [], [{'targets': ['initiator'],
                'attribute': 'energy', 'change_value': delta}],
                initiator=character, graph_manager=graph)
            assert action.execute()
            assert character.energy == node['energy'] == expected
            assert native_character_state(character, graph)['energy'] == expected
        assert [c.args for c in writes.call_args_list] == [
            (character, 'energy', 15), (character, 'energy', 25)]
print({'sequential_energy': [20, 15, 25], 'writes': 2})
""")

    def test_market_reward_is_not_lost_by_following_action(self):
        self.run_world("""
from unittest.mock import patch
from tiny_utility_functions import native_character_state
node = graph.G.nodes[character]
node['wealth_money'] = character.wealth_money = 20
with patch('tiny_gameplay_controller.random.randint', return_value=3) as random_reward, (
    patch.object(graph, 'update_node_attribute', wraps=graph.update_node_attribute)
) as writes:
    controller._update_economic_state_from_event('market')
    assert character.wealth_money == node['wealth_money'] == 23
    assert native_character_state(character, graph)['wealth_money'] == 23
    action = Action('Earn', [], [{'targets': ['initiator'],
        'attribute': 'money', 'change_value': 10}], initiator=character, graph_manager=graph)
    assert action.execute()
    assert character.wealth_money == node['wealth_money'] == 33
    assert [c.args for c in writes.call_args_list] == [
        (character, 'wealth_money', 23), (character, 'wealth_money', 33)]
    random_reward.assert_called_once_with(1, 5)
print({'market_and_action': [20, 23, 33]})
""")

    def test_scalar_setters_zero_and_removed_character_use_graph(self):
        self.run_world("""
from unittest.mock import patch
node = graph.G.nodes[character]
fields = ['health_status', 'hunger_level', 'wealth_money', 'mental_health',
          'social_wellbeing', 'job_performance', 'shelter', 'stability',
          'luxury', 'hope', 'success', 'control']
with patch.object(graph, 'update_node_attribute', wraps=graph.update_node_attribute) as writes:
    for field in fields:
        assert getattr(character, 'set_' + field)(15) == 15
        assert getattr(character, field) == node[field] == 15
    character.update_character(**dict.fromkeys(fields, 0))
    for field in fields:
        assert getattr(character, field) == node[field] == 0, field
    character.update_character()
    assert writes.call_count == 2 * len(fields)
    character.satisfaction = 12
    assert character.current_satisfaction == node['current_satisfaction'] == node['satisfaction'] == 12
    assert character.set_happiness(.5) == .5 and node['happiness'] == .5
graph.world_state.remove_node(character)
from world_state import UnknownNodeError
with patch.object(graph, 'update_node_attribute', wraps=graph.update_node_attribute) as writes:
    try:
        character.set_wealth_money(99)
    except UnknownNodeError:
        pass
    else:
        raise AssertionError('Removed Character accepted an object-only update')
    writes.assert_not_called()
assert character.wealth_money == 0 and character not in graph.G
character.graph_manager = None
try:
    character.set_wealth_money(99)
except ValueError:
    pass
else:
    raise AssertionError('Initialized native Character lost its state owner')
assert character.wealth_money == 0
from character_attribute_mapper import AttributeMapper
assert AttributeMapper.set_attribute_value(character, 'money', 99) is False
from tiny_building_manager import BuildingManager, BuildingService
manager = BuildingManager()
manager.BUILDING_SERVICES = {'stand': {'reward': BuildingService('Reward', cost=-5)}}
assert not manager.provide_service('stand-1', 'stand', 'reward', character)[0]
assert character.wealth_money == 0
""")

    def test_social_callbacks_stage_declared_field_and_commit_once(self):
        self.run_world("""
from types import SimpleNamespace
from unittest.mock import patch
node = graph.G.nodes[character]
character.social_wellbeing = 80
node['social_wellbeing'] = 20
character.personality_traits.agreeableness = 70
partner = SimpleNamespace(name='Partner', personality_traits=SimpleNamespace(agreeableness=70))
character.friendship_grid = {}
def respond():
    character.respond_to_talk(partner)
character.respond_test = respond
action = Action('Talk callback', [], [{'targets': ['initiator'],
    'attribute': 'social_wellbeing', 'change_value': 'respond_test'}],
    initiator=character, graph_manager=graph)
with patch.object(graph, 'update_node_attribute', wraps=graph.update_node_attribute) as writes:
    assert action.execute()
    assert math.isclose(character.social_wellbeing, 20.2)
    assert math.isclose(node['social_wellbeing'], 20.2)
    assert writes.call_count == 1
character.respond_to_greeting(partner)
assert math.isclose(character.social_wellbeing, 20.25)
assert math.isclose(node['social_wellbeing'], 20.25)
character.respond_to_compliment(partner, 'kindness')
assert math.isclose(character.social_wellbeing, 20.45)
assert math.isclose(node['social_wellbeing'], 20.45)
""")

    def test_nested_setters_restore_failed_field_and_preserve_other_commits(self):
        self.run_world("""
from unittest.mock import patch
node = graph.G.nodes[character]
node['wealth_money'] = 20
character.wealth_money = 40
node['money'] = 19
def earn():
    character.set_social_wellbeing(31)
    character.set_wealth_money(character.wealth_money + 3)
    character.set_wealth_money(character.wealth_money + 2)
character.earn_test = earn
action = Action('Nested earn', [], [{'targets': ['initiator'], 'attribute': 'money',
    'change_value': 'earn_test'}], initiator=character, graph_manager=graph)
original = graph.update_node_attribute
def fail_after_write(*args):
    original(*args)
    if args[1] == 'wealth_money':
        raise RuntimeError('Injected wealth commit failure')
with patch.object(graph, 'update_node_attribute', side_effect=fail_after_write) as writes:
    assert not action.execute()
    assert [c.args[1:] for c in writes.call_args_list] == [
        ('social_wellbeing', 31), ('wealth_money', 25)]
assert character.wealth_money == 40 and node['wealth_money'] == 20 and node['money'] == 19
assert character.social_wellbeing == node['social_wellbeing'] == 31
with patch.object(graph, 'update_node_attribute', wraps=original) as writes:
    assert action.execute()
    assert [c.args[1:] for c in writes.call_args_list] == [
        ('social_wellbeing', 31), ('wealth_money', 25)]
assert character.wealth_money == node['wealth_money'] == node['money'] == 25
""")

    def test_constructor_registers_final_values_without_early_scalar_writes(self):
        self.run_world("""
from unittest.mock import patch
with patch.object(graph, 'update_node_attribute', wraps=graph.update_node_attribute) as writes:
    from tiny_characters import PersonalityTraits
    other = Character('Initialized safely', 30, graph_manager=graph,
        action_system=character.action_system, gametime_manager=character.gametime_manager,
        personality_traits=PersonalityTraits(4, 4, 4, 4, 4),
        memory_mode='basic', energy=17, wealth_money=23)
    writes.assert_not_called()
assert other._initialized and graph.resolve_node(other.uuid, 'character') is other
assert graph.G.nodes[other]['energy'] == other.energy == 17
assert graph.G.nodes[other]['wealth_money'] == other.wealth_money == 23
""")

    def test_building_fees_effects_and_failures_use_authoritative_fields(self):
        self.run_world("""
from unittest.mock import patch
from tiny_building_manager import BuildingManager, BuildingService
manager = BuildingManager()
service = BuildingService('Recovery', cost=5,
    effects=[{'attribute': 'energy', 'change_value': 10},
             {'attribute': 'hunger_level', 'change_value': -5}])
manager.BUILDING_SERVICES = {'clinic': {'recovery': service}}
manager.register_building('clinic-1', 'clinic')
node = graph.G.nodes[character]
node.update(wealth_money=20, energy=20, hunger_level=60)
character.wealth_money = 0
character.energy = 90
character.hunger_level = 10
with patch.object(graph, 'update_node_attribute', wraps=graph.update_node_attribute) as writes:
    assert manager.provide_service('clinic-1', 'clinic', 'recovery', character)[0]
    assert [c.args[1:] for c in writes.call_args_list] == [
        ('wealth_money', 15), ('energy', 30), ('hunger_level', 55)]
assert character.wealth_money == node['wealth_money'] == 15
assert character.energy == node['energy'] == 30
assert character.hunger_level == node['hunger_level'] == 55
node['wealth_money'] = 0
character.wealth_money = 100
assert not manager.provide_service('clinic-1', 'clinic', 'recovery', character)[0]
assert len(manager.service_history) == 1
node['wealth_money'] = character.wealth_money = 20
original = graph.update_node_attribute
def fail_after_write(*args):
    original(*args)
    if args[1] == 'energy':
        raise RuntimeError('Injected service effect failure')
with patch.object(graph, 'update_node_attribute', side_effect=fail_after_write):
    assert not manager.provide_service('clinic-1', 'clinic', 'recovery', character)[0]
assert character.wealth_money == node['wealth_money'] == 15
assert character.energy == node['energy'] == 30
assert len(manager.service_history) == 1
assert character.hunger_level == node['hunger_level'] == 55
""")

    def test_mapper_and_event_updates_preserve_mapping_and_graph_coherence(self):
        self.run_world("""
from character_attribute_mapper import AttributeMapper
from effect_dispatcher import EffectDispatcher
from effect_schema import OperatorType
from tiny_event_handler import EventHandler
node = graph.G.nodes[character]
dispatcher = EffectDispatcher(graph)
handler = EventHandler(graph)
node['wealth_money'] = 20
character.wealth_money = 40
assert dispatcher._modify_entity_attribute(character, 'wealth', 3, OperatorType.ADD)
assert character.wealth_money == node['wealth_money'] == 23
handler._modify_entity_attribute(character, 'money', 2)
assert character.wealth_money == node['wealth_money'] == 25
assert AttributeMapper.set_attribute_value(character, 'money', 12)
assert character.wealth_money == node['wealth_money'] == 12
for invalid in (float('nan'), float('inf')):
    assert AttributeMapper.set_attribute_value(character, 'money', invalid) is False
    assert character.wealth_money == node['wealth_money'] == 12
node['social_wellbeing'] = character.social_wellbeing = 20
character.satisfaction = 7
assert AttributeMapper.set_attribute_value(character, 'satisfaction', 30)
assert character.social_wellbeing == node['social_wellbeing'] == 30
assert character.current_satisfaction == node['current_satisfaction'] == 7
assert dispatcher._modify_entity_attribute(character, 'happiness', 5, OperatorType.ADD)
assert character.social_wellbeing == node['social_wellbeing'] == 35
""")

    def test_scalar_calculation_noop_and_invalid_results_restore_staging(self):
        self.run_world("""
from actions import apply_attribute_change
from unittest.mock import patch
node = graph.G.nodes[character]
node['wealth_money'] = 20
character.wealth_money = 40
def noop(current):
    character.set_wealth_money(current + 3)
    return None
with patch.object(graph, 'update_node_attribute', wraps=graph.update_node_attribute) as writes:
    assert apply_attribute_change(character, graph, 'money', noop) is None
    writes.assert_not_called()
assert character.wealth_money == 40 and node['wealth_money'] == 20
for invalid in (float('nan'), float('inf'), True, 'bad'):
    try:
        apply_attribute_change(character, graph, 'money', lambda current: invalid)
    except ValueError:
        pass
    else:
        raise AssertionError('Invalid scalar committed')
    assert character.wealth_money == 40 and node['wealth_money'] == 20
assert apply_attribute_change(character, graph, 'money', lambda current: current + 5) == 25
assert character.wealth_money == node['wealth_money'] == 25
""")

    def test_native_mapper_and_dispatcher_preserve_nonnumeric_set(self):
        self.run_world("""
from character_attribute_mapper import AttributeMapper
from effect_dispatcher import EffectDispatcher
from effect_schema import OperatorType
node = graph.G.nodes[character]
assert AttributeMapper.set_attribute_value(character, 'job', 'Baker') is True
assert character.job == node['job'] == 'Baker'
assert EffectDispatcher(graph)._modify_entity_attribute(character, 'job', 'Farmer', OperatorType.SET)
assert character.job == node['job'] == 'Farmer'
assert AttributeMapper.set_attribute_value(character, 'recent_event', 'Festival')
assert character.recent_event == node['recent_event'] == 'Festival'
node['wealth_money'] = character.wealth_money = 20
for invalid in ('not money', float('nan'), float('inf'), True):
    assert not EffectDispatcher(graph)._modify_entity_attribute(character, 'money', invalid, OperatorType.SET)
    assert not AttributeMapper.set_attribute_value(character, 'money', invalid)
    assert character.wealth_money == node['wealth_money'] == 20
""")

    def test_native_service_thirst_retains_bounds_from_graph_value(self):
        self.run_world("""
from tiny_building_manager import BuildingManager
from unittest.mock import patch
node = graph.G.nodes[character]
manager = BuildingManager()
for graph_thirst, object_thirst, delta, expected in [(95, 20, 10, 100), (5, 80, -10, 0)]:
    node['thirst'] = graph_thirst
    character.thirst = object_thirst
    with patch.object(graph, 'update_node_attribute', wraps=graph.update_node_attribute) as writes:
        manager._apply_service_effect({'attribute': 'thirst', 'change_value': delta}, character)
        writes.assert_called_once_with(character, 'thirst', expected)
    assert character.thirst == node['thirst'] == expected
node['thirst'] = 95
character.thirst = 20
for invalid in (float('nan'), float('inf')):
    try:
        manager._apply_service_effect({'attribute': 'thirst', 'change_value': invalid}, character)
    except ValueError:
        pass
    else:
        raise AssertionError('Service bounds concealed a non-finite result')
    assert character.thirst == 20 and node['thirst'] == 95
""")

    def test_remaining_numeric_and_job_setters_keep_graph_state_current(self):
        self.run_world("""
from unittest.mock import patch
node = graph.G.nodes[character]
for field in ('age', 'beauty', 'community', 'material_goods'):
    node[field] = 20
    setattr(character, field, 40)
    with patch.object(graph, 'update_node_attribute', wraps=graph.update_node_attribute) as writes:
        assert getattr(character, 'set_' + field)(15) == 15
        assert getattr(character, field) == node[field] == 15
        assert Action('Follow setter', [], [{'targets': ['initiator'],
            'attribute': field, 'change_value': 10}], initiator=character, graph_manager=graph).execute()
        assert getattr(character, field) == node[field] == 25
        assert [c.args[1:] for c in writes.call_args_list] == [(field, 15), (field, 25)]
character.update_character(beauty=0, community=0, material_goods=0)
for field in ('beauty', 'community', 'material_goods'):
    assert getattr(character, field) == node[field] == 0
role = character.job
node['job'] = 'Baker'
with patch.object(graph, 'update_node_attribute', wraps=graph.update_node_attribute) as writes:
    assert character.set_job(role) is role
    assert character.job is node['job'] is role
    writes.assert_called_once_with(character, 'job', role)
selected = character.set_job('unemployed')
assert character.job is node['job'] is selected
with patch.object(graph, 'update_node_attribute', wraps=graph.update_node_attribute) as writes:
    try:
        character.set_job('NoSuchJob602')
    except ValueError:
        pass
    else:
        raise AssertionError('Unsupported job accepted')
    writes.assert_not_called()
assert character.job is node['job'] is selected
""")

    def test_caught_nested_failure_restores_savepoint_before_callback_continues(self):
        self.run_world("""
from actions import apply_attribute_change
from unittest.mock import patch
node = graph.G.nodes[character]
node['wealth_money'] = 20
character.wealth_money = 40
def earn():
    character.set_wealth_money(character.wealth_money + 2)
    def fail(current):
        character.wealth_money = 99
        raise ValueError('Aborted inner calculation')
    try:
        apply_attribute_change(character, graph, 'money', fail)
    except ValueError:
        pass
    assert character.wealth_money == 22
    character.set_wealth_money(character.wealth_money + 3)
character.earn_test = earn
with patch.object(graph, 'update_node_attribute', wraps=graph.update_node_attribute) as writes:
    assert Action('Nested savepoint', [], [{'targets': ['initiator'],
        'attribute': 'money', 'change_value': 'earn_test'}],
        initiator=character, graph_manager=graph).execute()
    writes.assert_called_once_with(character, 'wealth_money', 25)
assert character.wealth_money == node['wealth_money'] == 25
""")

    def test_reentrant_graph_commit_cannot_leave_object_and_graph_divergent(self):
        self.run_world("""
from actions import apply_attribute_change
from unittest.mock import patch
node = graph.G.nodes[character]
node['wealth_money'] = 20
node['money'] = 19
character.wealth_money = 40
original = graph.update_node_attribute
def reentrant_commit(*args):
    original(*args)
    character.set_wealth_money(99)
with patch.object(graph, 'update_node_attribute', side_effect=reentrant_commit) as writes:
    try:
        apply_attribute_change(character, graph, 'money', lambda current: current + 5)
    except RuntimeError:
        pass
    else:
        raise AssertionError('Commit accepted a conflicting same-field mutation')
    writes.assert_called_once_with(character, 'wealth_money', 25)
assert character.wealth_money == 40 and node['wealth_money'] == 20 and node['money'] == 19
assert apply_attribute_change(character, graph, 'money', lambda current: current + 5) == 25
assert character.wealth_money == node['wealth_money'] == node['money'] == 25
""")

    def test_graph_conditions_and_native_scalar_aliases(self):
        self.run_world("""
from unittest.mock import patch
node = graph.G.nodes[character]
character.energy = 0
character.social_wellbeing = 7
node.update(energy=20, health_status=80, hunger_level=30, wealth_money=15,
            job_performance=10, happiness=.4, social_wellbeing=7)
action = Action('Aliases', {'energy': Condition('Ready', 'energy', None, 20),
    'health': Condition('Healthy', 'health', None, 80)}, [
    {'targets': ['initiator'], 'attribute': 'health', 'change_value': 2},
    {'targets': ['initiator'], 'attribute': 'hunger', 'change_value': 'add:-3'},
    {'targets': ['initiator'], 'attribute': 'money', 'change_value': 'set:25'},
    {'targets': ['initiator'], 'attribute': 'wealth', 'change_value': 5},
    {'targets': ['initiator'], 'attribute': 'job_performance', 'change_value': 1},
    {'targets': ['initiator'], 'attribute': 'happiness', 'change_value': .2},
], graph_manager=graph)
with patch.object(graph, 'update_node_attribute', wraps=graph.update_node_attribute) as writes:
    assert action.execute(str(character.uuid))
    assert writes.call_count == 6
for field, value in dict(health_status=82, hunger_level=27, wealth_money=30,
                         job_performance=11, happiness=.6, social_wellbeing=7).items():
    assert math.isclose(getattr(character, field), value) and math.isclose(node[field], value), (field, node[field])
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
from tiny_gameplay_controller import _ticks_ms
quest = {'name': 'Practice', 'type': 'skill', 'progress': 0, 'assigned_time': _ticks_ms()}
controller.quest_system = {'active_quests': {character.uuid: [quest]}}
with patch.object(action, 'execute', wraps=action.execute) as execute, patch.object(controller, '_update_character_state_after_action',
                            side_effect=RuntimeError('bookkeeping failure')) as memory:
    assert controller._apply_strategy_to_character(character, [action])
    execute.assert_not_called()
    controller.update_game_state(1)
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
    execute.assert_not_called()
    controller.update_game_state(1)
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
with patch.object(broken, 'execute', wraps=broken.execute) as execute, patch.object(controller.action_resolver, 'get_fallback_action') as fallback:
    assert controller._apply_strategy_to_character(character, [broken])
    execute.assert_not_called()
    controller.update_game_state(1)
    execute.assert_called_once()
    fallback.assert_not_called()
assert len(history) == 5 and history[-1]['success'] is False
assert quest['progress'] == 15 and controller.game_statistics['actions_executed'] == 3
""")

    def test_invalid_action_and_empty_fallback_report_truthfully(self):
        self.run_world("""
from types import SimpleNamespace
from unittest.mock import patch

invalid = SimpleNamespace(name='Invalid action')
history = controller.action_resolver.execution_history
initial_failures = controller.game_statistics['actions_failed']
initial_successes = controller.game_statistics['actions_executed']
with patch.object(controller.strategy_manager, 'get_daily_actions', return_value=[invalid]), (
    patch.object(controller, 'provide_action_feedback')
) as feedback, patch.object(controller, '_invoke_action_execute') as invoke, (
    patch.object(graph, 'update_node_attribute', wraps=graph.update_node_attribute)
) as writes:
    assert controller._run_selected_action(invalid, character, feedback=True) is False
    invoke.assert_not_called()
    writes.assert_not_called()
    feedback.assert_called_once_with('Invalid action', False, character.name)
assert controller.game_statistics['actions_failed'] == initial_failures + 1
assert controller.game_statistics['actions_executed'] == initial_successes
assert [(entry['action_name'], entry['success']) for entry in history] == [
    ('Invalid action', False)
]

with patch.object(controller.strategy_manager, 'get_daily_actions', return_value=[]), (
    patch('tiny_gameplay_controller.logger.warning')
) as warning, patch.object(controller, 'provide_action_feedback') as feedback:
    assert controller._execute_fallback_character_action(character) is False
    warning.assert_called_once_with('No actions available for %s', character.name)
    feedback.assert_not_called()
assert controller.game_statistics['actions_failed'] == initial_failures + 1
assert len(history) == 1
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
from tiny_strategy_manager import PlanRequest
impossible = Goal('Unreachable', character, character, 1, 'unreachable',
    {False:[Condition('beyond clamp','energy',character,101)]},
    None,None,None,None,None,None,[],graph,'Basic Needs',{'energy':101})
controller.queue_plan_request(PlanRequest(character.uuid,impossible))
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

    def test_strategy_rest_uses_native_energy_and_clamps(self):
        self.run_world("""
from unittest.mock import patch
from tiny_strategy_manager import StrategyManager
strategy = StrategyManager(graph_manager=graph)
node = graph.G.nodes[character]
node.update(energy=20)
character.energy = 20
rest = next(a for a in strategy.get_daily_actions(character) if a.name == "Rest")
assert [(e.get("attribute"), e.get("change_value")) for e in rest.effects if isinstance(e, dict)] == [
    ("energy", 15), ("satisfaction", 8)]
with patch.object(graph, "update_node_attribute", wraps=graph.update_node_attribute) as writes:
    assert rest.execute(str(character.uuid))
assert node["energy"] == character.energy == 35
assert node["satisfaction"] == character.satisfaction == 8
node.update(energy=95)
character.energy = 95
with patch.object(graph, "update_node_attribute", wraps=graph.update_node_attribute) as writes2:
    assert rest.execute(str(character.uuid))
assert node["energy"] == character.energy == 100
print({"rest": [20, 35], "capped": [95, 100]})
""")

    def test_strategy_eat_hunger_reduction_uses_native_points(self):
        self.run_world("""
node = graph.G.nodes[character]
node.update(hunger_level=2)
character.hunger_level = 2
eat = Action("Eat Apple", [], [
    {"targets": ["initiator"], "attribute": "hunger", "change_value": -(20 * 0.1)}], graph_manager=graph)
assert eat.execute(str(character.uuid))
assert node["hunger_level"] == character.hunger_level == 0
print({"eat_hunger": [2, 0]})
""")

    def test_registered_health_event_reaches_both_stores(self):
        self.run_world("""
from effect_dispatcher import EffectDispatcher
from effect_schema import OperatorType
dispatcher = EffectDispatcher(graph_manager=graph)
node = graph.G.nodes[character]
node.update(health_status=80)
character.health_status = 80
assert dispatcher._modify_entity_attribute(character, "health", 20, OperatorType.SUBTRACT)
assert node["health_status"] == character.health_status == 60
print({"health_event": [80, 60]})
""")

    def test_graph_first_snapshot_feeds_utility_and_prompt(self):
        self.run_world("""
from tiny_strategy_manager import StrategyManager
from tiny_prompt_builder import PromptBuilder
strategy = StrategyManager(graph_manager=graph)
node = graph.G.nodes[character]
node.update(energy=77)
character.energy = 20
snapshot = strategy.get_character_state_dict(character)
assert snapshot["energy"] == 0.77, snapshot["energy"]
prompt = PromptBuilder(character, graph_manager=graph).generate_social_interaction_prompt(["greet villager"])
assert "Energy 77.0/100" in prompt, prompt[-300:]
assert "/10," not in prompt and "/10." not in prompt
print({"snapshot_energy": 0.77, "prompt_energy": "77.0/100"})
""")

    def test_native_delta_rollback_restores_both_stores(self):
        self.run_world("""
from unittest.mock import patch
node = graph.G.nodes[character]
node.update(energy=20)
character.energy = 20
rest = Action("Rest", [], [
    {"targets": ["initiator"], "attribute": "energy", "change_value": 15}], graph_manager=graph)
with patch.object(graph, "update_node_attribute", side_effect=RuntimeError("write failed")):
    assert not rest.execute(str(character.uuid))
assert node["energy"] == character.energy == 20
print({"rollback_restored": 20})
""")


class TestNamedTargetWorldIntegration(FreshPythonTestCase):
    run_world = test_goap_graph_contracts.TestSupportedPlannerPrimitives.run_world

    def run_nearby_world(self, source):
        self.run_world('''
from copy import copy
from types import SimpleNamespace
from uuid import uuid4
from unittest.mock import patch
from tiny_characters import PersonalityTraits
from tiny_locations import Location
from tiny_utility_functions import native_character_state
actor = character
room = actor.location
graph.add_location_node(room)
elsewhere = Location('Elsewhere602', *room.get_coordinates(), 5, 5, actor.action_system)
graph.add_location_node(elsewhere)
def new_character(name, location):
    return Character(name, 30, graph_manager=graph, action_system=actor.action_system,
        gametime_manager=actor.gametime_manager, memory_mode='basic', location=location,
        personality_traits=PersonalityTraits(4, 4, 4, 4, 4), energy=20)
partner = new_character('Nearby602', room)
other = new_character('Other602', room)
for entity in (actor, partner, other):
    entity.energy = graph.G.nodes[entity]['energy'] = 20
delta = {'targets': ['target_character_in_location'], 'attribute': 'energy', 'change_value': 5}
''' + source)

    def test_named_discovery_excludes_canonical_actor_alias(self):
        self.run_nearby_world('''
old_hash = hash(room)
room.current_visitors[:] = [copy(actor), partner, copy(partner), partner]
with patch.object(graph, 'update_node_attribute', wraps=graph.update_node_attribute) as writes:
    assert Action('Discover', [], [delta], initiator=str(actor.uuid), graph_manager=graph).execute()
    writes.assert_called_once_with(partner, 'energy', 25)
assert actor.energy == graph.G.nodes[actor]['energy'] == 20
assert partner.energy == graph.G.nodes[partner]['energy'] == 25
assert other.energy == graph.G.nodes[other]['energy'] == 20
assert hash(room) == old_hash
assert graph.resolve_node(room.uuid, 'location') is room
print({'canonical_actor': 20, 'nearby_partner': 25, 'writes': 1})
''')

    def test_registered_visitor_location_and_name_override_stale_handles(self):
        self.run_nearby_world('''
for actual_location, handle_location, expected in [(room, elsewhere, 25), (elsewhere, room, 20)]:
    partner.location = actual_location
    partner.energy = graph.G.nodes[partner]['energy'] = 20
    handle = copy(partner)
    handle.location = handle_location
    room.current_visitors[:] = [actor, handle]
    with patch.object(graph, 'update_node_attribute', wraps=graph.update_node_attribute) as writes:
        assert Action('Actual location', [], [delta], initiator=actor,
            target=partner.uuid, graph_manager=graph).execute()
        assert writes.call_count == (1 if expected == 25 else 0)
    assert partner.energy == graph.G.nodes[partner]['energy'] == expected
partner.location = room
partner.energy = graph.G.nodes[partner]['energy'] = 20
handle = copy(partner)
handle.name = 'Outdated602'
room.current_visitors[:] = [actor, handle]
assert Action('Actual name', [], [delta], initiator=actor, target=partner.name,
    graph_manager=graph).execute()
assert partner.energy == graph.G.nodes[partner]['energy'] == 25
assert actor.energy == other.energy == 20
''')

    def test_requested_identity_dedup_and_named_drain_recovery(self):
        self.run_nearby_world('''
room.current_visitors[:] = [copy(actor), other, partner, copy(partner), partner]
with patch.object(graph, 'update_node_attribute', wraps=graph.update_node_attribute) as writes:
    wrong_type = SimpleNamespace(uuid=partner.uuid)
    assert Action('Wrong object type', [], [delta], initiator=actor,
        target=wrong_type, graph_manager=graph).execute()
    writes.assert_not_called()
    for requested, expected in [(partner, 22), (partner.uuid, 24),
                                (str(partner.uuid), 26), (partner.name, 28)]:
        assert Action('Requested', [], [dict(delta, change_value=2)], initiator=actor,
            target=requested, graph_manager=graph).execute()
        assert partner.energy == graph.G.nodes[partner]['energy'] == expected
    assert Action('Overlap', [], [dict(delta, targets=['target',
        'target_character_in_location'], change_value=2)], initiator=actor,
        target=partner.uuid, graph_manager=graph).execute()
    assert [c.args for c in writes.call_args_list] == [
        (partner, 'energy', expected) for expected in (22, 24, 26, 28, 30)]
partner.energy = graph.G.nodes[partner]['energy'] = 20
with patch.object(graph, 'update_node_attribute', wraps=graph.update_node_attribute) as writes:
    for name, change, expected in [('Drain nearby', -5, 15), ('Recover nearby', 10, 25)]:
        assert Action(name, [], [dict(delta, targets=['target',
            'target_character_in_location'], change_value=change)], initiator=actor,
            target=partner.uuid, graph_manager=graph).execute()
        assert partner.energy == graph.G.nodes[partner]['energy'] == expected
        assert native_character_state(partner, graph)['energy'] == expected
    assert [c.args for c in writes.call_args_list] == [
        (partner, 'energy', 15), (partner, 'energy', 25)]
assert actor.energy == graph.G.nodes[actor]['energy'] == 20
assert other.energy == graph.G.nodes[other]['energy'] == 20
print({'nearby_energy': [20, 15, 25], 'actor_energy': 20, 'writes': 2})
''')

    def test_named_absence_fallback_and_required_target_failure(self):
        self.run_nearby_world('''
unknown = SimpleNamespace(name='Unregistered602', uuid=uuid4(), location=room)
room.current_visitors[:] = [actor, unknown, partner]
with patch.object(graph, 'update_node_attribute', wraps=graph.update_node_attribute) as writes:
    assert Action('Absent', [], [delta], initiator=actor, target='missing', graph_manager=graph).execute()
    writes.assert_not_called()
    assert Action('Skip unrelated stale visitor', [], [delta], initiator=actor,
        target=partner.uuid, graph_manager=graph).execute()
    writes.assert_called_once_with(partner, 'energy', 25)
    writes.reset_mock()
    assert not Action('Invalid matching visitor', [], [delta], initiator=actor,
        target=unknown.name, graph_manager=graph).execute()
    writes.assert_not_called()
    other.location = elsewhere
    room.current_visitors[:] = []
    assert Action('Registry fallback', [], [delta], initiator=actor,
        target=partner.name, graph_manager=graph).execute()
    writes.assert_called_once_with(partner, 'energy', 30)
    writes.reset_mock()
    room.current_visitors[:] = [actor, other]
    assert Action('Populated visitors', [], [delta], initiator=actor,
        target=partner.uuid, graph_manager=graph).execute()
    writes.assert_not_called()
    room.current_visitors[:] = [actor, partner, other]
    partner.location = elsewhere
    assert Action('Elsewhere', [], [delta], initiator=actor,
        target=partner.uuid, graph_manager=graph).execute()
    actor.location = None
    assert Action('No actor location', [], [delta], initiator=actor,
        target=partner.uuid, graph_manager=graph).execute()
    writes.assert_not_called()
    actor.location = room
    self_effect = dict(delta, targets=['initiator'], change_value=1)
    required = dict(delta, targets=['target'])
    assert not Action('Required missing', [], [self_effect, required], initiator=actor,
        target='missing', graph_manager=graph).execute()
    assert not Action('Blocked', [Condition('Blocked', 'energy', actor, 100)],
        [self_effect], initiator=actor, graph_manager=graph).execute()
    writes.assert_not_called()
    assert Action('Optional missing', [], [self_effect, delta], initiator=actor,
        target='missing', graph_manager=graph).execute()
    writes.assert_called_once_with(actor, 'energy', 21)
assert actor.energy == graph.G.nodes[actor]['energy'] == 21
assert partner.energy == graph.G.nodes[partner]['energy'] == 30
assert other.energy == graph.G.nodes[other]['energy'] == 20
''')

    def test_named_ambiguity_and_failed_field_preserve_exact_state(self):
        self.run_nearby_world('''
collision = new_character(str(partner.uuid), room)
room.current_visitors[:] = [actor, partner, collision]
self_effect = dict(delta, targets=['initiator'], change_value=1)
with patch.object(graph, 'update_node_attribute', wraps=graph.update_node_attribute) as writes:
    assert not Action('Ambiguous', [], [self_effect, delta], initiator=actor,
        target=str(partner.uuid), graph_manager=graph).execute()
    writes.assert_not_called()
assert actor.energy == partner.energy == 20
room.current_visitors[:] = [copy(actor), partner, copy(partner)]
original = graph.update_node_attribute
def fail_after_write(*args):
    original(*args)
    if args[1] == 'temporary602':
        raise RuntimeError('Injected later named field failure')
with patch.object(graph, 'update_node_attribute', side_effect=fail_after_write) as writes:
    assert not Action('Partial nearby', [], [delta, dict(delta, attribute='temporary602',
        change_value=1)], initiator=actor, target=partner.uuid, graph_manager=graph).execute()
    assert [c.args for c in writes.call_args_list] == [
        (partner, 'energy', 25), (partner, 'temporary602', 1)]
assert partner.energy == graph.G.nodes[partner]['energy'] == 25
assert not hasattr(partner, 'temporary602') and 'temporary602' not in graph.G.nodes[partner]
assert actor.energy == graph.G.nodes[actor]['energy'] == 20
assert other.energy == graph.G.nodes[other]['energy'] == 20
''')
