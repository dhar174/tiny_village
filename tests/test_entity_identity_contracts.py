from tests import test_goap_graph_contracts
from tests.test_graph_initialization import FreshPythonTestCase


class TestEntityIdentityContracts(FreshPythonTestCase):
    run_world = test_goap_graph_contracts.TestSupportedPlannerPrimitives.run_world

    def test_lookup_and_removal_survive_state_changes(self):
        self.run_world('''
from tiny_locations import Location
from actions import ActionSystem
room = Location('Room', 0, 0, 5, 5, ActionSystem(graph_manager=graph))
twin = Location('Room', 0, 0, 5, 5, ActionSystem(graph_manager=graph))
assert room != twin and room != (0, 0)
graph.add_location_node(room)
graph.G.add_edge(character, room)
hashes = hash(character), hash(room)
character.energy += 1
character.location = room
room.x += 1
room.current_visitors.extend([character, character])
assert (hash(character), hash(room)) == hashes
assert graph.G.has_edge(character, room)
for entity in (character, room):
    for identifier in (entity, entity.uuid, str(entity.uuid), entity.name):
        assert graph.resolve_node(identifier) is entity
    try:
        entity.uuid = room.uuid
    except AttributeError:
        pass
    else:
        raise AssertionError('UUID reassignment succeeded')
graph.add_location_node(room)
assert graph.G.has_edge(character, room)
from copy import copy
duplicate = copy(room)
duplicate.name = 'Other'
try:
    graph.add_location_node(duplicate)
except ValueError:
    pass
else:
    raise AssertionError('Duplicate UUID registered')
for raw in ('legacy-a', 'legacy-b'):
    graph.G.add_node(raw, type='location', name='Ambiguous')
try:
    graph.resolve_node('Ambiguous', 'location')
except ValueError as error:
    assert 'Ambiguous' in str(error)
else:
    raise AssertionError('Ambiguous name selected a node')
try:
    graph.add_location_node(twin)
except ValueError:
    pass
else:
    raise AssertionError('Duplicate name registered')
graph.world_state.remove_node(str(room.uuid))
assert room not in graph.G and character in graph.G
graph.world_state.remove_node(character.uuid)
assert character not in graph.G
''')

    def test_authority_empty_results_and_original_action(self):
        self.run_world('''
planner = GOAPPlanner(graph)
graph.G.nodes[character]['energy'] = 77
assert character.energy == 20
assert planner.get_current_world_state(character)['energy'] == 77
assert planner.get_available_actions(character) == []
try:
    planner.get_available_actions('missing-character')
except ValueError:
    pass
else:
    raise AssertionError('Missing graph entity hidden by fallback')
assert GOAPPlanner(None).get_available_actions(character)
from tiny_locations import Location
from actions import ActionSystem
room = Location('Room', 0, 0, 5, 5, ActionSystem(graph_manager=graph))
graph.add_location_node(room)
action = Action('Recovery', [], [{'targets': ['initiator'],
    'attribute': 'energy', 'change_value': 10}], cost=2, graph_manager=graph)
room.possible_interactions = [action]
graph.G.add_edge(character, room)
record, = graph.get_possible_actions(character)
assert record['action'] is action
assert record['cost'] == 2 and record['effects'] == action.effects
assert record['preconditions'] == action.preconditions
assert planner.get_available_actions(character)[0] is action
from tiny_strategy_manager import StrategyManager
strategy = StrategyManager(graph_manager=graph)
assert strategy._get_current_state_for_character(character)['energy'] == 77
assert strategy._get_actions_for_character(character)[0] is action
graph.G.remove_edge(character, room)
assert strategy._get_actions_for_character(character) == []
''')

    def test_exact_effect_and_failed_write_rollback(self):
        self.run_world('''
from unittest.mock import patch
action = Action('Recovery', [], [{'targets': ['initiator'],
    'attribute': 'energy', 'change_value': 10}], graph_manager=graph)
assert action.execute(character)
assert character.energy == graph.G.nodes[character]['energy'] == 30
graph.G.nodes[character]['energy'] = 40
assert action.execute(character)
assert character.energy == graph.G.nodes[character]['energy'] == 50
original = graph.update_node_attribute
def fail_after_write(*args):
    original(*args)
    raise RuntimeError('injected graph write failure')
with patch.object(graph, 'update_node_attribute', side_effect=fail_after_write):
    assert action.execute(character) is False
assert character.energy == graph.G.nodes[character]['energy'] == 50
two_effects = Action('Partial', [], action.effects + [{'targets': ['initiator'],
    'attribute': '799_temporary', 'change_value': 1}], graph_manager=graph)
def fail_second(node, attribute, value):
    original(node, attribute, value)
    if attribute == '799_temporary':
        raise RuntimeError('injected second effect failure')
with patch.object(graph, 'update_node_attribute', side_effect=fail_second):
    assert two_effects.execute(character) is False
assert character.energy == graph.G.nodes[character]['energy'] == 60
assert not hasattr(character, '799_temporary')
assert '799_temporary' not in graph.G.nodes[character]
blocked = Action('Blocked', [Condition('Need energy', 'energy', character, 100)],
                 action.effects, graph_manager=graph)
assert blocked.execute(character) is False
assert character.energy == graph.G.nodes[character]['energy'] == 60
''')


    def test_named_target_synchronizes_real_characters_once(self):
        self.run_python('''
import runpy

probe = runpy.run_path('docs/planning/issue-734/evidence/audit_probes.py')
probe['named_target']()
''')

    def test_named_targets_use_location_identity_and_reject_ambiguity(self):
        self.run_world('''
from actions import ActionSystem
from tiny_locations import Location
from types import SimpleNamespace
from uuid import uuid4
system = ActionSystem(graph_manager=graph)
room = Location('Room', 0, 0, 5, 5, system)
twin = Location('Room', 0, 0, 5, 5, system)
character.location = room
other = SimpleNamespace(name='Other', uuid=uuid4(), location=twin)
room.current_visitors.extend([character, other, other])
action = Action('Nearby', [], [{'targets': ['target_character_in_location'],
    'attribute': 'energy', 'change_value': 10}], initiator=character,
    target=other.uuid, graph_manager=graph)
assert action.execute() is True
assert graph.G.nodes[character]['energy'] == 20
other.location = room
collision = SimpleNamespace(name=str(other.uuid), uuid=uuid4(), location=room)
room.current_visitors.append(collision)
action.target = str(other.uuid)
assert action.execute() is False
assert not hasattr(other, 'energy') and not hasattr(collision, 'energy')
''')
