"""Unique State and Action contracts salvaged from PR #420."""

from tests.test_graph_initialization import FreshPythonTestCase


class TestActionContracts(FreshPythonTestCase):
    def test_dict_state_reads_defaults_and_writes(self):
        self.run_python('''
from actions import State
state = State({'energy': 5, 'focus': 3})
assert state['energy'] == 5
assert state.get('focus') == 3
assert state.get('missing_value', 99) == 99
state['energy'] = 8
assert state['energy'] == 8
print({'final_state': state.dict_or_obj})
''')

    def test_empty_preconditions_and_additive_state_effect(self):
        self.run_python('''
from actions import Action, State
action = Action(name='Rest', preconditions=[],
                effects=[{'attribute': 'energy', 'change_value': 2}], cost=1)
state = State({'energy': 5})
assert action.preconditions_met() is True
updated = action.apply_effects(state)
assert updated['energy'] == 7, updated['energy']
print({'initial_energy': 5, 'final_energy': updated['energy']})
''')

    def test_condition_preconditions_follow_complete_character_energy(self):
        self.run_python('''
from tiny_gameplay_controller import GameplayController
from tiny_graph_manager import GraphManager
from tiny_characters import Character
from actions import Action, Condition
controller = GameplayController(GraphManager(), {'headless': True,
    'memory': {'mode': 'basic'}, 'characters': {'count': 1, 'use_llm': False},
    'checkpoint': {'auto_enabled': False}})
try:
    assert len(controller.characters) == 1
    character = next(iter(controller.characters.values()))
    assert isinstance(character, Character) and character._initialized
    condition = Condition('HasEnergy', 'energy', character, 10, '>=')
    action = Action(name='RequiresEnergy', preconditions=[condition],
                    effects=[{'attribute': 'energy', 'change_value': 1}], cost=1)
    for energy, expected in ((12, True), (5, False)):
        character.energy = energy
        assert character.get_state()['energy'] == energy
        assert condition.check_condition() is expected, (energy, expected)
        assert action.preconditions_met() is expected, (energy, expected)
        print({'energy': energy, 'condition_and_preconditions': expected})
finally:
    controller.close()
''')
