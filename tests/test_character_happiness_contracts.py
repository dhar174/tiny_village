"""Character-local happiness outcomes using complete headless runtime objects."""

import textwrap

from tests.test_graph_initialization import FreshPythonTestCase


HAPPINESS_SETUP = '''
import json
import math
from numbers import Real
import random

from tiny_characters import Character, Motive, PersonalMotives
from tiny_gameplay_controller import GameplayController
from tiny_graph_manager import GraphManager

random.seed(736)
MOTIVE_NAMES = (
    'hunger', 'wealth', 'mental_health', 'social_wellbeing', 'happiness',
    'health', 'shelter', 'stability', 'luxury', 'hope', 'success', 'control',
    'job_performance', 'beauty', 'community', 'material_goods', 'family',
)
FIXED_INPUTS = {
    'hope': 40, 'success': 50, 'control': 50, 'beauty': 50,
    'community': 40, 'job_performance': 50, 'material_goods': 50,
    'social_wellbeing': 40, 'stability': 40, 'mental_health': 40,
}
graph = GraphManager()
controller = GameplayController(graph, {
    'headless': True,
    'memory': {'mode': 'basic'},
    'characters': {'count': 1, 'use_llm': False},
    'checkpoint': {'auto_enabled': False},
})
'''

HAPPINESS_INPUTS = '''
assert len(controller.characters) == 1
character = next(iter(controller.characters.values()))
assert isinstance(character, Character) and character._initialized
assert character.graph_manager is graph
assert character.memory_manager.mode == 'basic'
assert not character.use_llm_decisions
assert not controller.initialization_errors, controller.initialization_errors
assert not controller.checkpoint_manager.auto_checkpoint_enabled
character.set_motives(PersonalMotives(**{
    name + '_motive': Motive(name, 'Happiness contract fixture', 5)
    for name in MOTIVE_NAMES
}))
for name, value in FIXED_INPUTS.items():
    getattr(character, 'set_' + name)(value)
# These optional Character-local inputs have no production setters. Their
# contribution tests do not claim graph-derived relationship integration.
character.relationships = {}
character.romantic_partner = None
character.family_members = []

def evaluate(label):
    value = character.calculate_happiness()
    assert isinstance(value, Real) and not isinstance(value, bool), repr(value)
    assert math.isfinite(value), repr(value)
    print(json.dumps({
        'case': label,
        'happiness': float(value),
        'attributes': {name: getattr(character, 'get_' + name)()
                       for name in FIXED_INPUTS},
        'motive_scores': {name: getattr(character.motives, 'get_' + name + '_motive')().get_score()
                         for name in MOTIVE_NAMES},
        'relationships': character.relationships,
        'romantic_partner': character.romantic_partner,
        'family_members': character.family_members,
    }, sort_keys=True), flush=True)
    return value
'''


def happiness_source(case):
    """Assemble a fresh-process fixture without reproducing happiness logic."""
    return (
        HAPPINESS_SETUP + '\ntry:\n'
        + textwrap.indent(HAPPINESS_INPUTS + textwrap.dedent(case), '    ')
        + '\nfinally:\n    controller.close()\n'
    )


class TestCharacterHappinessContracts(FreshPythonTestCase):
    def test_supported_motive_caps_change_happiness(self):
        # This covers the existing five motive caps, not the additional
        # motive_satisfaction contribution retained in the blocked artifact.
        self.run_python(happiness_source('''
for name in ('success', 'control', 'beauty', 'job_performance', 'material_goods'):
    getattr(character.motives, 'get_' + name + '_motive')().set_score(2)
low = evaluate('supported_motive_caps_2')
for name in ('success', 'control', 'beauty', 'job_performance', 'material_goods'):
    getattr(character.motives, 'get_' + name + '_motive')().set_score(8)
high = evaluate('supported_motive_caps_8')
assert high > low, (low, high)
'''))

    def test_social_relationship_quality_changes_happiness(self):
        self.run_python(happiness_source('''
character.relationships = {'peer': 0.2}
low = evaluate('social_quality_0.2')
character.relationships = {'peer': 0.8}
high = evaluate('social_quality_0.8')
assert high > low, (low, high)
'''))

    def test_romantic_partner_adds_happiness(self):
        self.run_python(happiness_source('''
character.relationships = {'peer': 0.8}
without_partner = evaluate('same_relationship_without_partner')
character.romantic_partner = 'peer'
with_partner = evaluate('same_relationship_with_partner')
assert with_partner > without_partner, (without_partner, with_partner)
'''))

    def test_family_membership_adds_happiness(self):
        self.run_python(happiness_source('''
character.relationships = {'peer': 0.8}
without_family = evaluate('same_relationship_without_family')
character.family_members = ['peer']
with_family = evaluate('same_relationship_with_family')
assert with_family > without_family, (without_family, with_family)
'''))

    def test_empty_relationships_have_finite_happiness(self):
        self.run_python(happiness_source('''
assert character.relationships == {}
assert character.romantic_partner is None and character.family_members == []
evaluate('empty_relationships')
'''))

    def test_no_partner_has_finite_happiness(self):
        self.run_python(happiness_source('''
character.relationships = {'peer': 0.8}
assert character.romantic_partner is None
evaluate('no_partner')
'''))

    def test_no_family_has_finite_happiness(self):
        self.run_python(happiness_source('''
character.relationships = {'peer': 0.8}
character.romantic_partner = 'peer'
assert character.family_members == []
evaluate('no_family')
'''))
