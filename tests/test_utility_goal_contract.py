"""Stable utility metadata and the separate full-domain Goal bridge."""
from tests.test_graph_initialization import FreshPythonTestCase
from tests import test_goap_graph_contracts


class TestUtilityGoalContract(FreshPythonTestCase):
    def test_metadata_fields_defaults_and_explicit_zero(self):
        self.run_python('''
import tiny_utility_functions as utility
assert hasattr(utility, 'UtilityGoal'), 'Explicit production UtilityGoal is missing'
assert utility.Goal is utility.UtilityGoal
first = utility.UtilityGoal()
second = utility.UtilityGoal()
assert first.name == 'UnnamedGoal' and first.priority == first.score == first.urgency == .5
assert first.deadline is None and first.description == 'Goal: UnnamedGoal'
assert first.target_effects is not second.target_effects
assert first.attributes is not second.attributes
explicit = utility.UtilityGoal('Urgent', {'energy': .5}, 0, score=0,
    urgency=.9, deadline=7, description='Urgent metadata', attributes={'energy': 1})
assert explicit.priority == explicit.score == 0 and explicit.urgency == .9
assert explicit.deadline == 7 and explicit.description == 'Urgent metadata'
assert explicit.attributes == {'energy': 1}
assert utility.UtilityGoal(priority=.7, urgency=None).urgency == .7
assert not hasattr(explicit, 'check_completion')  # metadata never supplies full-domain completion
''')

    def test_import_orders_and_reload_preserve_role_and_scoring(self):
        for first in ('tiny_characters', 'tiny_utility_functions', 'tiny_strategy_manager'):
            with self.subTest(first=first):
                self.run_python('import ' + first + '''
import importlib, math
import tiny_utility_functions as utility
import tiny_characters as characters
from actions import Action
for reload_module in (False, True):
    if reload_module:
        importlib.reload(utility)
    assert hasattr(utility, 'UtilityGoal'), 'Explicit production UtilityGoal is missing'
    assert utility.Goal is utility.UtilityGoal and utility.Goal is not characters.Goal
    goal = utility.Goal('SatisfyHunger', {'hunger': -.8}, .7)
    assert goal.urgency == .7 and goal.deadline is None
    action = Action('EatFood', [], [{'attribute': 'hunger', 'change_value': -.7}], cost=.1)
    state = {'hunger': .9, 'energy': .5}
    before = state.copy()
    assert math.isclose(utility.calculate_action_utility(state, action, goal), 29.1)
    assert state == before
''')

    def test_urgency_multiplier_uses_explicit_metadata(self):
        self.run_python('''
import math
import tiny_utility_functions as utility
from actions import Action
assert hasattr(utility, 'UtilityGoal'), 'Explicit production UtilityGoal is missing'
action = Action('Recovery', [], [{'attribute': 'energy', 'change_value': .5}], cost=.1)
normal = utility.UtilityGoal('Recover', {'energy': .4}, .8, urgency=.5)
urgent = utility.UtilityGoal('Recover', {'energy': .4}, .8, urgency=.9)
evaluator = utility.UtilityEvaluator()
state = {'energy': .3}
before = state.copy()
normal_score = evaluator.evaluate_action_utility('actor', state, action, normal)
urgent_score = evaluator.evaluate_action_utility('actor', state, action, urgent)
assert normal_score > 0 and math.isclose(urgent_score, 2 * normal_score)
assert state == before
''')

    def test_full_native_goal_scoring_and_execution_remain_separate(self):
        test_goap_graph_contracts.TestSupportedPlannerPrimitives.run_world(self, '''
from tiny_utility_functions import calculate_native_action_utility
assert isinstance(character, Character) and character._initialized
assert graph.characters[character.name] is character
condition = Condition('rested', 'energy', character, 30)
goal = Goal('Recover energy', character, character, 1, 'recover', {False: [condition]},
    None, None, None, None, None, None, [], graph, 'basic', {'energy': 30})
action = Action('Recovery', [], [{'targets': ['initiator'], 'attribute': 'energy', 'change_value': 10}],
    cost=.1, initiator=character, graph_manager=graph)
state = {'energy': 20}
before = state.copy()
assert goal.check_completion() is False
# Need benefit .8*.1*15 + existing default goal bonus .5*25 - cost .1*10.
score = calculate_native_action_utility(state, action, goal)
assert math.isclose(score, 12.7), score
assert state == before and character.energy == graph.G.nodes[character]['energy'] == 20
assert action.execute(character=character)
assert character.energy == graph.G.nodes[character]['energy'] == 30
assert goal.check_completion() is True
assert math.isclose(calculate_native_action_utility({'energy': 30}, action, goal), .05)
''')
