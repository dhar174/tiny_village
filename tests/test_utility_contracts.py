"""Real-Action utility contracts from PR #417, including inherited golden coverage.

Goal here is the utility module's production export, including its lightweight
fallback. These cases do not claim the full tiny_characters.Goal contract.
"""

from tests.test_graph_initialization import FreshPythonTestCase


UTILITY_IMPORTS = '''
import math
from tiny_utility_functions import Goal, calculate_action_utility, calculate_plan_utility
from actions import Action
print({'utility_goal_class': Goal.__module__ + '.' + Goal.__name__})
'''

EDGE_INPUTS = '''
states = [
    {'hunger': 0.0, 'energy': 1.0, 'happiness': 0.5},
    {'hunger': 1.0, 'energy': 0.0, 'happiness': 0.0},
    {'hunger': 0.5, 'energy': 0.5, 'happiness': 1.0},
    {'hunger': 0.999, 'energy': 0.001, 'happiness': 0.001},
]
goal = Goal(name='SatisfyHunger', target_effects={'hunger': -0.2}, priority=0.7)
action = Action(name='EatAndWork', preconditions=[], effects=[
    {'attribute': 'hunger', 'change_value': -0.3},
    {'attribute': 'energy', 'change_value': -0.1}], cost=1.0)
'''


class TestUtilityContracts(FreshPythonTestCase):
    def test_real_action_goal_utility_matches_29_point_1(self):
        self.run_python(UTILITY_IMPORTS + '''
state = {'hunger': 0.9, 'energy': 0.5}
before = state.copy()
goal = Goal(name='SatisfyHunger', target_effects={'hunger': -0.8}, priority=0.7)
action = Action(name='EatFood', preconditions=[],
    effects=[{'attribute': 'hunger', 'change_value': -0.7}], cost=0.1)
utility = calculate_action_utility(state, action, current_goal=goal)
assert isinstance(utility, (int, float)) and math.isfinite(utility), utility
# Hunger benefit 12.6 + goal progress 17.5 - cost 1.0.
assert math.isclose(utility, 29.1, rel_tol=0, abs_tol=1e-9), utility
assert state == before, (before, state)
print({'state': state, 'utility': utility, 'expected': 29.1})
''')

    def test_edge_state_action_utilities_are_finite_and_preserve_input(self):
        self.run_python(UTILITY_IMPORTS + EDGE_INPUTS + '''
for state in states:
    before = state.copy()
    utility = calculate_action_utility(state, action, current_goal=goal)
    assert isinstance(utility, (int, float)) and math.isfinite(utility), (state, utility)
    assert abs(utility) < 1000, (state, utility)
    assert state == before, (before, state)
    print({'state': state, 'action_utility': utility})
''')

    def test_edge_state_plan_utilities_are_finite_and_preserve_input(self):
        self.run_python(UTILITY_IMPORTS + EDGE_INPUTS + '''
for state in states:
    for simulate in (False, True):
        before = state.copy()
        utility = calculate_plan_utility(state, [action], current_goal=goal,
                                         simulate_effects=simulate)
        assert isinstance(utility, (int, float)) and math.isfinite(utility), (state, utility)
        assert state == before, (before, state, simulate)
        print({'state': state, 'simulate_effects': simulate, 'plan_utility': utility})
''')

    def test_real_action_utility_tracks_hunger_and_goal_alignment(self):
        self.run_python(UTILITY_IMPORTS + '''
goal = Goal(name='SatisfyHunger', target_effects={'hunger': -0.2}, priority=0.7)
eat = Action(name='EatMeal', preconditions=[],
    effects=[{'attribute': 'hunger', 'change_value': -0.3}], cost=1.0)
work = Action(name='WorkShift', preconditions=[],
    effects=[{'attribute': 'energy', 'change_value': -0.2}], cost=1.0)
hungry = {'hunger': 0.9, 'energy': 0.7, 'health': 0.8}
satiated = {'hunger': 0.1, 'energy': 0.7, 'health': 0.8}
before = (hungry.copy(), satiated.copy())
hungry_utility = calculate_action_utility(hungry, eat, current_goal=goal)
satiated_utility = calculate_action_utility(satiated, eat, current_goal=goal)
misaligned_utility = calculate_action_utility(hungry, work, current_goal=goal)
assert all(math.isfinite(value) for value in
           (hungry_utility, satiated_utility, misaligned_utility))
assert hungry_utility > satiated_utility, (hungry_utility, satiated_utility)
assert misaligned_utility < hungry_utility, (misaligned_utility, hungry_utility)
assert (hungry, satiated) == before, (before, hungry, satiated)
print({'hungry': hungry, 'satiated': satiated, 'hungry_utility': hungry_utility,
       'satiated_utility': satiated_utility, 'misaligned_utility': misaligned_utility})
''')

    def test_real_action_plan_simulation_reduces_repeated_eating_value(self):
        self.run_python(UTILITY_IMPORTS + '''
state = {'hunger': 0.9, 'energy': 0.7, 'health': 0.8}
before = state.copy()
goal = Goal(name='SatisfyHunger', target_effects={'hunger': -0.2}, priority=0.7)
eat = Action(name='EatMeal', preconditions=[],
    effects=[{'attribute': 'hunger', 'change_value': -0.3}], cost=1.0)
single = calculate_action_utility(state, eat, current_goal=goal)
unsimulated = calculate_plan_utility(state, [eat, eat], current_goal=goal,
                                     simulate_effects=False)
assert state == before, (before, state)
simulated = calculate_plan_utility(state, [eat, eat], current_goal=goal,
                                   simulate_effects=True)
assert all(math.isfinite(value) for value in (single, unsimulated, simulated))
assert math.isclose(unsimulated, 2 * single, rel_tol=0, abs_tol=1e-9), unsimulated
assert simulated < 2 * single, (simulated, single)
assert state == before, (before, state)
print({'state': state, 'single_utility': single, 'unsimulated_plan': unsimulated,
       'simulated_plan': simulated})
''')
