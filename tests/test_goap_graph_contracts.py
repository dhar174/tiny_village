"""Supported Plan/planner primitives using complete domain objects.

These cases do not establish goal completion, graph-fed planning or world
execution. Required failures for those boundaries are retained separately in
docs/planning/issue-736/execution/blocked/.
"""

import textwrap

from tests.test_graph_initialization import FreshPythonTestCase


WORLD_SETUP = '''
import math
import random
random.seed(736)
from actions import Action, Condition, State
from tiny_characters import Character, Goal
from tiny_gameplay_controller import GameplayController
from tiny_goap_system import GOAPPlanner, Plan
from tiny_graph_manager import GraphManager
from networkx import MultiDiGraph
graph = GraphManager()
controller = GameplayController(graph, {
    'headless': True, 'memory': {'mode': 'basic'},
    'characters': {'count': 1, 'use_llm': False, 'use_custom': True,
        'character_data': [{'name': 'ContractActor', 'age': 30,
            'job': 'unemployed', 'energy': 20}]},
    'checkpoint': {'auto_enabled': False}})
'''


class TestSupportedPlannerPrimitives(FreshPythonTestCase):
    def run_world(self, source):
        self.run_python(
            WORLD_SETUP + "\ntry:\n    character = next(iter(controller.characters.values()))\n" + textwrap.indent(textwrap.dedent(source), '    ')
            + '\nfinally:\n    controller.close()\n'
        )

    def test_plan_retains_real_goal_and_action_queue_entries(self):
        self.run_world('''
assert isinstance(character, Character) and character._initialized
assert isinstance(graph.G, MultiDiGraph)
assert graph.characters[character.name] is character
assert any(node is character for node in graph.G)
condition = Condition('energy_target', 'energy', character, 30)
goal = Goal('Recover energy', character, character, 1, 'recover',
    {False: [condition]}, None, None, None, None, None, None, [],
    graph, 'Basic Needs', {'energy': 30})
first = Action('First', [], [], cost=1, graph_manager=graph)
second = Action('Second', [], [], cost=2, graph_manager=graph)
plan = Plan('ordered_recovery', graph_manager=graph)
plan.add_goal(goal)
plan.add_action(first, priority=1, dependencies=['prepare'])
plan.add_action(second, priority=1)
assert plan.name == 'ordered_recovery'
assert plan.goals == [goal] and plan.goals[0] is goal
assert len(plan.action_queue) == 2
assert plan.action_queue[0] == (1, 0, first, ['prepare'])
assert plan.action_queue[1] == (1, 1, second, [])
print({'queue': [entry[2].name for entry in plan.action_queue],
       'goal': goal.name, 'completion_checked': False})
''')

    def test_blocked_real_condition_prevents_action_execution(self):
        self.run_world('''
initial_energy = character.energy
condition = Condition('needs_more_energy', 'energy', character, initial_energy + 1)
action = Action('BlockedRecovery', [condition],
    [{'targets': ['initiator'], 'attribute': 'energy', 'change_value': 10}],
    initiator=character, graph_manager=graph)
assert condition.check_condition() is False
assert action.preconditions_met() is False
assert GOAPPlanner(graph)._action_applicable(action, character.get_state()) is False
assert action.execute(character=character, graph_manager=graph) is False
assert character.energy == initial_energy
print({'precondition': False, 'initial_energy': initial_energy,
       'final_energy': character.energy})
''')

    def test_planner_effect_simulation_isolated_from_input_and_character(self):
        self.run_world('''
action = Action('SimulatedRecovery', [],
    [{'targets': ['initiator'], 'attribute': 'energy', 'change_value': 10}],
    initiator=character, graph_manager=graph)
state = State({'energy': 20})
initial_energy = character.energy
result = GOAPPlanner(graph)._apply_action_effects(action, state)
assert result is not state and result.get('energy') == 30
assert state.dict_or_obj == {'energy': 20}
assert character.energy == initial_energy
applied = action.apply_effects(State({'energy': 20}))
assert applied.get('energy') == 30
print({'simulated_energy': result.get('energy'),
       'caller_energy': state.get('energy'), 'character_energy': character.energy})
''')

    def test_cost_and_heuristic_use_current_context_signatures(self):
        self.run_world('''
condition = Condition('energy_target', 'energy', character, 50)
goal = Goal('Recover energy', character, character, 1, 'recover',
    {False: [condition]}, None, None, None, None, None, None, [],
    graph, 'Basic Needs', {'energy': 50})
action = Action('Recovery', [],
    [{'attribute': 'energy', 'change_value': 10}], cost=2, graph_manager=graph)
planner = GOAPPlanner(graph)
cost = planner._calculate_action_cost(action, State({'energy': 20}), character, goal)
far = planner._estimate_cost_to_goal(State({'energy': 20}), goal, character)
near = planner._estimate_cost_to_goal(State({'energy': 40}), goal, character)
at_target = planner._estimate_cost_to_goal(State({'energy': 50}), goal, character)
assert isinstance(cost, (int, float)) and math.isfinite(cost) and cost > 0
assert math.isfinite(far) and math.isfinite(near) and 0 <= near < far
assert at_target == 0
assert planner._estimate_cost_to_goal(State({'energy': 20}), None, character) == 0
print({'cost': cost, 'far_estimate': far, 'near_estimate': near,
       'at_target_estimate': at_target, 'goal_completion_checked': False})
''')

    def test_planner_utility_selects_a_real_action_from_sequence(self):
        self.run_world('''
low = Action('LowerBenefit', [], [], cost=1, graph_manager=graph)
high = Action('HigherBenefit', [], [], cost=1, graph_manager=graph)
low.satisfaction, high.satisfaction = 1, 10
planner = GOAPPlanner(graph)
low_utility = planner.calculate_utility(low, character)
high_utility = planner.calculate_utility(high, character)
assert math.isfinite(low_utility) and math.isfinite(high_utility)
assert high_utility > low_utility
assert planner.evaluate_utility([low, high], character) is high
print({'lower_utility': low_utility, 'higher_utility': high_utility,
       'selected': high.name, 'goal_outcome_checked': False})
''')
