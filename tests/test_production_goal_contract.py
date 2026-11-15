"""Production Goal acceptance; no replacement Goal or Character fixtures."""
from tests import test_goap_graph_contracts
from tests.test_graph_initialization import FreshPythonTestCase

GOAL_SETUP = '''
assert isinstance(character, Character) and character._initialized
assert graph.characters[character.name] is character
def make_goal(conditions=None):
    if conditions is None:
        conditions = {False: [Condition('recover', 'energy', character, 30)]}
    return Goal('Recover energy', character, character, 1, 'recover', conditions,
                None, None, None, None, None, None, [], graph, 'basic', {'energy': 30})
planner = GOAPPlanner(graph)
'''


class TestProductionGoalContract(FreshPythonTestCase):
    def world(self, source):
        test_goap_graph_contracts.TestSupportedPlannerPrimitives.run_world(self, GOAL_SETUP + source)

    def test_container_and_repeated_evaluation(self):
        self.world('''
a = Condition('recover', 'energy', character, 30)
b = Condition('not hungry', 'hunger', character, 10, '<=')
goal = make_goal({False: a, True: (b,)})
assert list(goal.iter_completion_conditions()) == [a, b]
assert goal.completion_conditions == {False: [a], True: [b]}
assert goal.check_completion({'energy': 30, 'hunger_level': 10}) is True
assert goal.check_completion({'energy': 29.95, 'hunger_level': 10}) is False
assert goal.check_completion({'energy': 30, 'hunger_level': 11}) is False
assert goal.check_completion({'energy': 30, 'hunger_level': 10}) is True
assert make_goal({}).check_completion({}) is True
assert goal.completion_conditions == {False: [a], True: [b]}
''')

    def test_live_state_plan_and_graph_share_authority(self):
        self.world('''
goal = make_goal()
plan = Plan('recover', [goal], graph_manager=graph)
character.energy = 90
assert goal.check_completion() is False
assert plan.evaluate() is False
assert graph.is_goal_achieved(character, goal) is False
assert goal.check_completion(State({'energy': 30})) is True
assert goal.check_completion() is False
recovery = Action('Recover', [], [{'targets': ['initiator'], 'attribute': 'energy', 'change_value': 10}], initiator=character, graph_manager=graph)
assert recovery.execute(character=character)
assert character.energy == graph.G.nodes[character]['energy'] == 30
assert goal.check_completion() is plan.evaluate() is graph.is_goal_achieved(character, goal) is True
''')

    def test_planner_outcomes_feasibility_and_cost(self):
        self.world('''
goal = make_goal()
state = State({'energy': 20})
cheap = Action('Cheap', [], [{'attribute': 'energy', 'change_value': 10}], cost=1, initiator=character, graph_manager=graph)
expensive = Action('Expensive', [], cheap.effects, cost=5, initiator=character, graph_manager=graph)
blocked = Action('Blocked', [Condition('unavailable', 'energy', character, 21)], cheap.effects, initiator=character, graph_manager=graph)
for actions in ([expensive, cheap], [cheap, expensive]):
    selected = planner.plan_actions(character, goal, state, actions)
    assert selected == [cheap], selected
    final = selected[0].apply_effects(State(dict(state.dict_or_obj)))
    assert final.get('energy') == 30
    assert planner._goal_satisfied(goal, final) is True
assert planner.plan_actions(character, goal, State({'energy': 30}), []) == []
for actions in ([], [blocked]):
    assert planner.plan_actions(character, goal, state, actions) is None
    assert planner.evaluate_feasibility_of_goal(goal, state, actions=actions) is False
assert planner.evaluate_feasibility_of_goal(goal, state, actions=[cheap]) is True
assert planner.evaluate_feasibility_of_goal(goal, {'energy': 20}, actions=[cheap]) is True
assert planner.evaluate_feasibility_of_goal(goal, State({'energy': 30}), actions=[]) is True
assert planner.plan_actions(character, goal, state, [cheap]) == [cheap]  # positive control after negatives
assert state.dict_or_obj == {'energy': 20}
assert character.energy == graph.G.nodes[character]['energy'] == 20
half = Action('Half', [], [{'attribute': 'energy', 'change_value': 5}], initiator=character, graph_manager=graph)
planner.MAX_PLAN_LENGTH = 1
assert planner.evaluate_feasibility_of_goal(goal, state, actions=[half]) is False
planner.MAX_PLAN_LENGTH = 2
assert planner.evaluate_feasibility_of_goal(goal, state, actions=[half]) is True
assert planner.evaluate_feasibility_of_goal(goal, state) == (planner.plan_actions(character, goal, state) is not None)
''')

    def test_invalid_contracts_propagate(self):
        self.world('''
for invalid in ({False: [object()]}, {'energy': 30}, [Condition('energy', 'energy', character, 30)]):
    try:
        make_goal(invalid)
    except (TypeError, ValueError):
        pass
    else:
        raise AssertionError('Malformed Goal accepted')
goal = make_goal()
for evaluate in (lambda: goal.check_completion({}), lambda: planner._goal_satisfied(goal, State({})), lambda: planner.plan_actions(character, goal, State({}), []), lambda: planner.evaluate_feasibility_of_goal(goal, State({}), actions=[])):
    try:
        evaluate()
    except ValueError as error:
        assert 'recover' in str(error), str(error)
    else:
        raise AssertionError('Required attribute error swallowed')
goal.completion_conditions[False].append(None)
try:
    goal.check_completion({'energy': 30})
except (TypeError, ValueError):
    pass
else:
    raise AssertionError('Post-construction invalid condition ignored')
''')

    def test_partner_conditions_do_not_use_actor_snapshot(self):
        self.world('''
from tiny_characters import Character
partner = Character(name='GoalPartner', age=30, job='unemployed', graph_manager=graph, personality_traits=character.personality_traits, gametime_manager=character.gametime_manager, memory_mode='basic')
graph.add_character_node(partner)
graph.G.nodes[partner]['energy'] = 10
partner.energy = 90
goal = make_goal({False: [Condition('partner energy', 'energy', partner, 30)]})
assert goal.check_completion({'energy': 100}) is False
graph.G.nodes[partner]['energy'] = 30
assert goal.check_completion({'energy': 0}) is True
assert partner.energy == 90
actor_by_name = make_goal({False: [Condition('named actor', 'energy', character.name, 30)]})
assert actor_by_name.check_completion({'energy': 30}) is True
unresolved = make_goal({False: [Condition('missing actor', 'energy', 'NotRegistered', 30)]})
try:
    unresolved.check_completion({'energy': 30})
except ValueError as error:
    assert 'missing actor' in str(error)
else:
    raise AssertionError('Unresolved target accepted')
''')

    def test_inventory_literals(self):
        self.world('''
for expression in ("inventory.check_has_item_by_type(['food'])", 'inventory.check_has_item_by_type(["food"])'):
    condition = Condition('food', expression, character, True, '==')
    goal = make_goal({False: [condition]})
    assert goal.required_items == [({'item_type': 'food'}, True)]
    assert goal.check_completion() is True
    assert goal.check_completion(State({'inventory': character.inventory})) is True
    condition.satisfy_value = False
    assert goal.check_completion() is False
''')

    def test_import_orders_use_full_goal(self):
        for modules in ('tiny_characters, tiny_goap_system, tiny_utility_functions', 'tiny_utility_functions, tiny_goap_system, tiny_characters'):
            self.run_python('import ' + modules + '\nfrom tiny_characters import Goal\nassert hasattr(Goal, "iter_completion_conditions")\nassert "character" in __import__("inspect").signature(Goal).parameters\n')

    def test_registered_location_uses_its_graph_state(self):
        self.world('''
from tiny_locations import Location
place = Location('GoalRoom', 0, 0, 10, 10, controller.action_system)
graph.add_location_node(place)
place.open = False
graph.G.nodes[place]['open'] = True
goal = make_goal({False: [Condition('open room', 'open', place, True, '==')]})
assert goal.check_completion({'open': False}) is True
assert place.open is False
''')
