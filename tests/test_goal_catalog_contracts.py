"""Real-domain regressions for #821; no substitute domain classes."""
from tests import test_production_goal_contract as production
from tests.test_graph_initialization import FreshPythonTestCase


class TestGoalCatalogContracts(FreshPythonTestCase):
    world = production.TestProductionGoalContract.world
    def test_inventory_categories_and_attribute_queries(self):
        self.world('''
from tiny_items import ItemInventory, ItemObject, FoodItem
from actions import ActionSystem
for category in ('clothing', 'tools', 'weapons', 'medicine', 'misc'):
    empty = ItemInventory()
    assert empty.check_has_item_by_type(category) is False
    assert empty.check_has_item_by_type(category, 0, '==') is True
    stacks = [ItemObject(str(i), 'stack', 1, 1, qty, item_type=category) for i, qty in enumerate((2, 3))]
    inv = ItemInventory(**{category + '_items': stacks})
    assert inv.check_has_item_by_type(category, 5)
    assert not inv.check_has_item_by_type(category, 6)
raw = FoodItem('raw', 'raw', 1, False, 1, 1, ActionSystem(), cooked=False)
cooked = FoodItem('cooked', 'cooked', 1, False, 1, 1, ActionSystem(), cooked=True)
inv = ItemInventory(food_items=[raw, cooked])
assert inv.check_has_item_by_attribute_value('cooked', True, '==')
assert not inv.check_has_item_by_attribute_value('absent', True, '==')
assert not ItemInventory(food_items=[raw]).check_has_item_by_attribute_value('cooked', True, '==')
assert ItemInventory().check_has_item_by_type('food') is False
assert not inv.check_has_item_by_type('unknown')
assert inv.check_has_item_by_attribute_value('quantity', 1, 'ge')
assert inv.check_has_item_by_attribute_value('quantity', 1, '>=')
for oper in ('unsupported',):
    try: ItemInventory().check_has_item_by_attribute_value('cooked', True, oper)
    except KeyError: pass
    else: raise AssertionError('Invalid operator accepted')
''')

    def test_factory_cook_requires_owned_cooked_food(self):
        self.world('''
from tiny_characters import Motive, motive_to_goals
from tiny_items import ItemInventory, ItemObject, FoodItem
from actions import ActionSystem
goal = next(g for _, g in motive_to_goals(Motive('hunger', 'hunger', 5), character, graph, planner, character.prompt_builder) if g.name == 'Cook')
assert [c.name for c in goal.iter_completion_conditions()] == ['has_food', 'has_cooked_food']
raw = FoodItem('raw', 'raw', 1, False, 1, 1, ActionSystem(), cooked=False)
cooked = FoodItem('cooked', 'cooked', 1, False, 1, 0, ActionSystem(), cooked=True)
tool = ItemObject('tool', 'tool', 1, 1, 1, item_type='tools')
tool.cooked = True
for foods, tools, expected in (([], [], False), ([raw], [], False), ([raw], [tool], False), ([cooked], [], False)):
    graph.G.nodes[character]['inventory'] = ItemInventory(food_items=foods, tools_items=tools)
    assert goal.check_completion() is expected
cooked.quantity = 1
graph.G.nodes[character]['inventory'] = ItemInventory(food_items=[raw, cooked])
assert goal.check_completion() is True
''')

    def test_completion_dispatch_preserves_full_conditions(self):
        self.world('''
from tiny_utility_functions import UtilityGoal, is_goal_achieved
empty = UtilityGoal()
for check in (lambda: is_goal_achieved(empty, State({'energy': 20})), lambda: planner._goal_satisfied(empty, State({'energy': 20})), lambda: graph.is_goal_achieved(character, empty)):
    assert check() is False
goal = make_goal()
goal.target_effects = {'energy': 10}
assert is_goal_achieved(goal, State({'energy': 20})) is False
assert make_goal({}).check_completion({}) is True
empty.priority, empty.description = 10, 'wealth'
character.wealth_money = 999
assert graph.is_goal_achieved(character, empty) is False
''')

    def test_job_projection_and_hypothetical_isolation(self):
        self.world('''
goal = make_goal({False: [Condition('employment', 'has_job', character, True, '==')]})
graph.G.nodes[character]['job'] = 'Baker'
character.job = 'unemployed'
assert character.has_job() is True
assert goal.check_completion() is True
assert goal.check_completion({'job': None}) is False
assert goal.check_completion({'job': 'Baker'}) is True
assert planner._goal_satisfied(goal, planner._planning_snapshot({'job': graph.G.nodes[character]['job']})) is True
try: goal.check_completion({})
except ValueError: pass
else: raise AssertionError('Missing hypothetical job accepted')
graph.G.nodes[character]['job'] = 'unemployed'
character.job = 'Baker'
assert goal.check_completion() is character.has_job() is False
''')

    def test_scoring_callback_and_errors(self):
        self.world('''
from tiny_utility_functions import GoalEvaluationError
calls = []
def score(actor, goal, environment):
    calls.append((actor, goal, environment))
    return 7
goal = make_goal()
goal.evaluate_utility_function = score
character.goals = [(0, goal)]
assert character.evaluate_goals() == [(7, goal)]
assert len(calls) == 1 and calls[0][0] is character and calls[0][1] is goal and calls[0][2] is graph
for invalid in (True, float('nan'), float('inf'), 'bad'):
    goal.evaluate_utility_function = lambda *args, v=invalid: v
    try: goal.evaluate_utility()
    except GoalEvaluationError: pass
    else: raise AssertionError('Invalid score accepted')
from tiny_strategy_manager import StrategyManager
for error in (RuntimeError, IndexError, ZeroDivisionError):
    def fail(*args, e=error): raise e('controlled')
    goal.evaluate_utility_function = fail
    try: StrategyManager(graph_manager=graph)._gather_character_goals(character)
    except GoalEvaluationError as caught: assert isinstance(caught.__cause__, error)
    else: raise AssertionError('Scoring failure hidden')
    predicate = make_goal({False: [Condition('boom', 'explode', character, True, '==')]})
    try: predicate.check_completion({'explode': fail})
    except GoalEvaluationError as caught: assert isinstance(caught.__cause__, error) and 'boom' in str(caught)
    else: raise AssertionError('Predicate error hidden')
''')

    def test_equality_and_plan_inspection_are_pure(self):
        self.world('''
a, b = make_goal({}), make_goal({})
assert a == b
b.goal_type = 'different'
assert a != b
calls = []
original = a.check_completion
def counted(*args, **kwargs):
    calls.append(1)
    return original(*args, **kwargs)
a.check_completion = counted
p, q = Plan('p', [a], graph_manager=graph), Plan('p', [a], graph_manager=graph)
assert p == q
str(p); repr(p)
assert calls == []
for value in (a, p):
    try: hash(value)
    except TypeError: pass
    else: raise AssertionError('Mutable value remains hashable')
assert p.evaluate() is True and len(calls) == 1
''')

    def test_requests_track_employment_and_preserve_error_outcomes(self):
        self.world('''
from tiny_strategy_manager import StrategyManager, PlanRequest
from tiny_utility_functions import GoalEvaluationError, normalize_job_name
from copy import copy
strategy = StrategyManager(graph_manager=graph)
job = graph.G.nodes[character]['job']
name = normalize_job_name(job)
assert planner._planning_snapshot({'job': job}).get('job') == name
assert planner._planning_snapshot({'job': None}).dict_or_obj['job'] is None
assert normalize_job_name(copy(job)) == name
employment = make_goal({False: [Condition('employment', 'has_job', character, True, '==')]})
request = PlanRequest(character.uuid, employment)
before = strategy.request_fingerprint(request)
graph.G.nodes[character]['job'] = 'Baker'
after = strategy.request_fingerprint(request)
assert before != after
assert after == strategy.request_fingerprint(request)
graph.G.nodes[character]['job'] = object()
energy = make_goal()
assert strategy.plan_for_goal(character, energy, actions=[]) is None
assert strategy.plan_request(PlanRequest(character.uuid, energy)).outcome in ('PLAN', 'NO_PLAN')
assert strategy.plan_request(PlanRequest(character.uuid, employment)).outcome == 'ERROR'
# A valid unreachable target and an invalid predicate have different outcomes.
assert strategy.plan_for_goal(character, energy, actions=[]) is None
original_candidates = strategy.request_candidates
strategy.request_candidates = lambda request: []
assert strategy.plan_request(PlanRequest(character.uuid, energy)).outcome == 'NO_PLAN'
strategy.request_candidates = original_candidates
invalid = make_goal({False: [Condition('missing', 'absent_required', character, True, '==')]})
assert strategy.plan_request(PlanRequest(character.uuid, invalid)).outcome == 'ERROR'
graph.G.nodes[character]['job'] = job
recovery = Action('Recovery', [], [{'targets': ['initiator'], 'attribute': 'energy', 'change_value': 10}], initiator=character, graph_manager=graph)
sequence = strategy.plan_for_goal(character, energy, actions=[recovery])
assert sequence == [recovery]
for action in sequence: assert action.execute(character=character)
assert graph.G.nodes[character]['energy'] == character.energy == 30
assert energy.check_completion()
''')

    def test_partner_employment_fingerprints_and_typed_identity(self):
        self.world('''
from tiny_strategy_manager import StrategyManager, PlanRequest
from tiny_locations import Location
from actions import ActionSystem
# A distinct registered Character, not a copied actor.
from tiny_characters import PersonalityTraits
partner = Character('Partner821', 31, graph_manager=graph, action_system=character.action_system,
    gametime_manager=character.gametime_manager, memory_mode='basic', location=character.location,
    personality_traits=PersonalityTraits(4, 4, 4, 4, 4), energy=20)
goal = make_goal({False: [Condition('partner employment', 'has_job', partner, True, '==')]})
strategy = StrategyManager(graph_manager=graph)
request = PlanRequest(character.uuid, goal)
before = strategy.request_fingerprint(request)
graph.G.nodes[partner]['job'] = 'Baker'
assert before != strategy.request_fingerprint(request)
assert goal.check_completion({'energy': 20}) is True
room = Location(character.name, 0, 0, 1, 1, ActionSystem())
graph.add_location_node(room)
assert graph.resolve_node(character.name, 'character') is character
try: graph.resolve_node(character.name)
except ValueError: pass
else: raise AssertionError('Ambiguous name accepted')
# A missing graph job cannot fall back to the partner object.
del graph.G.nodes[partner]['job']
try: goal.check_completion()
except ValueError: pass
else: raise AssertionError('Missing graph job fell back to object')
''')

    def test_missing_authoritative_job_never_uses_object_fallback(self):
        self.world('''
from tiny_strategy_manager import StrategyManager, PlanRequest
from tiny_utility_functions import GoalEvaluationError
strategy = StrategyManager(graph_manager=graph)
# Both references agree before graph presence changes.
character.job = graph.G.nodes[character]['job'] = 'Baker'
goal = make_goal({False: [Condition('employment', 'has_job', character, True, '==')]})
request = PlanRequest(character.uuid, goal)
assert graph.is_goal_achieved(character, goal) is True
before = strategy.request_fingerprint(request)
del graph.G.nodes[character]['job']
assert before != strategy.request_fingerprint(request), 'Actor fingerprint retained stale object job'
for check in (goal.check_completion, lambda: graph.is_goal_achieved(character, goal)):
    try: check()
    except GoalEvaluationError: pass
    else: raise AssertionError('Achievement used stale object job')
assert strategy.plan_request(request).outcome == 'ERROR'
missing = strategy.request_fingerprint(request)
graph.G.nodes[character]['job'] = None
assert strategy.request_fingerprint(request) != missing
assert graph.is_goal_achieved(character, goal) is goal.check_completion() is False
# Other-entity dependencies preserve absent versus present None, too.
from tiny_characters import PersonalityTraits
partner = Character('Presence821', 31, graph_manager=graph, action_system=character.action_system,
    gametime_manager=character.gametime_manager, memory_mode='basic', location=character.location,
    personality_traits=PersonalityTraits(4, 4, 4, 4, 4), energy=20)
graph.G.nodes[character]['job'] = 'Baker'
partner_goal = make_goal({False: [Condition('partner employment', 'has_job', partner, True, '==')]})
partner_request = PlanRequest(character.uuid, partner_goal)
for job in ('Baker', None):
    partner.job = graph.G.nodes[partner]['job'] = job
    present = strategy.request_fingerprint(partner_request)
    del graph.G.nodes[partner]['job']
    assert present != strategy.request_fingerprint(partner_request), 'Partner fingerprint lost job presence'
    try: partner_goal.check_completion()
    except GoalEvaluationError: pass
    else: raise AssertionError('Partner completion used object fallback')
''')

    def test_duplicate_cross_type_uuid_keeps_actor_requests_valid(self):
        self.world('''
from tiny_strategy_manager import StrategyManager, PlanRequest
from tiny_locations import Location
from unittest.mock import patch
# Inject only generated identity; use real entity constructors and registrations.
with patch('tiny_locations.uuid.uuid4', return_value=character.uuid):
    room = Location('SharedUuid821', 0, 0, 1, 1, character.action_system)
try: graph.add_location_node(room)
except ValueError: pass
else: raise AssertionError('Duplicate UUID registration accepted')
strategy = StrategyManager(graph_manager=graph)
actor_goal = make_goal({False: [Condition('actor energy', 'energy', None, 30)]})
actor_request = PlanRequest(character.uuid, actor_goal)
assert strategy.goal_key(actor_request)
assert strategy.request_fingerprint(actor_request)
''')
