"""Closure checks for #740 using real graph-backed runtime objects."""

from tests import test_goap_graph_contracts
from tests.test_graph_initialization import FreshPythonTestCase


class TestRuntimeUnits(FreshPythonTestCase):
    run_world = test_goap_graph_contracts.TestSupportedPlannerPrimitives.run_world

    def test_native_dictionary_and_object_snapshots_agree(self):
        self.run_world('''
from tiny_strategy_manager import StrategyManager
manager = StrategyManager(graph_manager=graph)
node = graph.G.nodes[character]
node.update(energy=77, hunger_level=60, wealth_money=100)
snapshot = manager.get_character_state_dict(character)
dictionary = manager.get_character_state_dict(dict(node))
for key in ('energy', 'hunger', 'money'):
    assert snapshot[key] == dictionary[key], (key, snapshot, dictionary)
assert dictionary['energy'] == .77 and dictionary['hunger'] == .6
assert dictionary['money'] == 100
print({'dictionary_energy': dictionary['energy'], 'money': dictionary['money']})
''')

    def test_real_goap_utility_uses_native_state_boundary(self):
        self.run_world('''
from tiny_strategy_manager import StrategyManager
from tiny_goap_system import GOAPPlanner
graph.G.nodes[character]['energy'] = 20
character.energy = 90  # the graph remains authoritative
manager = StrategyManager(graph_manager=graph)
rest = next(a for a in manager.get_daily_actions(character) if a.name == 'Rest')
idle = Action('Idle', [], [], graph_manager=graph)
planner = GOAPPlanner(graph)
score = planner.calculate_utility(rest, character)
assert math.isclose(score, .8), score
assert planner.evaluate_utility([idle, rest], character) is rest
print({'real_planner_rest_utility': score})
''')

    def test_interpreter_and_building_satisfaction_share_canonical_field(self):
        self.run_world('''
from tiny_output_interpreter import OutputInterpreter
from tiny_strategy_manager import StrategyManager
node = graph.G.nodes[character]
node.update(current_satisfaction=0, energy=20)
character.current_satisfaction = 0
manager = StrategyManager(graph_manager=graph)
generic = OutputInterpreter(graph)._create_generic_action('Learn')(
    initiator_id=character, graph_manager=graph)
assert generic.execute(character=character)
assert node['current_satisfaction'] == character.current_satisfaction == 1
rest = next(a for a in manager.get_daily_actions(character) if a.name == 'Rest')
assert rest.execute(character=character)
building = Action('BuildingService', [], [
    {'targets': ['initiator'], 'attribute': 'current_satisfaction', 'change_value': 5}],
    graph_manager=graph)
assert building.execute(character=character)
assert node['current_satisfaction'] == character.current_satisfaction == 14
assert node['satisfaction'] == character.satisfaction == 14
assert manager.get_character_state_dict(character)['satisfaction'] == .14
print({'satisfaction': [0, 1, 9, 14], 'snapshot': .14})
''')

    def test_catalog_happiness_is_fractional_and_bounded(self):
        self.run_world('''
from actions import PursueHobbyAction, _AttributeEffect
node = graph.G.nodes[character]
node['happiness'] = character.happiness = .5
hobby = PursueHobbyAction(initiator_id=character, graph_manager=graph)
assert hobby.execute(character=character)
assert math.isclose(node['happiness'], .7), node['happiness']
assert node['happiness'] == character.happiness
for field in ('energy', 'hunger_level', 'health_status', 'mental_health',
              'social_wellbeing', 'job_performance', 'current_satisfaction'):
    assert _AttributeEffect({'attribute': field, 'change_value': 200}).scalar_result(95) == 100
assert _AttributeEffect({'attribute': 'happiness', 'change_value': .2}).scalar_result(.95) == 1
print({'hobby_happiness': character.happiness, 'need_cap': 100, 'fraction_cap': 1})
''')

    def test_canonical_need_effects_and_goals_share_utility_keys(self):
        self.run_world('''
from tiny_utility_functions import calculate_native_action_utility, calculate_native_plan_utility
from tiny_utility_functions import Goal, is_goal_achieved
native = {'hunger_level': 80, 'energy': 20, 'current_satisfaction': 20, 'wealth_money': 100}
before = native.copy()
eat = Action('Eat', [], [{'attribute': 'hunger_level', 'change_value': -20}],
             cost=.1, graph_manager=graph)
score = calculate_native_action_utility(native, eat)
assert math.isclose(score, 2.2), score
plan_score = calculate_native_plan_utility(native, [eat, eat])
assert math.isclose(plan_score, 3.6), plan_score
empty = Action('NoEffects', [], None, graph_manager=graph)
assert calculate_native_plan_utility(native, [empty]) == 0
assert native == before
goal = Goal(name='satisfied', target_effects={'satisfaction': 20, 'money': 100}, priority=1)
assert is_goal_achieved(goal, State(native))
assert GOAPPlanner(graph)._goal_satisfied(goal, State(native))
print({'eat_utility': score, 'two_meal_utility': plan_score, 'state_preserved': True})
''')

    def test_prompt_examples_compare_normalized_states(self):
        self.run_world('''
from tiny_prompt_builder import PromptBuilder
from tiny_utility_functions import build_utility_action_view
builder = PromptBuilder(character, graph_manager=graph)
node = graph.G.nodes[character]
node.update(current_satisfaction=80, satisfaction=10, wealth_money=100)
character.current_satisfaction = 20
native = builder._get_native_effect_state()
assert native['satisfaction'] == native['current_satisfaction'] == 80
cut = Action('Set', [], [{'attribute': 'satisfaction', 'change_value': 'set:25'}], graph_manager=graph)
view = build_utility_action_view(cut, native)
assert view.effects[0]['attribute'] == 'satisfaction'
assert view.effects[0]['change_value'] == -.55
examples = builder.few_shot_manager.get_relevant_examples(
    {'energy': .7, 'social_wellbeing': .3, 'mental_health': .5}, max_examples=1)
assert examples[0].decision_made == 'social_visit', examples
print({'prompt_satisfaction_delta': -.55, 'matching_example': 'social_visit'})
''')

    def test_event_need_writes_use_shared_native_bounds(self):
        self.run_world('''
from effect_dispatcher import EffectDispatcher
from effect_schema import OperatorType
dispatcher = EffectDispatcher(graph)
node = graph.G.nodes[character]
for field in ('mental_health', 'social_wellbeing', 'job_performance', 'current_satisfaction'):
    node[field] = 70
    setattr(character, field, 10)
    assert dispatcher._modify_entity_attribute(character, field, 10, OperatorType.ADD)
    assert node[field] == getattr(character, field) == 80
    assert dispatcher._modify_entity_attribute(character, field, 200, OperatorType.ADD)
    assert node[field] == getattr(character, field) == 100
print({'event_need_trace': [70, 80, 100]})
''')


class TestVisualRuntimeUnits(FreshPythonTestCase):
    run_world = test_goap_graph_contracts.TestSupportedPlannerPrimitives.run_world

    def test_panel_displays_authoritative_hunger(self):
        self.run_world('''
import pygame
from types import SimpleNamespace
from tiny_gameplay_controller import SelectedCharacterPanel
controller.headless = False
controller._initialize_rendering()
pygame.font.init()
node = graph.G.nodes[character]
node.update(energy=77, hunger_level=60, health_status=90)
character.energy = 5
character.hunger_level = 2
controller.map_controller = SimpleNamespace(selected_character=character)
labels = []
class RecordingFont:
    def __init__(self):
        self.font = pygame.font.Font(None, 18)
    def render(self, text, *args):
        labels.append(text)
        return self.font.render(text, *args)
font = RecordingFont()
panel = SelectedCharacterPanel('Needs', position=(0, 0))
panel._render_content(pygame.Surface((500, 700)), controller,
                      {'small': font, 'tiny': font})
assert '  Energy: 77/100' in labels, labels
assert '  Hunger: 60/100' in labels, labels
assert '  Health: 90/100' in labels, labels
print({'panel_labels': [s for s in labels if '/100' in s]})
''')
