#!/usr/bin/env python3

from dataclasses import dataclass
from math import isfinite
from types import SimpleNamespace
from unittest.mock import patch

from actions import Action, State
from goap_evaluator import GoapEvaluator, WorldState


@dataclass
class StubTraits:
    openness: float
    extraversion: float
    conscientiousness: float
    agreeableness: float
    neuroticism: float

    def get_openness(self):
        return self.openness

    def get_extraversion(self):
        return self.extraversion

    def get_conscientiousness(self):
        return self.conscientiousness

    def get_agreeableness(self):
        return self.agreeableness

    def get_neuroticism(self):
        return self.neuroticism


class StubCharacter:
    pass


class StubGoal:
    pass


@dataclass
class StubMotive:
    name: str
    description: str
    score: float


class StubPersonalMotives:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class StubCharacterImpl(StubCharacter):
    def __init__(self, name, energy=40, mental_health=8, traits=None):
        self.name = name
        self.energy = energy
        self.mental_health = mental_health
        self.personality_traits = traits or StubTraits(
            openness=5.0,
            extraversion=6.0,
            conscientiousness=7.0,
            agreeableness=5.5,
            neuroticism=3.0,
        )
        self.state_data = {
            "energy": energy,
            "hunger": 20,
            "happiness": 60,
        }

    def get_mental_health(self):
        return self.mental_health

    def get_state(self):
        return State(self.state_data.copy())


@dataclass(unsafe_hash=True)
class StubCondition:
    attribute: str
    operator: str
    satisfy_value: float
    target: StubCharacterImpl
    weight: float = 1.0

    def check_condition(self, state):
        current_value = state.get(self.attribute, 0)
        if self.operator == "ge":
            return current_value >= self.satisfy_value
        if self.operator == "gt":
            return current_value > self.satisfy_value
        if self.operator == "le":
            return current_value <= self.satisfy_value
        if self.operator == "lt":
            return current_value < self.satisfy_value
        return current_value == self.satisfy_value


class StableConditionsDict(dict):
    """Test helper alias for dict; uses standard dict iteration semantics."""
    pass


class StubGoalImpl(StubGoal):
    def __init__(self, target, completion_conditions):
        self.name = "restore_energy"
        self.target = target
        self.completion_conditions = completion_conditions
        self.criteria = {"energy": True}


def install_tiny_characters_stub():
    module = SimpleNamespace(
        Character=StubCharacter,
        Goal=StubGoal,
        Motive=StubMotive,
        PersonalMotives=StubPersonalMotives,
    )
    return patch.dict("sys.modules", {"tiny_characters": module})


def deterministic_gauss(mean, _std_dev):
    return mean


def test_calculate_motives_reflects_character_traits():
    evaluator = GoapEvaluator()
    outgoing = StubCharacterImpl(
        "Outgoing",
        traits=StubTraits(
            openness=6.0,
            extraversion=8.0,
            conscientiousness=6.0,
            agreeableness=6.0,
            neuroticism=2.0,
        ),
    )
    reserved = StubCharacterImpl(
        "Reserved",
        traits=StubTraits(
            openness=4.0,
            extraversion=2.0,
            conscientiousness=6.0,
            agreeableness=4.0,
            neuroticism=4.0,
        ),
    )

    with install_tiny_characters_stub(), patch(
        "goap_evaluator.random.gauss", side_effect=deterministic_gauss
    ):
        outgoing_motives = evaluator.calculate_motives(outgoing, WorldState())
        reserved_motives = evaluator.calculate_motives(reserved, WorldState())

    assert isinstance(outgoing_motives, StubPersonalMotives)
    assert isinstance(outgoing_motives.social_wellbeing_motive, StubMotive)
    assert outgoing_motives.social_wellbeing_motive.score > (
        reserved_motives.social_wellbeing_motive.score
    )
    assert outgoing_motives.hunger_motive.score >= 0
    assert reserved_motives.family_motive.score >= 0


def test_calculate_how_goal_impacts_character_uses_stateful_conditions():
    evaluator = GoapEvaluator()
    tired_character = StubCharacterImpl("Tired", energy=40)
    energized_character = StubCharacterImpl("Energized", energy=80)
    tired_goal = StubGoalImpl(
        tired_character,
        [StubCondition("energy", "ge", 60, tired_character)],
    )
    energized_goal = StubGoalImpl(
        energized_character,
        [StubCondition("energy", "ge", 60, energized_character)],
    )

    with install_tiny_characters_stub():
        low_impact = evaluator.calculate_how_goal_impacts_character(
            tired_goal, tired_character, WorldState()
        )
        high_impact = evaluator.calculate_how_goal_impacts_character(
            energized_goal, energized_character, WorldState()
        )

    assert low_impact == 0
    assert high_impact == 20


def test_calculate_action_effect_cost_penalizes_harmful_real_actions():
    evaluator = GoapEvaluator()
    character = StubCharacterImpl("Alice", energy=40)
    goal = StubGoalImpl(character, [StubCondition("energy", "ge", 60, character)])
    helpful_action = Action(
        name="Rest",
        preconditions=[],
        effects=[{"targets": "initiator", "attribute": "energy", "change_value": 25}],
        cost=1.0,
    )
    harmful_action = Action(
        name="Overwork",
        preconditions=[],
        effects=[{"targets": "initiator", "attribute": "energy", "change_value": -10}],
        cost=1.0,
    )

    with install_tiny_characters_stub():
        helpful_cost = evaluator.calculate_action_effect_cost(
            helpful_action, character, goal, WorldState()
        )
        harmful_cost = evaluator.calculate_action_effect_cost(
            harmful_action, character, goal, WorldState()
        )

    assert isfinite(helpful_cost)
    assert isfinite(harmful_cost)
    assert 0 <= helpful_cost < harmful_cost <= 1


def test_evaluate_action_plan_tracks_viability_and_goal_progress():
    evaluator = GoapEvaluator()
    character = StubCharacterImpl("Alice", energy=40)
    completion_condition = StubCondition("energy", "ge", 60, character)
    goal = StubGoalImpl(
        character,
        StableConditionsDict({False: [completion_condition]}),
    )
    action = Action(
        name="Rest",
        preconditions=[],
        effects=[{"targets": "initiator", "attribute": "energy", "change_value": 25}],
        cost=1.0,
    )

    with install_tiny_characters_stub():
        evaluation = evaluator.evaluate_action_plan(
            [action], character, goal, WorldState()
        )

    assert evaluation["cost"] == 1.5
    assert evaluation["viability"] == 1.0
    assert evaluation["success_probability"] == 1.0
    assert evaluation["conditions_satisfied"] == 1
    assert evaluation["total_conditions"] == 1


def test_goap_evaluator_instances_remain_stateless():
    first_evaluator = GoapEvaluator()
    second_evaluator = GoapEvaluator()
    character = StubCharacterImpl("Alice", energy=40)
    goal = StubGoalImpl(character, [StubCondition("energy", "ge", 60, character)])
    action = Action(
        name="Rest",
        preconditions=[],
        effects=[{"targets": "initiator", "attribute": "energy", "change_value": 25}],
        cost=1.0,
    )

    with install_tiny_characters_stub():
        first_cost = first_evaluator.calculate_action_effect_cost(
            action, character, goal, WorldState()
        )
        second_cost = second_evaluator.calculate_action_effect_cost(
            action, character, goal, WorldState()
        )

    assert first_cost == second_cost
    assert first_evaluator.dp_cache == second_evaluator.dp_cache == {}
