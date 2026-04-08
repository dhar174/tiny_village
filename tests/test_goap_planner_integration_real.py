import pytest

from actions import Action, State
from tiny_goap_system import ActionWrapper, GOAPPlanner, Plan
from tiny_strategy_manager import StrategyManager
from tiny_utility_functions import Goal


class HomeLocation:
    def __init__(self, name="Home"):
        self.name = name


class JobStub:
    def __init__(self, job_title="Programmer"):
        self.job_title = job_title


class CharacterStub:
    def __init__(
        self,
        name="Alice",
        *,
        energy=20,
        hunger=60,
        wealth_money=10,
        social_wellbeing=5,
        mental_health=5,
        location=None,
        job=None,
    ):
        self.name = name
        self.id = name
        self.energy = energy
        self.hunger_level = hunger
        self.wealth_money = wealth_money
        self.social_wellbeing = social_wellbeing
        self.mental_health = mental_health
        self.location = location or HomeLocation()
        self.job = job or JobStub()
        self.inventory = type("InventoryStub", (), {"get_food_items": lambda self: []})()

    def get_state(self):
        return State(
            {
                "energy": self.energy,
                "hunger": self.hunger_level,
                "money": self.wealth_money,
                "social_wellbeing": self.social_wellbeing,
                "mental_health": self.mental_health,
            }
        )


class GraphManagerStub:
    def __init__(self, state_by_name, actions_by_name):
        self._state_by_name = state_by_name
        self._actions_by_name = actions_by_name

    def get_character_state(self, name):
        return self._state_by_name[name]

    def get_possible_actions(self, name):
        return self._actions_by_name[name]


class EnergyGoal:
    def __init__(self, actor, target_energy):
        self.actor = actor
        self.target_energy = target_energy
        self.name = "restore_energy"

    def check_completion(self):
        return self.actor.energy >= self.target_energy


class EnergyAction(ActionWrapper):
    def __init__(self, actor, *, name, required_energy, energy_gain, target):
        super().__init__(
            name=name,
            cost=required_energy,
            effects=[{"attribute": "energy", "change_value": energy_gain}],
            preconditions={},
        )
        self.actor = actor
        self.required_energy = required_energy
        self.target = target
        self.initiator = actor
        self.calls = []

    def preconditions_met(self):
        return self.actor.energy >= self.required_energy

    def execute(self, target=None, initiator=None):
        self.calls.append((target, initiator))
        actor = initiator or self.actor
        if actor.energy < self.required_energy:
            return False
        energy_gain = next((effect.get("change_value", 0) for effect in self.effects if effect.get("attribute") == "energy"), 0)
        actor.energy = actor.energy - self.required_energy + energy_gain
        return True


def apply_plan(planner, plan, initial_state):
    state = initial_state
    for action in plan:
        state = planner._apply_action_effects(action, state)
    return state


def test_plan_actions_uses_graph_manager_state_and_actions_to_reach_goal():
    character = CharacterStub(name="GraphAlice", energy=20)
    graph_actions = {
        character.name: [
            Action(
                name="Rest",
                preconditions=[],
                effects=[{"attribute": "energy", "change_value": 10}],
                cost=1,
            ),
            Action(
                name="Sleep",
                preconditions=[],
                effects=[{"attribute": "energy", "change_value": 30}],
                cost=2,
            ),
        ]
    }
    planner = GOAPPlanner(
        GraphManagerStub(
            {character.name: {"energy": 20}},
            graph_actions,
        )
    )
    goal = Goal(name="restore_energy", target_effects={"energy": 50}, priority=1.0)

    plan = planner.plan_actions(character, goal)

    assert plan is not None
    assert [action.name for action in plan]
    final_state = apply_plan(planner, plan, planner.get_current_world_state(character))
    assert planner._goal_satisfied(goal, final_state)


def test_plan_for_character_without_graph_manager_uses_fallback_actions_meaningfully():
    character = CharacterStub(name="FallbackBob", energy=20, job=None)
    planner = GOAPPlanner(None)
    goal = Goal(name="recover", target_effects={"energy": 30}, priority=1.0)

    fallback_actions = planner.get_available_actions(character)
    plan = planner.plan_for_character(character, goal)

    assert [action.name for action in fallback_actions] == ["Rest", "Idle"]
    assert plan is not None
    final_state = apply_plan(planner, plan, planner.get_current_world_state(character))
    assert planner._goal_satisfied(goal, final_state)


def test_strategy_manager_generated_energy_plan_actually_raises_energy_to_goal():
    character = CharacterStub(name="Strategist", energy=2, hunger=8, wealth_money=10)
    manager = StrategyManager(use_llm=False)
    goal = Goal(name="restore_energy", target_effects={"energy": 0.6}, priority=0.9)
    current_state = State(manager.get_character_state_dict(character))
    actions = manager.get_daily_actions(character)

    plan = manager.goap_planner.plan_actions(character, goal, current_state, actions)

    assert plan is not None
    final_state = apply_plan(manager.goap_planner, plan, current_state)
    assert manager.goap_planner._goal_satisfied(goal, final_state)

    for action in plan:
        if "Sleep" in action.name:
            assert any(
                effect.get("attribute") == "energy" and effect.get("change_value", 0) > 0
                for effect in action.effects
            )


def test_plan_execute_passes_real_target_and_initiator_to_successful_action():
    actor = type("Actor", (), {"energy": 10})()
    goal = EnergyGoal(actor, target_energy=15)
    action = EnergyAction(
        actor,
        name="Nap",
        required_energy=2,
        energy_gain=8,
        target="bed",
    )
    plan = Plan("recover")
    plan.add_goal(goal)
    plan.add_action(action, priority=1)

    assert plan.execute() is True
    assert action.calls == [("bed", actor)]
    assert actor.energy == pytest.approx(16)


def test_plan_execute_fails_when_real_preconditions_are_not_met():
    actor = type("Actor", (), {"energy": 1})()
    goal = EnergyGoal(actor, target_energy=15)
    action = EnergyAction(
        actor,
        name="Nap",
        required_energy=2,
        energy_gain=8,
        target="bed",
    )
    plan = Plan("recover")
    plan.add_goal(goal)
    plan.add_action(action, priority=1)

    assert plan.execute() is False
    assert action.calls == []
    assert actor.energy == pytest.approx(1)
