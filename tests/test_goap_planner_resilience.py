import time
import unittest

from actions import State
from tiny_goap_system import ActionWrapper, GOAPPlanner
from tiny_strategy_manager import StrategyManager


class CharacterStub:
    def __init__(
        self,
        name="PlannerAlice",
        *,
        energy=0.2,
        hunger=0.4,
        wealth_money=10.0,
        social_wellbeing=0.4,
        mental_health=0.6,
        motives=None,
        personality=None,
    ):
        self.name = name
        self.id = name
        self.energy = energy * 10
        self.hunger_level = hunger * 10
        self.wealth_money = wealth_money
        self.social_wellbeing = social_wellbeing * 10
        self.mental_health = mental_health * 10
        self.motives = motives or {"energy_motive": 4.0, "social_motive": 1.0}
        self.personality = personality or {"conscientiousness": 2.0, "extroversion": 1.0}
        self._memory_notes = []

    def get_state(self):
        return State(
            {
                "energy": self.energy / 10.0,
                "hunger": self.hunger_level / 10.0,
                "money": self.wealth_money,
                "social_wellbeing": self.social_wellbeing / 10.0,
                "mental_health": self.mental_health / 10.0,
            }
        )

    def add_memory(self, note):
        self._memory_notes.append(note)


class GoalStub:
    def __init__(self, name, target_effects, priority=1.0):
        self.name = name
        self.target_effects = target_effects
        self.priority = priority

    def check_completion(self, state):
        return all(state.get(key, 0) >= value - 0.1 for key, value in self.target_effects.items())


class GraphManagerStub:
    def __init__(self, actions=None, relationships=None):
        self._actions = actions or []
        self._relationships = relationships or {}

    def get_possible_actions(self, _character_id):
        return self._actions

    def analyze_character_relationships(self, _character):
        return self._relationships

    def evaluate_relationship_strength(self, _character, relation_name):
        relation = self._relationships.get(relation_name, {})
        if isinstance(relation, dict):
            return relation.get("strength", 0.0)
        return getattr(relation, "strength", 0.0)


def build_warmup_actions():
    return [
        ActionWrapper(
            name="Gather Wood",
            cost=0.1,
            preconditions={"energy": 0.2},
            effects=[
                {"attribute": "firewood", "change_value": 1},
                {"attribute": "energy", "change_value": -0.05},
            ],
        ),
        ActionWrapper(
            name="Light Fire",
            cost=0.1,
            preconditions={"firewood": 1},
            effects=[{"attribute": "campfire", "change_value": 1}],
        ),
        ActionWrapper(
            name="Warm Up",
            cost=0.1,
            preconditions={"campfire": 1},
            effects=[{"attribute": "energy", "change_value": 0.45}],
        ),
    ]


class TestGOAPPlannerResilience(unittest.TestCase):
    def test_plan_actions_builds_multi_step_plan(self):
        planner = GOAPPlanner(None)
        character = CharacterStub()
        goal = GoalStub("restore_energy", {"energy": 0.6}, priority=0.9)
        current_state = character.get_state()

        plan = planner.plan_actions(character, goal, current_state, build_warmup_actions())

        self.assertEqual([action.name for action in plan], ["Gather Wood", "Light Fire", "Warm Up"])
        self.assertTrue(planner.validate_plan(plan, current_state))

    def test_validate_plan_detects_invalid_preconditions(self):
        planner = GOAPPlanner(None)
        invalid_plan = [
            ActionWrapper(
                name="Warm Up",
                cost=0.1,
                preconditions={"campfire": 1},
                effects=[{"attribute": "energy", "change_value": 0.45}],
            )
        ]

        self.assertFalse(planner.validate_plan(invalid_plan, State({"energy": 0.2, "campfire": 0})))

    def test_plan_cache_reuses_plan_and_invalidates_on_world_change(self):
        planner = GOAPPlanner(None)
        character = CharacterStub()
        goal = GoalStub("restore_energy", {"energy": 0.6}, priority=0.9)
        actions = build_warmup_actions()
        current_state = State({"energy": 0.2, "firewood": 0, "campfire": 0})

        first_plan = planner.plan_actions(character, goal, current_state, actions)
        second_plan = planner.plan_actions(character, goal, current_state, actions)

        self.assertEqual(planner.cache_hits, 1)
        self.assertEqual([action.name for action in first_plan], [action.name for action in second_plan])

        changed_state = State({"energy": 0.2, "firewood": 1, "campfire": 0})
        changed_plan = planner.plan_actions(character, goal, changed_state, actions)

        self.assertEqual(len(planner.plan_cache), 1)
        self.assertEqual([action.name for action in changed_plan], ["Light Fire", "Warm Up"])

    def test_evaluate_goal_importance_uses_needs_relationships_and_memory_signals(self):
        actions = [{"name": "Recharge", "effects": [{"attribute": "energy", "change_value": 0.3}], "cost": 1}]
        planner = GOAPPlanner(GraphManagerStub(actions=actions, relationships={"Bob": {"strength": 4.0}}))
        low_energy_character = CharacterStub(name="LowEnergy", energy=0.1)
        high_energy_character = CharacterStub(name="HighEnergy", energy=0.8)
        energy_goal = GoalStub("restore_energy", {"energy": 0.7}, priority=0.7)

        low_score = planner.evaluate_goal_importance(
            low_energy_character,
            energy_goal,
            planner.graph_manager,
            recent_successes=1,
        )
        high_score = planner.evaluate_goal_importance(
            high_energy_character,
            energy_goal,
            planner.graph_manager,
            recent_failures=2,
        )

        self.assertGreater(low_score, high_score)

    def test_replan_after_failure_avoids_failed_action(self):
        planner = GOAPPlanner(None)
        character = CharacterStub()
        goal = GoalStub("restore_energy", {"energy": 0.6}, priority=0.9)
        current_state = State({"energy": 0.2})
        risky_action = ActionWrapper(
            name="Risky Nap",
            cost=0.1,
            preconditions={},
            effects=[{"attribute": "energy", "change_value": 0.4}],
            utility=10.0,
        )
        safe_action = ActionWrapper(
            name="Safe Nap",
            cost=0.3,
            preconditions={},
            effects=[{"attribute": "energy", "change_value": 0.4}],
            utility=5.0,
        )

        initial_plan = planner.plan_actions(character, goal, current_state, [risky_action, safe_action])
        replanned = planner.replan_after_failure(
            character,
            goal,
            current_state=current_state,
            actions=[risky_action, safe_action],
            failed_action=risky_action,
        )

        self.assertEqual([action.name for action in initial_plan], ["Risky Nap"])
        self.assertEqual([action.name for action in replanned], ["Safe Nap"])

    def test_strategy_manager_replanning_flow_uses_goap_planner(self):
        planner = GOAPPlanner(None)
        manager = StrategyManager.__new__(StrategyManager)
        manager.goap_planner = planner
        character = CharacterStub()
        goal = GoalStub("restore_energy", {"energy": 0.6}, priority=0.9)
        current_state = State({"energy": 0.2})
        risky_action = ActionWrapper(
            name="Risky Nap",
            cost=0.1,
            preconditions={},
            effects=[{"attribute": "energy", "change_value": 0.4}],
            utility=10.0,
        )
        safe_action = ActionWrapper(
            name="Safe Nap",
            cost=0.3,
            preconditions={},
            effects=[{"attribute": "energy", "change_value": 0.4}],
            utility=5.0,
        )

        replanned = manager.handle_plan_execution_failure(
            character,
            goal,
            failed_action=risky_action,
            actions=[risky_action, safe_action],
            current_state=current_state,
        )

        self.assertEqual([action.name for action in replanned], ["Safe Nap"])

    def test_planner_runtime_stays_within_demo_constraints(self):
        planner = GOAPPlanner(None)
        character = CharacterStub()
        goal = GoalStub("restore_energy", {"energy": 0.6}, priority=0.9)
        actions = build_warmup_actions() + [
            ActionWrapper(
                name=f"Noise {index}",
                cost=0.5,
                preconditions={"energy": 0.0},
                effects=[{"attribute": f"noise_{index}", "change_value": 1}],
            )
            for index in range(20)
        ]

        start = time.perf_counter()
        plan = planner.plan_actions(character, goal, State({"energy": 0.2}), actions)
        duration = time.perf_counter() - start

        self.assertIsNotNone(plan)
        self.assertLess(duration, 1.0)


if __name__ == "__main__":
    unittest.main()
