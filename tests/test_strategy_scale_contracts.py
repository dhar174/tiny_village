"""#740 native-scale contracts for strategy, GOAP parity, and fixture conversion.

Maintained core regressions using real Characters, Actions, GraphManager, and the
controller world harness (FreshPythonTestCase.run_world). Exact values are asserted
in both stores (Character attributes AND graph node) per the #740 plan.
"""

import textwrap
import unittest

from tests.test_graph_initialization import FreshPythonTestCase
from tests.test_goap_graph_contracts import WORLD_SETUP


class TestStrategyScaleContracts(FreshPythonTestCase):
    def run_world(self, source):
        self.run_python(
            WORLD_SETUP
            + "\ntry:\n    character = next(iter(controller.characters.values()))\n"
            + textwrap.indent(textwrap.dedent(source), "    ")
            + "\nfinally:\n    controller.close()\n"
        )

    def test_sleep_offered_below_native_30_energy_at_home(self):
        self.run_world("""
from tiny_strategy_manager import StrategyManager
strategy = StrategyManager(graph_manager=graph)
class _Home:
    name = "Home"
character.location = _Home()
for energy_value, offered in ((29, True), (30, False)):
    graph.G.nodes[character].update(energy=energy_value)
    character.energy = energy_value
    names = [a.name for a in strategy.get_daily_actions(character)]
    assert ("Sleep" in names) == offered, (energy_value, names)
print({"sleep_gate": [29, 30]})
""")

    def test_work_requires_native_30_energy_and_raw_credits(self):
        self.run_world("""
from unittest.mock import patch
from tiny_strategy_manager import StrategyManager
strategy = StrategyManager(graph_manager=graph)
node = graph.G.nodes[character]
class _Home:
    name = "Home"
character.location = _Home()
character.job = "Baker"
node.update(energy=29)
character.energy = 29
# #818 review: blocked Work is excluded from daily selection entirely
names_low = [a.name for a in strategy.get_daily_actions(character)]
assert not any(n.startswith("Work as") for n in names_low), names_low
node.update(energy=30)
character.energy = 30
work = next(a for a in strategy.get_daily_actions(character) if a.name.startswith("Work as"))
assert [(e.get("attribute"), e.get("change_value")) for e in work.effects if isinstance(e, dict)] == [
    ("money", 20.0), ("energy", -30)]
start_credits = character.wealth_money
with patch.object(graph, "update_node_attribute", wraps=graph.update_node_attribute) as writes:
    assert work.execute(str(character.uuid))
assert node["wealth_money"] == character.wealth_money == start_credits + 20
assert node["energy"] == character.energy == 0
print({"work_gate": [29, 30], "energy": [30, 0]})
""")

    def test_native_snapshot_parity(self):
        self.run_world("""
from tiny_strategy_manager import StrategyManager
strategy = StrategyManager(graph_manager=graph)
node = graph.G.nodes[character]
node.update(energy=70, hunger_level=60, health_status=100, job_performance=50, wealth_money=88)
for field in ("energy", "hunger_level", "health_status", "job_performance", "wealth_money"):
    setattr(character, field, node[field])
snapshot = strategy.get_character_state_dict(character)
assert snapshot["energy"] == 0.7
assert snapshot["hunger"] == 0.6
assert snapshot["health"] == 1.0
assert snapshot["job_performance"] == 0.5
assert snapshot["money"] == 88
assert 0.0 <= snapshot["happiness"] <= 1.0
node.update(energy=100)
character.energy = 100
snapshot = strategy.get_character_state_dict(character)
assert snapshot["energy"] == 1.0
print({"snapshot_energy": [0.7, 1.0]})
""")

    def test_native_low_value_inputs_stay_native(self):
        self.run_world("""
from tiny_strategy_manager import StrategyManager
strategy = StrategyManager(graph_manager=graph)
node = graph.G.nodes[character]
node.update(energy=5, hunger_level=95)
character.energy = 5
character.hunger_level = 95
class _Home:
    name = "Home"
character.location = _Home()
names = [a.name for a in strategy.get_daily_actions(character)]
assert "Sleep" in names
snapshot = strategy.get_character_state_dict(character)
assert snapshot["energy"] == 0.05
assert snapshot["hunger"] == 0.95
print({"low_value": [0.05, 0.95]})
""")

    def test_hunger_targets_evaluate_as_maximums(self):
        self.run_world("""
from actions import State
from tiny_goap_system import GOAPPlanner
from tiny_utility_functions import Goal
planner = GOAPPlanner(graph)
goal = Goal(name="reduce_hunger", target_effects={"hunger_level": -60}, priority=1.0)
assert planner._goal_satisfied(goal, State({"hunger_level": 55}))
assert not planner._goal_satisfied(goal, State({"hunger_level": 65}))
assert not planner._goal_satisfied(goal, State({"hunger_level": 90}))
positive = Goal(name="lift_energy", target_effects={"energy": 65}, priority=1.0)
assert not planner._goal_satisfied(positive, State({"energy": 55}))
assert planner._goal_satisfied(positive, State({"energy": 65}))
print({"hunger_max_direction": True})
""")

    def test_second_action_eligible_only_in_simulated_successor(self):
        self.run_world("""
from actions import State
from tiny_goap_system import GOAPPlanner
from tiny_utility_functions import Goal
planner = GOAPPlanner(graph)
node = graph.G.nodes[character]
node.update(energy=20, hunger_level=90)
character.energy = 20
character.hunger_level = 90
rest = Action("Rest", [], [{"targets": ["initiator"], "attribute": "energy", "change_value": 30}], graph_manager=graph)
cook = Action("Cook", [Condition("HasEnergy", "energy", None, 45)], [
    {"targets": ["initiator"], "attribute": "hunger", "change_value": -40}], graph_manager=graph)
goal = Goal(name="reduce_hunger", target_effects={"hunger_level": -60}, priority=1.0)
plan = planner.plan_actions(character, goal, State({"energy": 20, "hunger_level": 90}), [rest, cook])
assert plan is not None and len(plan) == 2, plan
names = [a.name for a in plan]
assert names == ["Rest", "Cook"], names
assert cook.preconditions_met(State({"energy": 50})) is True
assert cook.preconditions_met() is False
assert node["energy"] == 20 and node["hunger_level"] == 90
print({"plan_order": names})
""")

    def test_projection_matches_execution_for_add_and_set(self):
        self.run_world("""
from actions import State
from tiny_goap_system import GOAPPlanner
planner = GOAPPlanner(graph)
node = graph.G.nodes[character]
node.update(energy=20)
character.energy = 20
boost = Action("Boost", [], [{"targets": ["initiator"], "attribute": "energy", "change_value": "add:30"}], graph_manager=graph)
simulated = planner._apply_action_effects(boost, State({"energy": 20, "hunger_level": 40}))
assert simulated["energy"] == 50, simulated
setfix = Action("SetFix", [], [{"targets": ["initiator"], "attribute": "energy", "change_value": "set:80"}], graph_manager=graph)
simulated = planner._apply_action_effects(setfix, simulated)
assert simulated["energy"] == 80, simulated
assert boost.execute(str(character.uuid))
assert node["energy"] == character.energy == 50
print({"projection_vs_execution": [50, 80]})
""")


class TestFixtureUnitConversion(unittest.TestCase):
    """Named village-economy-v1 fixture conversion boundary (#740)."""

    def test_fixture_units_roundtrip_and_directions(self):
        from actions import fixture_units_to_native, native_units_to_fixture

        self.assertEqual(fixture_units_to_native(7), 70.0)
        self.assertEqual(fixture_units_to_native(-6), -60.0)
        self.assertEqual(fixture_units_to_native(0), 0.0)
        self.assertEqual(native_units_to_fixture(10), 1.0)
        self.assertEqual(native_units_to_fixture(fixture_units_to_native(7)), 7.0)
        self.assertEqual(fixture_units_to_native(0.5), 5.0)
        self.assertEqual(native_units_to_fixture(fixture_units_to_native(10)), 10.0)

    def test_clamp_keeps_native_low_values_and_caps_high_values(self):
        from actions import _AttributeEffect

        self.assertEqual(_AttributeEffect({"attribute": "energy", "change_value": 200}).scalar_result(77), 100.0)
        self.assertEqual(_AttributeEffect({"attribute": "energy", "change_value": "set:200"}).scalar_result(77), 100.0)
        self.assertEqual(_AttributeEffect({"attribute": "hunger_level", "change_value": -2}).scalar_result(2), 0.0)
        self.assertEqual(_AttributeEffect({"attribute": "health_status", "change_value": 15}).scalar_result(95), 100.0)
        self.assertEqual(_AttributeEffect({"attribute": "wealth_money", "change_value": 200}).scalar_result(50), 250.0)


if __name__ == "__main__":
    unittest.main()


class TestReviewContractRows(FreshPythonTestCase):
    """Regression rows for the PR #818 review fixes (#818 review dispositions)."""

    def run_world(self, source):
        self.run_python(
            WORLD_SETUP
            + "\ntry:\n    character = next(iter(controller.characters.values()))\n"
            + textwrap.indent(textwrap.dedent(source), "    ")
            + "\nfinally:\n    controller.close()\n"
        )

    def test_non_numeric_event_value_rejected_before_synchronized_write(self):
        self.run_world("""
from effect_dispatcher import EffectDispatcher
from effect_schema import OperatorType
dispatcher = EffectDispatcher(graph_manager=graph)
node = graph.G.nodes[character]
node.update(energy=20)
character.energy = 20
from unittest.mock import patch
with patch.object(graph, "update_node_attribute", wraps=graph.update_node_attribute) as writes:
    applied = dispatcher._modify_entity_attribute(character, "energy", "corrupt", OperatorType.SET)
assert applied is False
assert node["energy"] == character.energy == 20
assert writes.call_count == 0
print({"non_numeric_set": "rejected"})
""")

    def test_bound_target_condition_resolves_live_under_supplied_state(self):
        self.run_world("""
from actions import State, Action, Condition
from tiny_time_manager import GameTimeManager
node = graph.G.nodes[character]
node.update(energy=20)
character.energy = 20
other = Character(name="OtherActor", age=30, energy=90,
                  personality_traits={"extraversion": 50, "openness": 80, "conscientiousness": 70, "agreeableness": 40, "neuroticism": 20},
                  gametime_manager=GameTimeManager)
action = Action("MixedTargets", [Condition("PartnerFresh", "energy", other, 10, ">=")], [
    {"targets": ["initiator"], "attribute": "energy", "change_value": 5}], graph_manager=graph)
supplied = State({"energy": 1})
assert action.preconditions_met(supplied) is True
assert action.preconditions_met() is True
tired_partner = Character(name="TiredActor", age=30, energy=5,
                          personality_traits={"extraversion": 50, "openness": 80, "conscientiousness": 70, "agreeableness": 40, "neuroticism": 20},
                          gametime_manager=GameTimeManager)
action_tired = Action("MixedTargetsTired", [Condition("PartnerFresh", "energy", tired_partner, 10, ">=")], [
    {"targets": ["initiator"], "attribute": "energy", "change_value": 5}], graph_manager=graph)
assert action_tired.preconditions_met(supplied) is False
print({"bound_target_live_resolution": True})
""")

    def test_effective_delta_clamps_near_cap(self):
        from actions import Action as _Action
        from tiny_utility_functions import Goal, build_utility_action_view, build_signed_remaining_goal
        Action = _Action

        rest = Action("Rest", [], [{"targets": ["initiator"], "attribute": "energy", "change_value": 15}], graph_manager=None)
        view = build_utility_action_view(rest, native_state={"energy": 95})
        assert view.effects[0]["change_value"] == 0.05
        default_view = build_utility_action_view(rest)
        assert default_view.effects[0]["change_value"] == 0.15
        money_view = build_utility_action_view(
            Action("Work", [], [{"targets": ["initiator"], "attribute": "money", "change_value": 20.0}], graph_manager=None),
            native_state={"wealth_money": 100.0},
        )
        assert money_view.effects[0]["change_value"] == 20.0
        goal_view = build_signed_remaining_goal(
            Goal(name="g", target_effects={"energy": 65}, priority=0.8), {"energy": 0.7}
        )
        assert goal_view.target_effects["energy"] == 0.0
        print({"effective_delta": [0.05, 0.15, 20.0], "satisfied_minimum_clamped": 0.0})

    def test_money_alias_projection_satisfies_money_goals(self):
        self.run_world("""
from actions import State
from tiny_goap_system import GOAPPlanner
planner = GOAPPlanner(graph)
node = graph.G.nodes[character]
node.update(wealth_money=10)
character.wealth_money = 10
work = Action("Work", [], [{"targets": ["initiator"], "attribute": "money", "change_value": 20.0}], graph_manager=graph)
simulated = planner._apply_action_effects(work, State({"wealth_money": 10}))
assert simulated["wealth_money"] == 30
assert simulated["money"] == 30
print({"alias_projection": [30, 30]})
""")

    def test_sleep_gate_uses_authoritative_energy_snapshot(self):
        self.run_world("""
from tiny_strategy_manager import StrategyManager
strategy = StrategyManager(graph_manager=graph)
node = graph.G.nodes[character]
class _Home:
    name = "Home"
character.location = _Home()
node.update(energy=20)
character.energy = 90
names = [a.name for a in strategy.get_daily_actions(character)]
assert "Sleep" in names
node.update(energy=90)
character.energy = 20
names = [a.name for a in strategy.get_daily_actions(character)]
assert "Sleep" not in names
print({"sleep_gate_snapshot": True})
""")

    def test_actor_bound_condition_uses_simulated_successor(self):
        self.run_world("""
from tiny_time_manager import GameTimeManager
from tiny_utility_functions import Goal
node = graph.G.nodes[character]
node.update(energy=20)
character.energy = 20
rest = Action("Rest", [], [{"targets": ["initiator"], "attribute": "energy", "change_value": 30}], graph_manager=graph)
work = Action("Work", [Condition("Ready", "energy", character, 45)], [
    {"targets": ["initiator"], "attribute": "wealth_money", "change_value": 20.0}],
    initiator=character, graph_manager=graph)
assert work.preconditions_met(State({"energy": 50})) is True
assert work.preconditions_met(State({"energy": 40})) is False
planner = GOAPPlanner(graph)
plan = planner.plan_actions(character, Goal(name="earn", target_effects={"wealth_money": 25}, priority=1.0),
    State({"energy": 20, "wealth_money": 5}), [rest, work])
assert plan is not None, plan
assert [a.name for a in plan] == ["Rest", "Work"], plan
print({"actor_bound_successor_plan": ["Rest", "Work"]})
""")

    def test_view_effects_project_sequentially_with_descriptors(self):
        from actions import Action as _Action
        from tiny_utility_functions import build_utility_action_view
        Action = _Action
        boost = Action("Boost", [], [
            {"targets": ["initiator"], "attribute": "energy", "change_value": 10},
            {"targets": ["initiator"], "attribute": "energy", "change_value": 10}], graph_manager=None)
        view = build_utility_action_view(boost, native_state={"energy": 95})
        assert [e["change_value"] for e in view.effects] == [0.05, 0.0]
        descriptor = Action("Mixed", [], [
            {"targets": ["initiator"], "attribute": "energy", "change_value": "add:15"},
            {"targets": ["initiator"], "attribute": "energy", "change_value": "set:100"}], graph_manager=None)
        view = build_utility_action_view(descriptor, native_state={"energy": 95})
        assert [e["change_value"] for e in view.effects] == [0.05, 0.0]  # add:15 reaches the cap; set:100 is then a no-op
        money = Action("Pay", [], [
            {"targets": ["initiator"], "attribute": "money", "change_value": 20.0}], graph_manager=None)
        view = build_utility_action_view(money, native_state={"wealth_money": 100.0})
        assert [e["change_value"] for e in view.effects] == [20.0]
        print({"sequential": [0.05, 0.0], "descriptors": [0.05, 0.0], "money_passthrough": 20.0})

    def test_alias_sync_keeps_all_keys_current(self):
        self.run_world("""
from actions import State
from tiny_goap_system import GOAPPlanner
from tiny_utility_functions import Goal
planner = GOAPPlanner(graph)
node = graph.G.nodes[character]
node.update(wealth_money=10)
character.wealth_money = 10
mixed = Action("MixedMoney", [], [
    {"targets": ["initiator"], "attribute": "money", "change_value": 20.0},
    {"targets": ["initiator"], "attribute": "wealth_money", "change_value": -20.0}], graph_manager=graph)
simulated = planner._apply_action_effects(mixed, State({"wealth_money": 10}))
assert simulated["wealth_money"] == 10
assert simulated["money"] == 10
goal = Goal(name="earn", target_effects={"money": 30}, priority=1.0)
assert not planner._goal_satisfied(goal, simulated)
print({"alias_sync": [10, 10], "money_goal_not_satisfied": True})
""")

    def test_prompt_effect_state_carries_resource_starting_values(self):
        from actions import Action as _Action
        from tiny_characters import Character as _Character
        from tiny_time_manager import GameTimeManager as _GameTimeManager
        from tiny_utility_functions import build_utility_action_view
        from tiny_prompt_builder import PromptBuilder
        Action = _Action
        Character = _Character
        GameTimeManager = _GameTimeManager
        probe = Character(name="PromptMoneyProbe", age=30, wealth_money=100, energy=20,
                          personality_traits={"extraversion": 50, "openness": 80, "conscientiousness": 70, "agreeableness": 40, "neuroticism": 20},
                          gametime_manager=GameTimeManager)
        builder = PromptBuilder(probe)
        snapshot = builder._get_native_effect_state()
        assert snapshot["money"] == snapshot["wealth_money"] == 100
        cut = Action("CutPrice", [], [
            {"targets": ["initiator"], "attribute": "money", "change_value": "set:25"}], graph_manager=None)
        view = build_utility_action_view(cut, native_state=snapshot)
        assert view.effects[0]["change_value"] == -75.0  # was +25 before the complete snapshot (#818 round 3)
        print({"prompt_set25_at_100": -75.0})
