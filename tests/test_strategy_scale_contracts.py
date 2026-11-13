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
from tiny_goap_system import GOAPPlanner
from tiny_strategy_manager import StrategyManager
strategy = StrategyManager(graph_manager=graph)
planner_free = GOAPPlanner(graph)
node = graph.G.nodes[character]
class _Home:
    name = "Home"
character.location = _Home()
character.job = "Baker"
node.update(energy=29)
character.energy = 29
work_low = next(a for a in strategy.get_daily_actions(character) if a.name.startswith("Work as"))
assert not work_low.preconditions_met()
from actions import State as _State
assert not planner_free._action_applicable(work_low, _State({"energy": 29}))
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
