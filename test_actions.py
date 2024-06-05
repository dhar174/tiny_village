from re import T
from symbol import parameters
import unittest
from actions import ActionSystem, State, ActionTemplate, Condition


class TestChar:
    def __init__(self, name, state: State):
        self.name = name
        self.state = state


class TestActionSystem(unittest.TestCase):
    def setUp(self):
        self.action_system = ActionSystem()

    def test_setup_actions(self):
        self.action_system.setup_actions()
        self.assertEqual(len(self.action_system.action_generator.templates), 3)

    def test_generate_actions(self):
        character = TestChar("Emma", State({"energy": 50}))
        self.action_system.setup_actions()
        actions = self.action_system.generate_actions(character)
        self.assertEqual(len(actions), 3)

    def test_execute_action(self):
        state = State({"energy": 50})
        parameters = {"initiator": "Emma", "target": "seesaw"}
        action = ActionTemplate(
            "Test",
            self.action_system.create_precondition("energy", "gt", 20),
            {"energy": -10},
            1,
        ).instantiate(parameters)
        self.assertTrue(self.action_system.execute_action(action, state))
        self.assertEqual(state["energy"], 39)

    def test_create_precondition(self):
        precondition = self.action_system.create_precondition("energy", "gt", 20)[
            "energy"
        ]
        li = precondition(State({"energy": 30}))
        print(f"For energy of 30, precondition is {li}")
        self.assertTrue(precondition(State({"energy": 30})))
        ll = precondition(State({"energy": 10}))
        print(f"For energy of 10, precondition is {ll}")
        self.assertFalse(
            precondition(State({"energy": 10})),
            msg=f"{precondition} failed, energy is 10",
        )

    def test_instantiate_condition(self):
        condition_dict = {
            "name": "Test",
            "attribute": "energy",
            "satisfy_value": 20,
            "operator": "gt",
        }
        condition = self.action_system.instantiate_condition(condition_dict)
        self.assertIsInstance(condition, Condition)
        self.assertEqual(condition.name, "Test")

    def test_instantiate_conditions(self):
        conditions_list = [
            {
                "name": "Test1",
                "attribute": "energy",
                "satisfy_value": 20,
                "operator": "gt",
            },
            {
                "name": "Test2",
                "attribute": "happiness",
                "satisfy_value": 10,
                "operator": "ge",
            },
        ]
        conditions = self.action_system.instantiate_conditions(conditions_list)
        self.assertEqual(len(conditions), 2)


if __name__ == "__main__":
    unittest.main()
