import unittest

from tiny_jobs import JobManager, JobRoles


class DummyCharacter:
    def __init__(
        self,
        name="Avery",
        job="unemployed",
        job_performance=50,
        energy=8,
        hunger_level=2,
        wealth_money=100.0,
        material_goods=0.0,
        community=5.0,
        skills=None,
        job_experience=None,
    ):
        self.name = name
        self.job = job
        self.job_performance = job_performance
        self.energy = energy
        self.hunger_level = hunger_level
        self.wealth_money = wealth_money
        self.material_goods = material_goods
        self.community = community
        self.skills = skills or {}
        self.job_experience = job_experience or {}


class TestTinyJobs(unittest.TestCase):
    def setUp(self):
        self.manager = JobManager()
        self.blacksmith = JobRoles(
            "Blacksmith",
            "Blacksmith",
            "Forge tools for the village.",
            "$52,000",
            ["Metalworking", "Craftsmanship"],
            "Apprenticeship",
            "2 years",
            ["wealth motive", "job performance motive"],
        )
        self.farmer = JobRoles(
            "Farmer",
            "Farmer",
            "Grow crops and maintain fields.",
            "$36,000",
            ["Farming", "Stamina"],
            "None",
            "1 year",
            ["wealth motive", "community motive"],
        )
        self.junior_smith = JobRoles(
            "Junior Smith",
            "Junior Smith",
            "Entry-level metalworker.",
            "$32,000",
            ["Metalworking"],
            "Apprenticeship",
            "1 year",
            ["wealth motive"],
        )
        self.senior_smith = JobRoles(
            "Senior Smith",
            "Senior Smith",
            "Lead the smithing workshop.",
            "$65,000",
            ["Metalworking", "Leadership"],
            "Advanced Apprenticeship",
            "4 years",
            ["wealth motive", "success motive"],
        )
        self.manager.job_rules.ValidJobRoles = [
            self.blacksmith,
            self.farmer,
            self.junior_smith,
            self.senior_smith,
        ]

    def test_job_role_parses_salary_and_hourly_rate(self):
        self.assertEqual(self.blacksmith.get_job_salary_value(), 52000.0)
        self.assertGreater(self.blacksmith.get_job_hourly_rate(), 0.0)

    def test_apply_for_job_assigns_qualified_character(self):
        character = DummyCharacter(
            skills={"Metalworking": 3, "Craftsmanship": 2},
        )

        result = self.manager.apply_for_job(character, "Blacksmith")

        self.assertTrue(result["qualified"])
        self.assertTrue(result["assigned"])
        self.assertEqual(character.job.get_job_name(), "Blacksmith")
        self.assertEqual(character.job.employee, character)

    def test_leave_job_restores_unemployed_state(self):
        character = DummyCharacter(skills={"Metalworking": 3})
        self.manager.assign_character_to_job(character, "Blacksmith")

        previous_job = self.manager.leave_job(character)

        self.assertEqual(previous_job.get_job_name(), "Blacksmith")
        self.assertEqual(character.job, "unemployed")
        self.assertTrue(previous_job.is_open())

    def test_get_job_actions_returns_specific_role_actions(self):
        blacksmith_actions = self.manager.get_job_actions(self.blacksmith)
        farmer_actions = self.manager.get_job_actions(self.farmer)

        self.assertEqual(blacksmith_actions[0]["name"], "CraftToolAtBlacksmith")
        self.assertEqual(farmer_actions[0]["name"], "WorkAtFarm")
        self.assertGreater(blacksmith_actions[0]["energy_cost"], 0)
        self.assertEqual(blacksmith_actions[1]["name"], "ImproveBlacksmithSkills")

    def test_perform_job_updates_character_and_village_economy(self):
        character = DummyCharacter(
            skills={"Metalworking": 3, "Craftsmanship": 2},
            job_performance=55,
        )
        self.manager.assign_character_to_job(character, "Blacksmith")

        outcome = self.manager.perform_job(character)
        economy = self.manager.get_village_economy_summary()

        self.assertTrue(outcome["success"])
        self.assertGreater(character.wealth_money, 100.0)
        self.assertLess(character.energy, 8)
        self.assertGreater(character.job_performance, 55)
        self.assertGreater(character.material_goods, 0.0)
        self.assertIn("tools", outcome["resource_output"])
        self.assertGreater(economy["total_wages_paid"], 0.0)
        self.assertGreater(economy["goods_produced"]["tools"], 0.0)
        self.assertIn("Metalworking", outcome["skill_updates"])

    def test_progress_career_promotes_character_when_ready(self):
        character = DummyCharacter(
            skills={"Metalworking": 5, "Leadership": 1},
            job_performance=82,
            job_experience={"Junior Smith": 80},
        )
        self.manager.assign_character_to_job(character, "Junior Smith")

        result = self.manager.progress_career(character)

        self.assertTrue(result["promoted"])
        self.assertEqual(result["previous_job"], "Junior Smith")
        self.assertEqual(result["new_job"], "Senior Smith")
        self.assertEqual(character.job.get_job_name(), "Senior Smith")
        self.assertGreater(result["salary_increase"], 0.0)


if __name__ == "__main__":
    unittest.main()
