#!/usr/bin/env python3
"""
Simplified test for the enhanced generate_decision_prompt method.
This test focuses on testing the prompt generation with a mock character.
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))


def test_enhanced_prompt_with_mock():
    """Test the enhanced prompt generation with a mock character object."""
    print("🧪 Testing Enhanced Prompt Generation (Mock Character)")
    print("=" * 60)

    try:
        from tiny_prompt_builder import PromptBuilder

        # Create a mock character object with just the needed attributes
        class MockCharacter:
            def __init__(self):
                self.name = "Alice"
                self.age = 28
                self.occupation = "Teacher"
                self.job = "Teacher"  # Add job attribute

                # Direct attributes that the prompt generation expects
                self.health_status = 8
                self.hunger_level = 7
                self.energy = 4
                self.mental_health = 6
                self.social_wellbeing = 7
                self.wealth_money = 150.0
                self.recent_event = "had a good day teaching students"

                # Mock methods that might be called
                self.goals = []
                self.motives = MockMotives()

            def evaluate_goals(self):
                """Mock goals evaluation."""
                return [
                    (
                        8.5,
                        MockGoal(
                            "Prepare engaging lesson plans",
                            "Help students learn better",
                            priority=0.85,
                            target_effects={"job_performance": 0.8, "energy": -0.2}
                        ),
                    ),
                    (
                        7.2,
                        MockGoal(
                            "Maintain work-life balance",
                            "Avoid burnout and stay healthy",
                            priority=0.72,
                            target_effects={"health": 0.8, "mental_health": 0.7}
                        ),
                    ),
                    (
                        6.8,
                        MockGoal(
                            "Build stronger relationships",
                            "Connect with friends and family",
                            priority=0.68,
                            target_effects={"social_wellbeing": 0.8, "happiness": 0.6}
                        ),
                    ),
                ]

            # Add all the missing getter methods
            def get_health_status(self):
                return self.health_status

            def get_hunger_level(self):
                return self.hunger_level

            def get_energy(self):
                return self.energy

            def get_wealth_money(self):
                return self.wealth_money

            def get_social_wellbeing(self):
                return self.social_wellbeing

            def get_mental_health(self):
                return self.mental_health

            def get_age(self):
                return self.age

            def get_job_performance(self):
                return 75  # Mock job performance

            def get_community(self):
                return 5  # Mock community rating

            def get_motives(self):
                """Return the motives object."""
                return self.motives

            # All the additional getter methods that the needs priority calculator expects
            def get_wealth(self):
                return 6.5  # Mock wealth value

            def get_happiness(self):
                return 7.0  # Mock happiness value

            def get_shelter(self):
                return 8.0  # Mock shelter value

            def get_stability(self):
                return 6.0  # Mock stability value

            def get_luxury(self):
                return 4.0  # Mock luxury value

            def get_hope(self):
                return 7.5  # Mock hope value

            def get_success(self):
                return 6.8  # Mock success value

            def get_control(self):
                return 5.5  # Mock control value

            def get_beauty(self):
                return 6.2  # Mock beauty value

            def get_material_goods(self):
                return 4.5  # Mock material goods value



            def get_family_motive_value(self):
                return 8.0  # Mock family motive value

            def get_friendship_grid(self):
                # Return mock friendship grid data that varies based on social wellbeing
                # This makes the test more realistic and avoids hardcoded values
                base_friendship_score = self.social_wellbeing * 10  # Convert to 0-100 scale
                return [
                    {
                        "character_name": "Bob",
                        "friendship_score": base_friendship_score + 5,
                        "emotional_impact": 7,
                        "trust_level": 6,
                    },
                    {
                        "character_name": "Carol", 
                        "friendship_score": base_friendship_score - 10,
                        "emotional_impact": 5,
                        "trust_level": 4,
                    },
                ]

        # Enhanced MockGoal with proper interface matching real Goal class
        class MockGoal:
            def __init__(self, name, description, priority=0.5, target_effects=None):
                self.name = name
                self.description = description
                self.priority = priority
                self.score = priority
                self.target_effects = target_effects if target_effects else {}
                self.completed = False
                
                # Additional attributes for realistic testing
                self.character = None
                self.target = None
                self.completion_conditions = {}
                self.criteria = []
                self.required_items = []
                self.goal_type = "test"
                
            def check_completion(self, state=None):
                """Check goal completion - matches real Goal interface."""
                return self.completed
                
            def get_name(self):
                """Getter method found in real Goal class."""
                return self.name
                
            def get_score(self):
                """Getter method found in real Goal class."""
                return self.score

        class MockMotives:
            def __init__(self):
                self.motives_data = {
                    "hunger": 8.5,
                    "wealth": 6.2,
                    "mental_health": 7.1,
                    "social_wellbeing": 5.8,
                    "happiness": 6.9,
                    "health": 7.5,
                    "shelter": 4.2,
                    "stability": 8.1,
                    "luxury": 3.5,
                    "hope": 7.8,
                    "success": 6.4,
                    "control": 5.5,
                    "job_performance": 7.2,
                    "beauty": 4.8,
                    "community": 6.7,
                    "material_goods": 4.1,
                    "family": 8.9,
                }

            def to_dict(self):
                return self.motives_data

            # All the getter methods that PersonalMotives has
            # These should return numeric values for needs priority calculations
            def get_hunger_motive(self):
                return self.motives_data["hunger"]

            def get_wealth_motive(self):
                return self.motives_data["wealth"]

            def get_mental_health_motive(self):
                return self.motives_data["mental_health"]

            def get_social_wellbeing_motive(self):
                return self.motives_data["social_wellbeing"]

            def get_happiness_motive(self):
                return self.motives_data["happiness"]

            def get_health_motive(self):
                return self.motives_data["health"]

            def get_shelter_motive(self):
                return self.motives_data["shelter"]

            def get_stability_motive(self):
                return self.motives_data["stability"]

            def get_luxury_motive(self):
                return self.motives_data["luxury"]

            def get_hope_motive(self):
                return self.motives_data["hope"]

            def get_success_motive(self):
                return self.motives_data["success"]

            def get_control_motive(self):
                return self.motives_data["control"]

            def get_job_performance_motive(self):
                return self.motives_data["job_performance"]

            def get_beauty_motive(self):
                return self.motives_data["beauty"]

            def get_community_motive(self):
                return self.motives_data["community"]

            def get_material_goods_motive(self):
                return self.motives_data["material_goods"]

            def get_family_motive(self):
                return self.motives_data["family"]



        class MockMotive:
            """Mock individual motive class."""

            def __init__(self, name, description, score):
                self.name = name
                self.description = description
                self.score = score

            def get_score(self):
                return self.score

        # Create mock character
        character = MockCharacter()
        print(f"✅ Created mock character: {character.name}")

        # Create prompt builder
        prompt_builder = PromptBuilder(character)
        print(f"✅ Created PromptBuilder for {character.name}")

        # Create test action choices
        action_choices = [
            "1. Eat breakfast (Utility: 8.5) - Effects: hunger: -3.0, energy: +1.0",
            "2. Take a nap (Utility: 7.2) - Effects: energy: +4.0",
            "3. Go for a walk (Utility: 6.1) - Effects: energy: -0.5, mental_health: +1.0",
            "4. Call a friend (Utility: 5.8) - Effects: social_wellbeing: +2.0",
            "5. Work on lesson plans (Utility: 4.3) - Effects: energy: -2.0, money: +10.0",
        ]
        print(f"✅ Created {len(action_choices)} test actions")

        # Create character state dict
        character_state_dict = {
            "hunger": character.hunger_level / 10.0,
            "energy": character.energy / 10.0,
            "health": character.health_status / 10.0,
            "mental_health": character.mental_health / 10.0,
            "social_wellbeing": character.social_wellbeing / 10.0,
            "money": character.wealth_money,
        }
        print(f"✅ Created character state dict")

        # Generate the enhanced prompt
        print("\n🔄 Generating enhanced decision prompt...")
        prompt = prompt_builder.generate_decision_prompt(
            time="morning",
            weather="sunny",
            action_choices=action_choices,
            character_state_dict=character_state_dict,
        )

        print("✅ Successfully generated enhanced prompt!")

        # Analyze the prompt content
        print("\n📊 Prompt Analysis:")
        print("-" * 30)

        # Check for key sections and content
        checks = [
            ("Character name", character.name in prompt),
            ("Goals section", "Goals" in prompt or "goals" in prompt),
            ("Needs section", "Needs" in prompt or "needs" in prompt),
            ("Motives section", "Motives" in prompt or "motives" in prompt),
            ("Current state", "State" in prompt or "status" in prompt),
            ("Action choices", "Actions" in prompt or "choices" in prompt),
            (
                "Character attributes",
                any(attr in prompt.lower() for attr in ["hunger", "energy", "health"]),
            ),
            (
                "Decision guidance",
                "decide" in prompt.lower() or "choose" in prompt.lower(),
            ),
        ]

        for check_name, check_result in checks:
            status = "✅" if check_result else "❌"
            print(f"{status} {check_name}: {'Found' if check_result else 'Missing'}")

        print(f"\n📏 Prompt length: {len(prompt)} characters")
        print(f"📄 Prompt lines: {len(prompt.split(chr(10)))} lines")

        # Show a preview of the prompt
        print("\n📖 Prompt Preview (first 800 characters):")
        print("-" * 50)
        preview = prompt[:800].replace("\\n", "\n")
        print(preview + "..." if len(prompt) > 800 else preview)

        # Test if helper methods work
        print("\n🔧 Testing Helper Methods:")
        print("-" * 30)

        try:
            # Test need description helper
            need_desc = prompt_builder._get_need_description(7.5)
            print(f"✅ Need description (7.5): '{need_desc}'")
        except Exception as e:
            print(f"❌ Need description test failed: {e}")

        try:
            # Test motive description helper
            motive_desc = prompt_builder._get_motive_description(8.2)
            print(f"✅ Motive description (8.2): '{motive_desc}'")
        except Exception as e:
            print(f"❌ Motive description test failed: {e}")

        # Check if the prompt contains enhanced information
        enhanced_features = [
            "utility" in prompt.lower(),
            "priority" in prompt.lower() or "score" in prompt.lower(),
            len([line for line in prompt.split("\n") if line.strip()])
            > 10,  # More than 10 non-empty lines
            "teacher" in prompt.lower(),  # Character occupation
        ]

        enhancement_score = sum(enhanced_features)
        print(f"\n🎯 Enhancement Score: {enhancement_score}/4")

        if enhancement_score >= 3:
            print("🎉 Enhanced prompt generation is working well!")
            return True
        else:
            print("⚠️  Prompt may need more enhancement")
            return False

    except Exception as e:
        print(f"❌ Error in prompt generation test: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    """Run the simplified test."""
    print("🚀 Starting Simplified Enhanced Prompt Test")
    print("=" * 60)

    success = test_enhanced_prompt_with_mock()

    print("\n📊 Test Summary:")
    print("=" * 30)
    if success:
        print("✅ Test passed! Enhanced prompt system is working.")
    else:
        print("❌ Test failed. Check the output above for details.")

    return success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
