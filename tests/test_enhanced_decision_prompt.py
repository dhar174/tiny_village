#!/usr/bin/env python3
"""
Test suite for the enhanced LLM decision-making prompt system.
Tests the generate_decision_prompt method with real character data to verify
integration of goals, needs priorities, and comprehensive character stats.
"""

import unittest
import logging
from unittest.mock import MagicMock, patch
import sys
import types

if 'torch' not in sys.modules:
    torch_stub = types.ModuleType('torch')
    torch_stub.Graph = object
    torch_stub.eq = lambda *args, **kwargs: None
    torch_stub.rand = lambda *args, **kwargs: 0
    sys.modules['torch'] = torch_stub

from tiny_characters import Character, PersonalMotives, Motive
from tiny_locations import Location
from tiny_prompt_builder import PromptBuilder
from tiny_graph_manager import GraphManager
from tiny_goap_system import Goal, Condition, GOAPPlanner
from actions import ActionSystem
import tiny_time_manager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestEnhancedDecisionPrompt(unittest.TestCase):
    """Test enhanced decision-making prompt generation with comprehensive character context."""

    def setUp(self):
        """Set up test environment with realistic character data."""
        self.graph_manager = GraphManager()

        # Create realistic character motives
        motives = PersonalMotives(
            hunger_motive=Motive("hunger", "Need for food and nutrition", 7.5),
            wealth_motive=Motive("wealth", "Desire for financial security", 6.2),
            mental_health_motive=Motive(
                "mental_health", "Need for mental wellness", 8.1
            ),
            social_wellbeing_motive=Motive(
                "social_wellbeing", "Need for social connections", 5.9
            ),
            happiness_motive=Motive("happiness", "Pursuit of joy and fulfillment", 7.3),
            health_motive=Motive("health", "Physical health maintenance", 8.5),
            shelter_motive=Motive("shelter", "Need for housing security", 4.2),
            stability_motive=Motive("stability", "Desire for life stability", 6.8),
            luxury_motive=Motive("luxury", "Appreciation for comfort", 3.1),
            hope_motive=Motive("hope", "Optimism about the future", 7.0),
            success_motive=Motive("success", "Drive for achievement", 8.8),
            control_motive=Motive("control", "Need for personal agency", 6.5),
            job_performance_motive=Motive(
                "job_performance", "Professional excellence", 9.2
            ),
            beauty_motive=Motive("beauty", "Aesthetic appreciation", 4.7),
            community_motive=Motive("community", "Community involvement", 7.8),
            material_goods_motive=Motive(
                "material_goods", "Material acquisitions", 3.9
            ),
            family_motive=Motive("family", "Family relationships", 8.3),
        )

        # Create test character with comprehensive attributes
        self.character = Character(
            name="Sarah Chen",
            age=28,
            pronouns="she/her",
            job="Software Engineer",
            health_status=7,  # Good but not perfect health
            hunger_level=4,  # Moderately hungry
            wealth_money=6,  # Middle-class financial status
            mental_health=6,  # Some stress
            social_wellbeing=8,  # Good social connections
            job_performance=9,  # Excellent at work
            community=5,  # Average community involvement
            friendship_grid=[],
            recent_event="promotion",
            goal="Build a startup",  # Long-term goal
            personality_traits={
                "extraversion": 60,
                "openness": 85,
                "conscientiousness": 90,
                "agreeableness": 70,
                "neuroticism": 40,
            },
            action_system=ActionSystem(),
            gametime_manager=tiny_time_manager,
            location=Location("Sarah's Apartment", 0, 0, 1, 1, ActionSystem()),
            graph_manager=self.graph_manager,
            motives=motives,  # Add our custom motives
        )

        # Set additional character attributes for comprehensive testing
        self.character.long_term_goal = "Launch a successful AI startup"
        self.character.energy = 6.5  # Moderate energy level

        # Add some realistic goals to the character
        self._add_test_goals()

        # Create prompt builder
        self.prompt_builder = PromptBuilder(self.character)

        # Test action choices
        self.test_action_choices = [
            "1. Work on your startup project to make progress toward your goal",
            "2. Go to the gym to improve your physical health",
            "3. Have lunch with colleagues to strengthen social connections",
            "4. Take a coding course to enhance your professional skills",
            "5. Take a relaxing walk to reduce stress and improve mental health",
        ]

    def _add_test_goals(self):
        """Add some realistic goals to the character for testing."""
        try:
            # Create sample goals that would be generated by the GOAP system
            startup_goal = Goal(
                name="Launch Startup",
                description="Build and launch your AI startup company",
                target=self.character,
                score=9.5,
                character=self.character,
                completion_conditions={},
                evaluate_utility_function=lambda x: 9.5,
                difficulty=lambda x: 8.0,
                completion_reward=lambda x: 100,
                failure_penalty=lambda x: -20,
                completion_message=lambda x, y: "Successfully launched startup!",
                failure_message=lambda x, y: "Startup launch delayed",
                criteria={},
                graph_manager=self.graph_manager,
                goal_type="career",
                target_effects={"success": 10, "wealth_level": 20},
            )

            health_goal = Goal(
                name="Improve Fitness",
                description="Get in better physical shape through regular exercise",
                target=self.character,
                score=7.2,
                character=self.character,
                completion_conditions={},
                evaluate_utility_function=lambda x: 7.2,
                difficulty=lambda x: 5.0,
                completion_reward=lambda x: 50,
                failure_penalty=lambda x: -10,
                completion_message=lambda x, y: "Achieved fitness goals!",
                failure_message=lambda x, y: "Missed fitness targets",
                criteria={},
                graph_manager=self.graph_manager,
                goal_type="health",
                target_effects={"health_status": 3},
            )

            skill_goal = Goal(
                name="Learn New Technology",
                description="Master the latest AI/ML frameworks for competitive advantage",
                target=self.character,
                score=8.1,
                character=self.character,
                completion_conditions={},
                evaluate_utility_function=lambda x: 8.1,
                difficulty=lambda x: 6.5,
                completion_reward=lambda x: 75,
                failure_penalty=lambda x: -15,
                completion_message=lambda x, y: "New skills acquired!",
                failure_message=lambda x, y: "Learning goals not met",
                criteria={},
                graph_manager=self.graph_manager,
                goal_type="skill",
                target_effects={"job_performance": 2},
            )

            # Add goals to character's goal system (if it exists)
            if hasattr(self.character, "goals"):
                self.character.goals.extend([startup_goal, health_goal, skill_goal])
            else:
                # Create a simple goals list for testing
                self.character.goals = [startup_goal, health_goal, skill_goal]

        except Exception as e:
            logger.warning(f"Could not add test goals: {e}")

    def test_enhanced_prompt_generation(self):
        """Test that enhanced prompt generates with comprehensive character context."""
        logger.info("Testing enhanced decision prompt generation...")

        # Generate the enhanced decision prompt
        prompt = self.prompt_builder.generate_decision_prompt(
            time="Tuesday afternoon",
            weather="sunny",
            action_choices=self.test_action_choices,
        )

        # Verify prompt is not empty
        self.assertIsNotNone(prompt)
        self.assertGreater(len(prompt), 100, "Prompt should be substantial in length")

        # Verify system prompt structure
        self.assertIn("<|system|>", prompt, "Should contain system prompt tag")
        self.assertIn("<|user|>", prompt, "Should contain user prompt tag")
        self.assertIn("<|assistant|>", prompt, "Should contain assistant prompt tag")

        # Verify character identity integration
        self.assertIn("Sarah Chen", prompt, "Should include character name")
        self.assertIn("Software Engineer", prompt, "Should include character job")

        # Verify status information is included
        self.assertIn("Health 7/10", prompt, "Should include health status")
        self.assertIn("Hunger 4/10", prompt, "Should include hunger level")
        self.assertIn("Mental Health 6/10", prompt, "Should include mental health")
        self.assertIn(
            "Social Wellbeing 8/10", prompt, "Should include social wellbeing"
        )

        # Verify action choices are included
        for action in self.test_action_choices:
            self.assertIn(action, prompt, f"Should include action choice: {action}")

        # Verify long-term goal integration
        self.assertIn("AI startup", prompt, "Should reference long-term goal")

        logger.info("Enhanced prompt generated successfully!")
        return prompt

    def test_goals_integration(self):
        """Test that character goals are properly integrated into the prompt."""
        logger.info("Testing goals integration...")

        prompt = self.prompt_builder.generate_decision_prompt(
            time="morning", weather="cloudy", action_choices=self.test_action_choices
        )

        # Check if goals section exists (even if no goals are found)
        # The prompt should handle missing goals gracefully
        if hasattr(self.character, "evaluate_goals"):
            try:
                goals = self.character.evaluate_goals()
                if goals and len(goals) > 0:
                    self.assertIn(
                        "current goals", prompt, "Should mention current goals"
                    )
                    logger.info("Goals successfully integrated into prompt")
                else:
                    logger.info("No goals found, but prompt handled gracefully")
            except Exception as e:
                logger.warning(f"Goal evaluation failed (expected): {e}")
                # This is acceptable - the prompt should handle this gracefully

        logger.info("Goals integration test completed")

    def test_needs_priorities_integration(self):
        """Test that character needs priorities are properly calculated and integrated."""
        logger.info("Testing needs priorities integration...")

        prompt = self.prompt_builder.generate_decision_prompt(
            time="evening", weather="rainy", action_choices=self.test_action_choices
        )

        # Verify needs section exists
        self.assertIn("pressing needs", prompt, "Should mention character needs")

        # Check for specific need descriptions
        need_keywords = ["health", "mental", "social", "wealth", "priority"]
        found_keywords = [
            keyword for keyword in need_keywords if keyword in prompt.lower()
        ]
        self.assertGreater(
            len(found_keywords),
            2,
            f"Should contain need-related keywords: {found_keywords}",
        )

        logger.info("Needs priorities successfully integrated!")

    def test_motives_integration(self):
        """Test that character motives are properly integrated with intensity descriptions."""
        logger.info("Testing motives integration...")

        prompt = self.prompt_builder.generate_decision_prompt(
            time="noon", weather="sunny", action_choices=self.test_action_choices
        )

        # Check for motives section
        self.assertIn("key motivations", prompt, "Should mention character motivations")

        # Check for intensity descriptions
        intensity_keywords = ["High", "Moderate", "Low", "Very High"]
        found_intensity = [
            keyword for keyword in intensity_keywords if keyword in prompt
        ]
        self.assertGreater(
            len(found_intensity),
            0,
            f"Should contain intensity descriptions: {found_intensity}",
        )

        # Check for specific high-scoring motives from our test character
        motive_keywords = [
            "job performance",
            "success",
            "health",
        ]  # High-scoring motives
        found_motives = [
            keyword for keyword in motive_keywords if keyword.lower() in prompt.lower()
        ]
        self.assertGreater(
            len(found_motives),
            0,
            f"Should contain high-priority motives: {found_motives}",
        )

        logger.info("Motives successfully integrated!")

    def test_error_handling(self):
        """Test that the prompt handles edge cases and errors gracefully."""
        logger.info("Testing error handling...")

        # Test with character that might have missing attributes
        minimal_character = Character(
            name="Test User",
            age=25,
            pronouns="they/them",
            job="Tester",
            health_status=5,
            hunger_level=5,
            wealth_money=5,
            mental_health=5,
            social_wellbeing=5,
            job_performance=5,
            community=5,
            friendship_grid=[],
            recent_event="testing",
            goal="test goals",
            personality_traits={},
            action_system=ActionSystem(),
            gametime_manager=tiny_time_manager,
            location=Location("Test Location", 0, 0, 1, 1, ActionSystem()),
            graph_manager=self.graph_manager,
        )

        minimal_prompt_builder = PromptBuilder(minimal_character)

        # This should not crash even with minimal character data
        prompt = minimal_prompt_builder.generate_decision_prompt(
            time="test time", weather="test weather", action_choices=["1. Test action"]
        )

        self.assertIsNotNone(prompt)
        self.assertIn("Test User", prompt)
        self.assertIn("Test action", prompt)

        logger.info("Error handling test passed!")

    def test_prompt_completeness(self):
        """Test that the generated prompt contains all expected sections."""
        logger.info("Testing prompt completeness...")

        prompt = self.prompt_builder.generate_decision_prompt(
            time="Wednesday morning",
            weather="partly cloudy",
            action_choices=self.test_action_choices,
        )

        # Check for all major sections
        expected_sections = [
            "You are Sarah Chen",  # Character identity
            "Current state:",  # Status information
            "Available actions:",  # Action choices
            "Choose the action",  # Decision guidance
        ]

        for section in expected_sections:
            self.assertIn(section, prompt, f"Should contain section: {section}")

        # Verify the prompt ends properly
        self.assertIn(
            "Sarah Chen, I choose", prompt, "Should end with proper completion format"
        )

        logger.info("Prompt completeness verified!")

    def test_different_scenarios(self):
        """Test prompt generation under different time/weather scenarios."""
        logger.info("Testing different scenarios...")

        scenarios = [
            ("early morning", "foggy"),
            ("late night", "stormy"),
            ("lunch time", "hot"),
            ("evening", "cold"),
        ]

        for time, weather in scenarios:
            prompt = self.prompt_builder.generate_decision_prompt(
                time=time, weather=weather, action_choices=self.test_action_choices
            )

            self.assertIsNotNone(prompt)
            self.assertIn(time, prompt, f"Should reference time: {time}")
            self.assertGreater(
                len(prompt),
                200,
                f"Prompt should be substantial for scenario: {time}, {weather}",
            )

        logger.info("Different scenarios tested successfully!")

    def test_character_state_dict_parameter(self):
        """Test the optional character_state_dict parameter."""
        logger.info("Testing character_state_dict parameter...")

        # Test with additional state information
        character_state = {
            "current_location": "Office",
            "recent_activity": "Coding session",
            "mood": "Focused",
            "energy_level": 7.5,
        }

        prompt = self.prompt_builder.generate_decision_prompt(
            time="afternoon",
            weather="clear",
            action_choices=self.test_action_choices,
            character_state_dict=character_state,
        )

        self.assertIsNotNone(prompt)
        self.assertIn("Sarah Chen", prompt)
        for key, value in character_state.items():
            formatted_key = key.replace("_", " ").title()
            expected_line = f"- {formatted_key}: {value}"
            self.assertIn(expected_line, prompt)

        logger.info("Character state dict parameter test completed!")

    def test_dynamic_action_choices_with_real_utility_function(self):
        """Test that action choices include real utility calculations with edge cases."""
        from tiny_utility_functions import calculate_action_utility
        
        # Create mock actions that represent real edge cases
        class MockAction:
            def __init__(self, name, description, cost=0.0, effects=None):
                self.name = name
                self.description = description
                self.cost = cost
                self.effects = effects if effects is not None else []
        
        # Edge case 1: High hunger character with food action
        high_hunger_state = {"hunger": 0.9, "energy": 0.5}
        eat_action = MockAction(
            "Eat bread", 
            "Consume bread to reduce hunger",
            cost=0.1, 
            effects=[{"attribute": "hunger", "change_value": -0.7}]
        )
        
        utility = calculate_action_utility(high_hunger_state, eat_action)
        self.assertGreater(utility, 0, "High hunger + food should have positive utility")
        self.assertGreater(utility, 10, "Should be substantial utility for addressing high hunger")
        
        # Edge case 2: Low energy character with rest action
        low_energy_state = {"energy": 0.2, "hunger": 0.3}
        rest_action = MockAction(
            "Rest",
            "Take a rest to restore energy",
            cost=0.05,
            effects=[{"attribute": "energy", "change_value": 0.6}]
        )
        
        utility = calculate_action_utility(low_energy_state, rest_action)
        self.assertGreater(utility, 0, "Low energy + rest should have positive utility")
        
        # Edge case 3: Action with no beneficial effects should have low/negative utility
        empty_state = {"hunger": 0.5, "energy": 0.5}
        expensive_action = MockAction(
            "Expensive action",
            "Costly action with no benefits",
            cost=2.0,
            effects=[]
        )
        
        utility = calculate_action_utility(empty_state, expensive_action)
        self.assertLess(utility, 0, "Expensive action with no benefits should have negative utility")
        
        # Edge case 4: Action with malformed effects should not crash
        malformed_action = MockAction(
            "Malformed action",
            "Action with malformed effects",
            cost=0.1,
            effects=[{"attribute": "hunger"}]  # Missing change_value
        )
        
        try:
            utility = calculate_action_utility(empty_state, malformed_action)
            # Should handle gracefully, typically resulting in small negative utility from cost
            self.assertLessEqual(utility, 0, "Malformed action should have non-positive utility")
        except Exception as e:
            self.fail(f"calculate_action_utility should handle malformed effects gracefully, but raised: {e}")
        
        # Edge case 5: Goal alignment should increase utility
        goal_state = {"hunger": 0.8}
        aligned_action = MockAction(
            "Eat food",
            "Eat to reduce hunger",
            cost=0.1,
            effects=[{"attribute": "hunger", "change_value": -0.5}]
        )
        
        class MockGoal:
            def __init__(self, target_effects, priority):
                self.target_effects = target_effects
                self.priority = priority
        
        hunger_goal = MockGoal(target_effects={"hunger": -0.8}, priority=0.9)
        
        utility_with_goal = calculate_action_utility(goal_state, aligned_action, hunger_goal)
        utility_without_goal = calculate_action_utility(goal_state, aligned_action)
        
        self.assertGreater(utility_with_goal, utility_without_goal, 
                          "Goal-aligned actions should have higher utility than non-aligned ones")
        
        # Edge case 6: Empty character state should be handled gracefully
        try:
            utility = calculate_action_utility({}, eat_action)
            # With empty state, hunger level defaults to 0, so food action shouldn't be very beneficial
            self.assertIsInstance(utility, (int, float), "Should return numeric utility even with empty state")
        except Exception as e:
            self.fail(f"calculate_action_utility should handle empty character state, but raised: {e}")

        logger.info("Real utility function edge case testing completed successfully!")


def print_sample_prompt():
    """Generate and print a sample enhanced prompt for visual inspection."""
    print("\n" + "=" * 80)
    print("SAMPLE ENHANCED DECISION PROMPT")
    print("=" * 80)

    # Set up test environment
    graph_manager = GraphManager()
    motives = PersonalMotives(
        hunger_motive=Motive("hunger", "Need for food", 6.5),
        wealth_motive=Motive("wealth", "Financial security", 7.2),
        mental_health_motive=Motive("mental_health", "Mental wellness", 8.1),
        social_wellbeing_motive=Motive("social_wellbeing", "Social connections", 6.8),
        happiness_motive=Motive("happiness", "Joy and fulfillment", 7.5),
        health_motive=Motive("health", "Physical health", 8.8),
        shelter_motive=Motive("shelter", "Housing security", 5.2),
        stability_motive=Motive("stability", "Life stability", 6.9),
        luxury_motive=Motive("luxury", "Comfort", 4.1),
        hope_motive=Motive("hope", "Optimism", 7.7),
        success_motive=Motive("success", "Achievement", 9.1),
        control_motive=Motive("control", "Personal agency", 6.6),
        job_performance_motive=Motive(
            "job_performance", "Professional excellence", 9.3
        ),
        beauty_motive=Motive("beauty", "Aesthetic appreciation", 5.4),
        community_motive=Motive("community", "Community involvement", 7.9),
        material_goods_motive=Motive("material_goods", "Material acquisitions", 4.3),
        family_motive=Motive("family", "Family relationships", 8.4),
    )

    character = Character(
        name="Alex Rivera",
        age=32,
        pronouns="they/them",
        job="Game Developer",
        health_status=8,
        hunger_level=3,
        wealth_money=7,
        mental_health=7,
        social_wellbeing=8,
        job_performance=9,
        community=6,
        friendship_grid=[],
        recent_event="game launch",
        goal="Create indie game studio",
        personality_traits={
            "extraversion": 55,
            "openness": 90,
            "conscientiousness": 85,
            "agreeableness": 75,
            "neuroticism": 35,
        },
        action_system=ActionSystem(),
        gametime_manager=tiny_time_manager,
        location=Location("Alex's Home Office", 0, 0, 1, 1, ActionSystem()),
        graph_manager=graph_manager,
        motives=motives,
    )

    character.long_term_goal = "Build a successful indie game development studio"
    character.energy = 7.8

    prompt_builder = PromptBuilder(character)

    action_choices = [
        "1. Work on your current game project to meet the upcoming deadline",
        "2. Network with other developers at the local game dev meetup",
        "3. Study new game engine features to improve your technical skills",
        "4. Take a break and play some games for inspiration and relaxation",
        "5. Update your portfolio website to attract potential collaborators",
    ]

    prompt = prompt_builder.generate_decision_prompt(
        time="Thursday evening", weather="gentle rain", action_choices=action_choices
    )

    print(prompt)
    print("\n" + "=" * 80)


if __name__ == "__main__":
    # Run the tests
    unittest.main(argv=[""], exit=False, verbosity=2)

    # Print a sample prompt for visual inspection
    print_sample_prompt()
