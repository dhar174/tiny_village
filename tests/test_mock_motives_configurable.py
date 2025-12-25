#!/usr/bin/env python3
"""
Test demonstrating the fix for issue #329: MockMotives configurable values.

This test file specifically validates that the MockMotives class no longer
returns hardcoded values (50, 30, 70, 50) and instead provides configurable
values based on character state, ensuring tests will fail if PromptBuilder
doesn't handle varying inputs correctly.
"""

import unittest
from test_crisis_prompt import MockCharacter, MockMotives


class TestMockMotivesConfigurable(unittest.TestCase):
    """Test that MockMotives provides configurable values instead of hardcoded ones."""
    
    def test_motives_vary_with_character_state(self):
        """Test that motive values change based on character state."""
        
        # Create character with very poor conditions
        poor_char = MockCharacter()
        poor_char.health_status = 1      # Very unhealthy
        poor_char.wealth_money = 0       # No money
        poor_char.mental_health = 1      # Poor mental health
        poor_char.social_wellbeing = 1   # Poor social life
        poor_char.hunger_level = 10      # Very hungry
        
        # Create character with excellent conditions  
        excellent_char = MockCharacter()
        excellent_char.health_status = 10    # Perfect health
        excellent_char.wealth_money = 1000   # Rich
        excellent_char.mental_health = 10    # Excellent mental health
        excellent_char.social_wellbeing = 10 # Great social life
        excellent_char.hunger_level = 1      # Well fed
        
        poor_motives = poor_char.get_motives()
        excellent_motives = excellent_char.get_motives()
        
        # Characters in poor condition should have higher motives (more urgent needs)
        self.assertGreater(
            poor_motives.get_health_motive(),
            excellent_motives.get_health_motive(),
            "Poor health character should have higher health motive"
        )
        
        self.assertGreater(
            poor_motives.get_wealth_motive(),
            excellent_motives.get_wealth_motive(),
            "Poor wealth character should have higher wealth motive"
        )
        
        self.assertGreater(
            poor_motives.get_mental_health_motive(),
            excellent_motives.get_mental_health_motive(),
            "Poor mental health character should have higher mental health motive"
        )
        
        self.assertGreater(
            poor_motives.get_hunger_motive(),
            excellent_motives.get_hunger_motive(),
            "Hungry character should have higher hunger motive"
        )
    
    def test_no_hardcoded_values_issue_329(self):
        """Test that the specific hardcoded values from issue #329 are not returned."""
        
        # Test with multiple different character configurations
        test_configs = [
            {"health_status": 5, "wealth_money": 10, "mental_health": 6, "social_wellbeing": 6},
            {"health_status": 1, "wealth_money": 1, "mental_health": 1, "social_wellbeing": 1},
            {"health_status": 10, "wealth_money": 100, "mental_health": 10, "social_wellbeing": 10},
            {"health_status": 3, "wealth_money": 50, "mental_health": 8, "social_wellbeing": 4},
        ]
        
        hardcoded_values = [50, 30, 70, 50]  # The problematic values from issue #329
        
        for config in test_configs:
            with self.subTest(config=config):
                char = MockCharacter()
                for attr, value in config.items():
                    setattr(char, attr, value)
                
                motives = char.get_motives()
                actual_values = [
                    motives.get_health_motive(),
                    motives.get_wealth_motive(),
                    motives.get_mental_health_motive(),
                    motives.get_social_wellbeing_motive()
                ]
                
                self.assertNotEqual(
                    actual_values,
                    hardcoded_values,
                    f"MockMotives should not return hardcoded values {hardcoded_values} "
                    f"for character config {config}, got {actual_values}"
                )
    
    def test_motive_values_within_bounds(self):
        """Test that all motive values are within reasonable bounds."""
        
        # Test with extreme character configurations
        extreme_configs = [
            {"health_status": 0, "wealth_money": 0, "mental_health": 0, "social_wellbeing": 0},
            {"health_status": 100, "wealth_money": 10000, "mental_health": 100, "social_wellbeing": 100},
        ]
        
        for config in extreme_configs:
            with self.subTest(config=config):
                char = MockCharacter()
                for attr, value in config.items():
                    setattr(char, attr, value)
                
                motives = char.get_motives()
                
                # Test all motive getter methods
                motive_methods = [
                    'get_health_motive', 'get_wealth_motive', 'get_mental_health_motive',
                    'get_social_wellbeing_motive', 'get_hunger_motive', 'get_happiness_motive',
                    'get_shelter_motive', 'get_stability_motive', 'get_luxury_motive',
                    'get_hope_motive', 'get_success_motive', 'get_control_motive',
                    'get_job_performance_motive', 'get_beauty_motive', 'get_community_motive',
                    'get_material_goods_motive', 'get_friendship_grid_motive'
                ]
                
                for method_name in motive_methods:
                    if hasattr(motives, method_name):
                        method = getattr(motives, method_name)
                        value = method()
                        self.assertTrue(
                            10 <= value <= 100,
                            f"{method_name}() returned {value}, should be between 10-100 "
                            f"for config {config}"
                        )
    
    def test_character_state_affects_calculations(self):
        """Test that changing character state affects motive calculations."""
        
        char = MockCharacter()
        
        # Test health motive changes with health status
        char.health_status = 1
        low_health_motive = char.get_motives().get_health_motive()
        
        char.health_status = 10
        high_health_motive = char.get_motives().get_health_motive()
        
        self.assertNotEqual(
            low_health_motive, 
            high_health_motive,
            "Health motive should change when character's health status changes"
        )
        
        # Test wealth motive changes with wealth
        char.wealth_money = 1
        low_wealth_motive = char.get_motives().get_wealth_motive()
        
        char.wealth_money = 100
        high_wealth_motive = char.get_motives().get_wealth_motive()
        
        self.assertNotEqual(
            low_wealth_motive,
            high_wealth_motive,
            "Wealth motive should change when character's wealth changes"
        )
    
    def test_standalone_motives_fallback(self):
        """Test that MockMotives works without character reference (fallback mode)."""
        
        standalone_motives = MockMotives(character=None)
        
        # Should return fallback values that are still reasonable
        health_motive = standalone_motives.get_health_motive()
        wealth_motive = standalone_motives.get_wealth_motive()
        mental_health_motive = standalone_motives.get_mental_health_motive()
        social_motive = standalone_motives.get_social_wellbeing_motive()
        
        # Verify fallback values are within bounds
        for value in [health_motive, wealth_motive, mental_health_motive, social_motive]:
            self.assertTrue(
                10 <= value <= 100,
                f"Fallback motive value {value} should be between 10-100"
            )
        
        # Verify these are not the old hardcoded problematic values
        fallback_values = [health_motive, wealth_motive, mental_health_motive, social_motive]
        self.assertNotEqual(
            fallback_values,
            [50, 30, 70, 50],
            f"Fallback values should not be the old hardcoded values, got {fallback_values}"
        )


if __name__ == "__main__":
    unittest.main()