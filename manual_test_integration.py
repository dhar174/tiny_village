#!/usr/bin/env python3
"""
Manual validation test for PromptBuilder enhancements.
Tests the core integration without requiring full character dependencies.
"""

import sys
from unittest.mock import MagicMock


def test_prompt_builder_features():
    """Test PromptBuilder features with minimal mocking."""
    print("Testing PromptBuilder feature integration...")
    
    # Mock the tiny_characters module
    mock_tc = MagicMock()
    
    class MockCharacter:
        def __init__(self):
            self.name = "TestAlice"
            self.job = "Engineer"
            self.health_status = 8
            self.hunger_level = 4
            self.wealth_money = 15
            self.mental_health = 7
            self.social_wellbeing = 6
            self.energy = 5
            self.recent_event = "craft fair"
            self.long_term_goal = "career advancement"
            self.personality_traits = {
                "extraversion": 75,
                "conscientiousness": 80,
                "openness": 70
            }
            
        def evaluate_goals(self):
            # Mock goal evaluation
            mock_goal = MagicMock()
            mock_goal.name = "Learn new skills"
            mock_goal.description = "Improve technical abilities"
            return [(0.8, mock_goal)]
    
    mock_tc.Character = MockCharacter
    sys.modules['tiny_characters'] = mock_tc
    
    # Now import PromptBuilder
    try:
        from tiny_prompt_builder import PromptBuilder, DescriptorMatrices
        
        # Create test character and builder
        character = MockCharacter()
        builder = PromptBuilder(character)
        
        print("✓ PromptBuilder initialized successfully")
        
        # Test 1: Character voice initialization
        voice_traits = builder.character_voice_traits
        assert 'speech_style' in voice_traits
        assert 'analytical' in voice_traits['speech_style']  # Engineer trait
        print("✓ Character voice traits initialized correctly")
        
        # Test 2: Conversation history integration
        builder.record_conversation_turn(
            "What should I do?", 
            "I choose to work", 
            "work", 
            "Earned money"
        )
        context = builder.conversation_history.get_recent_context(character.name)
        assert len(context) == 1
        assert context[0].action_taken == "work"
        print("✓ Conversation history recording works")
        
        # Test 3: Few-shot example integration
        builder.add_few_shot_example(
            "Character needed money",
            {"money": 5, "energy": 8},
            "work",
            "Earned 10 coins",
            0.85
        )
        examples = builder.few_shot_manager.get_relevant_examples({"money": 4, "energy": 7})
        assert len(examples) > 0
        print("✓ Few-shot example management works")
        
        # Test 4: Character voice application
        test_prompt = "<|system|>Test<|user|>What do you do?"
        enhanced_prompt = builder.apply_character_voice(test_prompt)
        assert "methodical" in enhanced_prompt or "analytical" in enhanced_prompt
        print("✓ Character voice application works")
        
        # Test 5: Enhanced descriptor matrices
        descriptors = DescriptorMatrices()
        voice_desc = descriptors.get_character_voice_descriptor("Engineer", "speech_pattern")
        assert voice_desc in ["analytical", "precise", "methodical", "logical"]
        print("✓ Enhanced DescriptorMatrices works")
        
        # Test 6: Structured output schemas
        from tiny_prompt_builder import OutputSchema
        decision_schema = OutputSchema.get_decision_schema()
        assert "JSON" in decision_schema
        assert "reasoning" in decision_schema
        print("✓ Structured output schemas work")
        
        print("\n" + "=" * 50)
        print("ALL TESTS PASSED!")
        print("PromptBuilder enhancements are working correctly.")
        print("=" * 50)
        
        return True
        
    except Exception as e:
        print(f"✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_enhanced_prompt_generation():
    """Test that enhanced prompt generation methods work."""
    print("\nTesting enhanced prompt generation...")
    
    try:
        # Mock dependencies first
        mock_tc = MagicMock()
        
        class MockCharacter:
            def __init__(self):
                self.name = "TestCharacter"
                self.job = "Engineer"
                self.health_status = 8
                self.hunger_level = 4
                self.wealth_money = 15
                self.mental_health = 7
                self.social_wellbeing = 6
                self.recent_event = "craft fair"
                self.long_term_goal = "career advancement"
                self.personality_traits = {"extraversion": 75}
                
            def evaluate_goals(self):
                return []
                
            # Mock all the getter methods needed by calculate_needs_priorities
            def get_health_status(self): return self.health_status
            def get_hunger_level(self): return self.hunger_level
            def get_wealth(self): return self.wealth_money
            def get_mental_health(self): return self.mental_health
            def get_social_wellbeing(self): return self.social_wellbeing
            def get_happiness(self): return 7
            def get_shelter(self): return 8
            def get_stability(self): return 6
            def get_luxury(self): return 5
            def get_hope(self): return 7
            def get_success(self): return 6
            def get_control(self): return 7
            def get_job_performance(self): return 8
            def get_beauty(self): return 6
            def get_community(self): return 5
            def get_material_goods(self): return 6
            def get_motives(self):
                # Return a mock motives object
                mock_motives = MagicMock()
                mock_motives.get_health_motive.return_value = 5
                mock_motives.get_hunger_motive.return_value = 3
                mock_motives.get_wealth_motive.return_value = 6
                mock_motives.get_mental_health_motive.return_value = 4
                mock_motives.get_social_wellbeing_motive.return_value = 5
                mock_motives.get_happiness_motive.return_value = 6
                mock_motives.get_shelter_motive.return_value = 3
                mock_motives.get_stability_motive.return_value = 4
                mock_motives.get_luxury_motive.return_value = 2
                mock_motives.get_hope_motive.return_value = 7
                mock_motives.get_success_motive.return_value = 8
                mock_motives.get_control_motive.return_value = 6
                mock_motives.get_job_performance_motive.return_value = 9
                mock_motives.get_beauty_motive.return_value = 3
                mock_motives.get_community_motive.return_value = 4
                mock_motives.get_material_goods_motive.return_value = 3
                mock_motives.get_friendship_grid_motive.return_value = 5
                return mock_motives
                
        mock_tc.Character = MockCharacter
        sys.modules['tiny_characters'] = mock_tc
        
        from tiny_prompt_builder import PromptBuilder
        
        character = MockCharacter()
        builder = PromptBuilder(character)
        
        # Test enhanced daily routine prompt
        routine_prompt = builder.generate_daily_routine_prompt(
            "morning", 
            "sunny",
            include_conversation_context=False,  # Disable to avoid empty context
            include_few_shot_examples=False,     # Disable to keep test simple
            output_format="json"
        )
        
        # Verify the prompt structure
        assert isinstance(routine_prompt, str)
        assert "TestCharacter" in routine_prompt
        assert "JSON" in routine_prompt
        assert "analytical" in routine_prompt or "methodical" in routine_prompt  # Voice consistency
        print("✓ Enhanced daily routine prompt generation works")
        
        # Test decision prompt with structured output
        decision_prompt = builder.generate_decision_prompt(
            "afternoon",
            "cloudy", 
            ["work", "eat_food", "sleep"],
            include_conversation_context=False,
            include_few_shot_examples=False,
            output_format="json"
        )
        
        assert isinstance(decision_prompt, str)
        assert "TestCharacter" in decision_prompt
        assert "reasoning" in decision_prompt  # JSON schema
        print("✓ Enhanced decision prompt generation works")
        
        # Test crisis prompt
        crisis_prompt = builder.generate_crisis_response_prompt(
            "Emergency: Building fire nearby!",
            "high",
            include_conversation_context=False,
            include_few_shot_examples=False
        )
        
        assert isinstance(crisis_prompt, str)
        assert "TestCharacter" in crisis_prompt
        assert "Immediate Action" in crisis_prompt  # Crisis schema
        print("✓ Enhanced crisis response prompt generation works")
        
        print("✓ All prompt generation methods work correctly")
        return True
        
    except Exception as e:
        print(f"✗ Prompt generation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all validation tests."""
    print("Manual Validation of PromptBuilder Enhancements")
    print("=" * 60)
    
    success = True
    
    success &= test_prompt_builder_features()
    success &= test_enhanced_prompt_generation()
    
    if success:
        print("\n🎉 ALL VALIDATIONS PASSED!")
        print("The LLM integration features are working correctly.")
    else:
        print("\n❌ Some validations failed.")
        print("Please check the implementation.")


if __name__ == "__main__":
    main()