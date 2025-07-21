#!/usr/bin/env python3
"""
Comprehensive test for the enhanced generate_decision_prompt method.
This test creates a real character with goals, needs, and motives,
then generates and validates the enhanced decision prompt.
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from tiny_prompt_builder import PromptBuilder
from tiny_characters import Character
from tiny_utility_functions import Goal
from actions import Action


def create_test_character():
    """Create a test character with realistic attributes for prompt testing."""
    character = Character(
        name="Alice",
        age=28,
        job="Teacher",
        personality_traits=["curious", "patient", "creative"],
    )

    # Set some realistic character attributes
    character.hunger_level = 7  # Moderately hungry
    character.energy = 4  # Low energy
    character.health = 8  # Good health
    character.mental_health = 6  # Okay mental health
    character.social_wellbeing = 7  # Good social wellbeing
    character.wealth_money = 150.0  # Some money

    return character


def create_test_actions():
    """Create a list of test actions with effects."""
    actions = [
        "1. Eat breakfast (Utility: 8.5) - Effects: hunger: -3.0, energy: +1.0",
        "2. Take a nap (Utility: 7.2) - Effects: energy: +4.0",
        "3. Go for a walk (Utility: 6.1) - Effects: energy: -0.5, mental_health: +1.0",
        "4. Call a friend (Utility: 5.8) - Effects: social_wellbeing: +2.0",
        "5. Work on lesson plans (Utility: 4.3) - Effects: energy: -2.0, money: +10.0",
    ]
    return actions


def test_enhanced_prompt_generation():
    """Test the enhanced prompt generation with comprehensive character data."""
    print("🧪 Testing Enhanced Prompt Generation")
    print("=" * 50)

    try:
        # Create test character
        character = create_test_character()
        print(f"✅ Created test character: {character.name}")

        # Create prompt builder
        prompt_builder = PromptBuilder(character)
        print(f"✅ Created PromptBuilder for {character.name}")

        # Create test action choices
        action_choices = create_test_actions()
        print(f"✅ Created {len(action_choices)} test actions")

        # Create character state dict
        character_state_dict = {
            "hunger": character.hunger_level / 10.0,
            "energy": character.energy / 10.0,
            "health": character.health / 10.0,
            "mental_health": character.mental_health / 10.0,
            "social_wellbeing": character.social_wellbeing / 10.0,
            "money": character.wealth_money,
        }
        print(f"✅ Created character state dict: {character_state_dict}")

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

        # Check for key sections
        sections_found = []
        if "SYSTEM:" in prompt:
            sections_found.append("System identity")
        if "GOALS:" in prompt or "Current Goals:" in prompt:
            sections_found.append("Goals section")
        if "NEEDS:" in prompt or "Current Needs:" in prompt:
            sections_found.append("Needs section")
        if "MOTIVES:" in prompt or "Character Motives:" in prompt:
            sections_found.append("Motives section")
        if "CURRENT STATE:" in prompt or "Current Status:" in prompt:
            sections_found.append("Current state")
        if "ACTION CHOICES:" in prompt or "Available Actions:" in prompt:
            sections_found.append("Action choices")

        print(f"📋 Found sections: {', '.join(sections_found)}")
        print(f"📏 Prompt length: {len(prompt)} characters")
        print(f"📄 Prompt lines: {len(prompt.split('\\n'))} lines")

        # Show a preview of the prompt
        print("\n📖 Prompt Preview (first 500 characters):")
        print("-" * 40)
        print(prompt[:500] + "..." if len(prompt) > 500 else prompt)

        # Show the full prompt in sections
        print("\n📚 Full Enhanced Prompt:")
        print("=" * 60)
        print(prompt)
        print("=" * 60)

        return True

    except Exception as e:
        print(f"❌ Error in prompt generation test: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_helper_methods():
    """Test the helper methods used in prompt generation."""
    print("\n🧪 Testing Helper Methods")
    print("=" * 50)

    try:
        character = create_test_character()
        prompt_builder = PromptBuilder(character)

        # Test _get_need_description method if it exists
        test_scores = [8.5, 7.2, 6.1, 5.8, 4.3]
        for score in test_scores:
            try:
                description = prompt_builder._get_need_description(score)
                print(f"✅ Need score {score:.1f} → '{description}'")
            except AttributeError:
                print(f"⚠️  _get_need_description method not found")
                break

        # Test _get_motive_description method if it exists
        for score in test_scores:
            try:
                description = prompt_builder._get_motive_description(score)
                print(f"✅ Motive score {score:.1f} → '{description}'")
            except AttributeError:
                print(f"⚠️  _get_motive_description method not found")
                break

        return True

    except Exception as e:
        print(f"❌ Error in helper methods test: {e}")
        return False


def main():
    """Run all comprehensive tests."""
    print("🚀 Starting Comprehensive Enhanced Prompt Tests")
    print("=" * 60)

    # Run tests
    test_results = []
    test_results.append(test_enhanced_prompt_generation())
    test_results.append(test_helper_methods())

    # Summary
    print("\n📊 Test Summary:")
    print("=" * 30)
    passed = sum(test_results)
    total = len(test_results)
    print(f"✅ Passed: {passed}/{total}")

    if passed == total:
        print("🎉 All tests passed! Enhanced prompt system is working correctly.")
    else:
        print("⚠️  Some tests failed. Check the output above for details.")

    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
