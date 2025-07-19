#!/usr/bin/env python3
"""
Demo script showing the new LLM integration features for tiny_prompt_builder.py
This demonstrates the core functionality without requiring heavy dependencies.
"""

import sys
from typing import Dict, Any

# Add the current directory to path so we can import our test modules
sys.path.insert(0, '.')

# Import our isolated test classes which contain the same implementations
from test_llm_features_isolated import (
    ConversationHistory, 
    FewShotExampleManager, 
    FewShotExample,
    OutputSchema
)


def demo_conversation_history():
    """Demonstrate multi-turn conversation management."""
    print("=" * 60)
    print("1. MULTI-TURN CONVERSATION MANAGEMENT")
    print("=" * 60)
    
    history = ConversationHistory(max_history_length=5)
    
    # Simulate a conversation sequence for Alice
    print("Simulating conversation sequence for Alice:")
    
    # Turn 1
    history.add_turn(
        "Alice", 
        "What should I do? I'm hungry and low on energy.",
        "I choose to eat food to restore my hunger.",
        "eat_food",
        "Hunger reduced from 8/10 to 4/10"
    )
    print("Turn 1: Alice chooses to eat food")
    
    # Turn 2  
    history.add_turn(
        "Alice",
        "I feel better now, what's next?", 
        "I choose to work to earn money.",
        "work",
        "Earned 15 coins, energy reduced to 6/10"
    )
    print("Turn 2: Alice chooses to work")
    
    # Turn 3
    history.add_turn(
        "Alice",
        "I'm getting tired, should I continue working?",
        "I choose to sleep to restore energy.",
        "sleep", 
        "Energy restored to 9/10, feeling refreshed"
    )
    print("Turn 3: Alice chooses to sleep")
    
    # Show formatted context
    print("\nFormatted conversation context for next prompt:")
    print("-" * 40)
    context = history.format_context_for_prompt("Alice")
    print(context)
    
    # Show how this preserves context across decisions
    recent_context = history.get_recent_context("Alice", 2)
    print(f"Alice's last 2 decisions: {[turn.action_taken for turn in recent_context]}")


def demo_few_shot_examples():
    """Demonstrate few-shot learning examples."""
    print("\n" + "=" * 60)
    print("2. FEW-SHOT LEARNING EXAMPLES") 
    print("=" * 60)
    
    manager = FewShotExampleManager()
    
    print("Adding custom example from character experience:")
    # Add a new example from character experience
    manager.add_example(FewShotExample(
        situation_context="Character was stressed (mental_health: 3/10) and needed relaxation",
        character_state={"mental_health": 3, "energy": 7, "social_wellbeing": 4},
        decision_made="pursue_hobby",
        outcome="Mental health improved to 7/10, felt more creative and relaxed",
        success_rating=0.9
    ))
    
    print("Current state: High stress, need relaxation")
    current_state = {"mental_health": 4, "energy": 8, "social_wellbeing": 3}
    
    # Get relevant examples
    relevant_examples = manager.get_relevant_examples(current_state, max_examples=2)
    
    print(f"\nFound {len(relevant_examples)} relevant examples:")
    formatted_examples = manager.format_examples_for_prompt(relevant_examples)
    print("-" * 40)
    print(formatted_examples)
    
    print("These examples would be included in the prompt to guide the LLM's decision.")


def demo_structured_output():
    """Demonstrate structured output formatting."""
    print("\n" + "=" * 60)
    print("3. STRUCTURED OUTPUT FORMATTING")
    print("=" * 60)
    
    print("Decision Schema (JSON format):")
    print("-" * 30)
    print(OutputSchema.get_decision_schema())
    
    print("\nRoutine Schema (Markdown format):")
    print("-" * 30)
    print(OutputSchema.get_routine_schema())
    
    print("\nCrisis Schema (Structured format):")
    print("-" * 30)
    print(OutputSchema.get_crisis_schema())
    
    print("\nThese schemas ensure consistent, parseable responses from the LLM.")


def demo_character_voice_consistency():
    """Demonstrate character voice consistency features."""
    print("\n" + "=" * 60)
    print("4. CHARACTER VOICE CONSISTENCY")
    print("=" * 60)
    
    # Simulate character voice descriptors for different jobs
    voice_examples = {
        "Engineer": {
            "speech_pattern": "analytical and precise",
            "decision_approach": "methodical and logical", 
            "personality_tone": "problem-solving oriented",
            "sample_voice": "I approach this systematically by analyzing the data and considering the most efficient solution."
        },
        "Farmer": {
            "speech_pattern": "practical and down-to-earth",
            "decision_approach": "grounded and patient",
            "personality_tone": "connected to nature", 
            "sample_voice": "Let me think about this practically - what's the most sensible approach for the long term?"
        },
        "Waitress": {
            "speech_pattern": "friendly and service-oriented",
            "decision_approach": "people-focused and empathetic",
            "personality_tone": "socially aware",
            "sample_voice": "I want to make sure everyone is taken care of - how can I help others while meeting my own needs?"
        }
    }
    
    for job, traits in voice_examples.items():
        print(f"\n{job} Character Voice:")
        print(f"  Speech: {traits['speech_pattern']}")
        print(f"  Decisions: {traits['decision_approach']}")
        print(f"  Personality: {traits['personality_tone']}")
        print(f"  Sample: \"{traits['sample_voice']}\"")
    
    print("\nThese voice patterns are applied consistently across all prompts for each character.")


def demo_integration_example():
    """Show how all features work together."""
    print("\n" + "=" * 60)
    print("5. INTEGRATED EXAMPLE")
    print("=" * 60)
    
    print("Simulating an enhanced prompt with all features:")
    print("-" * 50)
    
    # Simulate a complete enhanced prompt
    prompt_parts = []
    
    # System prompt with character voice
    prompt_parts.append("<|system|>")
    prompt_parts.append("You are Alice, a methodical and analytical Engineer in a small town.")
    prompt_parts.append("You approach decisions in a systematic and logical manner.")
    prompt_parts.append("Your communication is analytical and precise.")
    
    # Conversation context
    history = ConversationHistory()
    history.add_turn("Alice", "Previous prompt", "I worked to earn money", "work", "Earned 15 coins")
    context = history.format_context_for_prompt("Alice")
    if context:
        prompt_parts.append(context)
    
    # Few-shot examples
    manager = FewShotExampleManager()
    current_state = {"hunger": 6, "energy": 5, "money": 20}
    examples = manager.get_relevant_examples(current_state, max_examples=1)
    if examples:
        examples_text = manager.format_examples_for_prompt(examples)
        prompt_parts.append(examples_text)
    
    # Current situation
    prompt_parts.append("<|user|>")
    prompt_parts.append("Alice, it's afternoon and the weather is sunny.")
    prompt_parts.append("Current state: Health 8/10, Hunger 6/10, Energy 5/10")
    prompt_parts.append("What do you choose to do next?")
    
    # Output schema
    prompt_parts.append(OutputSchema.get_decision_schema())
    
    # Response start
    prompt_parts.append("<|assistant|>")
    prompt_parts.append("Alice, I choose")
    
    # Show the complete prompt
    full_prompt = "\n".join(prompt_parts)
    print(full_prompt)
    
    print("\n" + "=" * 60)
    print("This prompt now includes:")
    print("✓ Character voice consistency")
    print("✓ Conversation history context") 
    print("✓ Relevant few-shot examples")
    print("✓ Structured output format")
    print("✓ Enhanced character understanding")


def main():
    """Run all demonstrations."""
    print("LLM Integration Features Demo")
    print("Showcasing enhanced tiny_prompt_builder.py capabilities")
    
    demo_conversation_history()
    demo_few_shot_examples()
    demo_structured_output()
    demo_character_voice_consistency()
    demo_integration_example()
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print("The enhanced PromptBuilder now supports:")
    print("1. Multi-turn conversation management for context continuity")
    print("2. Few-shot learning examples for better decision guidance") 
    print("3. Structured output formatting for consistent LLM responses")
    print("4. Character voice consistency for immersive roleplay")
    print("\nAll features work together to create richer, more contextual")
    print("prompts that lead to better LLM performance and character behavior.")


if __name__ == "__main__":
    main()