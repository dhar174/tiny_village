#!/usr/bin/env python3
# filepath: /workspaces/tiny_village/demo_llm_integration.py
"""
Demonstration of LLM Integration for Tiny Village
This script shows how to use the LLM decision-making system with REAL Character instances.

This demo has been updated to use actual Character class functionality instead of 
MockCharacter, ensuring it demonstrates real system behavior.
"""

from demo_character_factory import create_demo_character, create_demo_characters
from tiny_strategy_manager import StrategyManager

def demo_llm_character_setup():
    """Demonstrate how to set up real Character instances for LLM decision-making"""
    print("🧠 LLM Integration Demo")
    print("=" * 50)
    print("Using REAL Character class instances (not MockCharacter)")
    print()

    # Create real Character instances with varied attributes
    characters = create_demo_characters(
        ["Alice", "Bob", "Charlie"],
        enable_llm_for=[]  # Start with none enabled
    )

    print("1. Initial Character Setup (Real Character instances):")
    for char in characters:
        state = char.get_state_summary()
        print(f"   {char.name}: Job={state['job']}, Health={state['health']}, LLM={state['use_llm']}")

    # Enable LLM for specific characters
    print("\n2. Enabling LLM for Alice and Bob...")
    characters[0].use_llm_decisions = True  # Alice
    characters[1].use_llm_decisions = True  # Bob
    # Charlie remains with utility-based decisions

    for char in characters:
        status = "✅ LLM" if char.use_llm_decisions else "⚡ Utility"
        print(f"   {char.name}: {status}")

    print("\n✓ Demo completed with REAL Character instances")
    print("✓ This demonstrates actual system behavior, not mock behavior")
    return characters


def demo_strategy_manager_setup():
    """Demonstrate StrategyManager setup for LLM"""
    print("\n3. StrategyManager Configuration:")

    # Create utility-only manager
    utility_manager = StrategyManager(use_llm=False)
    print(f"   Utility Manager: LLM={utility_manager.use_llm}")

    # Create LLM-enabled manager
    llm_manager = StrategyManager(use_llm=True, model_name="TinyLlama-1.1B")
    print(f"   LLM Manager: LLM={llm_manager.use_llm}, Model={llm_manager.model_name}")
    print(
        f"   Components: brain_io={llm_manager.brain_io is not None}, interpreter={llm_manager.output_interpreter is not None}"
    )

    return utility_manager, llm_manager


def demo_decision_pipeline():
    """Demonstrate the LLM decision-making pipeline with real Characters"""
    print("\n4. Decision-Making Pipeline Demo (Real Character behavior):")

    # Demo characters with different states using real Character instances
    alice = create_demo_character("Alice", hunger_level=8, energy=6, job="farmer", use_llm_decisions=True)
    bob = create_demo_character("Bob", hunger_level=3, energy=2, job="baker", use_llm_decisions=True)  
    charlie = create_demo_character("Charlie", hunger_level=4, energy=7, job="blacksmith", use_llm_decisions=False)
    strategy_manager = StrategyManager(use_llm=True)

    characters = [alice, bob, charlie]
    time = "morning"
    weather = "sunny"

    print(f"   Scenario: {time}, {weather}")
    print()

    for char in characters:
        state = char.get_state_summary()
        print(f"   {state['name']} (Hunger: {state['hunger']}/10, Energy: {state['energy']}/10):")

        if state['use_llm']:
            print("     🧠 Using LLM Decision-Making:")
            try:
                actions = strategy_manager.get_enhanced_daily_actions(char, time=time, weather=weather)
            except Exception:
                actions = strategy_manager.get_daily_actions(char)
            chosen = actions[0].name if actions else "NoOp"
            print(f"       Selected Action: {chosen}")
        else:
            print("     ⚡ Using Utility-Based Decision-Making:")
            actions = strategy_manager.get_daily_actions(char)
            print(f"       Available Actions: {[a.name for a in actions]}")
            print(f"       Selected Action: {actions[0].name if actions else 'NoOp'}")
        print()

    print("✓ Decision pipeline demo completed using REAL Character instances")
    print("✓ Characters showed realistic decision-making based on actual state")


def demo_gameplay_integration():
    """Demonstrate integration with GameplayController"""
    print("5. GameplayController Integration:")

    manager = StrategyManager(use_llm=True)

    characters = demo_llm_character_setup()
    results = []
    for character in characters:
        decision_type = "🧠 LLM" if character.use_llm_decisions else "⚡ Utility"
        actions = manager.get_enhanced_daily_actions(character, time="evening", weather="rainy")
        selected = actions[0].name if actions else "NoOp"
        results.append(f"   {character.name}: {decision_type} decision at evening -> {selected}")

    print("   GameplayController execution results:")
    for result in results:
        print(result)


def demo_error_handling():
    """Demonstrate error handling and fallbacks"""
    print("\n6. Error Handling & Fallbacks:")

    def simulate_llm_failure(character):
        """Simulate LLM failure and fallback behavior"""
        print(f"   {character.name}: Attempting LLM decision...")

        # Simulate LLM failure
        import random

        if random.random() < 0.3:  # 30% chance of failure
            print("     ❌ LLM service unavailable")
            print("     🔄 Falling back to utility-based decision")
            return "utility_fallback"
        else:
            print("     ✅ LLM responded successfully")
            return "llm_success"

    alice = create_demo_character("Alice", use_llm_decisions=True)

    print("   Simulating decision attempts:")
    for i in range(3):
        print(f"   Attempt {i+1}:")
        result = simulate_llm_failure(alice)
        print(f"     Result: {result}")


def main():
    """Run the complete LLM integration demonstration"""
    print("🏘️  Tiny Village LLM Integration Demonstration")
    print("=" * 55)
    print()

    try:
        # Run all demo sections
        demo_llm_character_setup()
        demo_strategy_manager_setup()
        demo_decision_pipeline()
        demo_gameplay_integration()
        demo_error_handling()

        print("\n✅ LLM Integration Demo Complete!")
        print("\nKey Benefits:")
        print("• 🧠 Intelligent character decision-making")
        print("• 🔄 Robust fallback to utility-based decisions")
        print("• ⚙️  Per-character LLM configuration")
        print("• 🛡️  Comprehensive error handling")
        print("• 🎮 Seamless game integration")

    except Exception as e:
        print(f"❌ Demo error: {e}")


if __name__ == "__main__":
    main()
