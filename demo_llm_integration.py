#!/usr/bin/env python3
# filepath: /workspaces/tiny_village/demo_llm_integration.py
"""
Demonstration of LLM Integration for Tiny Village
This script shows how to use the LLM decision-making system.
"""


class DemoCharacter:
    """Demo character class for testing LLM integration"""

    def __init__(self, name):
        self.name = name
        self.id = name
        self.hunger_level = 6.0  # 0-10 scale
        self.energy = 4.0  # 0-10 scale
        self.wealth_money = 25.0
        self.social_wellbeing = 7.0
        self.mental_health = 6.0
        self.use_llm_decisions = False

        # Mock location and inventory
        self.location = type("Location", (), {"name": "Home"})()
        self.inventory = type(
            "Inventory",
            (),
            {
                "get_food_items": lambda: [
                    type("Food", (), {"name": "apple", "calories": 80})(),
                    type("Food", (), {"name": "bread", "calories": 120})(),
                ]
            },
        )()
        self.job = "farmer"


def demo_llm_character_setup():
    """Demonstrate how to set up characters for LLM decision-making"""
    print("🧠 LLM Integration Demo")
    print("=" * 50)

    # Create demo characters
    characters = [
        DemoCharacter("Alice"),
        DemoCharacter("Bob"),
        DemoCharacter("Charlie"),
    ]

    print("1. Initial Character Setup:")
    for char in characters:
        print(f"   {char.name}: LLM decisions = {char.use_llm_decisions}")

    # Enable LLM for specific characters
    print("\n2. Enabling LLM for Alice and Bob...")
    characters[0].use_llm_decisions = True  # Alice
    characters[1].use_llm_decisions = True  # Bob
    # Charlie remains with utility-based decisions

    for char in characters:
        status = "✅ LLM" if char.use_llm_decisions else "⚡ Utility"
        print(f"   {char.name}: {status}")

    return characters


def demo_strategy_manager_setup():
    """Demonstrate StrategyManager setup for LLM"""
    print("\n3. StrategyManager Configuration:")

    # Simulate the StrategyManager setup (mock since we can't import due to dependencies)
    class MockStrategyManager:
        def __init__(self, use_llm=False, model_name=None):
            self.use_llm = use_llm
            self.model_name = model_name
            if use_llm:
                print(f"   🔧 Initializing LLM components with model: {model_name}")
                self.brain_io = f"MockBrainIO({model_name})"
                self.output_interpreter = "MockOutputInterpreter()"
            else:
                self.brain_io = None
                self.output_interpreter = None

    # Create utility-only manager
    utility_manager = MockStrategyManager(use_llm=False)
    print(f"   Utility Manager: LLM={utility_manager.use_llm}")

    # Create LLM-enabled manager
    llm_manager = MockStrategyManager(use_llm=True, model_name="TinyLlama-1.1B")
    print(f"   LLM Manager: LLM={llm_manager.use_llm}, Model={llm_manager.model_name}")
    print(
        f"   Components: brain_io={llm_manager.brain_io is not None}, interpreter={llm_manager.output_interpreter is not None}"
    )

    return utility_manager, llm_manager


def demo_decision_pipeline():
    """Demonstrate the LLM decision-making pipeline"""
    print("\n4. Decision-Making Pipeline Demo:")

    # Mock the decision pipeline components
    class MockPipeline:
        @staticmethod
        def utility_based_actions(character):
            """Simulate utility-based action generation"""
            actions = []
            if character.hunger_level > 5:
                actions.append("Eat apple (hunger relief)")
            if character.energy < 5:
                actions.append("Sleep (energy restoration)")
            if hasattr(character, "job") and character.job:
                actions.append(f"Work as {character.job} (income)")
            actions.append("Wander around (exploration)")
            return actions

        @staticmethod
        def llm_decision(character, time, weather):
            """Simulate LLM decision-making"""
            context = f"Character: {character.name}, Hunger: {character.hunger_level}/10, Energy: {character.energy}/10"
            prompt = f"Given {context}, time: {time}, weather: {weather}, what should {character.name} do?"

            # Mock LLM response based on character state
            if character.hunger_level > 7:
                response = f"I choose to eat because {character.name} is very hungry (hunger: {character.hunger_level}/10)"
                action = "Eat apple"
            elif character.energy < 3:
                response = f"I choose to sleep because {character.name} needs energy (energy: {character.energy}/10)"
                action = "Sleep"
            else:
                response = f"I choose to work because {character.name} has decent energy and needs income"
                action = f"Work as {character.job}"

            return response, action

    # Demo characters with different states
    alice = DemoCharacter("Alice")
    alice.hunger_level = 8.0  # Very hungry
    alice.energy = 6.0
    alice.use_llm_decisions = True

    bob = DemoCharacter("Bob")
    bob.hunger_level = 3.0
    bob.energy = 2.0  # Very tired
    bob.use_llm_decisions = True

    charlie = DemoCharacter("Charlie")
    charlie.hunger_level = 4.0
    charlie.energy = 7.0  # Good state
    charlie.use_llm_decisions = False  # Uses utility-based

    characters = [alice, bob, charlie]
    time = "morning"
    weather = "sunny"

    print(f"   Scenario: {time}, {weather}")
    print()

    for char in characters:
        print(
            f"   {char.name} (Hunger: {char.hunger_level}/10, Energy: {char.energy}/10):"
        )

        if char.use_llm_decisions:
            # Simulate LLM decision-making
            print("     🧠 Using LLM Decision-Making:")
            response, action = MockPipeline.llm_decision(char, time, weather)
            print(f'       LLM Response: "{response}"')
            print(f"       Selected Action: {action}")
        else:
            # Simulate utility-based decision-making
            print("     ⚡ Using Utility-Based Decision-Making:")
            actions = MockPipeline.utility_based_actions(char)
            print(f"       Available Actions: {actions}")
            print(f"       Selected Action: {actions[0] if actions else 'NoOp'}")
        print()


def demo_gameplay_integration():
    """Demonstrate integration with GameplayController"""
    print("5. GameplayController Integration:")

    def mock_execute_character_actions(characters, time, weather):
        """Mock the GameplayController character action execution"""
        results = []
        for char in characters:
            if hasattr(char, "use_llm_decisions") and char.use_llm_decisions:
                decision_type = "🧠 LLM"
                # Would call: strategy_manager.decide_action_with_llm(char, time, weather)
            else:
                decision_type = "⚡ Utility"
                # Would call: strategy_manager.get_daily_actions(char)

            results.append(f"   {char.name}: {decision_type} decision at {time}")
        return results

    characters = demo_llm_character_setup()
    results = mock_execute_character_actions(characters, "evening", "rainy")

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

    alice = DemoCharacter("Alice")
    alice.use_llm_decisions = True

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
