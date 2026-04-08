"""
Utility functions for managing LLM decision-making in characters
"""


def enable_llm_decisions(character, enable=True):
    """
    Enable or disable LLM decision-making for a character.

    Args:
        character: Character object to modify
        enable: Boolean indicating whether to enable (True) or disable (False) LLM decisions
    """
    character.use_llm_decisions = enable
    return character


def enable_llm_for_characters(characters, character_names=None):
    """
    Enable LLM decision-making for specific characters or all characters.

    Args:
        characters: List of character objects
        character_names: List of character names to enable LLM for. If None, enables for all.

    Returns:
        List of characters with LLM decisions enabled
    """
    enabled_characters = []

    for character in characters:
        if character_names is None or character.name in character_names:
            enable_llm_decisions(character, True)
            enabled_characters.append(character)
        else:
            enable_llm_decisions(character, False)

    return enabled_characters


def get_llm_enabled_characters(characters):
    """
    Get a list of characters that have LLM decision-making enabled.

    Args:
        characters: List of character objects

    Returns:
        List of characters with LLM decisions enabled
    """
    return [
        char
        for char in characters
        if hasattr(char, "use_llm_decisions") and char.use_llm_decisions
    ]


def configure_strategy_manager_for_llm(
    strategy_manager, model_name="alexredna/TinyLlama-1.1B-Chat-v1.0-reasoning-v2"
):
    """
    Configure a StrategyManager instance to use LLM decision-making.

    Args:
        strategy_manager: StrategyManager instance to configure
        model_name: Name of the LLM model to use

    Returns:
        Configured StrategyManager instance
    """
    if not strategy_manager.use_llm:
        # If not already configured for LLM, we need to reinitialize
        strategy_manager.use_llm = True

        # Import here to avoid dependency issues
        try:
            from tiny_brain_io import TinyBrainIO
            from tiny_output_interpreter import OutputInterpreter

            strategy_manager.brain_io = TinyBrainIO(model_name)
            strategy_manager.output_interpreter = OutputInterpreter()

        except ImportError as e:
            print(f"Warning: Could not initialize LLM components: {e}")
            strategy_manager.use_llm = False
            strategy_manager.brain_io = None
            strategy_manager.output_interpreter = None

    return strategy_manager


def create_llm_enabled_strategy_manager(
    model_name="alexredna/TinyLlama-1.1B-Chat-v1.0-reasoning-v2",
):
    """
    Create a new StrategyManager instance with LLM decision-making enabled.

    Args:
        model_name: Name of the LLM model to use

    Returns:
        StrategyManager instance with LLM enabled, or None if dependencies unavailable
    """
    try:
        # Try to import and catch any dependency issues
        import importlib
        strategy_module = importlib.import_module("tiny_strategy_manager")
        StrategyManager = strategy_module.StrategyManager
        return StrategyManager(use_llm=True, model_name=model_name)
    except Exception as e:
        print(f"Note: Could not create LLM-enabled StrategyManager due to dependencies: {type(e).__name__}")
        print("This is expected in demo environments without full ML dependencies.")
        return None


def setup_full_llm_integration(
    characters,
    character_names=None,
    model_name="alexredna/TinyLlama-1.1B-Chat-v1.0-reasoning-v2",
):
    """
    Complete setup for LLM integration including characters and strategy manager.

    Args:
        characters: List of character objects
        character_names: List of character names to enable LLM for. If None, enables for all.
        model_name: Name of the LLM model to use

    Returns:
        Tuple of (enabled_characters, strategy_manager)
    """
    # Enable LLM for specified characters
    enabled_characters = enable_llm_for_characters(characters, character_names)

    # Create LLM-enabled strategy manager
    strategy_manager = create_llm_enabled_strategy_manager(model_name)

    return enabled_characters, strategy_manager


# Example usage functions


def example_basic_llm_setup():
    """
    Example of basic LLM setup for real Character instances.
    This demonstrates actual Character class behavior instead of MockCharacter.
    """
    print("Example: Basic LLM Setup")
    print("-" * 30)

    # Import the real character factory instead of creating MockCharacter
    from demo_character_factory import create_demo_characters

    # Create real Character instances instead of MockCharacter
    characters = create_demo_characters(["Alice", "Bob", "Charlie"])
    
    print("Created REAL Character instances (not MockCharacter):")
    for char in characters:
        state = char.get_state_summary()
        print(f"  {state['name']}: Job={state['job']}, LLM={state['use_llm']}")

    # Enable LLM for specific characters
    enable_llm_for_characters(characters, ["Alice", "Bob"])

    # Show results
    for char in characters:
        state = char.get_state_summary()
        llm_status = "enabled" if state['use_llm'] else "disabled"
        print(f"  {state['name']}: LLM {llm_status}")

    print(
        f"\nLLM-enabled characters: {[char.name for char in get_llm_enabled_characters(characters)]}"
    )
    print("✓ Completed with REAL Character instances, not MockCharacter")


def example_full_integration_setup():
    """
    Example of complete LLM integration setup using real Character instances.
    This replaces MockCharacter usage with actual Character class functionality.
    """
    print("\nExample: Full Integration Setup")
    print("-" * 35)

    # Import the real character factory instead of creating MockCharacter
    from demo_character_factory import create_demo_characters

    # Create real Character instances instead of MockCharacter
    characters = create_demo_characters(["Emma", "David"])
    
    print("Created REAL Character instances:")
    for char in characters:
        state = char.get_state_summary()
        print(f"  {state['name']}: Job={state['job']}, Health={state['health']}")

    # Full setup
    enabled_chars, strategy_manager = setup_full_llm_integration(
        characters,
        character_names=["Emma"],  # Only enable for Emma
        model_name="test-model",
    )

    print(f"Characters with LLM enabled: {[c.name for c in enabled_chars]}")

    if strategy_manager:
        print(f"Strategy Manager LLM status: {strategy_manager.use_llm}")
        print(f"Strategy Manager has brain_io: {strategy_manager.brain_io is not None}")
        print(
            f"Strategy Manager has output_interpreter: {strategy_manager.output_interpreter is not None}"
        )
    else:
        print("Strategy Manager creation failed (expected in demo environment)")
        
    print("✓ Completed with REAL Character instances, demonstrating actual behavior")


if __name__ == "__main__":
    example_basic_llm_setup()
    example_full_integration_setup()
