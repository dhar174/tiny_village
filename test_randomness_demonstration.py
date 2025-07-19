#!/usr/bin/env python3
"""
Demonstration test showing why the old torch.rand behavior (always returning 0) 
was problematic and how the new behavior fixes the issue.
"""

import sys
import os
sys.path.insert(0, '/home/runner/work/tiny_village/tiny_village')

from torch import rand


def demonstrate_randomness_testing_issue():
    """
    This test demonstrates why always returning 0 from rand() was problematic.
    """
    print("Demonstrating why the fix was necessary...")
    print("=" * 60)
    
    # Simulate a function that should behave differently with random inputs
    def function_that_depends_on_randomness():
        """A function that should produce different outputs with random inputs."""
        random_factor = rand()  # This should be different each time
        
        # Some computation that depends on randomness
        if random_factor < 0.5:
            return "low_value_behavior"
        else:
            return "high_value_behavior"
    
    # Test the function multiple times
    results = []
    for i in range(10):
        result = function_that_depends_on_randomness()
        results.append(result)
        print(f"Call {i+1}: {result}")
    
    # Count different behaviors
    unique_behaviors = set(results)
    print(f"\\nUnique behaviors observed: {len(unique_behaviors)}")
    print(f"Behaviors: {list(unique_behaviors)}")
    
    # With the OLD behavior (always returning 0), we would ALWAYS get:
    # - random_factor would always be 0
    # - 0 < 0.5 is True
    # - So we'd always get "low_value_behavior"
    # - This would hide bugs where the randomness affects logic
    
    # With the NEW behavior (actual random values), we should get:
    # - Different random_factor values
    # - Sometimes < 0.5, sometimes >= 0.5
    # - Different behaviors, properly testing the randomness-dependent code
    
    print("\\nAnalysis:")
    if len(unique_behaviors) == 1:
        print("❌ Only one behavior observed - this would indicate the old problematic behavior")
        print("   (All random values were the same, likely all 0)")
    else:
        print("✅ Multiple behaviors observed - randomness is working correctly!")
        print("   This means our tests will properly catch issues in randomness-dependent code")
    
    # The test should pass if we see variation (indicating proper randomness)
    assert len(unique_behaviors) > 1, "Should see different behaviors due to randomness"
    
    return len(unique_behaviors)


def demonstrate_embedding_variance():
    """
    Demonstrate how random embeddings should vary (important for ML testing).
    """
    print("\\n" + "=" * 60)
    print("Demonstrating embedding variance (important for ML code)...")
    
    # Generate multiple "embeddings" like the code does
    embeddings = []
    for i in range(5):
        embedding = rand(1, 768)  # Common pattern from test_tiny_memories.py
        embeddings.append(embedding)
        
        # Extract first few values for comparison
        if hasattr(embedding, 'data') and isinstance(embedding.data, list):
            first_values = embedding.data[0][:3] if isinstance(embedding.data[0], list) else embedding.data[:3]
            print(f"Embedding {i+1} first 3 values: {first_values}")
    
    # Check if embeddings are different (they should be with proper randomness)
    print("\\nAnalysis:")
    print("With the OLD behavior (always 0), all embeddings would be identical")
    print("With the NEW behavior, embeddings should be different")
    
    # For verification, check that embeddings have different values
    all_same = True
    if len(embeddings) > 1:
        first_embedding_data = embeddings[0].data
        for embedding in embeddings[1:]:
            if embedding.data != first_embedding_data:
                all_same = False
                break
    
    if all_same:
        print("❌ All embeddings are identical - this indicates the old problematic behavior")
    else:
        print("✅ Embeddings are different - proper randomness for testing!")
    
    assert not all_same, "Embeddings should be different due to randomness"


if __name__ == "__main__":
    try:
        num_behaviors = demonstrate_randomness_testing_issue()
        demonstrate_embedding_variance()
        
        print("\\n" + "=" * 60)
        print("🎉 DEMONSTRATION COMPLETE")
        print("\\nKey points proven:")
        print("1. ✅ torch.rand now returns actual random values")
        print("2. ✅ Functions depending on randomness show varied behavior")
        print("3. ✅ Mock embeddings are properly varied for realistic testing")
        print("4. ✅ Tests will now fail appropriately when randomness-dependent code breaks")
        print("\\nThe original issue has been successfully resolved!")
        
    except AssertionError as e:
        print(f"\\n❌ Demonstration failed: {e}")
        print("This would indicate the torch.rand stub is still problematic")
        sys.exit(1)
    except Exception as e:
        print(f"\\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)