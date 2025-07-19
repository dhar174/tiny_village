#!/usr/bin/env python3
"""
Test to verify that torch.rand now returns random values instead of always 0.

This test addresses the issue: "The torch.rand stub always returns 0, which could 
mask issues where randomness is expected. This violates the principle of writing 
tests that fail when functions don't work as expected."
"""

import sys
import os

# Add the project directory to the path
sys.path.insert(0, '/home/runner/work/tiny_village/tiny_village')

from torch import rand, ones, zeros


def test_rand_returns_random_values():
    """Test that torch.rand returns actual random values, not always 0."""
    print("Testing torch.rand behavior...")
    
    # Generate multiple random values to verify they're not all the same
    random_values = []
    for i in range(10):
        val = rand()
        random_values.append(val)
        print(f"rand() call {i+1}: {val}")
    
    # Check that we got different values (not all zeros)
    unique_values = set(random_values)
    print(f"\\nGenerated {len(unique_values)} unique values out of 10 calls")
    
    # Test should pass if we get more than 1 unique value
    assert len(unique_values) > 1, "torch.rand should return different random values, not the same value every time"
    
    # Test that no values are exactly 0 (which was the old problematic behavior)
    zero_count = sum(1 for val in random_values if val == 0.0)
    print(f"Number of exact zeros: {zero_count}")
    
    # We shouldn't get all zeros (the old broken behavior)
    assert zero_count < len(random_values), "torch.rand should not always return 0"
    
    print("✓ torch.rand is returning random values as expected")


def test_rand_tensor_generation():
    """Test that torch.rand generates tensors with random values."""
    print("\\nTesting tensor generation...")
    
    # Test 1D tensor
    tensor_1d = rand(5)
    print(f"1D tensor: {tensor_1d}")
    
    # Test 2D tensor  
    tensor_2d = rand(2, 3)
    print(f"2D tensor: {tensor_2d}")
    
    # Verify tensors have the expected shape
    assert tensor_1d.shape == (5,), f"Expected shape (5,), got {tensor_1d.shape}"
    assert tensor_2d.shape == (2, 3), f"Expected shape (2, 3), got {tensor_2d.shape}"
    
    # Check that tensor values are not all the same (indicating randomness)
    def extract_all_values(tensor_data):
        """Extract all numeric values from nested tensor data."""
        values = []
        if isinstance(tensor_data, (list, tuple)):
            for item in tensor_data:
                if isinstance(item, (list, tuple)):
                    values.extend(extract_all_values(item))
                else:
                    values.append(item)
        else:
            values.append(tensor_data)
        return values
    
    values_1d = extract_all_values(tensor_1d.data)
    values_2d = extract_all_values(tensor_2d.data)
    
    # Check for randomness
    unique_1d = set(values_1d)
    unique_2d = set(values_2d)
    
    print(f"1D tensor unique values: {len(unique_1d)} out of {len(values_1d)}")
    print(f"2D tensor unique values: {len(unique_2d)} out of {len(values_2d)}")
    
    # We should have some variation in values (not all the same)
    assert len(unique_1d) > 1, "1D tensor should have different random values"
    assert len(unique_2d) > 1, "2D tensor should have different random values"
    
    print("✓ Tensor generation is working correctly with random values")


def test_comparison_with_zeros_and_ones():
    """Test that rand generates values different from zeros() and ones()."""
    print("\\nTesting difference from zeros and ones...")
    
    # Generate some random values
    rand_vals = [rand() for _ in range(5)]
    zero_val = zeros()
    one_val = ones()
    
    print(f"Random values: {rand_vals}")
    print(f"Zero value: {zero_val}")
    print(f"One value: {one_val}")
    
    # Random values should not all be 0 (the old broken behavior)
    not_all_zeros = any(val != 0.0 for val in rand_vals)
    assert not_all_zeros, "Random values should not all be zero"
    
    # Random values should not all be 1
    not_all_ones = any(val != 1.0 for val in rand_vals)
    assert not_all_ones, "Random values should not all be one"
    
    print("✓ Random values are properly distinguished from zeros and ones")


if __name__ == "__main__":
    try:
        test_rand_returns_random_values()
        test_rand_tensor_generation()
        test_comparison_with_zeros_and_ones()
        print("\\n🎉 All tests passed! torch.rand is now working correctly.")
        print("\\nThe issue has been resolved:")
        print("- torch.rand no longer always returns 0")
        print("- Random values are generated that will properly test randomness expectations")
        print("- Tests will now fail appropriately when functions don't work as expected")
    except AssertionError as e:
        print(f"\\n❌ Test failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)