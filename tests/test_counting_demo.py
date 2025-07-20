import unittest
import math


def simple_counter(start_value, increment):
    """A simple counting function for demonstration purposes."""
    return start_value + increment


def factorial(n):
    """Calculate factorial of a number."""
    if n < 0:
        raise ValueError("Factorial is not defined for negative numbers")
    if n == 0 or n == 1:
        return 1
    result = 1
    for i in range(2, n + 1):
        result *= i
    return result


class TestCountingDemo(unittest.TestCase):
    """Demonstration tests that actually validate function behavior."""

    def test_method_one(self):
        """First test method - tests simple_counter function behavior."""
        # Test basic counting functionality
        result = simple_counter(5, 3)
        self.assertEqual(result, 8, "simple_counter(5, 3) should return 8")
        
        # Test with negative increment
        result = simple_counter(10, -2)
        self.assertEqual(result, 8, "simple_counter(10, -2) should return 8")
        
        # Test with zero
        result = simple_counter(0, 0)
        self.assertEqual(result, 0, "simple_counter(0, 0) should return 0")

    def test_factorial_calculation(self):
        """Test factorial function with various inputs."""
        # Test basic cases
        self.assertEqual(factorial(0), 1, "factorial(0) should return 1")
        self.assertEqual(factorial(1), 1, "factorial(1) should return 1")
        self.assertEqual(factorial(5), 120, "factorial(5) should return 120")
        
        # Test error case
        with self.assertRaises(ValueError):
            factorial(-1)

    def test_edge_cases(self):
        """Test edge cases that could reveal implementation issues."""
        # Test large increments
        result = simple_counter(1000000, 1000000)
        self.assertEqual(result, 2000000, "Should handle large numbers correctly")
        
        # Test float arithmetic (potential precision issues)
        result = simple_counter(0.1, 0.2)
        self.assertAlmostEqual(result, 0.3, places=10, 
                              msg="Should handle float arithmetic correctly")


if __name__ == '__main__':
    unittest.main()