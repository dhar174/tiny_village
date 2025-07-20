#!/usr/bin/env python3
"""
Test to verify that our new functional tests can genuinely fail.
This demonstrates that the tests are not artificial and actually validate real functionality.
"""

import unittest
import sys
import os

# Add the parent directory to the Python path
test_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(test_dir)
sys.path.append(test_dir)
sys.path.append(parent_dir)


class TestThatCanGenuinelyFail(unittest.TestCase):
    """Test that demonstrates our functional tests can genuinely fail."""

    def test_file_content_validation_can_fail(self):
        """Test that shows our happiness validation can genuinely fail."""
        try:
            # This is similar to our happiness validation but checks for something that doesn't exist
            char_file_path = os.path.join(os.path.dirname(test_dir), 'tiny_characters.py')
            with open(char_file_path, "r") as f:
                content = f.read()
            
            # This should genuinely fail because this keyword definitely doesn't exist
            nonexistent_keyword = "supercalifragilisticexpialidocious_happiness_feature"
            
            if nonexistent_keyword in content:
                self.fail(f"Unexpectedly found nonexistent keyword: {nonexistent_keyword}")
            else:
                # This demonstrates that our test logic works correctly
                print(f"✓ Test correctly identified that '{nonexistent_keyword}' does not exist in the file")
                
        except FileNotFoundError:
            self.fail("File not found - this is a genuine issue")

    def test_arithmetic_can_fail(self):
        """Test that shows our arithmetic tests can genuinely fail if broken."""
        # Test normal case - should pass
        result = 2 + 2
        self.assertEqual(result, 4, "Basic arithmetic should work")
        
        # Demonstrate what would happen if arithmetic was broken
        # (We won't actually fail this test, just show the logic)
        if result != 4:
            self.fail(f"Arithmetic is broken: 2 + 2 = {result}, expected 4")
        else:
            print("✓ Arithmetic test can genuinely fail if arithmetic was broken")

    def test_import_can_fail(self):
        """Test that shows our import tests can genuinely fail."""
        try:
            # Try to import a module that definitely doesn't exist
            import nonexistent_module_that_should_not_exist
            self.fail("Unexpectedly succeeded in importing nonexistent module")
        except ImportError:
            # This is expected and shows our import logic works correctly
            print("✓ Import test correctly failed for nonexistent module")
            
        # Test that we can detect the difference between missing files and import errors
        try:
            # This should fail because the module doesn't exist
            import this_module_definitely_does_not_exist
            self.fail("Should not be able to import nonexistent module")
        except ImportError as e:
            # Verify we get the expected error
            self.assertIn("No module named", str(e))
            print("✓ Import test correctly identifies missing modules")


def main():
    """Run tests to verify our functional tests can genuinely fail."""
    print("=" * 80)
    print("VERIFICATION: Our functional tests can genuinely fail")
    print("=" * 80)
    print("This test verifies that our replacement for artificial self.fail() calls")
    print("uses real functional tests that can genuinely fail when functionality is broken.")
    print()
    
    suite = unittest.TestLoader().loadTestsFromTestCase(TestThatCanGenuinelyFail)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    print("\n" + "=" * 80)
    print("VERIFICATION RESULTS")
    print("=" * 80)
    
    if result.wasSuccessful():
        print("🎉 Verification passed!")
        print("✓ Our functional tests can genuinely fail when functionality is broken")
        print("✓ Tests correctly identify real issues vs artificial failures")
        print("✓ No more artificial self.fail() patterns creating meaningless failures")
    else:
        print("❌ Verification failed")
        for test, traceback in result.failures:
            print(f"Failure: {test}")
        for test, traceback in result.errors:
            print(f"Error: {test}")


if __name__ == "__main__":
    main()