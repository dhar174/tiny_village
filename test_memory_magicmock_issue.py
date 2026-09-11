#!/usr/bin/env python3
"""
Test to demonstrate the memory MagicMock antipattern issue.

This test shows the problem with using MagicMock for memory objects
in PromptBuilder tests, and the solution using real memory objects.
"""

import unittest
from unittest.mock import MagicMock
import sys
import os

# Add the project root to the path so we can import modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


class TestMemoryMagicMockIssue(unittest.TestCase):
    """Demonstrates the MagicMock antipattern for memory testing."""

    def test_magicmock_antipattern_demonstration(self):
        """Demonstrates why MagicMock is problematic for memory testing."""
        print("\n=== DEMONSTRATING MAGICMOCK ANTIPATTERN ===")
        
        # This is the PROBLEMATIC pattern mentioned in the issue
        mem1 = MagicMock(description="Met Bob yesterday", importance_score=5)
        mem2 = MagicMock(description="Lost keys at market", importance_score=3)
        
        # The problem: MagicMock allows ANY attribute access, even invalid ones
        fake_attribute = mem1.nonexistent_memory_field  # This should fail but doesn't!
        fake_method = mem1.invalid_memory_method()  # This should fail but doesn't!
        
        print(f"❌ MagicMock allows invalid attribute: {fake_attribute}")
        print(f"❌ MagicMock allows invalid method: {fake_method}")
        
        # Test memory processing like PromptBuilder does
        def format_memory_for_prompt(memory):
            """Simulates how PromptBuilder.generate_decision_prompt formats memories."""
            # This is the actual line from PromptBuilder:
            # desc = getattr(mem, "description", str(mem))
            desc = getattr(memory, "description", str(memory))
            return f"- {desc}"
        
        # This works with MagicMock but doesn't validate real memory structure
        prompt_line1 = format_memory_for_prompt(mem1)
        prompt_line2 = format_memory_for_prompt(mem2)
        
        print(f"✓ Formatted memory 1: {prompt_line1}")
        print(f"✓ Formatted memory 2: {prompt_line2}")
        
        # But the test passes even if the memory processing logic is broken!
        # For example, if PromptBuilder expected an 'importance_score' method:
        def broken_memory_processor(memory):
            """Simulates broken memory processing that expects methods."""
            # This would fail with real memory objects but passes with MagicMock
            return memory.get_importance_level()  # Method that doesn't exist
        
        # MagicMock makes this "pass" even though it shouldn't
        fake_importance = broken_memory_processor(mem1)
        print(f"❌ MagicMock makes broken logic 'pass': {fake_importance}")
        
        # Assertions that demonstrate the problem
        self.assertIsNotNone(fake_attribute)  # Should fail with real objects
        self.assertIsNotNone(fake_method)     # Should fail with real objects  
        self.assertIsNotNone(fake_importance) # Should fail with real objects
        
        print("❌ All problematic assertions passed with MagicMock!")

    def test_real_memory_objects_solution(self):
        """Demonstrates the correct approach using real memory objects."""
        print("\n=== DEMONSTRATING REAL MEMORY SOLUTION ===")
        
        # Create simple memory classes that simulate the real Memory structure
        class TestMemory:
            """Simplified memory class that mimics tiny_memories.Memory."""
            def __init__(self, description):
                self.description = description
                self.creation_time = "2024-01-01T00:00:00"
                self.last_access_time = "2024-01-01T00:00:00"

        class TestSpecificMemory(TestMemory):
            """Simplified specific memory that mimics tiny_memories.SpecificMemory."""
            def __init__(self, description, importance_score):
                super().__init__(description)
                self.importance_score = importance_score

        # Create real memory objects
        mem1 = TestSpecificMemory("Met Bob yesterday", 5)
        mem2 = TestSpecificMemory("Lost keys at market", 3)
        
        # Test memory processing like PromptBuilder does
        def format_memory_for_prompt(memory):
            """Simulates how PromptBuilder.generate_decision_prompt formats memories."""
            desc = getattr(memory, "description", str(memory))
            return f"- {desc}"
        
        # This works correctly with real memory objects
        prompt_line1 = format_memory_for_prompt(mem1)
        prompt_line2 = format_memory_for_prompt(mem2)
        
        print(f"✓ Formatted memory 1: {prompt_line1}")
        print(f"✓ Formatted memory 2: {prompt_line2}")
        
        # Verify the memory objects have the expected attributes
        self.assertEqual(mem1.description, "Met Bob yesterday")
        self.assertEqual(mem1.importance_score, 5)
        self.assertEqual(mem2.description, "Lost keys at market") 
        self.assertEqual(mem2.importance_score, 3)
        
        # Now test that invalid attribute access properly fails
        with self.assertRaises(AttributeError):
            _ = mem1.nonexistent_memory_field
            
        # Test that invalid method calls properly fail  
        with self.assertRaises(AttributeError):
            _ = mem1.invalid_memory_method()
            
        print("✓ Real memory objects properly reject invalid attributes/methods")
        
        # Test more realistic memory processing scenarios
        def get_memory_importance(memory):
            """Gets importance score, handling cases where it might not exist."""
            return getattr(memory, "importance_score", 0)
        
        def process_memory_details(memory):
            """Processes memory with multiple attributes."""
            details = {
                "description": getattr(memory, "description", "Unknown"),
                "importance": getattr(memory, "importance_score", 0),
                "created": getattr(memory, "creation_time", "Unknown")
            }
            return details
        
        # Test with specific memories
        importance1 = get_memory_importance(mem1)
        importance2 = get_memory_importance(mem2)
        details1 = process_memory_details(mem1)
        details2 = process_memory_details(mem2)
        
        self.assertEqual(importance1, 5)
        self.assertEqual(importance2, 3)
        self.assertEqual(details1["description"], "Met Bob yesterday")
        self.assertEqual(details2["importance"], 3)
        
        print(f"✓ Memory 1 importance: {importance1}")
        print(f"✓ Memory 2 details: {details2}")
        
        # Test with general memory (no importance_score)
        general_mem = TestMemory("General memory without importance")
        general_importance = get_memory_importance(general_mem)
        general_details = process_memory_details(general_mem)
        
        self.assertEqual(general_importance, 0)  # Default fallback
        self.assertEqual(general_details["importance"], 0)
        self.assertEqual(general_details["description"], "General memory without importance")
        
        print(f"✓ General memory fallback: {general_details}")
        print("✅ Real memory objects provide accurate testing!")

    def test_promptbuilder_memory_integration_simulation(self):
        """Simulates PromptBuilder memory integration with real objects."""
        print("\n=== TESTING PROMPTBUILDER MEMORY INTEGRATION ===")
        
        # Create memory classes that match the actual structure
        class MockSpecificMemory:
            def __init__(self, description, importance_score):
                self.description = description
                self.importance_score = importance_score
        
        class MockGeneralMemory:
            def __init__(self, description):
                self.description = description
                # Note: GeneralMemory doesn't have importance_score
        
        # Create a mix of memory types
        memories = [
            MockSpecificMemory("Had coffee with Sarah", 4),
            MockGeneralMemory("Visited the market"),
            MockSpecificMemory("Completed work project", 8),
        ]
        
        # Simulate the actual PromptBuilder memory formatting logic
        def format_memories_for_prompt(memories):
            """Simulates the actual PromptBuilder.generate_decision_prompt memory section."""
            if not memories:
                return ""
            
            prompt_section = "\nRecent memories influencing you:\n"
            for mem in memories[:2]:  # PromptBuilder only uses first 2
                desc = getattr(mem, "description", str(mem))
                prompt_section += f"- {desc}\n"
            return prompt_section
        
        # Test the formatting
        memory_prompt = format_memories_for_prompt(memories)
        
        # Verify the correct memories were included
        self.assertIn("Had coffee with Sarah", memory_prompt)
        self.assertIn("Visited the market", memory_prompt)
        self.assertNotIn("Completed work project", memory_prompt)  # Only first 2
        
        print(f"✓ Memory prompt section:\n{memory_prompt}")
        
        # Test edge cases
        empty_prompt = format_memories_for_prompt([])
        self.assertEqual(empty_prompt, "")
        
        none_prompt = format_memories_for_prompt(None)
        self.assertEqual(none_prompt, "")
        
        # Test with broken memory object (missing description)
        class BrokenMemory:
            pass
        
        broken_memories = [BrokenMemory()]
        broken_prompt = format_memories_for_prompt(broken_memories)
        
        # Should fall back to str(mem) when description is missing
        self.assertIn("BrokenMemory", broken_prompt)
        
        print(f"✓ Broken memory fallback: {broken_prompt}")
        print("✅ PromptBuilder memory integration simulation passed!")


if __name__ == "__main__":
    print("Memory MagicMock Antipattern Issue Demonstration")
    print("=" * 55)
    print("This test demonstrates the problem with using MagicMock")
    print("for memory objects in PromptBuilder tests, and shows the")
    print("solution using real memory object structures.")
    print("=" * 55)
    
    unittest.main(verbosity=2)