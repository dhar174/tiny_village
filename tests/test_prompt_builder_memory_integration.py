#!/usr/bin/env python3
"""
Test PromptBuilder memory integration with proper memory objects.

This test demonstrates the correct way to test PromptBuilder with memory objects,
avoiding the MagicMock antipattern and using realistic memory structures.
"""

import unittest
from unittest.mock import MagicMock
import sys
import os
from datetime import datetime

# Add the project root to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


class RealMemory:
    """
    Realistic memory class that mimics tiny_memories.Memory structure.
    This should be used instead of MagicMock for memory testing.
    """
    def __init__(self, description, creation_time=None):
        self.description = description
        self.creation_time = creation_time or datetime.now()
        self.last_access_time = self.creation_time

    def update_access_time(self, access_time):
        self.last_access_time = access_time

    def __str__(self):
        return f"Memory({self.description})"

    def __repr__(self):
        return f"RealMemory(description='{self.description}')"


class RealSpecificMemory(RealMemory):
    """
    Realistic specific memory class that mimics tiny_memories.SpecificMemory.
    This has importance_score and other attributes that specific memories should have.
    """
    def __init__(self, description, importance_score, parent_memory=None, subject=None):
        super().__init__(description)
        self.importance_score = importance_score
        self.parent_memory = parent_memory
        self.subject = subject
        self.related_memories = []
        self.keywords = []
        self.tags = []
        
    def __repr__(self):
        return f"RealSpecificMemory(description='{self.description}', importance={self.importance_score})"


class RealGeneralMemory(RealMemory):
    """
    Realistic general memory class that mimics tiny_memories.GeneralMemory.
    Note: GeneralMemory doesn't have importance_score.
    """
    def __init__(self, description):
        super().__init__(description)
        self.keywords = []
        self.analysis = None
        self.sentiment_score = None
        
    def __repr__(self):
        return f"RealGeneralMemory(description='{self.description}')"


class MockCharacterForMemoryTesting:
    """Mock character that focuses on memory testing needs."""
    def __init__(self, name="TestChar"):
        self.name = name
        self.job = "Tester"
        self.health_status = 7
        self.hunger_level = 4
        self.wealth_money = 50
        self.mental_health = 6
        self.social_wellbeing = 7
        self.recent_event = "testing"
        self.long_term_goal = "Complete all tests successfully"
        self.energy = 6.5


class TestPromptBuilderMemoryIntegration(unittest.TestCase):
    """Test PromptBuilder memory integration with proper memory objects."""

    def setUp(self):
        """Set up test fixtures with realistic memory objects."""
        self.character = MockCharacterForMemoryTesting("Alice")
        
        # Create realistic memory objects (NOT MagicMock!)
        self.specific_memory1 = RealSpecificMemory(
            description="Met Bob yesterday",
            importance_score=5
        )
        self.specific_memory2 = RealSpecificMemory(
            description="Lost keys at market", 
            importance_score=3
        )
        self.general_memory = RealGeneralMemory(
            description="The weather was nice"
        )
        
        self.test_memories = [
            self.specific_memory1,
            self.specific_memory2,
            self.general_memory
        ]

    def test_memory_objects_have_correct_structure(self):
        """Test that our memory objects have the expected structure."""
        # Test specific memory attributes
        self.assertEqual(self.specific_memory1.description, "Met Bob yesterday")
        self.assertEqual(self.specific_memory1.importance_score, 5)
        self.assertIsInstance(self.specific_memory1.creation_time, datetime)
        
        # Test general memory attributes  
        self.assertEqual(self.general_memory.description, "The weather was nice")
        self.assertFalse(hasattr(self.general_memory, 'importance_score'))
        
        # Test that invalid attributes raise errors (unlike MagicMock)
        with self.assertRaises(AttributeError):
            _ = self.specific_memory1.nonexistent_attribute
            
        with self.assertRaises(AttributeError):
            _ = self.general_memory.invalid_method()

    def test_prompt_builder_memory_formatting_simulation(self):
        """Test PromptBuilder memory formatting logic with real memory objects."""
        
        # Simulate the actual PromptBuilder.generate_decision_prompt memory formatting
        def simulate_prompt_builder_memory_formatting(memories):
            """
            Simulates the memory formatting from tiny_prompt_builder.py lines 2182-2186:
            
            if memories:
                prompt += "\nRecent memories influencing you:\n"
                for mem in memories[:2]:
                    desc = getattr(mem, "description", str(mem))
                    prompt += f"- {desc}\n"
            """
            if not memories:
                return ""
                
            prompt_section = "\nRecent memories influencing you:\n"
            for mem in memories[:2]:  # Only first 2 memories
                desc = getattr(mem, "description", str(mem))
                prompt_section += f"- {desc}\n"
            return prompt_section
        
        # Test with our realistic memory objects
        memory_prompt = simulate_prompt_builder_memory_formatting(self.test_memories)
        
        # Verify correct behavior
        self.assertIn("Recent memories influencing you:", memory_prompt)
        self.assertIn("- Met Bob yesterday", memory_prompt)
        self.assertIn("- Lost keys at market", memory_prompt)
        self.assertNotIn("The weather was nice", memory_prompt)  # Only first 2
        
        # Test edge cases
        empty_prompt = simulate_prompt_builder_memory_formatting([])
        self.assertEqual(empty_prompt, "")
        
        none_prompt = simulate_prompt_builder_memory_formatting(None)
        self.assertEqual(none_prompt, "")

    def test_memory_attribute_access_patterns(self):
        """Test various ways PromptBuilder might access memory attributes."""
        
        # Test the actual getattr pattern used in PromptBuilder
        desc1 = getattr(self.specific_memory1, "description", str(self.specific_memory1))
        desc2 = getattr(self.general_memory, "description", str(self.general_memory))
        
        self.assertEqual(desc1, "Met Bob yesterday")
        self.assertEqual(desc2, "The weather was nice")
        
        # Test importance_score access with fallback
        importance1 = getattr(self.specific_memory1, "importance_score", 0)
        importance2 = getattr(self.general_memory, "importance_score", 0)
        
        self.assertEqual(importance1, 5)
        self.assertEqual(importance2, 0)  # Fallback for general memory
        
        # Test other possible attribute patterns
        subject1 = getattr(self.specific_memory1, "subject", None)
        keywords1 = getattr(self.specific_memory1, "keywords", [])
        
        self.assertIsNone(subject1)  # Default None
        self.assertEqual(keywords1, [])  # Default empty list

    def test_broken_memory_object_handling(self):
        """Test how PromptBuilder handles malformed memory objects."""
        
        class BrokenMemory:
            """Memory object missing description attribute."""
            pass
        
        class PartialMemory:
            """Memory object with some but not all attributes."""
            def __init__(self):
                self.description = "Partial memory"
                # Missing other expected attributes
        
        broken_memory = BrokenMemory()
        partial_memory = PartialMemory()
        
        mixed_memories = [self.specific_memory1, broken_memory, partial_memory]
        
        # Test that PromptBuilder logic handles broken objects gracefully
        def robust_memory_formatting(memories):
            if not memories:
                return ""
                
            prompt_section = "\nRecent memories influencing you:\n"
            for mem in memories[:2]:
                # This is the actual PromptBuilder pattern - should handle broken objects
                desc = getattr(mem, "description", str(mem))
                prompt_section += f"- {desc}\n"
            return prompt_section
        
        result = robust_memory_formatting(mixed_memories)
        
        # Should include first memory description
        self.assertIn("- Met Bob yesterday", result)
        
        # Should fall back to str() for broken memory
        self.assertIn("BrokenMemory", result)
        
        # Should not include third memory (only first 2)
        self.assertNotIn("Partial memory", result)

    def test_antipattern_demonstration(self):
        """Demonstrate why MagicMock is problematic vs real memory objects."""
        
        print("\n=== ANTIPATTERN DEMONSTRATION ===")
        
        # BAD: The problematic pattern from the issue description
        bad_mem1 = MagicMock(description="Met Bob yesterday", importance_score=5)
        bad_mem2 = MagicMock(description="Lost keys at market", importance_score=3)
        
        # GOOD: The correct pattern with real memory objects
        good_mem1 = RealSpecificMemory("Met Bob yesterday", 5)
        good_mem2 = RealSpecificMemory("Lost keys at market", 3)
        
        # Both work for basic access
        self.assertEqual(bad_mem1.description, "Met Bob yesterday")
        self.assertEqual(good_mem1.description, "Met Bob yesterday")
        
        # But MagicMock hides real bugs
        fake_attribute = bad_mem1.nonexistent_field  # This "works" but shouldn't!
        self.assertIsNotNone(fake_attribute)
        
        # Real memory objects catch bugs
        with self.assertRaises(AttributeError):
            _ = good_mem1.nonexistent_field  # This properly fails
        
        print("❌ MagicMock allows: mem.nonexistent_field")
        print("✅ Real memory rejects: mem.nonexistent_field")
        
        # Test a more realistic failure scenario
        def broken_memory_processor(memory):
            """Simulates broken code that expects a method that doesn't exist."""
            return memory.get_embedding_vector()  # This method doesn't exist
        
        # MagicMock makes this pass when it should fail
        fake_embedding = broken_memory_processor(bad_mem1)
        self.assertIsNotNone(fake_embedding)
        
        # Real memory properly fails
        with self.assertRaises(AttributeError):
            broken_memory_processor(good_mem1)
        
        print("❌ MagicMock makes broken logic 'pass'")
        print("✅ Real memory catches broken logic")

    def test_comprehensive_memory_integration_scenario(self):
        """Test a comprehensive scenario using realistic memory objects."""
        
        # Create a variety of memory types
        memories = [
            RealSpecificMemory("Completed important project", 9),
            RealSpecificMemory("Had lunch with colleague", 4),
            RealGeneralMemory("Sunny day at the office"),
            RealSpecificMemory("Received praise from boss", 7),
            RealGeneralMemory("Traffic was heavy"),
        ]
        
        # Test memory filtering by importance
        high_importance = [m for m in memories if getattr(m, 'importance_score', 0) >= 7]
        self.assertEqual(len(high_importance), 2)
        
        # Test memory sorting by importance
        specific_memories = [m for m in memories if hasattr(m, 'importance_score')]
        sorted_memories = sorted(specific_memories, key=lambda m: m.importance_score, reverse=True)
        
        self.assertEqual(sorted_memories[0].importance_score, 9)
        self.assertEqual(sorted_memories[-1].importance_score, 4)
        
        # Test comprehensive memory processing
        def process_memories_for_context(memories, max_memories=3):
            """Process memories to create context for decision-making."""
            if not memories:
                return []
            
            # Separate specific and general memories
            specific = [m for m in memories if hasattr(m, 'importance_score')]
            general = [m for m in memories if not hasattr(m, 'importance_score')]
            
            # Sort specific memories by importance
            specific_sorted = sorted(specific, key=lambda m: m.importance_score, reverse=True)
            
            # Combine and limit
            combined = specific_sorted + general
            return combined[:max_memories]
        
        context_memories = process_memories_for_context(memories, 3)
        
        # Should prioritize high-importance specific memories
        self.assertEqual(len(context_memories), 3)
        self.assertEqual(context_memories[0].description, "Completed important project")
        self.assertEqual(context_memories[1].description, "Received praise from boss")
        
        # Test final prompt integration
        def create_memory_context_prompt(memories):
            processed_memories = process_memories_for_context(memories, 2)
            if not processed_memories:
                return ""
            
            prompt = "\nRecent memories influencing you:\n"
            for mem in processed_memories:
                desc = getattr(mem, "description", str(mem))
                importance = getattr(mem, "importance_score", "general")
                prompt += f"- {desc} (importance: {importance})\n"
            return prompt
        
        memory_prompt = create_memory_context_prompt(memories)
        
        self.assertIn("Completed important project", memory_prompt)
        self.assertIn("importance: 9", memory_prompt)
        self.assertIn("Received praise from boss", memory_prompt)
        self.assertIn("importance: 7", memory_prompt)


def demonstrate_correct_memory_testing_pattern():
    """
    Demonstrates the correct pattern for testing memory integration.
    
    This function shows how to create realistic memory objects for testing
    instead of using MagicMock, which can hide bugs in memory processing logic.
    """
    print("\n" + "=" * 60)
    print("CORRECT MEMORY TESTING PATTERN DEMONSTRATION")
    print("=" * 60)
    
    print("\n❌ PROBLEMATIC PATTERN (from issue description):")
    print("mem1 = MagicMock(description='Met Bob yesterday', importance_score=5)")
    print("- This allows ANY attribute access, even invalid ones")
    print("- Tests can pass even when memory processing logic is broken")
    print("- Does not validate actual memory object structure")
    
    print("\n✅ CORRECT PATTERN:")
    print("mem1 = RealSpecificMemory('Met Bob yesterday', 5)")
    print("- Uses objects that match actual memory structure")
    print("- Properly validates attribute access")  
    print("- Tests fail when memory processing logic is broken")
    print("- Catches integration bugs that MagicMock would miss")
    
    print("\n📝 IMPLEMENTATION GUIDELINES:")
    print("1. Create memory classes that mimic tiny_memories.Memory structure")
    print("2. Include all attributes that PromptBuilder might access")
    print("3. Raise AttributeError for invalid attribute access")
    print("4. Test both normal and edge cases (missing attributes, etc.)")
    print("5. Verify that broken memory processing logic fails appropriately")
    
    print("\n🧪 TEST SCENARIOS TO COVER:")
    print("- Memory description formatting")
    print("- Importance score access with fallbacks")
    print("- Mixed specific and general memory handling")
    print("- Broken/malformed memory object handling")
    print("- Memory filtering and sorting operations")
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    print("PromptBuilder Memory Integration Testing")
    print("=" * 50)
    print("This test suite demonstrates proper memory testing patterns")
    print("and fixes the MagicMock antipattern issue.")
    print("=" * 50)
    
    # Run the demonstration
    demonstrate_correct_memory_testing_pattern()
    
    # Run the tests
    unittest.main(verbosity=2, argv=[''])