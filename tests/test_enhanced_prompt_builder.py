"""Tests for enhanced PromptBuilder functionality including memory integration and versioning."""

import unittest
from unittest.mock import MagicMock, patch
import sys
import types
from datetime import datetime

# Create minimal character stub to avoid heavy dependencies
tc_stub = types.ModuleType('tiny_characters')
class MockCharacter:
    def __init__(self):
        self.name = "TestCharacter"
        self.job = "Engineer" 
        self.health_status = 8
        self.hunger_level = 5
        self.mental_health = 7
        self.social_wellbeing = 6
        self.energy = 8
        self.wealth_money = 50
        self.long_term_goal = "become a better engineer"
        self.recent_event = "default"
        self.inventory = MagicMock()
        
    def evaluate_goals(self):
        return [(0.8, MagicMock(name="Goal1", description="Learn new technology"))]

tc_stub.Character = MockCharacter
sys.modules['tiny_characters'] = tc_stub

# Mock the descriptors to avoid import issues
mock_descriptors = MagicMock()
mock_descriptors.get_job_adjective.return_value = "skilled"
mock_descriptors.get_job_pronoun.return_value = "engineer"
mock_descriptors.get_job_enjoys_verb.return_value = "building"
mock_descriptors.get_job_verb_acts_on_noun.return_value = "systems"
mock_descriptors.get_job_currently_working_on.return_value = "a new project"
mock_descriptors.get_job_place.return_value = "at the office"
mock_descriptors.get_job_planning_to_attend.return_value = "tech conference"
mock_descriptors.get_job_hoping_to_there.return_value = "network"
mock_descriptors.get_weather_description.return_value = "sunny weather"
mock_descriptors.get_feeling_health.return_value = "healthy"
mock_descriptors.get_feeling_hunger.return_value = "satisfied"
mock_descriptors.get_event_recent.return_value = "Recently"
mock_descriptors.get_financial_situation.return_value = "you have some money"
mock_descriptors.get_motivation.return_value = "You're motivated to"
mock_descriptors.get_routine_question_framing.return_value = "What do you choose to do?"

with patch('tiny_prompt_builder.descriptors', mock_descriptors):
    from tiny_prompt_builder import PromptBuilder


class TestEnhancedPromptBuilder(unittest.TestCase):
    """Test enhanced PromptBuilder functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.character = MockCharacter()
        self.mock_memory_manager = MagicMock()
        
        # Mock strategy manager to avoid import issues
        with patch('tiny_prompt_builder.StrategyManager'):
            self.prompt_builder = PromptBuilder(self.character, self.mock_memory_manager)
            
    def test_initialization_with_memory_manager(self):
        """Test PromptBuilder initialization with memory manager."""
        self.assertIsNotNone(self.prompt_builder.context_manager)
        self.assertEqual(self.prompt_builder.memory_manager, self.mock_memory_manager)
        self.assertIn('version', self.prompt_builder.prompt_metadata)
        self.assertEqual(self.prompt_builder.prompt_version, "1.0.0")
        
    def test_integrate_relevant_memories(self):
        """Test memory integration functionality."""
        # Setup mock memories
        mock_memory1 = MagicMock()
        mock_memory1.description = "Memory about coding"
        mock_memory2 = MagicMock()
        mock_memory2.description = "Memory about debugging"
        
        self.prompt_builder.context_manager.gather_memory_context = MagicMock(
            return_value=[mock_memory1, mock_memory2]
        )
        
        memories = self.prompt_builder.integrate_relevant_memories("coding skills", 2)
        
        self.assertEqual(len(memories), 2)
        self.prompt_builder.context_manager.gather_memory_context.assert_called_once_with("coding skills", 2)
        
    def test_format_memories_for_prompt(self):
        """Test memory formatting for prompt inclusion."""
        mock_memory1 = MagicMock()
        mock_memory1.description = "Memory about coding"
        mock_memory2 = MagicMock()
        mock_memory2.description = "Memory about debugging"
        
        memories = [mock_memory1, mock_memory2]
        formatted = self.prompt_builder.format_memories_for_prompt(memories)
        
        self.assertIn("Relevant memories to consider:", formatted)
        self.assertIn("1. Memory about coding", formatted)
        self.assertIn("2. Memory about debugging", formatted)
        
    def test_format_memories_empty_list(self):
        """Test memory formatting with empty list."""
        formatted = self.prompt_builder.format_memories_for_prompt([])
        self.assertEqual(formatted, "")
        
    def test_format_memories_different_object_types(self):
        """Test memory formatting with different object types."""
        # Memory with content attribute instead of description
        mock_memory1 = MagicMock()
        mock_memory1.content = "Memory content"
        del mock_memory1.description  # Remove description attribute
        
        # Memory that's just a string
        mock_memory2 = "Simple string memory"
        
        memories = [mock_memory1, mock_memory2]
        formatted = self.prompt_builder.format_memories_for_prompt(memories)
        
        self.assertIn("Memory content", formatted)
        self.assertIn("Simple string memory", formatted)
        
    def test_add_prompt_metadata(self):
        """Test prompt metadata creation."""
        metadata = self.prompt_builder.add_prompt_metadata(
            "test_prompt", 
            {"test_key": "test_value"}
        )
        
        self.assertEqual(metadata["prompt_version"], "1.0.0")
        self.assertEqual(metadata["prompt_type"], "test_prompt")
        self.assertEqual(metadata["character_name"], "TestCharacter")
        self.assertEqual(metadata["test_key"], "test_value")
        self.assertIn("timestamp", metadata)
        
    def test_collect_performance_feedback(self):
        """Test performance feedback collection."""
        self.prompt_builder.collect_performance_feedback(
            "daily_routine", 
            0.8, 
            response_quality=0.9, 
            user_feedback="Good prompt"
        )
        
        metrics = self.prompt_builder.prompt_metadata["performance_metrics"]["daily_routine"]
        self.assertEqual(len(metrics), 1)
        
        feedback = metrics[0]
        self.assertEqual(feedback["success_rating"], 0.8)
        self.assertEqual(feedback["response_quality"], 0.9)
        self.assertEqual(feedback["user_feedback"], "Good prompt")
        self.assertEqual(feedback["version"], "1.0.0")
        
    def test_collect_multiple_performance_feedback(self):
        """Test collecting multiple performance feedback entries."""
        # Add multiple feedback entries
        self.prompt_builder.collect_performance_feedback("daily_routine", 0.8)
        self.prompt_builder.collect_performance_feedback("daily_routine", 0.9)
        self.prompt_builder.collect_performance_feedback("decision", 0.7)
        
        # Check daily_routine has 2 entries
        daily_metrics = self.prompt_builder.prompt_metadata["performance_metrics"]["daily_routine"]
        self.assertEqual(len(daily_metrics), 2)
        
        # Check decision has 1 entry
        decision_metrics = self.prompt_builder.prompt_metadata["performance_metrics"]["decision"]
        self.assertEqual(len(decision_metrics), 1)
        
    @patch('tiny_prompt_builder.descriptors', mock_descriptors)
    def test_generate_daily_routine_prompt_with_memories(self):
        """Test enhanced daily routine prompt with memory integration."""
        # Setup mock context manager
        mock_context = {
            'character': {
                'basic_info': {
                    'name': 'TestCharacter',
                    'job': 'Engineer'
                }
            },
            'memories': [MagicMock(description="Test memory")]
        }
        
        self.prompt_builder.context_manager.assemble_complete_context = MagicMock(
            return_value=mock_context
        )
        
        # Mock action options to avoid import issues
        self.prompt_builder.action_options.prioritize_actions = MagicMock(return_value=[])
        
        prompt = self.prompt_builder.generate_daily_routine_prompt(
            "morning", "sunny", include_memories=True
        )
        
        # Check that version metadata is included
        self.assertIn("Prompt Version: 1.0.0", prompt)
        
        # Check that context manager was called
        self.prompt_builder.context_manager.assemble_complete_context.assert_called_once()
        
        # Check basic prompt structure
        self.assertIn("TestCharacter", prompt)
        self.assertIn("<|system|>", prompt)
        self.assertIn("<|user|>", prompt)
        
    @patch('tiny_prompt_builder.descriptors', mock_descriptors) 
    def test_generate_daily_routine_prompt_without_memories(self):
        """Test daily routine prompt without memory integration."""
        mock_context = {
            'character': {
                'basic_info': {
                    'name': 'TestCharacter',
                    'job': 'Engineer'
                }
            },
            'memories': []
        }
        
        self.prompt_builder.context_manager.assemble_complete_context = MagicMock(
            return_value=mock_context
        )
        self.prompt_builder.action_options.prioritize_actions = MagicMock(return_value=[])
        
        prompt = self.prompt_builder.generate_daily_routine_prompt(
            "morning", "sunny", include_memories=False
        )
        
        # Context manager should be called with no memory query
        args, kwargs = self.prompt_builder.context_manager.assemble_complete_context.call_args
        self.assertIsNone(kwargs.get('memory_query'))
        
    @patch('tiny_prompt_builder.descriptors', mock_descriptors)
    def test_generate_decision_prompt_with_memory_integration(self):
        """Test enhanced decision prompt with memory integration."""
        mock_context = {
            'character': {
                'basic_info': {
                    'name': 'TestCharacter',
                    'job': 'Engineer'
                }
            },
            'goals': {
                'active_goals': [],
                'needs_priorities': {}
            },
            'memories': [MagicMock(description="Decision memory")]
        }
        
        self.prompt_builder.context_manager.assemble_complete_context = MagicMock(
            return_value=mock_context
        )
        
        action_choices = ["Action 1", "Action 2"]
        
        prompt = self.prompt_builder.generate_decision_prompt(
            "afternoon", "rainy", action_choices, include_memory_integration=True
        )
        
        # Check version metadata
        self.assertIn("Prompt Version: 1.0.0", prompt)
        
        # Check memory query was made
        args, kwargs = self.prompt_builder.context_manager.assemble_complete_context.call_args
        self.assertIsNotNone(kwargs.get('memory_query'))
        self.assertIn("decision making", kwargs['memory_query'])
        
    def test_memory_integration_fallback_to_legacy(self):
        """Test that legacy memories parameter still works when memory integration is disabled."""
        mock_context = {
            'character': {
                'basic_info': {
                    'name': 'TestCharacter', 
                    'job': 'Engineer'
                }
            },
            'goals': {
                'active_goals': [],
                'needs_priorities': {}
            },
            'memories': []
        }
        
        self.prompt_builder.context_manager.assemble_complete_context = MagicMock(
            return_value=mock_context
        )
        
        legacy_memories = [MagicMock(description="Legacy memory")]
        self.prompt_builder.format_memories_for_prompt = MagicMock(return_value="Formatted legacy memories")
        
        prompt = self.prompt_builder.generate_decision_prompt(
            "morning", "sunny", ["Action 1"], 
            memories=legacy_memories, 
            include_memory_integration=False
        )
        
        # Should use legacy memories when memory integration is disabled
        self.prompt_builder.format_memories_for_prompt.assert_called_with(legacy_memories)


if __name__ == '__main__':
    unittest.main()