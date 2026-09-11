"""Tests for the new ContextManager functionality in tiny_prompt_builder.py"""

import unittest
from unittest.mock import MagicMock, patch
import sys
import types

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
        self.personality_traits = {"extraversion": 60, "conscientiousness": 80}
        
    def evaluate_goals(self):
        return [(0.8, MagicMock(name="Goal1", description="Learn new technology")),
                (0.6, MagicMock(name="Goal2", description="Improve health"))]

tc_stub.Character = MockCharacter
sys.modules['tiny_characters'] = tc_stub

from tiny_prompt_builder import ContextManager, ParameterizedTemplateEngine


class TestContextManager(unittest.TestCase):
    """Test the ContextManager class for systematic context gathering."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.character = MockCharacter()
        self.mock_memory_manager = MagicMock()
        self.context_manager = ContextManager(self.character, self.mock_memory_manager)
        
    def test_gather_character_context(self):
        """Test gathering comprehensive character context."""
        context = self.context_manager.gather_character_context()
        
        # Test basic info structure
        self.assertIn('basic_info', context)
        self.assertEqual(context['basic_info']['name'], "TestCharacter")
        self.assertEqual(context['basic_info']['job'], "Engineer")
        
        # Test current state structure
        self.assertIn('current_state', context)
        self.assertEqual(context['current_state']['health_status'], 8)
        self.assertEqual(context['current_state']['hunger_level'], 5)
        self.assertEqual(context['current_state']['wealth_money'], 50)
        
        # Test motivations structure
        self.assertIn('motivations', context)
        self.assertEqual(context['motivations']['long_term_goal'], "become a better engineer")
        
    def test_gather_environmental_context(self):
        """Test gathering environmental context."""
        context = self.context_manager.gather_environmental_context("morning", "sunny")
        
        self.assertEqual(context['time'], "morning")
        self.assertEqual(context['weather'], "sunny")
        self.assertEqual(context['time_formatted'], "it's morning")
        self.assertIn('weather_formatted', context)
        
    def test_gather_memory_context_with_manager(self):
        """Test memory context gathering when memory manager is available."""
        # Setup mock memory manager to return test memories
        mock_memory1 = MagicMock()
        mock_memory1.description = "Memory about coding"
        mock_memory2 = MagicMock()
        mock_memory2.description = "Memory about problem solving"
        
        self.mock_memory_manager.search_memories.return_value = {
            mock_memory1: 0.9,
            mock_memory2: 0.7
        }
        
        memories = self.context_manager.gather_memory_context("coding skills", 2)
        
        self.assertEqual(len(memories), 2)
        self.mock_memory_manager.search_memories.assert_called_once_with("coding skills")
        
    def test_gather_memory_context_without_manager(self):
        """Test memory context gathering when no memory manager is available."""
        context_manager = ContextManager(self.character, None)
        memories = context_manager.gather_memory_context("test query")
        
        self.assertEqual(memories, [])
        
    def test_gather_goal_context(self):
        """Test gathering goal and priority context."""
        context = self.context_manager.gather_goal_context()
        
        self.assertIn('active_goals', context)
        self.assertIn('needs_priorities', context)
        
        # Test that goals are gathered
        self.assertEqual(len(context['active_goals']), 2)
        
    def test_assemble_complete_context(self):
        """Test assembling all context types together."""
        context = self.context_manager.assemble_complete_context(
            "afternoon", "rainy", "test query"
        )
        
        # Test all major context sections are present
        self.assertIn('character', context)
        self.assertIn('environment', context) 
        self.assertIn('goals', context)
        self.assertIn('memories', context)
        self.assertIn('timestamp', context)
        
        # Test environment context
        self.assertEqual(context['environment']['time'], "afternoon")
        self.assertEqual(context['environment']['weather'], "rainy")


class TestParameterizedTemplateEngine(unittest.TestCase):
    """Test the ParameterizedTemplateEngine for dynamic template system."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.character = MockCharacter()
        self.template_engine = ParameterizedTemplateEngine()
        
    def test_base_templates_loaded(self):
        """Test that base templates are loaded on initialization."""
        self.assertIn('character_intro', self.template_engine.templates)
        self.assertIn('health_status', self.template_engine.templates)
        self.assertIn('weather_context', self.template_engine.templates)
        
    def test_set_character_parameters(self):
        """Test setting character-specific parameters."""
        self.template_engine.set_character_parameters(self.character)
        
        self.assertEqual(self.template_engine.parameters['character_name'], "TestCharacter")
        self.assertEqual(self.template_engine.parameters['character_role'], "engineer")
        self.assertIn('health_descriptor', self.template_engine.parameters)
        self.assertIn('hunger_descriptor', self.template_engine.parameters)
        
    def test_set_environmental_parameters(self):
        """Test setting environmental parameters."""
        self.template_engine.set_environmental_parameters("morning", "sunny")
        
        self.assertEqual(self.template_engine.parameters['time_period'], "morning")
        self.assertEqual(self.template_engine.parameters['weather_description'], "sunny")
        
    def test_generate_text_with_parameters(self):
        """Test generating text from templates with parameter substitution."""
        self.template_engine.set_character_parameters(self.character)
        
        # Test character intro generation
        intro_text = self.template_engine.generate_text('character_intro')
        self.assertIn("TestCharacter", intro_text)
        self.assertIn("engineer", intro_text)
        
    def test_generate_text_missing_template(self):
        """Test handling of missing template."""
        result = self.template_engine.generate_text('nonexistent_template')
        self.assertIn("Template 'nonexistent_template' not found", result)
        
    def test_generate_text_missing_parameter(self):
        """Test handling of missing parameters."""
        # Try to generate text without setting parameters
        result = self.template_engine.generate_text('character_intro')
        self.assertIn("Missing parameter", result)
        
    def test_add_custom_template(self):
        """Test adding custom templates."""
        self.template_engine.add_custom_template(
            'test_template', 
            'Hello {name}, you are {role}'
        )
        
        self.assertIn('test_template', self.template_engine.templates)
        
        # Test generation with custom template
        result = self.template_engine.generate_text(
            'test_template', 
            {'name': 'Alice', 'role': 'developer'}
        )
        self.assertEqual(result, "Hello Alice, you are developer")
        
    def test_modify_template(self):
        """Test modifying existing templates at runtime."""
        original_template = self.template_engine.templates['character_intro']
        new_template = "Modified: {character_name} is a {character_role}"
        
        self.template_engine.modify_template('character_intro', new_template)
        
        self.assertEqual(self.template_engine.templates['character_intro'], new_template)
        self.assertNotEqual(self.template_engine.templates['character_intro'], original_template)
        
    def test_personality_modifier_application(self):
        """Test applying personality modifiers to templates."""
        self.template_engine.set_character_parameters(self.character, 'analytical')
        
        # Should have analytical personality traits
        self.assertIn(self.template_engine.parameters.get('character_adjective', ''), 
                     ['methodical', 'precise', 'logical'])
        
    def test_dynamic_descriptors_health(self):
        """Test dynamic descriptor generation based on character state."""
        # Test high health
        self.character.health_status = 9
        self.template_engine.set_character_parameters(self.character)
        self.assertEqual(self.template_engine.parameters['health_descriptor'], 'excellent')
        
        # Test low health
        self.character.health_status = 3
        self.template_engine.set_character_parameters(self.character)
        self.assertEqual(self.template_engine.parameters['health_descriptor'], 'unwell')
        
    def test_dynamic_descriptors_wealth(self):
        """Test dynamic wealth descriptors."""
        # Test high wealth
        self.character.wealth_money = 150
        self.template_engine.set_character_parameters(self.character)
        self.assertEqual(self.template_engine.parameters['financial_descriptor'], 
                        'you are financially comfortable')
        
        # Test low wealth
        self.character.wealth_money = 5
        self.template_engine.set_character_parameters(self.character)
        self.assertEqual(self.template_engine.parameters['financial_descriptor'], 
                        'money is tight')


if __name__ == '__main__':
    unittest.main()