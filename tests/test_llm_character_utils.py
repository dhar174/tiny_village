"""
Unit tests for llm_character_utils.py
"""

import unittest
from unittest.mock import MagicMock, patch
import os
import sys

# Add the root directory to sys.path to import llm_character_utils
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
# Add the tests directory to sys.path to import mock_character
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import llm_character_utils
from mock_character import MockCharacter


class TestLLMCharacterUtils(unittest.TestCase):
    """Test cases for LLM character utility functions."""

    def test_enable_llm_decisions(self):
        """Test that enable_llm_decisions correctly sets the attribute."""
        char = MockCharacter("Alice")
        llm_character_utils.enable_llm_decisions(char, True)
        self.assertTrue(char.use_llm_decisions)

        llm_character_utils.enable_llm_decisions(char, False)
        self.assertFalse(char.use_llm_decisions)

    def test_enable_llm_for_characters_all(self):
        """Test that enable_llm_for_characters enables LLM for all characters when no names provided."""
        chars = [MockCharacter("Alice"), MockCharacter("Bob")]
        result = llm_character_utils.enable_llm_for_characters(chars)
        self.assertTrue(chars[0].use_llm_decisions)
        self.assertTrue(chars[1].use_llm_decisions)
        self.assertEqual(len(result), 2)

    def test_enable_llm_for_characters_selective(self):
        """Test that enable_llm_for_characters enables LLM only for specified character names."""
        chars = [
            MockCharacter("Alice"),
            MockCharacter("Bob"),
            MockCharacter("Charlie"),
        ]
        result = llm_character_utils.enable_llm_for_characters(
            chars, ["Alice", "Charlie"]
        )

        self.assertTrue(chars[0].use_llm_decisions)
        self.assertFalse(chars[1].use_llm_decisions)
        self.assertTrue(chars[2].use_llm_decisions)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0].name, "Alice")
        self.assertEqual(result[1].name, "Charlie")

    def test_get_llm_enabled_characters_mixed(self):
        """Test filtering of enabled and disabled characters."""
        chars = [
            MockCharacter("Alice", use_llm_decisions=True),
            MockCharacter("Bob", use_llm_decisions=False),
            MockCharacter("Charlie", use_llm_decisions=True),
        ]
        enabled = llm_character_utils.get_llm_enabled_characters(chars)
        self.assertEqual(len(enabled), 2)
        self.assertEqual(enabled[0].name, "Alice")
        self.assertEqual(enabled[1].name, "Charlie")

    def test_get_llm_enabled_characters_empty(self):
        """Test handling of an empty list."""
        enabled = llm_character_utils.get_llm_enabled_characters([])
        self.assertEqual(enabled, [])

    def test_get_llm_enabled_characters_none_enabled(self):
        """Test handling when no characters have LLM enabled."""
        chars = [
            MockCharacter("Alice", use_llm_decisions=False),
            MockCharacter("Bob", use_llm_decisions=False),
        ]
        enabled = llm_character_utils.get_llm_enabled_characters(chars)
        self.assertEqual(enabled, [])

    def test_get_llm_enabled_characters_missing_attr(self):
        """Test handling of objects without the use_llm_decisions attribute."""

        class SimpleObj:
            def __init__(self, name):
                self.name = name

        chars = [MockCharacter("Alice", use_llm_decisions=True), SimpleObj("Bob")]
        enabled = llm_character_utils.get_llm_enabled_characters(chars)
        self.assertEqual(len(enabled), 1)
        self.assertEqual(enabled[0].name, "Alice")

    def test_configure_strategy_manager_for_llm(self):
        """Test configuring a StrategyManager for LLM."""
        mock_sm = MagicMock()
        mock_sm.use_llm = False

        with patch("tiny_brain_io.TinyBrainIO") as mock_brain:
            with patch("tiny_output_interpreter.OutputInterpreter") as mock_oi:
                llm_character_utils.configure_strategy_manager_for_llm(mock_sm)
                self.assertTrue(mock_sm.use_llm)
                self.assertIsNotNone(mock_sm.brain_io)
                self.assertIsNotNone(mock_sm.output_interpreter)

    def test_configure_strategy_manager_already_configured(self):
        """Test configuring a StrategyManager that is already configured."""
        mock_sm = MagicMock()
        mock_sm.use_llm = True
        mock_sm.brain_io = "existing_brain"

        llm_character_utils.configure_strategy_manager_for_llm(mock_sm)
        self.assertEqual(mock_sm.brain_io, "existing_brain")

    def test_create_llm_enabled_strategy_manager(self):
        """Test creating a new LLM-enabled StrategyManager."""
        with patch("importlib.import_module") as mock_import:
            mock_module = MagicMock()
            mock_module.StrategyManager = MagicMock()
            mock_import.return_value = mock_module

            llm_character_utils.create_llm_enabled_strategy_manager()
            mock_import.assert_called_with("tiny_strategy_manager")
            mock_module.StrategyManager.assert_called_with(
                use_llm=True, model_name="alexredna/TinyLlama-1.1B-Chat-v1.0-reasoning-v2"
            )

    def test_setup_full_llm_integration(self):
        """Test the complete setup for LLM integration."""
        chars = [MockCharacter("Alice"), MockCharacter("Bob")]

        with patch(
            "llm_character_utils.create_llm_enabled_strategy_manager"
        ) as mock_create_sm:
            mock_sm = MagicMock()
            mock_create_sm.return_value = mock_sm

            enabled_chars, strategy_manager = llm_character_utils.setup_full_llm_integration(
                chars, ["Alice"]
            )

            self.assertTrue(chars[0].use_llm_decisions)
            self.assertFalse(chars[1].use_llm_decisions)
            self.assertEqual(len(enabled_chars), 1)
            self.assertEqual(enabled_chars[0].name, "Alice")
            self.assertEqual(strategy_manager, mock_sm)


if __name__ == "__main__":
    unittest.main()
