import unittest
from unittest.mock import MagicMock
import pygame
from tiny_gameplay_controller import GameplayController

class TestAntiAliasing(unittest.TestCase):
    def setUp(self):
        # Mock pygame.display.set_mode and other pygame calls to avoid GUI during tests
        pygame.display.set_mode = MagicMock()
        pygame.font.Font = MagicMock()
        pygame.font.init = MagicMock()
        pygame.init = MagicMock()

        # Mock GraphManager
        self.mock_graph_manager = MagicMock()

    def test_aa_initialization_default(self):
        # Default AA should be True
        gc = GameplayController(graph_manager=self.mock_graph_manager)
        self.assertTrue(gc.anti_aliasing)

    def test_aa_initialization_from_config(self):
        # AA should be False if config says so
        config = {"render": {"anti_aliasing": False}}
        gc = GameplayController(graph_manager=self.mock_graph_manager, config=config)
        self.assertFalse(gc.anti_aliasing)

    def test_aa_toggle(self):
        gc = GameplayController(graph_manager=self.mock_graph_manager)
        self.assertTrue(gc.anti_aliasing)

        # Mock event for F2 key
        event = MagicMock()
        event.key = pygame.K_F2

        # Mock add_event_notification
        gc.add_event_notification = MagicMock()

        # Handle keydown
        gc._handle_keydown(event)

        self.assertFalse(gc.anti_aliasing)
        gc.add_event_notification.assert_called_with("Anti-aliasing: Disabled")

        # Toggle back
        gc._handle_keydown(event)
        self.assertTrue(gc.anti_aliasing)
        gc.add_event_notification.assert_called_with("Anti-aliasing: Enabled")

if __name__ == "__main__":
    unittest.main()
