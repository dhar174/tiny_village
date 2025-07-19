#!/usr/bin/env python3
"""
Test file for map interactivity enhancements.
Tests the new context menu and information panel functionality without requiring pygame to be installed.
"""

import unittest
from unittest.mock import Mock, MagicMock, patch
import sys
import os

# Mock pygame before importing our modules
class MockPygame:
    QUIT = 12
    MOUSEBUTTONDOWN = 5
    MOUSEMOTION = 4
    KEYDOWN = 2
    K_ESCAPE = 27
    
    class Rect:
        def __init__(self, x, y, width, height):
            self.x = x
            self.y = y
            self.width = width
            self.height = height
            self.left = x
            self.top = y
            self.right = x + width
            self.bottom = y + height
            self.centerx = x + width // 2
            self.centery = y + height // 2
            
        def collidepoint(self, pos):
            x, y = pos
            return self.x <= x <= self.x + self.width and self.y <= y <= self.y + self.height
    
    class math:
        @staticmethod
        class Vector2:
            def __init__(self, x, y):
                self.x = x
                self.y = y
            
            def distance_to(self, other):
                return ((self.x - other.x) ** 2 + (self.y - other.y) ** 2) ** 0.5
    
    class font:
        @staticmethod
        def Font(font_file, size):
            return Mock()
        
        @staticmethod
        def SysFont(name, size):
            return Mock()
    
    class image:
        @staticmethod
        def load(path):
            return Mock()
    
    class display:
        @staticmethod
        def set_mode(size):
            return Mock()
    
    class mouse:
        @staticmethod
        def get_pos():
            return (100, 100)
    
    @staticmethod
    def init():
        pass
    
    @staticmethod
    def draw():
        pass

# Patch pygame in sys.modules before importing
sys.modules['pygame'] = MockPygame()
sys.modules['pygame.math'] = MockPygame.math()
sys.modules['pygame.font'] = MockPygame.font()
sys.modules['pygame.image'] = MockPygame.image()
sys.modules['pygame.display'] = MockPygame.display()
sys.modules['pygame.mouse'] = MockPygame.mouse()

# Now we can import our modules
from tiny_map_controller import InfoPanel, ContextMenu, MapController


class TestInfoPanel(unittest.TestCase):
    """Test the InfoPanel class functionality."""
    
    def setUp(self):
        self.panel = InfoPanel(10, 10, 200, 150)
    
    def test_initial_state(self):
        """Test that panel starts in hidden state."""
        self.assertFalse(self.panel.visible)
        self.assertEqual(self.panel.content, {})
    
    def test_show_panel(self):
        """Test showing panel with content."""
        content = {'name': 'Test Building', 'type': 'House', 'size': '50x30'}
        self.panel.show(content, (100, 100))
        
        self.assertTrue(self.panel.visible)
        self.assertEqual(self.panel.content, content)
        self.assertEqual(self.panel.x, 110)  # 100 + 10 offset
        self.assertEqual(self.panel.y, 110)  # 100 + 10 offset
    
    def test_hide_panel(self):
        """Test hiding panel."""
        content = {'name': 'Test Building'}
        self.panel.show(content, (100, 100))
        self.panel.hide()
        
        self.assertFalse(self.panel.visible)
        self.assertEqual(self.panel.content, {})
    
    def test_position_boundary_checking(self):
        """Test that panel position is adjusted to stay on screen."""
        # Test positioning near right edge
        self.panel.show({'name': 'Test'}, (750, 100))
        self.assertEqual(self.panel.x, 600)  # 800 - 200 (width)
        
        # Test positioning near bottom edge
        self.panel.show({'name': 'Test'}, (100, 550))
        self.assertEqual(self.panel.y, 450)  # 600 - 150 (height)


class TestContextMenu(unittest.TestCase):
    """Test the ContextMenu class functionality."""
    
    def setUp(self):
        self.menu = ContextMenu()
    
    def test_initial_state(self):
        """Test that menu starts in hidden state."""
        self.assertFalse(self.menu.visible)
        self.assertEqual(self.menu.options, [])
        self.assertEqual(self.menu.selected_option, -1)
    
    def test_show_menu(self):
        """Test showing menu with options."""
        options = [
            {'label': 'Enter Building', 'action': 'enter'},
            {'label': 'View Details', 'action': 'details'}
        ]
        self.menu.show(options, (100, 100), Mock())
        
        self.assertTrue(self.menu.visible)
        self.assertEqual(self.menu.options, options)
        self.assertEqual(self.menu.height, 60)  # 2 options * 25 + 10
    
    def test_hide_menu(self):
        """Test hiding menu."""
        options = [{'label': 'Test', 'action': 'test'}]
        self.menu.show(options, (100, 100), Mock())
        self.menu.hide()
        
        self.assertFalse(self.menu.visible)
        self.assertEqual(self.menu.options, [])
        self.assertIsNone(self.menu.target_object)
    
    def test_mouse_motion_handling(self):
        """Test mouse motion handling for option highlighting."""
        options = [
            {'label': 'Option 1', 'action': 'action1'},
            {'label': 'Option 2', 'action': 'action2'}
        ]
        self.menu.show(options, (100, 100), Mock())
        
        # Test hovering over first option
        self.menu.handle_mouse_motion((125, 110))  # Within menu bounds, first option
        self.assertEqual(self.menu.selected_option, 0)
        
        # Test hovering over second option
        self.menu.handle_mouse_motion((125, 135))  # Within menu bounds, second option
        self.assertEqual(self.menu.selected_option, 1)
        
        # Test hovering outside menu
        self.menu.handle_mouse_motion((200, 200))  # Outside menu bounds
        self.assertEqual(self.menu.selected_option, -1)
    
    def test_click_handling(self):
        """Test click handling for option selection."""
        options = [
            {'label': 'Option 1', 'action': 'action1'},
            {'label': 'Option 2', 'action': 'action2'}
        ]
        self.menu.show(options, (100, 100), Mock())
        
        # Test clicking on first option
        result = self.menu.handle_click((125, 110))
        self.assertEqual(result, options[0])
        self.assertFalse(self.menu.visible)  # Menu should hide after selection
        
        # Test clicking outside menu
        self.menu.show(options, (100, 100), Mock())
        result = self.menu.handle_click((200, 200))
        self.assertIsNone(result)


class TestMapControllerInteractivity(unittest.TestCase):
    """Test the MapController interactivity enhancements."""
    
    def setUp(self):
        # Create mock map data
        self.map_data = {
            'width': 800,
            'height': 600,
            'buildings': [
                {
                    'name': 'Town Hall',
                    'type': 'government',
                    'rect': MockPygame.Rect(100, 100, 50, 50)
                },
                {
                    'name': 'General Store',
                    'type': 'shop',
                    'rect': MockPygame.Rect(200, 150, 40, 30)
                }
            ]
        }
        
        # Mock the image loading
        with patch('pygame.image.load'):
            self.controller = MapController('dummy_path.png', self.map_data)
    
    def test_building_detection(self):
        """Test building detection at click position."""
        # Test clicking on Town Hall
        building = self.controller.is_building((125, 125))
        self.assertIsNotNone(building)
        self.assertEqual(building['name'], 'Town Hall')
        
        # Test clicking on empty space
        building = self.controller.is_building((500, 500))
        self.assertIsNone(building)
    
    def test_building_info_generation(self):
        """Test building information generation."""
        building = self.map_data['buildings'][0]  # Town Hall
        info = self.controller.get_building_info(building)
        
        self.assertEqual(info['name'], 'Town Hall')
        self.assertEqual(info['type'], 'government')
        self.assertEqual(info['position'], '(100, 100)')
        self.assertEqual(info['size'], '50 x 50')
        self.assertEqual(info['area'], 2500)
    
    def test_character_info_generation(self):
        """Test character information generation."""
        # Create a mock character
        mock_character = Mock()
        mock_character.name = 'John Doe'
        mock_character.position = MockPygame.math.Vector2(200, 300)
        mock_character.energy = 75
        mock_character.health = 90
        mock_character.mood = 'Happy'
        
        info = self.controller.get_character_info(mock_character)
        
        self.assertEqual(info['name'], 'John Doe')
        self.assertEqual(info['type'], 'Character')
        self.assertEqual(info['position'], '(200, 300)')
        self.assertEqual(info['energy'], 75)
        self.assertEqual(info['health'], 90)
        self.assertEqual(info['mood'], 'Happy')
    
    def test_building_context_menu_options(self):
        """Test building context menu option generation."""
        # Test general building
        building = self.map_data['buildings'][0]  # Town Hall
        with patch.object(self.controller.context_menu, 'show') as mock_show:
            self.controller.show_building_context_menu(building, (100, 100))
            
            # Check that show was called with correct options
            mock_show.assert_called_once()
            args = mock_show.call_args[0]
            options = args[0]
            
            # Check that basic options are present
            option_labels = [opt['label'] for opt in options]
            self.assertIn('Enter Building', option_labels)
            self.assertIn('View Details', option_labels)
            self.assertIn('Get Directions', option_labels)
        
        # Test shop building
        shop_building = self.map_data['buildings'][1]  # General Store
        with patch.object(self.controller.context_menu, 'show') as mock_show:
            self.controller.show_building_context_menu(shop_building, (100, 100))
            
            args = mock_show.call_args[0]
            options = args[0]
            option_labels = [opt['label'] for opt in options]
            
            # Check that shop-specific option is present
            self.assertIn('Browse Items', option_labels)
    
    def test_ui_element_hiding(self):
        """Test hiding of UI elements."""
        # Show some UI elements
        self.controller.info_panel.show({'name': 'Test'}, (100, 100))
        self.controller.context_menu.show([{'label': 'Test', 'action': 'test'}], (100, 100), Mock())
        
        # Hide UI elements
        self.controller.hide_ui_elements()
        
        self.assertFalse(self.controller.info_panel.visible)
        self.assertFalse(self.controller.context_menu.visible)
    
    def test_selection_clearing(self):
        """Test clearing of selections."""
        # Set some selections
        self.controller.selected_character = Mock()
        self.controller.selected_building = Mock()
        self.controller.selected_location = Mock()
        
        # Clear selections
        self.controller.clear_selections()
        
        self.assertIsNone(self.controller.selected_character)
        self.assertIsNone(self.controller.selected_building)
        self.assertIsNone(self.controller.selected_location)


class TestActionExecution(unittest.TestCase):
    """Test action execution from context menus."""
    
    def setUp(self):
        map_data = {
            'width': 800,
            'height': 600,
            'buildings': []
        }
        with patch('pygame.image.load'):
            self.controller = MapController('dummy_path.png', map_data)
    
    def test_enter_building_action(self):
        """Test enter building action execution."""
        building = {'name': 'Test Building', 'type': 'house'}
        option = {'action': 'enter', 'target': building}
        
        with patch('builtins.print') as mock_print:
            self.controller.execute_context_action(option)
            mock_print.assert_called_with('Entering Test Building')
    
    def test_details_action(self):
        """Test view details action execution."""
        building = {'name': 'Test Building', 'type': 'house', 'rect': MockPygame.Rect(100, 100, 50, 50)}
        option = {'action': 'details', 'target': building}
        
        with patch.object(self.controller.info_panel, 'show') as mock_show:
            self.controller.execute_context_action(option)
            mock_show.assert_called_once()
    
    def test_unknown_action(self):
        """Test handling of unknown actions."""
        option = {'action': 'unknown_action', 'target': Mock()}
        
        with patch('builtins.print') as mock_print:
            self.controller.execute_context_action(option)
            mock_print.assert_called_with('Unknown action: unknown_action')


if __name__ == '__main__':
    # Run the tests
    unittest.main(verbosity=2)