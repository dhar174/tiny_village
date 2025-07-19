#!/usr/bin/env python3
"""
Final integration test demonstrating that the map interactivity enhancements
work correctly and integrate well with the existing codebase.
"""

import sys
from unittest.mock import Mock, patch

# Mock pygame completely for this test
class MockPygame:
    QUIT = 12
    MOUSEBUTTONDOWN = 5
    MOUSEMOTION = 4
    KEYDOWN = 2
    K_ESCAPE = 27
    
    class Rect:
        def __init__(self, x, y, width, height):
            self.x, self.y, self.width, self.height = x, y, width, height
            self.left, self.top = x, y
            self.right, self.bottom = x + width, y + height
            self.centerx, self.centery = x + width // 2, y + height // 2
            
        def collidepoint(self, pos):
            x, y = pos
            return self.x <= x <= self.x + self.width and self.y <= y <= self.y + self.height
    
    class math:
        class Vector2:
            def __init__(self, x, y):
                self.x, self.y = x, y
            def distance_to(self, other):
                return ((self.x - other.x) ** 2 + (self.y - other.y) ** 2) ** 0.5
    
    class font:
        @staticmethod
        def Font(font_file, size): return Mock()
        @staticmethod
        def SysFont(name, size): return Mock()
    
    class image:
        @staticmethod
        def load(path): return Mock()
    
    class mouse:
        @staticmethod
        def get_pos(): return (300, 300)
    
    @staticmethod
    def init(): pass

# Mock all pygame modules
sys.modules['pygame'] = MockPygame()
sys.modules['pygame.math'] = MockPygame.math()
sys.modules['pygame.font'] = MockPygame.font()
sys.modules['pygame.image'] = MockPygame.image()
sys.modules['pygame.mouse'] = MockPygame.mouse()

from tiny_map_controller import MapController, InfoPanel, ContextMenu


def test_complete_interaction_flow():
    """Test a complete user interaction flow from start to finish."""
    
    print("🧪 Complete Integration Test")
    print("=" * 50)
    
    # Create realistic game data
    map_data = {
        'width': 800,
        'height': 600,
        'buildings': [
            {
                'name': 'Village Inn', 
                'type': 'social',
                'rect': MockPygame.Rect(100, 100, 60, 40),
                'owner': 'Martha',
                'capacity': 25,
                'description': 'A cozy inn with warm fires'
            },
            {
                'name': 'Weapon Shop',
                'type': 'shop', 
                'rect': MockPygame.Rect(300, 200, 50, 50),
                'owner': 'Blacksmith Joe',
                'speciality': 'Fine weapons and armor'
            }
        ]
    }
    
    # Initialize controller
    controller = MapController('test_map.png', map_data)
    
    # Add mock characters
    mock_character = Mock()
    mock_character.name = 'Adventurer Alice'
    mock_character.position = MockPygame.math.Vector2(250, 150)
    mock_character.energy = 80
    mock_character.health = 95
    mock_character.mood = 'Excited'
    mock_character.job = 'Explorer'
    mock_character.color = (0, 255, 0)
    
    controller.characters = {'alice': mock_character}
    
    print("✓ Map controller initialized with buildings and characters")
    
    # Test 1: Left-click on building shows info panel
    print("\n📋 Test 1: Left-click information panel")
    inn = map_data['buildings'][0]
    controller.select_building(inn, (130, 120))
    
    assert controller.selected_building == inn
    assert controller.info_panel.visible == True
    assert 'Village Inn' in controller.info_panel.content['name']
    print("✓ Building selection and info panel display working")
    
    # Test 2: Right-click on shop shows context menu with shop options
    print("\n🔧 Test 2: Right-click context menu with building-specific options")
    weapon_shop = map_data['buildings'][1]
    controller.show_building_context_menu(weapon_shop, (325, 225))
    
    assert controller.context_menu.visible == True
    option_labels = [opt['label'] for opt in controller.context_menu.options]
    assert 'Browse Items' in option_labels  # Shop-specific option
    assert 'Enter Building' in option_labels  # General option
    print("✓ Context menu with shop-specific options working")
    
    # Test 3: Character interaction
    print("\n👤 Test 3: Character interaction")
    char_info = controller.get_character_info(mock_character)
    expected_fields = ['name', 'position', 'energy', 'health', 'mood', 'job']
    
    for field in expected_fields:
        assert field in char_info
    print("✓ Character information generation working")
    
    # Test 4: Context menu action execution
    print("\n⚡ Test 4: Action execution")
    enter_action = {'action': 'enter', 'target': inn}
    browse_action = {'action': 'browse', 'target': weapon_shop}
    talk_action = {'action': 'talk', 'target': mock_character}
    
    # Capture print output to verify actions
    with patch('builtins.print') as mock_print:
        controller.execute_context_action(enter_action)
        controller.execute_context_action(browse_action)
        controller.execute_context_action(talk_action)
        
        # Verify actions were executed
        print_calls = [call.args[0] for call in mock_print.call_args_list]
        assert any('Entering Village Inn' in call for call in print_calls)
        assert any('Browsing items in Weapon Shop' in call for call in print_calls)
        assert any('Starting conversation with Adventurer Alice' in call for call in print_calls)
    
    print("✓ Action execution working correctly")
    
    # Test 5: UI state management
    print("\n🖥️  Test 5: UI state management")
    # Show both UI elements
    controller.info_panel.show({'name': 'Test'}, (100, 100))
    controller.context_menu.show([{'label': 'Test', 'action': 'test'}], (200, 200), None)
    
    assert controller.info_panel.visible == True
    assert controller.context_menu.visible == True
    
    # Hide all elements
    controller.hide_ui_elements()
    
    assert controller.info_panel.visible == False
    assert controller.context_menu.visible == False
    print("✓ UI state management working")
    
    # Test 6: Screen boundary handling
    print("\n🖼️  Test 6: Screen boundary positioning")
    # Test near screen edges
    controller.info_panel.show({'name': 'Edge Test'}, (750, 550))  # Near bottom-right
    
    # Panel should be repositioned to stay on screen
    assert controller.info_panel.x <= 800 - controller.info_panel.width
    assert controller.info_panel.y <= 600 - controller.info_panel.height
    print("✓ Screen boundary handling working")
    
    print("\n🎉 All integration tests passed!")
    return True


def test_backwards_compatibility():
    """Test that our changes don't break existing functionality."""
    
    print("\n🔄 Backwards Compatibility Test")
    print("=" * 40)
    
    map_data = {
        'width': 800,
        'height': 600,
        'buildings': [
            {'name': 'Test Building', 'rect': MockPygame.Rect(100, 100, 50, 50)}
        ]
    }
    
    controller = MapController('test_map.png', map_data)
    
    # Test that original methods still work
    building = controller.is_building((125, 125))
    assert building is not None
    assert building['name'] == 'Test Building'
    
    # Test that pathfinding system is intact
    assert hasattr(controller, 'pathfinder')
    assert hasattr(controller, 'find_path_cached')
    
    # Test that character management works
    assert hasattr(controller, 'characters')
    assert hasattr(controller, 'update_character_position')
    
    print("✓ All existing functionality preserved")
    return True


def test_error_handling():
    """Test error handling and edge cases."""
    
    print("\n🛡️  Error Handling Test")
    print("=" * 30)
    
    controller = MapController('test_map.png', {'width': 800, 'height': 600, 'buildings': []})
    
    # Test with empty data
    info = controller.get_building_info({})
    assert 'name' in info
    assert info['name'] == 'Unknown Building'
    
    # Test character info with minimal data
    minimal_char = Mock()
    minimal_char.name = 'Test'
    minimal_char.position = MockPygame.math.Vector2(0, 0)
    
    info = controller.get_character_info(minimal_char)
    assert info['name'] == 'Test'
    
    # Test unknown action
    with patch('builtins.print') as mock_print:
        controller.execute_context_action({'action': 'unknown', 'target': None})
        assert mock_print.called
    
    print("✓ Error handling working correctly")
    return True


def main():
    """Run all integration tests."""
    
    print("🚀 Map Interactivity Integration Tests")
    print("=" * 60)
    
    try:
        # Run all test suites
        test_complete_interaction_flow()
        test_backwards_compatibility() 
        test_error_handling()
        
        print("\n" + "=" * 60)
        print("🎯 ALL INTEGRATION TESTS PASSED!")
        print("The map interactivity enhancements are working correctly")
        print("and integrate seamlessly with the existing codebase.")
        print("=" * 60)
        
        return True
        
    except Exception as e:
        print(f"\n❌ Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)