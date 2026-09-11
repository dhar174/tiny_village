#!/usr/bin/env python3
"""
Demo script for map interactivity enhancements.
Shows how the new context menus and information panels work using real pygame components.
"""

import sys

try:
    import pygame
    pygame.init()
    pygame.font.init()
except ImportError:
    print("pygame is required for this demo. Please install pygame to run map interactivity showcase.")
    sys.exit(1)

from tiny_map_controller import MapController, InfoPanel, ContextMenu


def demo_info_panel():
    """Demonstrate InfoPanel functionality."""
    print("=== InfoPanel Demo ===")
    
    panel = InfoPanel(10, 10, 300, 200)
    print(f"Initial state - Visible: {panel.visible}")
    
    # Show panel with building information
    building_info = {
        'name': 'Town Hall',
        'type': 'Government Building',
        'position': '(100, 100)',
        'size': '50 x 50',
        'area': 2500,
        'owner': 'City Council',
        'capacity': 50
    }
    
    panel.show(building_info, (150, 150))
    print(f"After showing - Visible: {panel.visible}")
    print(f"Panel position: ({panel.x}, {panel.y})")
    print(f"Panel content: {panel.content}")
    
    panel.hide()
    print(f"After hiding - Visible: {panel.visible}")
    print()


def demo_context_menu():
    """Demonstrate ContextMenu functionality."""
    print("=== ContextMenu Demo ===")
    
    menu = ContextMenu()
    print(f"Initial state - Visible: {menu.visible}")
    
    # Show context menu with building options
    options = [
        {'label': 'Enter Building', 'action': 'enter'},
        {'label': 'View Details', 'action': 'details'},
        {'label': 'Get Directions', 'action': 'directions'},
        {'label': 'Browse Items', 'action': 'browse'}
    ]
    
    menu.show(options, (200, 200), options[0])
    print(f"After showing - Visible: {menu.visible}")
    print(f"Menu position: ({menu.x}, {menu.y})")
    print(f"Menu options: {[opt['label'] for opt in menu.options]}")
    
    # Simulate mouse motion
    menu.handle_mouse_motion((225, 215))  # Hover over first option
    print(f"Selected option after hover: {menu.selected_option}")
    
    # Simulate click
    selected = menu.handle_click((225, 215))
    print(f"Clicked option: {selected}")
    print(f"Menu visible after click: {menu.visible}")
    print()


def demo_map_controller():
    """Demonstrate MapController interactivity."""
    print("=== MapController Interactivity Demo ===")
    
    # Create sample map data
    map_data = {
        'width': 800,
        'height': 600,
        'buildings': [
            {
                'name': 'Town Hall',
                'type': 'government',
                'rect': pygame.Rect(100, 100, 50, 50),
                'capacity': 50,
                'owner': 'City Council'
            },
            {
                'name': 'General Store',
                'type': 'shop',
                'rect': pygame.Rect(200, 150, 40, 30),
                'owner': 'Bob Smith',
                'inventory': 25
            },
            {
                'name': 'Village Tavern',
                'type': 'social',
                'rect': pygame.Rect(300, 200, 45, 35),
                'owner': 'Alice Johnson',
                'capacity': 30
            }
        ]
    }
    
    # Create map controller (with mocked image loading)
    controller = MapController('dummy_map.png', map_data)
    
    # Demo building detection
    print("Building Detection:")
    town_hall_pos = (125, 125)  # Inside Town Hall
    building = controller.is_building(town_hall_pos)
    if building:
        print(f"  Found building at {town_hall_pos}: {building['name']}")
    
    empty_pos = (500, 500)  # Empty space
    building = controller.is_building(empty_pos)
    print(f"  Found building at {empty_pos}: {building}")
    
    # Demo building info generation
    print("\nBuilding Information:")
    town_hall = map_data['buildings'][0]
    info = controller.get_building_info(town_hall)
    for key, value in info.items():
        print(f"  {key}: {value}")
    
    # Demo character info generation
    print("\nCharacter Information:")
    class DemoCharacter:
        def __init__(self):
            self.name = 'John Doe'
            self.position = pygame.math.Vector2(200, 300)
            self.energy = 75
            self.health = 90
            self.mood = 'Happy'
            self.job = 'Blacksmith'
    demo_character = DemoCharacter()
    
    char_info = controller.get_character_info(demo_character)
    for key, value in char_info.items():
        print(f"  {key}: {value}")
    
    # Demo context menu actions
    print("\nContext Menu Actions:")
    
    # Test building actions
    print("  Building actions:")
    actions = [
        {'action': 'enter', 'target': town_hall},
        {'action': 'details', 'target': town_hall},
        {'action': 'directions', 'target': town_hall}
    ]
    
    for action in actions:
        print(f"    Executing {action['action']} action:")
        controller.execute_context_action(action)
    
    # Test character actions
    print("  Character actions:")
    char_actions = [
        {'action': 'talk', 'target': demo_character},
        {'action': 'follow', 'target': demo_character},
        {'action': 'trade', 'target': demo_character}
    ]
    
    for action in char_actions:
        print(f"    Executing {action['action']} action:")
        controller.execute_context_action(action)
    
    print()


def demo_interaction_scenarios():
    """Demonstrate typical interaction scenarios."""
    print("=== Interaction Scenarios Demo ===")
    
    map_data = {
        'width': 800,
        'height': 600,
        'buildings': [
            {
                'name': 'Blacksmith Shop',
                'type': 'shop',
                'rect': pygame.Rect(150, 200, 60, 40),
                'owner': 'Master Smith',
                'speciality': 'Weapons & Tools'
            }
        ]
    }
    
    controller = MapController('dummy_map.png', map_data)
    
    print("Scenario 1: Right-clicking on a shop building")
    shop = map_data['buildings'][0]
    
    # Simulate showing context menu for shop
    print("  - Showing context menu for Blacksmith Shop...")
    # This would normally be called by handle_right_click
    options = [
        {'label': 'Enter Building', 'action': 'enter', 'target': shop},
        {'label': 'Browse Items', 'action': 'browse', 'target': shop},
        {'label': 'View Details', 'action': 'details', 'target': shop},
        {'label': 'Get Directions', 'action': 'directions', 'target': shop},
    ]
    
    controller.context_menu.show(options, (175, 220), shop)
    print(f"  - Context menu shown with {len(options)} options")
    
    # Simulate selecting "Browse Items"
    browse_option = options[1]
    print(f"  - User selects: {browse_option['label']}")
    controller.execute_context_action(browse_option)
    
    print("\nScenario 2: Left-clicking on a building for details")
    print("  - Left-clicking on Blacksmith Shop...")
    controller.select_building(shop, (175, 220))
    print(f"  - Selected building: {controller.selected_building['name']}")
    print(f"  - Info panel visible: {controller.info_panel.visible}")
    
    print("\nScenario 3: Hiding UI elements with Escape key")
    print("  - Pressing Escape key...")
    controller.hide_ui_elements()
    print(f"  - Info panel visible: {controller.info_panel.visible}")
    print(f"  - Context menu visible: {controller.context_menu.visible}")
    
    print()


def main():
    """Run all demos."""
    print("Map Interactivity Enhancement Demo")
    print("==================================")
    print()
    
    demo_info_panel()
    demo_context_menu()
    demo_map_controller()
    demo_interaction_scenarios()
    
    print("Demo completed!")
    print()
    print("Key Features Demonstrated:")
    print("- InfoPanel: Shows detailed information about buildings and characters")
    print("- ContextMenu: Provides action options for right-clicked objects")
    print("- Enhanced MapController: Handles complex interactions beyond basic clicks")
    print("- Building-specific context options (shop, house, social buildings)")
    print("- Character interaction options (talk, follow, trade)")
    print("- UI state management (show/hide, selections)")


if __name__ == '__main__':
    main()
