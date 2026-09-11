# Map Interactivity Enhancements

This document describes the enhanced map interactivity features that expand beyond basic click-to-select/enter functionality by providing contextual menus and information panels for locations and buildings.

## Overview

The map interactivity system has been significantly enhanced to provide rich, contextual interactions with game objects. Users can now:

- **Left-click** objects to view detailed information in an information panel
- **Right-click** objects to access contextual action menus
- **Press ESC** to hide all UI elements
- Interact with different object types in contextually appropriate ways

## New Components

### InfoPanel Class

The `InfoPanel` displays detailed information about selected objects near the mouse cursor.

**Features:**
- Automatic positioning with screen boundary checking
- Displays object name, type, position, size, and custom attributes  
- Hide/show functionality
- Customizable appearance (colors, fonts, dimensions)

**Usage:**
```python
panel = InfoPanel(x=0, y=0, width=300, height=200)
panel.show(content_dict, mouse_position)
panel.hide()
```

### ContextMenu Class

The `ContextMenu` provides right-click action menus for interactive objects.

**Features:**
- Mouse hover highlighting
- Click handling and option selection
- Automatic height calculation based on options
- Screen boundary positioning

**Usage:**
```python
menu = ContextMenu()
options = [
    {'label': 'Enter Building', 'action': 'enter', 'target': building},
    {'label': 'View Details', 'action': 'details', 'target': building}
]
menu.show(options, mouse_position, target_object)
```

## Enhanced MapController

The `MapController` has been extended with comprehensive interactivity features.

### New Event Handling

**Left Click:**
- Selects objects and shows information panels
- Clears selections when clicking empty areas
- Handles context menu option selection

**Right Click:**
- Shows contextual action menus
- Different menus for buildings, characters, and empty areas
- Building-type specific options

**Mouse Motion:**
- Updates context menu hover states

**Keyboard:**
- ESC key hides all UI elements

### Building Interactions

Buildings now support rich contextual interactions based on their type:

**General Buildings:**
- Enter Building
- View Details  
- Get Directions

**Shop Buildings:**
- Browse Items (in addition to general options)

**House Buildings:**
- Knock on Door (in addition to general options)

**Social Buildings:**
- Join Activity (in addition to general options)

### Character Interactions

Characters support various social and gameplay interactions:

- **Talk to Character** - Initiate conversations
- **View Details** - Show character information
- **Follow Character** - Track character movement
- **Trade with Character** - Open trading interface

### Information Display

Detailed information is automatically generated for different object types:

**Buildings:**
- Name, type, position, size, area
- Owner, capacity, value (if available)
- Custom attributes from building data

**Characters:**
- Name, type, position
- Energy, health, mood, job (if available)
- Custom character attributes

## Integration Points

The enhanced system integrates with existing game systems:

### Action System
- Context menu actions can trigger existing game actions
- Actions support different target types (buildings, characters, positions)

### Pathfinding System
- "Get Directions" shows pathfinding information
- "Move Here" can update character pathfinding

### Building System
- Reads building type to determine available actions
- Supports custom building attributes in information display

### Character System
- Displays character attributes in information panels
- Supports character-specific interactions

## API Reference

### InfoPanel Methods

- `show(content: Dict, mouse_pos: Tuple[int, int])` - Show panel with content
- `hide()` - Hide the panel
- `render(surface)` - Render panel to pygame surface

### ContextMenu Methods

- `show(options: List[Dict], mouse_pos: Tuple[int, int], target_object)` - Show menu
- `hide()` - Hide the menu
- `handle_mouse_motion(mouse_pos: Tuple[int, int])` - Update hover states
- `handle_click(mouse_pos: Tuple[int, int]) -> Optional[Dict]` - Handle selection
- `render(surface)` - Render menu to pygame surface

### MapController Methods

- `handle_left_click(position)` - Handle left mouse clicks
- `handle_right_click(position)` - Handle right mouse clicks
- `show_building_context_menu(building, position)` - Show building menu
- `show_character_context_menu(char_id, position)` - Show character menu
- `execute_context_action(option)` - Execute selected menu action
- `get_building_info(building) -> Dict` - Generate building information
- `get_character_info(character) -> Dict` - Generate character information

## Examples

### Basic Usage

```python
# Create map controller with enhanced interactivity
controller = MapController(map_image_path, map_data)

# Handle pygame events
for event in pygame.event.get():
    controller.handle_event(event)

# Render with UI elements
controller.render(screen)
```

### Custom Building Types

```python
# Add custom building with specific type
building = {
    'name': 'Magic Shop',
    'type': 'shop',  # Enables 'Browse Items' option
    'rect': pygame.Rect(100, 100, 50, 50),
    'owner': 'Wizard Bob',
    'speciality': 'Enchanted Items'
}
```

### Extending Actions

```python
# Add custom action handling
def execute_context_action(self, option):
    action = option.get('action')
    target = option.get('target')
    
    if action == 'custom_action':
        # Handle custom action
        self.handle_custom_action(target)
    else:
        # Call original handler
        super().execute_context_action(option)
```

## Testing

The system includes comprehensive unit tests covering:

- InfoPanel functionality (show/hide, positioning, content)
- ContextMenu interactions (hover, click, option selection)  
- MapController event handling and object detection
- Action execution and information generation

Run tests with:
```bash
python test_map_interactivity.py
```

Run demo with:
```bash
python demo_map_interactivity.py
```

## Performance Considerations

- UI elements are only rendered when visible
- Screen boundary checking prevents off-screen rendering
- Event handling short-circuits for hidden elements
- Information generation is lightweight and cached-friendly

## Future Enhancements

Potential areas for future expansion:

- Drag-and-drop interactions
- Multi-select functionality
- Customizable key bindings
- Animation transitions for UI elements
- Touch/gesture support for mobile platforms
- Accessibility features (screen reader support, high contrast)