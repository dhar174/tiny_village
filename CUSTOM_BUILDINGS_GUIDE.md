# Custom Buildings Configuration Guide

This guide explains how to define custom buildings for your Tiny Village simulation using the `custom_buildings.json` file.

## Overview

Custom buildings can be loaded from a JSON configuration file, allowing you to design unique village layouts without modifying code. The system supports various building types, each with specific interactions and properties.

## File Format

Buildings are defined in a JSON file with the following structure:

```json
{
  "buildings": [
    {
      "name": "Building Name",
      "type": "building_type",
      "x": 100,
      "y": 150,
      "width": 50,
      "height": 50,
      ...additional properties...
    }
  ]
}
```

## Required Properties

Every building must have these properties:

- **`name`** (string): The display name of the building
- **`x`** (integer): X coordinate on the map
- **`y`** (integer): Y coordinate on the map

## Optional Properties

These properties have sensible defaults if not specified:

- **`type`** (string, default: `"building"`): The building type (see Building Types below)
- **`width`** (integer, default: `40`): Width in pixels
- **`height`** (integer, default: `40`): Height in pixels
- **`length`** (integer, default: same as `height`): Length/depth of the building (3D dimension)
- **`stories`** (integer, default: `1`): Number of floors
- **`num_rooms`** (integer, default: `1`): Number of rooms in the building
- **`address`** (string, default: `""`): Street address
- **`owner`** (string, default: `null`): Name of the building owner
- **`description`** (string, default: `""`): Descriptive text about the building
- **`door`** (object, default: `null`): Door location, e.g., `{"x": 110, "y": 100}`

## Building Types

Each building type determines what interactions are available to characters:

### Civic Buildings (`civic`, `office`)
**Type:** `"civic"`
**Interactions:**
- Enter Building
- Attend Meeting
- Get Information
- File Complaint

**Use for:** Town halls, government offices, community centers

### Commercial Buildings (`commercial`, `shop`)
**Type:** `"commercial"` or `"shop"`
**Interactions:**
- Enter Building
- Browse Goods
- Buy Items
- Trade with Merchants

**Use for:** Marketplaces, shops, trading posts

### Social Buildings (`social`, `tavern`)
**Type:** `"social"` or `"tavern"`
**Interactions:**
- Enter Building
- Socialize with Patrons
- Get a Drink
- Join Activity

**Use for:** Taverns, inns, social clubs, gathering places

### Crafting Buildings (`crafting`, `workshop`)
**Type:** `"crafting"` or `"workshop"`
**Interactions:**
- Enter Building
- Commission Item
- Learn Crafting
- Use Equipment

**Use for:** Blacksmiths, workshops, artisan studios

### Agricultural Buildings (`agricultural`, `farm`)
**Type:** `"agricultural"` or `"farm"`
**Interactions:**
- Enter Building
- Help with Crops
- Gather Food
- Tend Animals

**Use for:** Farms, barns, agricultural facilities

### Educational Buildings (`educational`, `school`, `library`)
**Type:** `"educational"`, `"school"`, or `"library"`
**Interactions:**
- Enter Building
- Attend Class (school only)
- Study Books
- Access Resources

**Use for:** Schools, libraries, learning centers

### Residential Buildings (`residential`, `house`)
**Type:** `"residential"` or `"house"`
**Interactions:**
- Enter Building
- Rest Inside
- Visit Residents
- Use Facilities

**Use for:** Houses, homes, residential buildings

### Generic Buildings
**Type:** `"building"` (or any unrecognized type)
**Interactions:**
- Enter Building (only)

**Use for:** Custom or special-purpose buildings

## Complete Example

Here's a comprehensive example showing various building types:

```json
{
  "buildings": [
    {
      "name": "Grand Town Hall",
      "type": "civic",
      "x": 100,
      "y": 150,
      "width": 60,
      "height": 55,
      "length": 55,
      "stories": 2,
      "num_rooms": 8,
      "address": "1 Main Square",
      "description": "The heart of village governance",
      "owner": "Village Council"
    },
    {
      "name": "Riverside Market",
      "type": "commercial",
      "x": 200,
      "y": 100,
      "width": 45,
      "height": 45,
      "length": 45,
      "stories": 1,
      "num_rooms": 4,
      "address": "12 Commerce Street",
      "description": "A bustling marketplace"
    },
    {
      "name": "The Golden Tavern",
      "type": "tavern",
      "x": 300,
      "y": 200,
      "width": 50,
      "height": 48,
      "length": 48,
      "stories": 2,
      "num_rooms": 6,
      "address": "7 Tavern Lane",
      "description": "Where villagers gather to socialize",
      "owner": "Innkeeper Gerald"
    },
    {
      "name": "Cozy Cottage",
      "type": "residential",
      "x": 250,
      "y": 400,
      "width": 35,
      "height": 30,
      "length": 30,
      "stories": 1,
      "num_rooms": 3,
      "address": "88 Cottage Lane",
      "description": "A small comfortable home"
    }
  ]
}
```

## Custom Properties

You can add your own custom properties beyond the standard ones. These will be preserved when loading buildings and can be used by custom game logic:

```json
{
  "name": "Mysterious Tower",
  "type": "building",
  "x": 500,
  "y": 500,
  "custom_marker": "quest_location",
  "magic_level": 75,
  "tags": ["mystical", "dangerous", "landmark"]
}
```

## Error Handling

The loading system is robust and handles various error conditions:

- **Missing file**: Returns empty building list with warning
- **Invalid JSON**: Returns empty list with error message
- **Missing required fields**: Skips that building with warning
- **Invalid numeric values**: Uses defaults with warning
- **Unrecognized building type**: Uses default interactions with info message

## Loading Custom Buildings

### From Configuration

In your game configuration, specify the buildings file:

```python
config = {
    "map": {
        "buildings_file": "custom_buildings.json"
    }
}
controller = GameplayController(config=config)
```

### Default Location

The system looks for `custom_buildings.json` in the project root by default. You can specify a different path:

```json
{
  "map": {
    "buildings_file": "configs/my_buildings.json"
  }
}
```

## Tips and Best Practices

1. **Spacing**: Leave adequate space between buildings (at least 10-20 pixels) to allow character movement
2. **Clustering**: Group related buildings together (e.g., commercial district, residential area)
3. **Scaling**: Larger important buildings (town halls, markets) should have larger dimensions
4. **Stories**: Multi-story buildings typically have more rooms
5. **Addressing**: Use consistent addressing schemes for immersion
6. **Descriptions**: Add descriptions to enhance the narrative experience

## Validation

After creating or modifying `custom_buildings.json`, you can validate it:

```bash
python -m json.tool custom_buildings.json
```

This will check for JSON syntax errors.

## Testing

Two test suites currently exist for building loading:

- `tests/test_custom_building_loading.py` is the preferred controller-level
  regression example. It exercises a real `GameplayController` while patching
  only the pygame display boundary.
- `tests/test_building_loading_unit.py` is an older, more mock-heavy direct
  loader suite. Keep it as supplementary coverage rather than the primary
  example to follow.

Preferred validation command:

```bash
python -m unittest tests.test_custom_building_loading
```

## Troubleshooting

### Buildings Not Appearing
- Check the JSON syntax is valid
- Verify coordinates are within map bounds
- Ensure required fields (name, x, y) are present

### Wrong Interactions
- Check the `type` field matches a recognized type
- Review the Building Types section for correct type names

### Performance Issues
- Reduce the number of buildings if experiencing lag
- Ensure building coordinates don't overlap excessively

## See Also

- `tiny_buildings.py` - Building class implementation
- `tiny_gameplay_controller.py` - Loading system implementation
- `tests/test_custom_building_loading.py` - Preferred controller-level examples
- `tests/test_building_loading_unit.py` - Legacy direct-loader examples
