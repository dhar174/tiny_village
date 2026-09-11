# Building Functionality System Documentation

## Overview

The Building Functionality System adds economic depth to Tiny Village by implementing:
- **Resource production and consumption** for different building types
- **Services offered** by buildings that characters can use
- **Economic transactions** between characters and buildings
- **Time-based resource generation** tied to the game loop

## Components

### 1. BuildingManager (`tiny_building_manager.py`)

The central manager for all building functionality. Handles:
- Resource inventory tracking for each building
- Production/consumption cycles
- Service provision to characters
- Economic transactions

### 2. ResourcePool

Manages a building's resource inventory with six resource types:
- **Food**: Consumable for hunger reduction
- **Materials**: Input for production processes
- **Tools**: Craftable items and equipment
- **Goods**: Tradeable commodities
- **Services**: Intangible offerings (socializing, education)
- **Knowledge**: Educational resources

### 3. BuildingService

Defines services that buildings can provide to characters:
- **Cost**: Money required from character
- **Resource Requirements**: Resources consumed from building
- **Resource Outputs**: Resources added to building
- **Effects**: Changes applied to character attributes

## Building Types and Their Functionality

### Market / Commercial Buildings
- **Production**: Goods (5/tick)
- **Consumption**: Materials (2/tick)
- **Production Interval**: 10 game ticks
- **Services**:
  - **Buy Goods** (10 money): Purchase goods, gain satisfaction
  - **Sell Goods** (-5 money): Sell goods to market, earn money

### Tavern / Social Buildings
- **Production**: Food (3/tick), Services (5/tick)
- **Consumption**: Materials (1/tick)
- **Production Interval**: 5 game ticks
- **Services**:
  - **Buy Meal** (5 money): Reduces hunger by 3, increases satisfaction
  - **Buy Drink** (3 money): Reduces thirst by 10, increases social wellbeing
  - **Socialize** (Free): Increases social wellbeing by 8

### Blacksmith / Crafting Buildings
- **Production**: Tools (3/tick), Goods (2/tick for workshops)
- **Consumption**: Materials (4-5/tick)
- **Production Interval**: 12-15 game ticks
- **Services**:
  - **Repair Tools** (8 money): Repairs equipment, increases satisfaction
  - **Commission Weapon** (20 money): Crafts weapons, high satisfaction
  - **Craft Item** (10 money): Create goods, improve crafting skill

### Farm / Agricultural Buildings
- **Production**: Food (10/tick)
- **Consumption**: None
- **Production Interval**: 20 game ticks
- **Services**:
  - **Buy Food** (3 money): Purchase food, reduces hunger by 5

### School / Educational Buildings
- **Production**: Knowledge (5-8/tick)
- **Consumption**: None
- **Production Interval**: 10-15 game ticks
- **Services**:
  - **Take Lesson** (5 money): Increases intelligence by 2
  - **Study** (Free, Library): Increases knowledge by 5, costs energy

## Integration with Game Systems

### GameplayController Integration

The BuildingManager is initialized in `GameplayController.initialize_game_systems()`:

```python
self.building_manager = BuildingManager()

# Register existing buildings
for building in self.map_controller.buildings:
    building_id = str(building.get('uuid', building.get('name', 'unknown')))
    building_type = building.get('type', 'building')
    self.building_manager.register_building(building_id, building_type)
```

### Update Loop Integration

Building production is processed in `GameplayController.update_game_state()`:

```python
# Process production for all buildings each update
current_tick = pygame.time.get_ticks()
for building in self.map_controller.buildings:
    building_id = str(building.get('uuid'))
    building_type = building.get('type')
    self.building_manager.process_production(building_id, building_type, current_tick)
```

### Building Class Integration

The `Building` class now integrates with the `BuildingManager`:

```python
# Create a building with manager integration
building = Building(
    name="Village Market",
    x=10, y=20,
    height=10, width=20, length=15,
    building_type="market",
    building_manager=building_manager  # Pass manager reference
)

# Use building services
services = building.get_available_services(character)
success, message = building.provide_service("buy_goods", character)
resources = building.get_resource_levels()
```

## Usage Examples

### Creating Buildings with Functionality

```python
from tiny_building_manager import BuildingManager
from tiny_buildings import Building

# Initialize the manager
manager = BuildingManager()

# Create a tavern
tavern = Building(
    name="Cozy Tavern",
    x=30, y=40,
    height=10, width=20, length=15,
    building_type="tavern",
    building_manager=manager
)

# Tavern is automatically registered and has initial resources
resources = tavern.get_resource_levels()
print(f"Tavern food: {resources['food']}")  # e.g., 30
```

### Processing Production Cycles

```python
# Simulate game ticks
current_tick = 0
for tick in range(0, 100, 5):
    current_tick = tick
    # Process production for the tavern
    tavern.process_production(current_tick)
    
# Check updated resources
resources = tavern.get_resource_levels()
print(f"After production - Food: {resources['food']}, Services: {resources['services']}")
```

### Characters Using Building Services

```python
# Character visits tavern
character = Character(name="Hungry Joe", wealth_money=50, hunger_level=8)

# Get available services
services = tavern.get_available_services(character)
print(f"Available services: {[s.name for s in services]}")

# Character buys a meal
success, message = tavern.provide_service("buy_meal", character)
if success:
    print(f"Success! {character.name} hunger: {character.hunger_level}")
    print(f"Remaining money: {character.wealth_money}")
```

### Economic Transactions

```python
# Create a market
market = Building(
    name="Village Market",
    x=10, y=20,
    height=10, width=20, length=15,
    building_type="market",
    building_manager=manager
)

# Merchant sells goods to market
merchant = Character(name="Trader Alice", wealth_money=50)
success, message = market.provide_service("sell_goods", merchant)
print(f"After selling - Wealth: {merchant.wealth_money}")  # Increased by 5

# Customer buys goods from market
customer = Character(name="Customer Bob", wealth_money=100)
success, message = market.provide_service("buy_goods", customer)
print(f"After buying - Wealth: {customer.wealth_money}")  # Decreased by 10
```

## Testing

The system includes comprehensive test coverage:

### Unit Tests (`tests/test_building_manager.py`)
- 39 tests covering:
  - ResourcePool operations
  - BuildingService validation
  - BuildingManager functionality
  - Production cycles
  - Service provision
  - All building types

### Integration Tests (`tests/test_building_gameplay_integration.py`)
- 11 tests covering:
  - Building-Manager integration
  - Character-Building interactions
  - Economic transaction flows
  - Resource depletion and production
  - Complete village economy scenarios

Run tests with:
```bash
python -m unittest tests.test_building_manager
python -m unittest tests.test_building_gameplay_integration
```

## Extension Points

The system is designed for easy extension:

### Adding New Building Types

1. Add production configuration to `BuildingManager.BUILDING_PRODUCTION`:
```python
"library": {
    "produces": {ResourceType.KNOWLEDGE: 5},
    "consumes": {},
    "production_interval": 10
}
```

2. Add services to `BuildingManager.BUILDING_SERVICES`:
```python
"library": {
    "study": BuildingService(
        name="Study",
        cost=0,
        resource_requirements={ResourceType.KNOWLEDGE: 2},
        effects=[...]
    )
}
```

3. Add to `BUILDING_TYPE_INTERACTIONS` in `tiny_buildings.py`

### Adding New Resource Types

1. Add to `ResourceType` enum
2. Add field to `ResourcePool` dataclass
3. Update initialization in `BuildingManager._initialize_building_resources()`

### Adding New Services

Add to the appropriate building type in `BUILDING_SERVICES`:
```python
"tavern": {
    "rent_room": BuildingService(
        name="Rent Room",
        cost=10,
        resource_requirements={},
        effects=[
            {"targets": ["initiator"], "attribute": "energy", "change_value": 30}
        ]
    )
}
```

## Performance Considerations

- **Resource Pools**: Lightweight dataclasses with minimal overhead
- **Production Caching**: Production intervals prevent excessive calculations
- **Service History**: Limited to last 1000 transactions
- **Registration**: Buildings registered once during initialization

## Future Enhancements

Potential improvements for the system:
1. **Dynamic Pricing**: Adjust service costs based on supply/demand
2. **Building Upgrades**: Improve production rates or add services
3. **Quality Levels**: Different quality tiers for resources
4. **Trading Networks**: Buildings trade resources with each other
5. **Specialized Workers**: NPCs assigned to work at buildings
6. **Seasonal Effects**: Production varies by time of year
7. **Building Deterioration**: Maintenance requirements over time
8. **Resource Storage Limits**: Maximum capacity per building

## Conclusion

The Building Functionality System provides a robust economic foundation for Tiny Village, enabling:
- Meaningful character-building interactions
- Resource-based economy
- Time-based production cycles
- Service-based gameplay mechanics

The system integrates seamlessly with existing game systems while remaining extensible for future features.
