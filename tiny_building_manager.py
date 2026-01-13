"""
Building Manager System for Tiny Village

This module manages building functionality including:
- Resource production and consumption
- Services offered by different building types
- Economic transactions
- Building state tracking
"""

import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class ResourceType(Enum):
    """Types of resources that can be produced/consumed."""
    FOOD = "food"
    MATERIALS = "materials"
    TOOLS = "tools"
    GOODS = "goods"
    SERVICES = "services"
    KNOWLEDGE = "knowledge"


@dataclass
class ResourcePool:
    """Represents a pool of resources at a building."""
    food: int = 0
    materials: int = 0
    tools: int = 0
    goods: int = 0
    services: int = 0
    knowledge: int = 0
    
    def add(self, resource_type: ResourceType, amount: int) -> None:
        """Add resources to the pool."""
        if amount < 0:
            raise ValueError("Cannot add negative resources")
        current = getattr(self, resource_type.value)
        setattr(self, resource_type.value, current + amount)
    
    def consume(self, resource_type: ResourceType, amount: int) -> bool:
        """
        Try to consume resources from the pool.
        Returns True if successful, False if insufficient resources.
        """
        if amount < 0:
            raise ValueError("Cannot consume negative resources")
        current = getattr(self, resource_type.value)
        if current >= amount:
            setattr(self, resource_type.value, current - amount)
            return True
        return False
    
    def get(self, resource_type: ResourceType) -> int:
        """Get the current amount of a resource."""
        return getattr(self, resource_type.value)
    
    def to_dict(self) -> Dict[str, int]:
        """Convert to dictionary representation."""
        return {
            "food": self.food,
            "materials": self.materials,
            "tools": self.tools,
            "goods": self.goods,
            "services": self.services,
            "knowledge": self.knowledge
        }


@dataclass
class BuildingService:
    """Represents a service offered by a building."""
    name: str
    cost: int  # Cost in money
    resource_requirements: Dict[ResourceType, int] = field(default_factory=dict)
    resource_outputs: Dict[ResourceType, int] = field(default_factory=dict)
    duration: int = 1  # Time in game ticks
    effects: List[Dict[str, Any]] = field(default_factory=list)
    
    def can_provide(self, character, building_resources: ResourcePool) -> bool:
        """Check if service can be provided based on character wealth and building resources."""
        # Check character wealth
        if hasattr(character, 'wealth_money') and character.wealth_money < self.cost:
            return False
        
        # Check building has required resources
        for resource_type, required_amount in self.resource_requirements.items():
            if building_resources.get(resource_type) < required_amount:
                return False
        
        return True


class BuildingManager:
    """
    Manages building functionality for the game.
    Handles resource production, consumption, and services.
    """
    
    # Building type to production configuration
    BUILDING_PRODUCTION = {
        "market": {
            "produces": {ResourceType.GOODS: 5},
            "consumes": {ResourceType.MATERIALS: 2},
            "production_interval": 10  # Every 10 game ticks
        },
        "commercial": {
            "produces": {ResourceType.GOODS: 5},
            "consumes": {ResourceType.MATERIALS: 2},
            "production_interval": 10
        },
        "tavern": {
            "produces": {ResourceType.FOOD: 3, ResourceType.SERVICES: 5},
            "consumes": {ResourceType.MATERIALS: 1},
            "production_interval": 5
        },
        "social": {
            "produces": {ResourceType.SERVICES: 5},
            "consumes": {},
            "production_interval": 5
        },
        "blacksmith": {
            "produces": {ResourceType.TOOLS: 3},
            "consumes": {ResourceType.MATERIALS: 5},
            "production_interval": 15
        },
        "crafting": {
            "produces": {ResourceType.TOOLS: 3, ResourceType.GOODS: 2},
            "consumes": {ResourceType.MATERIALS: 4},
            "production_interval": 12
        },
        "workshop": {
            "produces": {ResourceType.TOOLS: 3, ResourceType.GOODS: 2},
            "consumes": {ResourceType.MATERIALS: 4},
            "production_interval": 12
        },
        "farm": {
            "produces": {ResourceType.FOOD: 10},
            "consumes": {},
            "production_interval": 20
        },
        "agricultural": {
            "produces": {ResourceType.FOOD: 10},
            "consumes": {},
            "production_interval": 20
        },
        "school": {
            "produces": {ResourceType.KNOWLEDGE: 8},
            "consumes": {},
            "production_interval": 15
        },
        "educational": {
            "produces": {ResourceType.KNOWLEDGE: 8},
            "consumes": {},
            "production_interval": 15
        },
        "library": {
            "produces": {ResourceType.KNOWLEDGE: 5},
            "consumes": {},
            "production_interval": 10
        }
    }
    
    # Building type to services configuration
    BUILDING_SERVICES = {
        "market": {
            "buy_goods": BuildingService(
                name="Buy Goods",
                cost=10,
                resource_requirements={ResourceType.GOODS: 1},
                resource_outputs={},
                effects=[
                    {"targets": ["initiator"], "attribute": "current_satisfaction", "change_value": 5}
                ]
            ),
            "sell_goods": BuildingService(
                name="Sell Goods",
                cost=-5,  # Negative cost = gain money
                resource_requirements={},
                resource_outputs={ResourceType.GOODS: 1},
                effects=[
                    {"targets": ["initiator"], "attribute": "wealth_money", "change_value": 5}
                ]
            ),
        },
        "commercial": {
            "buy_goods": BuildingService(
                name="Buy Goods",
                cost=10,
                resource_requirements={ResourceType.GOODS: 1},
                resource_outputs={},
                effects=[
                    {"targets": ["initiator"], "attribute": "current_satisfaction", "change_value": 5}
                ]
            ),
        },
        "tavern": {
            "buy_meal": BuildingService(
                name="Buy Meal",
                cost=5,
                resource_requirements={ResourceType.FOOD: 2},
                resource_outputs={},
                effects=[
                    {"targets": ["initiator"], "attribute": "hunger_level", "change_value": -3},
                    {"targets": ["initiator"], "attribute": "current_satisfaction", "change_value": 3}
                ]
            ),
            "buy_drink": BuildingService(
                name="Buy Drink",
                cost=3,
                resource_requirements={ResourceType.FOOD: 1},
                resource_outputs={},
                effects=[
                    {"targets": ["initiator"], "attribute": "thirst", "change_value": -10},
                    {"targets": ["initiator"], "attribute": "social_wellbeing", "change_value": 2}
                ]
            ),
        },
        "social": {
            "socialize": BuildingService(
                name="Socialize",
                cost=0,
                resource_requirements={},
                resource_outputs={},
                effects=[
                    {"targets": ["initiator"], "attribute": "social_wellbeing", "change_value": 8}
                ]
            ),
        },
        "blacksmith": {
            "repair_tools": BuildingService(
                name="Repair Tools",
                cost=8,
                resource_requirements={ResourceType.MATERIALS: 2},
                resource_outputs={ResourceType.TOOLS: 1},
                effects=[
                    {"targets": ["initiator"], "attribute": "current_satisfaction", "change_value": 5}
                ]
            ),
            "commission_weapon": BuildingService(
                name="Commission Weapon",
                cost=20,
                resource_requirements={ResourceType.MATERIALS: 5, ResourceType.TOOLS: 1},
                resource_outputs={ResourceType.TOOLS: 2},
                effects=[
                    {"targets": ["initiator"], "attribute": "current_satisfaction", "change_value": 10}
                ]
            ),
        },
        "crafting": {
            "craft_item": BuildingService(
                name="Craft Item",
                cost=10,
                resource_requirements={ResourceType.MATERIALS: 3},
                resource_outputs={ResourceType.GOODS: 2},
                effects=[
                    {"targets": ["initiator"], "attribute": "skills.crafting", "change_value": 1},
                    {"targets": ["initiator"], "attribute": "current_satisfaction", "change_value": 5}
                ]
            ),
        },
        "workshop": {
            "craft_item": BuildingService(
                name="Craft Item",
                cost=10,
                resource_requirements={ResourceType.MATERIALS: 3},
                resource_outputs={ResourceType.GOODS: 2},
                effects=[
                    {"targets": ["initiator"], "attribute": "skills.crafting", "change_value": 1},
                    {"targets": ["initiator"], "attribute": "current_satisfaction", "change_value": 5}
                ]
            ),
        },
        "farm": {
            "buy_food": BuildingService(
                name="Buy Food",
                cost=3,
                resource_requirements={ResourceType.FOOD: 5},
                resource_outputs={},
                effects=[
                    {"targets": ["initiator"], "attribute": "hunger_level", "change_value": -5}
                ]
            ),
        },
        "agricultural": {
            "buy_food": BuildingService(
                name="Buy Food",
                cost=3,
                resource_requirements={ResourceType.FOOD: 5},
                resource_outputs={},
                effects=[
                    {"targets": ["initiator"], "attribute": "hunger_level", "change_value": -5}
                ]
            ),
        },
        "school": {
            "take_lesson": BuildingService(
                name="Take Lesson",
                cost=5,
                resource_requirements={ResourceType.KNOWLEDGE: 3},
                resource_outputs={},
                effects=[
                    {"targets": ["initiator"], "attribute": "intelligence", "change_value": 2},
                    {"targets": ["initiator"], "attribute": "current_satisfaction", "change_value": 3}
                ]
            ),
        },
        "educational": {
            "take_lesson": BuildingService(
                name="Take Lesson",
                cost=5,
                resource_requirements={ResourceType.KNOWLEDGE: 3},
                resource_outputs={},
                effects=[
                    {"targets": ["initiator"], "attribute": "intelligence", "change_value": 2},
                    {"targets": ["initiator"], "attribute": "current_satisfaction", "change_value": 3}
                ]
            ),
        },
        "library": {
            "study": BuildingService(
                name="Study",
                cost=0,
                resource_requirements={ResourceType.KNOWLEDGE: 2},
                resource_outputs={},
                effects=[
                    {"targets": ["initiator"], "attribute": "knowledge", "change_value": 5},
                    {"targets": ["initiator"], "attribute": "energy", "change_value": -3}
                ]
            ),
        },
    }
    
    def __init__(self):
        """Initialize the building manager."""
        self.building_resources: Dict[str, ResourcePool] = {}
        self.last_production_tick: Dict[str, int] = {}
        self.service_history: List[Dict[str, Any]] = []
        
    def register_building(self, building_id: str, building_type: str) -> None:
        """
        Register a building with the manager.
        
        Args:
            building_id: Unique identifier for the building
            building_type: Type of building (market, tavern, etc.)
        """
        if building_id not in self.building_resources:
            self.building_resources[building_id] = ResourcePool()
            self.last_production_tick[building_id] = 0
            logger.info(f"Registered building {building_id} of type {building_type}")
            
            # Initialize with some starting resources
            self._initialize_building_resources(building_id, building_type)
    
    def _initialize_building_resources(self, building_id: str, building_type: str) -> None:
        """Initialize building with starting resources based on type."""
        resources = self.building_resources[building_id]
        
        # Give buildings an initial stockpile
        if building_type in ["market", "commercial"]:
            resources.add(ResourceType.GOODS, 20)
            resources.add(ResourceType.MATERIALS, 10)
        elif building_type in ["tavern", "social"]:
            resources.add(ResourceType.FOOD, 30)
            resources.add(ResourceType.SERVICES, 20)
        elif building_type in ["blacksmith", "crafting", "workshop"]:
            resources.add(ResourceType.MATERIALS, 50)
            resources.add(ResourceType.TOOLS, 10)
        elif building_type in ["farm", "agricultural"]:
            resources.add(ResourceType.FOOD, 100)
        elif building_type in ["school", "educational", "library"]:
            resources.add(ResourceType.KNOWLEDGE, 50)
    
    def process_production(self, building_id: str, building_type: str, current_tick: int) -> bool:
        """
        Process resource production for a building.
        
        Args:
            building_id: Building identifier
            building_type: Type of building
            current_tick: Current game tick
            
        Returns:
            True if production occurred, False otherwise
        """
        if building_id not in self.building_resources:
            self.register_building(building_id, building_type)
            return False
        
        # Get production config for this building type
        prod_config = self.BUILDING_PRODUCTION.get(building_type)
        if not prod_config:
            return False
        
        # Check if it's time to produce
        last_tick = self.last_production_tick.get(building_id, 0)
        if current_tick - last_tick < prod_config["production_interval"]:
            return False
        
        # Process consumption first
        resources = self.building_resources[building_id]
        for resource_type, amount in prod_config.get("consumes", {}).items():
            if not resources.consume(resource_type, amount):
                logger.debug(f"Building {building_id} cannot produce - insufficient {resource_type.value}")
                return False
        
        # Process production
        for resource_type, amount in prod_config.get("produces", {}).items():
            resources.add(resource_type, amount)
        
        self.last_production_tick[building_id] = current_tick
        logger.debug(f"Building {building_id} produced resources at tick {current_tick}")
        return True
    
    def get_available_services(self, building_type: str, character) -> List[BuildingService]:
        """
        Get list of services available at a building type for a character.
        
        Args:
            building_type: Type of building
            character: Character requesting services
            
        Returns:
            List of available services
        """
        services_config = self.BUILDING_SERVICES.get(building_type, {})
        return list(services_config.values())
    
    def provide_service(
        self,
        building_id: str,
        building_type: str,
        service_name: str,
        character
    ) -> Tuple[bool, str]:
        """
        Provide a service to a character.
        
        Args:
            building_id: Building identifier
            building_type: Type of building
            service_name: Name of service to provide
            character: Character receiving service
            
        Returns:
            Tuple of (success, message)
        """
        # Get service configuration
        services = self.BUILDING_SERVICES.get(building_type, {})
        service = services.get(service_name.lower().replace(" ", "_"))
        
        if not service:
            return False, f"Service '{service_name}' not available at {building_type}"
        
        # Get building resources
        if building_id not in self.building_resources:
            self.register_building(building_id, building_type)
        
        resources = self.building_resources[building_id]
        
        # Check if service can be provided
        if not service.can_provide(character, resources):
            return False, "Insufficient resources or money for service"
        
        # Process service transaction
        try:
            # Ensure character can participate in monetary transactions when required
            if service.cost != 0 and not hasattr(character, 'wealth_money'):
                logger.error(
                    "Character '%s' lacks 'wealth_money' attribute required for service '%s' "
                    "with non-zero cost at building '%s'",
                    getattr(character, 'name', 'Unknown'),
                    service_name,
                    building_id,
                )
                return False, "Character cannot participate in monetary transactions for this service"

            # Deduct or pay cost to character
            if service.cost > 0:
                character.wealth_money = max(0, character.wealth_money - service.cost)
            elif service.cost < 0:
                # Service pays the character
                character.wealth_money += abs(service.cost)
            
            # Consume building resources
            for resource_type, amount in service.resource_requirements.items():
                resources.consume(resource_type, amount)
            
            # Add output resources to building
            for resource_type, amount in service.resource_outputs.items():
                resources.add(resource_type, amount)
            
            # Apply effects to character
            for effect in service.effects:
                self._apply_service_effect(effect, character)
            
            # Record service history
            self.service_history.append({
                "building_id": building_id,
                "building_type": building_type,
                "service": service_name,
                "character": getattr(character, 'name', 'Unknown'),
                "cost": service.cost
            })
            
            # Keep history limited
            if len(self.service_history) > 1000:
                self.service_history = self.service_history[-1000:]
            
            return True, f"Service '{service_name}' provided successfully"
            
        except Exception as e:
            logger.error(f"Error providing service: {e}")
            return False, f"Error providing service: {str(e)}"
    
    def _apply_service_effect(self, effect: Dict[str, Any], character) -> None:
        """Apply a service effect to a character."""
        try:
            attribute = effect.get("attribute")
            change_value = effect.get("change_value", 0)
            
            if not attribute:
                return
            
            # Handle nested attributes (e.g., "skills.crafting")
            if "." in attribute:
                parts = attribute.split(".")
                obj = character
                for part in parts[:-1]:
                    if hasattr(obj, part):
                        obj = getattr(obj, part)
                    else:
                        return
                
                final_attr = parts[-1]
                if hasattr(obj, final_attr):
                    current_value = getattr(obj, final_attr, 0)
                    setattr(obj, final_attr, current_value + change_value)
            else:
                # Direct attribute
                if hasattr(character, attribute):
                    current_value = getattr(character, attribute, 0)
                    new_value = current_value + change_value
                    
                    # Apply bounds if applicable
                    if attribute in ["hunger_level", "thirst"]:
                        new_value = max(0, min(10, new_value))
                    elif attribute in ["energy", "social_wellbeing", "current_satisfaction"]:
                        new_value = max(0, min(100, new_value))
                    
                    setattr(character, attribute, new_value)
                    
        except Exception as e:
            logger.warning(f"Error applying service effect: {e}")
    
    def get_building_resources(self, building_id: str) -> Optional[Dict[str, int]]:
        """
        Get current resource levels for a building.
        
        Args:
            building_id: Building identifier
            
        Returns:
            Dictionary of resource levels or None if building not found
        """
        if building_id in self.building_resources:
            return self.building_resources[building_id].to_dict()
        return None
    
    def get_building_info(self, building_id: str, building_type: str) -> Dict[str, Any]:
        """
        Get comprehensive information about a building.
        
        Args:
            building_id: Building identifier
            building_type: Type of building
            
        Returns:
            Dictionary with building information
        """
        resources = self.get_building_resources(building_id)
        services = [service.name for service in self.get_available_services(building_type, None)]
        
        prod_config = self.BUILDING_PRODUCTION.get(building_type, {})
        
        return {
            "building_id": building_id,
            "building_type": building_type,
            "resources": resources,
            "available_services": services,
            "production_interval": prod_config.get("production_interval", 0),
            "produces": {rt.value: amt for rt, amt in prod_config.get("produces", {}).items()},
            "consumes": {rt.value: amt for rt, amt in prod_config.get("consumes", {}).items()},
        }
