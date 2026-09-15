import logging
from typing import Dict, Iterable, List, Optional, Protocol, TypedDict

from tiny_building_manager import BuildingManager, ResourceType
from tiny_items import ItemInventory


logger = logging.getLogger(__name__)


class EconomicItem:
    def __init__(
        self,
        name: str,
        description: str,
        value: int,
        weight: int,
        quantity: int,
        item_type: str = "misc",
        item_subtype: Optional[str] = None,
        status: str = "new",
        coordinates_location=(0, 0),
    ):
        self.name = name
        self.description = description
        self.value = value
        self.weight = weight
        self.quantity = quantity
        self.item_type = item_type
        self.item_subtype = item_subtype or name
        self.status = status
        self.coordinates_location = coordinates_location
        self.type_specific_attributes = False
        self.usability = True
        self.ownership_history = ["economic_simulation"]

    def get_name(self):
        return self.name

    def get_description(self):
        return self.description

    def get_value(self):
        return self.value

    def get_weight(self):
        return self.weight

    def get_quantity(self):
        return self.quantity


class EconomicFoodItem(EconomicItem):
    def __init__(
        self,
        name: str,
        description: str,
        value: int,
        weight: int,
        quantity: int,
        calories: int = 200,
        perishable: bool = True,
        cooked: bool = False,
        coordinates_location=(0, 0),
    ):
        super().__init__(
            name=name,
            description=description,
            value=value,
            weight=weight,
            quantity=quantity,
            item_type="food",
            item_subtype=name,
            coordinates_location=coordinates_location,
        )
        self.calories = calories
        self.perishable = perishable
        self.cooked = cooked
        self.type_specific_attributes = True

    def get_calories(self):
        return self.calories


class JobOutputConfig(TypedDict, total=False):
    resource_type: ResourceType
    item_name: str
    quantity: int
    value: int
    weight: int
    calories: int


class EconomicActor(Protocol):
    name: str
    job: object
    wealth_money: int
    hunger_level: int
    inventory: ItemInventory


class EconomicSimulation:
    """Lightweight economic bridge for jobs, needs, and trading."""

    DEFAULT_JOB_OUTPUTS = {
        "farmer": {
            "resource_type": ResourceType.FOOD,
            "item_name": "Farm Produce",
            "quantity": 2,
            "value": 3,
            "weight": 1,
            "calories": 250,
        },
        "merchant": {
            "resource_type": ResourceType.GOODS,
            "item_name": "Trade Goods",
            "quantity": 2,
            "value": 6,
            "weight": 1,
        },
        "blacksmith": {
            "resource_type": ResourceType.TOOLS,
            "item_name": "Forged Tools",
            "quantity": 1,
            "value": 12,
            "weight": 4,
        },
        "artisan": {
            "resource_type": ResourceType.GOODS,
            "item_name": "Crafted Goods",
            "quantity": 1,
            "value": 8,
            "weight": 2,
        },
        "teacher": {
            "resource_type": ResourceType.KNOWLEDGE,
            "item_name": "Lesson Notes",
            "quantity": 1,
            "value": 5,
            "weight": 1,
        },
        "librarian": {
            "resource_type": ResourceType.KNOWLEDGE,
            "item_name": "Research Notes",
            "quantity": 1,
            "value": 5,
            "weight": 1,
        },
        "builder": {
            "resource_type": ResourceType.MATERIALS,
            "item_name": "Building Materials",
            "quantity": 2,
            "value": 4,
            "weight": 3,
        },
    }

    DEFAULT_NEED_RULES = {
        "hunger_level": {
            "threshold": 5,
            "item_type": "food",
            "reduction": 3,
        }
    }

    RESOURCE_ITEM_TYPES = {
        ResourceType.FOOD: "food",
        ResourceType.TOOLS: "tools",
        ResourceType.GOODS: "misc",
        ResourceType.MATERIALS: "misc",
        ResourceType.SERVICES: "misc",
        ResourceType.KNOWLEDGE: "misc",
    }

    def __init__(
        self,
        building_manager: Optional[BuildingManager] = None,
        production_interval: int = 10,
    ):
        self.building_manager = building_manager
        self.production_interval = production_interval
        self.item_availability: Dict[str, int] = {}
        self.job_outputs = dict(self.DEFAULT_JOB_OUTPUTS)
        self.need_rules = dict(self.DEFAULT_NEED_RULES)
        self._last_job_production: Dict[str, int] = {}

    def register_job_output(
        self,
        job_name: str,
        resource_type: ResourceType,
        quantity: int = 1,
        item_name: Optional[str] = None,
        value: int = 1,
        weight: int = 1,
        calories: int = 0,
    ) -> None:
        self.job_outputs[self._normalize_name(job_name)] = {
            "resource_type": resource_type,
            "item_name": item_name or resource_type.value.replace("_", " ").title(),
            "quantity": quantity,
            "value": value,
            "weight": weight,
            "calories": calories,
        }

    def process_economy(
        self,
        characters: Iterable[EconomicActor],
        current_tick: int,
        building_manager: Optional[BuildingManager] = None,
    ) -> Dict[str, int]:
        for character in characters:
            self.produce_items_for_job(character, current_tick=current_tick)
            self.consume_items_for_needs(character)
        return self.sync_item_availability(building_manager=building_manager, characters=characters)

    def produce_items_for_job(
        self,
        character: EconomicActor,
        current_tick: Optional[int] = None,
    ) -> List[EconomicItem]:
        job_key = self._get_job_key(getattr(character, "job", None))
        if not job_key:
            return []

        config = self._resolve_job_output(job_key)
        if not config:
            return []

        character_key = self._get_character_key(character)
        if current_tick is not None:
            last_tick = self._last_job_production.get(character_key)
            if last_tick is not None and current_tick - last_tick < self.production_interval:
                return []

        produced_item = self._create_item(
            config,
            action_system=getattr(character, "action_system", None),
        )
        inventory = self._get_or_create_inventory(character)
        self._inventory_add(inventory, produced_item)

        if current_tick is not None:
            self._last_job_production[character_key] = current_tick

        logger.debug(
            "Produced %s x%s for %s via job %s",
            produced_item.get_name(),
            produced_item.get_quantity(),
            getattr(character, "name", "unknown"),
            job_key,
        )
        return [produced_item]

    def consume_items_for_needs(self, character: EconomicActor) -> Optional[EconomicItem]:
        inventory = self._get_or_create_inventory(character)

        for need_name, rule in self.need_rules.items():
            current_value = getattr(character, need_name, 0)
            if current_value <= rule["threshold"]:
                continue

            item = self._find_first_item_of_type(inventory, rule["item_type"])
            if item is None:
                continue

            consumed_item = self._clone_item(item, 1, action_system=getattr(character, "action_system", None))
            self._inventory_remove(inventory, consumed_item)

            reduction = rule["reduction"]
            if hasattr(item, "get_calories"):
                reduction = max(reduction, max(1, int(item.get_calories() / 100)))

            setattr(character, need_name, max(0, current_value - reduction))
            logger.debug(
                "Consumed %s for %s; %s reduced from %s to %s",
                consumed_item.get_name(),
                getattr(character, "name", "unknown"),
                need_name,
                current_value,
                getattr(character, need_name, 0),
            )
            return consumed_item

        return None

    def trade_item(
        self,
        seller: EconomicActor,
        buyer: EconomicActor,
        item_name: str,
        quantity: int = 1,
        unit_price: Optional[int] = None,
    ) -> tuple[bool, str]:
        if quantity <= 0:
            return False, "Quantity must be positive"

        seller_inventory = self._get_or_create_inventory(seller)
        buyer_inventory = self._get_or_create_inventory(buyer)
        item = self._find_item_by_name(seller_inventory, item_name)
        if item is None or item.get_quantity() < quantity:
            return False, f"Seller lacks {quantity} {item_name}"

        price = unit_price if unit_price is not None else item.get_value()
        total_cost = price * quantity
        if getattr(buyer, "wealth_money", 0) < total_cost:
            return False, "Buyer cannot afford trade"

        transferred_item = self._clone_item(
            item,
            quantity,
            action_system=getattr(buyer, "action_system", None),
        )
        self._inventory_remove(seller_inventory, transferred_item)
        self._inventory_add(buyer_inventory, transferred_item)

        seller.wealth_money = getattr(seller, "wealth_money", 0) + total_cost
        buyer.wealth_money = getattr(buyer, "wealth_money", 0) - total_cost

        logger.debug(
            "Traded %s x%s from %s to %s for %s",
            item_name,
            quantity,
            getattr(seller, "name", "seller"),
            getattr(buyer, "name", "buyer"),
            total_cost,
        )
        return True, f"Traded {quantity} {item_name} for {total_cost}"

    def sync_item_availability(
        self,
        building_manager: Optional[BuildingManager] = None,
        characters: Optional[Iterable[EconomicActor]] = None,
    ) -> Dict[str, int]:
        availability: Dict[str, int] = {}
        manager = building_manager or self.building_manager

        if manager is not None:
            for pool in manager.building_resources.values():
                for resource_name, amount in pool.to_dict().items():
                    availability[resource_name] = availability.get(resource_name, 0) + amount

        if characters is not None:
            for character in characters:
                inventory = self._get_or_create_inventory(character)
                for item in inventory.get_all_items():
                    availability[item.get_name()] = availability.get(item.get_name(), 0) + item.get_quantity()

        self.item_availability = availability
        return availability

    def _resolve_job_output(self, job_key: str) -> Optional[JobOutputConfig]:
        if job_key in self.job_outputs:
            return self.job_outputs[job_key]

        for known_job, config in self.job_outputs.items():
            if known_job in job_key or job_key in known_job:
                return config

        return None

    def _get_or_create_inventory(self, character) -> ItemInventory:
        inventory = getattr(character, "inventory", None)
        if inventory is None:
            inventory = self._create_inventory()
            setattr(character, "inventory", inventory)
        return inventory

    def _find_first_item_of_type(self, inventory: ItemInventory, item_type: str) -> Optional[EconomicItem]:
        for item in self._refresh_inventory(inventory):
            if getattr(item, "item_type", None) == item_type and item.get_quantity() > 0:
                return item
        return None

    def _find_item_by_name(self, inventory: ItemInventory, item_name: str) -> Optional[EconomicItem]:
        for item in self._refresh_inventory(inventory):
            if item.get_name() == item_name and item.get_quantity() > 0:
                return item
        return None

    def _create_item(self, config: JobOutputConfig, action_system=None) -> EconomicItem:
        quantity = config["quantity"]
        item_name = config["item_name"]
        resource_type = config["resource_type"]
        value = config.get("value", 1)
        weight = config.get("weight", 1)

        if resource_type == ResourceType.FOOD:
            return EconomicFoodItem(
                name=item_name,
                description=f"{item_name} produced through village labor",
                value=value,
                weight=weight,
                quantity=quantity,
                calories=config.get("calories", 200),
            )

        return EconomicItem(
            name=item_name,
            description=f"{item_name} produced through village labor",
            value=value,
            weight=weight,
            quantity=quantity,
            item_type=self.RESOURCE_ITEM_TYPES.get(resource_type, "misc"),
        )

    def _clone_item(self, item: EconomicItem, quantity: int, action_system=None) -> EconomicItem:
        if getattr(item, "item_type", None) == "food":
            return EconomicFoodItem(
                name=item.get_name(),
                description=item.get_description(),
                value=item.get_value(),
                weight=item.get_weight(),
                quantity=quantity,
                calories=item.get_calories(),
                perishable=getattr(item, "perishable", True),
                cooked=getattr(item, "cooked", False),
                coordinates_location=getattr(item, "coordinates_location", (0, 0)),
            )

        return EconomicItem(
            name=item.get_name(),
            description=item.get_description(),
            value=item.get_value(),
            weight=item.get_weight(),
            quantity=quantity,
            item_type=getattr(item, "item_type", "misc"),
            item_subtype=getattr(item, "item_subtype", None),
            status=getattr(item, "status", "new"),
            coordinates_location=getattr(item, "coordinates_location", (0, 0)),
        )

    def _get_job_key(self, job) -> str:
        if job is None:
            return ""
        if isinstance(job, str):
            return self._normalize_name(job)
        if hasattr(job, "get_job_name"):
            return self._normalize_name(job.get_job_name())
        if hasattr(job, "job_name"):
            return self._normalize_name(job.job_name)
        return self._normalize_name(str(job))

    def _get_character_key(self, character) -> str:
        if hasattr(character, "uuid"):
            return str(character.uuid)
        return str(id(character))

    def _normalize_name(self, value: str) -> str:
        return value.strip().lower().replace("_", " ")

    def _create_inventory(self) -> ItemInventory:
        return ItemInventory([], [], [], [], [], [])

    def _inventory_add(self, inventory: ItemInventory, item: EconomicItem) -> None:
        item_list = self._get_inventory_list(inventory, getattr(item, "item_type", "misc"))
        for existing_item in item_list:
            if existing_item.get_name() == item.get_name():
                existing_item.quantity += item.get_quantity()
                self._refresh_inventory(inventory)
                return
        item_list.append(item)
        self._refresh_inventory(inventory)

    def _inventory_remove(self, inventory: ItemInventory, item: EconomicItem) -> bool:
        item_list = self._get_inventory_list(inventory, getattr(item, "item_type", "misc"))
        for existing_item in item_list:
            if existing_item.get_name() == item.get_name():
                existing_item.quantity -= item.get_quantity()
                if existing_item.quantity <= 0:
                    item_list.remove(existing_item)
                self._refresh_inventory(inventory)
                return True
        return False

    def _get_inventory_list(self, inventory: ItemInventory, item_type: str):
        item_lists = {
            "food": inventory.food_items,
            "clothing": inventory.clothing_items,
            "tools": inventory.tools_items,
            "weapons": inventory.weapons_items,
            "medicine": inventory.medicine_items,
            "misc": inventory.misc_items,
        }
        return item_lists.get(item_type, inventory.misc_items)

    def _refresh_inventory(self, inventory: ItemInventory) -> List[EconomicItem]:
        return inventory.get_all_items()
