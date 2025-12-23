"""
Effect Schema v2 - Typed and validated event effects system.

This module provides a formalized schema for event effects with validation,
replacing the ad-hoc dictionary-based approach with a structured, type-safe system.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
from enum import Enum
import logging


class EffectType(str, Enum):
    """Supported effect types."""
    ATTRIBUTE_CHANGE = "attribute_change"
    RELATIONSHIP_CHANGE = "relationship_change"
    LOCATION_CHANGE = "location_change"
    WORLD_STATE_CHANGE = "world_state_change"


class TargetSpec(str, Enum):
    """Standard target specifications."""
    PARTICIPANTS = "participants"
    LOCATION = "location"
    WORLD = "world"
    # Custom targets can be specified as strings (character names, location names, etc.)


class OperatorType(str, Enum):
    """Supported operators for attribute modification."""
    ADD = "add"
    SUBTRACT = "subtract"
    MULTIPLY = "multiply"
    SET = "set"
    MIN = "min"
    MAX = "max"


@dataclass
class EffectCondition:
    """
    Optional conditions that must be met for an effect to apply.
    
    Example:
        EffectCondition(
            attribute="energy",
            operator=">=",
            threshold=50
        )
    """
    attribute: str
    operator: str  # ">=", ">", "<=", "<", "==", "!="
    threshold: Union[int, float, str]
    
    def __post_init__(self):
        """Validate the condition after initialization."""
        valid_operators = [">=", ">", "<=", "<", "==", "!="]
        if self.operator not in valid_operators:
            raise ValueError(
                f"Invalid condition operator: {self.operator}. "
                f"Must be one of {valid_operators}"
            )
    
    def evaluate(self, value: Any) -> bool:
        """Evaluate the condition against a value."""
        if self.operator == ">=":
            return value >= self.threshold
        elif self.operator == ">":
            return value > self.threshold
        elif self.operator == "<=":
            return value <= self.threshold
        elif self.operator == "<":
            return value < self.threshold
        elif self.operator == "==":
            return value == self.threshold
        elif self.operator == "!=":
            return value != self.threshold
        else:
            logging.warning(f"Unknown operator: {self.operator}, defaulting to True")
            return True


@dataclass
class EffectV2:
    """
    Version 2 of the effect schema with full validation and type safety.
    
    Canonical Examples:
    
    1. Simple attribute change:
        EffectV2(
            type=EffectType.ATTRIBUTE_CHANGE,
            targets=["participants"],
            attribute="happiness",
            change_value=10
        )
    
    2. Conditional effect with stacking:
        EffectV2(
            type=EffectType.ATTRIBUTE_CHANGE,
            targets=["participants"],
            attribute="energy",
            change_value=-5,
            conditions=[EffectCondition("energy", ">=", 10)],
            stacking=True
        )
    
    3. Relationship change with chain:
        EffectV2(
            type=EffectType.RELATIONSHIP_CHANGE,
            targets=["participants"],
            attribute="trust",
            change_value=5,
            chain=["friendship_level"]
        )
    """
    
    # Required fields
    type: EffectType
    targets: List[str]
    attribute: str
    
    # Value change (required for most effects)
    change_value: Union[int, float] = 0
    
    # Optional fields for advanced effects
    operator: OperatorType = OperatorType.ADD
    conditions: List[EffectCondition] = field(default_factory=list)
    stacking: bool = True  # Whether multiple instances of this effect stack
    chain: List[str] = field(default_factory=list)  # Attributes to chain/cascade to
    
    # Metadata
    description: Optional[str] = None
    priority: int = 0  # Higher priority effects apply first
    
    def __post_init__(self):
        """Validate the effect after initialization."""
        self.validate()
    
    def validate(self):
        """
        Validate the effect configuration.
        
        Raises:
            ValueError: If the effect configuration is invalid
        """
        # Validate type
        if not isinstance(self.type, EffectType):
            try:
                self.type = EffectType(self.type)
            except ValueError:
                raise ValueError(
                    f"Invalid effect type: {self.type}. "
                    f"Must be one of {[e.value for e in EffectType]}"
                )
        
        # Validate targets
        if not self.targets:
            raise ValueError("Effect must have at least one target")
        
        if not isinstance(self.targets, list):
            raise ValueError(f"targets must be a list, got {type(self.targets)}")
        
        # Validate attribute
        if not self.attribute or not isinstance(self.attribute, str):
            raise ValueError(f"attribute must be a non-empty string, got {self.attribute}")
        
        # Validate operator first before using it in comparisons
        if not isinstance(self.operator, OperatorType):
            try:
                self.operator = OperatorType(self.operator)
            except ValueError:
                raise ValueError(
                    f"Invalid operator: {self.operator}. "
                    f"Must be one of {[e.value for e in OperatorType]}"
                )
        
        # Validate change_value for numeric operations (after operator is validated)
        if self.operator in [OperatorType.ADD, OperatorType.SUBTRACT, OperatorType.MULTIPLY]:
            if not isinstance(self.change_value, (int, float)):
                raise ValueError(
                    f"change_value must be numeric for operator {self.operator}, "
                    f"got {type(self.change_value)}"
                )
        
        # Validate conditions
        if self.conditions and not isinstance(self.conditions, list):
            raise ValueError("conditions must be a list")
        
        for condition in self.conditions:
            if not isinstance(condition, EffectCondition):
                raise ValueError(f"Invalid condition type: {type(condition)}")
        
        # Validate chain
        if self.chain and not isinstance(self.chain, list):
            raise ValueError("chain must be a list of attribute names")
    
    def should_apply(self, entity: Any) -> bool:
        """
        Check if this effect should apply to the given entity based on conditions.
        
        Args:
            entity: The entity to check conditions against
            
        Returns:
            bool: True if all conditions are met, False otherwise
        """
        if not self.conditions:
            return True
        
        for condition in self.conditions:
            # Get attribute value from entity
            value = None
            
            # Try get_state if it exists and returns a dict
            if hasattr(entity, "get_state") and callable(entity.get_state):
                try:
                    state = entity.get_state()
                    if isinstance(state, dict):
                        value = state.get(condition.attribute)
                except (AttributeError, TypeError, KeyError):
                    # get_state may fail or not return expected dict, continue to try direct access
                    pass
            
            # Try direct attribute access if we haven't got a value yet
            if value is None and hasattr(entity, condition.attribute):
                try:
                    value = getattr(entity, condition.attribute)
                except (AttributeError, TypeError):
                    # Attribute access may fail, treat as condition not met
                    pass
            
            if value is None:
                # If attribute doesn't exist, condition fails
                return False
            
            if not condition.evaluate(value):
                return False
        
        return True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert effect to dictionary format (for backward compatibility)."""
        result = {
            "type": self.type.value if isinstance(self.type, EffectType) else self.type,
            "targets": self.targets,
            "attribute": self.attribute,
            "change_value": self.change_value,
            "operator": self.operator.value if isinstance(self.operator, OperatorType) else self.operator,
            "stacking": self.stacking,
        }
        
        if self.conditions:
            result["conditions"] = [
                {
                    "attribute": c.attribute,
                    "operator": c.operator,
                    "threshold": c.threshold
                }
                for c in self.conditions
            ]
        
        if self.chain:
            result["chain"] = self.chain
        
        if self.description:
            result["description"] = self.description
        
        if self.priority != 0:
            result["priority"] = self.priority
        
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EffectV2":
        """
        Create an EffectV2 from a dictionary (for backward compatibility).
        
        Args:
            data: Dictionary containing effect data
            
        Returns:
            EffectV2: Validated effect instance
            
        Raises:
            ValueError: If the data is invalid
        """
        # Extract required fields
        effect_type = data.get("type")
        targets = data.get("targets", [])
        attribute = data.get("attribute")
        
        # Extract optional fields with defaults
        change_value = data.get("change_value", 0)
        operator = data.get("operator", "add")
        stacking = data.get("stacking", True)
        chain = data.get("chain", [])
        description = data.get("description")
        priority = data.get("priority", 0)
        
        # Parse conditions if present
        conditions = []
        if "conditions" in data and data["conditions"]:
            for cond_data in data["conditions"]:
                conditions.append(
                    EffectCondition(
                        attribute=cond_data["attribute"],
                        operator=cond_data["operator"],
                        threshold=cond_data["threshold"]
                    )
                )
        
        return cls(
            type=effect_type,
            targets=targets,
            attribute=attribute,
            change_value=change_value,
            operator=operator,
            conditions=conditions,
            stacking=stacking,
            chain=chain,
            description=description,
            priority=priority
        )


def validate_effect_dict(effect_dict: Dict[str, Any]) -> bool:
    """
    Validate an effect dictionary without creating an EffectV2 instance.
    
    Args:
        effect_dict: Dictionary to validate
        
    Returns:
        bool: True if valid, False otherwise
    """
    try:
        EffectV2.from_dict(effect_dict)
        return True
    except Exception as e:
        logging.error(f"Effect validation failed: {e}")
        return False


def create_canonical_effects() -> Dict[str, EffectV2]:
    """
    Create canonical example effects for documentation and testing.
    
    Returns:
        Dictionary of example effects by name
    """
    return {
        "happiness_boost": EffectV2(
            type=EffectType.ATTRIBUTE_CHANGE,
            targets=["participants"],
            attribute="happiness",
            change_value=10,
            description="Increases participant happiness by 10"
        ),
        
        "conditional_energy_drain": EffectV2(
            type=EffectType.ATTRIBUTE_CHANGE,
            targets=["participants"],
            attribute="energy",
            change_value=-5,
            conditions=[EffectCondition("energy", ">=", 10)],
            description="Drains 5 energy from participants with at least 10 energy"
        ),
        
        "relationship_trust_boost": EffectV2(
            type=EffectType.RELATIONSHIP_CHANGE,
            targets=["participants"],
            attribute="trust",
            change_value=5,
            chain=["friendship_level", "loyalty"],
            description="Increases trust between participants and chains to friendship"
        ),
        
        "location_development": EffectV2(
            type=EffectType.LOCATION_CHANGE,
            targets=["location"],
            attribute="development_level",
            change_value=2,
            operator=OperatorType.ADD,
            description="Increases location development level by 2"
        ),
        
        "world_economy_boost": EffectV2(
            type=EffectType.WORLD_STATE_CHANGE,
            targets=["world"],
            attribute="economic_activity",
            change_value=15,
            stacking=True,
            description="Increases global economic activity by 15 (stacks)"
        ),
    }
