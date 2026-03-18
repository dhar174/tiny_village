"""
Character Attribute Mapper

Maps event effect attribute names to actual Character class attributes.
Handles aliases and provides safe attribute access.
"""

from typing import Any, Optional, Dict, Tuple
import logging


class AttributeMapper:
    """
    Maps generic effect attribute names to actual Character attributes.
    
    This handles the mismatch between event template attribute names
    (e.g., "happiness", "health") and actual Character field names
    (e.g., "social_wellbeing", "health_status").
    """
    
    # Attribute mapping with bounds (min, max) and default values
    ATTRIBUTE_MAP = {
        # Template name -> (actual_attribute, min, max, default)
        
        # Health-related
        "health": ("health_status", 0, 10, 5),
        "health_status": ("health_status", 0, 10, 5),
        
        # Energy-related
        "energy": ("energy", 0, 10, 8),
        
        # Mental/Social wellbeing
        "happiness": ("social_wellbeing", 0, 10, 5),
        "social_wellbeing": ("social_wellbeing", 0, 10, 5),
        "mental_health": ("mental_health", 0, 10, 7),
        "morale": ("mental_health", 0, 10, 7),
        
        # Resources
        "wealth": ("wealth_money", 0, None, 50),
        "wealth_money": ("wealth_money", 0, None, 50),
        "money": ("wealth_money", 0, None, 50),
        
        # Hunger
        "hunger": ("hunger_level", 0, 10, 3),
        "hunger_level": ("hunger_level", 0, 10, 3),
        
        # Job/Work
        # Note: The default of 20 for job_performance here is an effect/mapper default
        # and is intentionally higher than the DemoRealCharacter baseline of 7 used
        # in demo_character_factory.py. DemoRealCharacter models an initial character
        # state, while this mapper provides a generic default when events reference
        # job_performance without an existing value on the character.
        "job_performance": ("job_performance", 0, 100, 20),
        "productivity": ("job_performance", 0, 100, 20),
        
        # Community/Social
        "community": ("community", 0, 10, 5),
        "community_standing": ("community", 0, 10, 5),
        
        # Additional common aliases
        "satisfaction": ("social_wellbeing", 0, 10, 5),
        "social_satisfaction": ("social_wellbeing", 0, 10, 5),
        "safety": ("health_status", 0, 10, 5),
        "curiosity": ("mental_health", 0, 10, 7),
        "helpfulness": ("community", 0, 10, 5),
        "competitive_spirit": ("mental_health", 0, 10, 7),
        "confidence": ("mental_health", 0, 10, 7),
        "knowledge": ("mental_health", 0, 10, 7),
        "expertise": ("job_performance", 0, 100, 20),
        "skill_improvement": ("job_performance", 0, 100, 20),
        "career_prospects": ("job_performance", 0, 100, 20),
        "skill_level": ("job_performance", 0, 100, 20),
        "community_pride": ("community", 0, 10, 5),
        "reputation": ("community", 0, 10, 5),
        "mutual_support": ("social_wellbeing", 0, 10, 5),
        "adventure_opportunity": ("mental_health", 0, 10, 7),
    }
    
    @classmethod
    def map_attribute(cls, template_attr: str) -> Tuple[str, Optional[int], Optional[int], Any]:
        """
        Map a template attribute name to the actual Character attribute.
        
        Args:
            template_attr: The attribute name used in effect templates
            
        Returns:
            Tuple of (actual_attribute_name, min_value, max_value, default_value)
            If not mapped, returns (template_attr, None, None, 0)
        """
        mapping = cls.ATTRIBUTE_MAP.get(template_attr.lower())
        if mapping:
            return mapping
        
        # If not in map, return the original attribute with no bounds
        logging.debug(f"Attribute '{template_attr}' not in mapping, using as-is")
        return (template_attr, None, None, 0)
    
    @classmethod
    def get_attribute_value(cls, entity: Any, template_attr: str) -> Optional[Any]:
        """
        Get the value of an attribute from an entity using mapping.
        
        Args:
            entity: The entity (typically a Character) to get the attribute from
            template_attr: The template attribute name
            
        Returns:
            The attribute value, or None if not found
        """
        actual_attr, _, _, default = cls.map_attribute(template_attr)
        
        # Try get_state if it exists and returns a dict
        if hasattr(entity, "get_state") and callable(entity.get_state):
            try:
                state = entity.get_state()
                if isinstance(state, dict):
                    value = state.get(actual_attr)
                    if value is not None:
                        return value
            except (AttributeError, TypeError, KeyError):
                # If get_state is missing or returns an unexpected structure,
                # fall back to other attribute access methods below.
                pass
        
        # Try direct attribute access with mapped name
        if hasattr(entity, actual_attr):
            try:
                return getattr(entity, actual_attr)
            except (AttributeError, TypeError):
                pass
        
        # Try template attribute name for backward compatibility
        if actual_attr != template_attr and hasattr(entity, template_attr):
            try:
                return getattr(entity, template_attr)
            except (AttributeError, TypeError):
                pass
        
        # Attribute doesn't exist, return default
        return default
    
    @classmethod
    def set_attribute_value(
        cls,
        entity: Any,
        template_attr: str,
        value: Any,
        apply_bounds: bool = True
    ) -> bool:
        """
        Set the value of an attribute on an entity using mapping.
        
        Args:
            entity: The entity (typically a Character) to set the attribute on
            template_attr: The template attribute name
            value: The value to set
            apply_bounds: Whether to clamp the value to min/max bounds
            
        Returns:
            True if successful, False otherwise
        """
        actual_attr, min_val, max_val, default = cls.map_attribute(template_attr)
        
        # Determine which attribute we'll actually set
        # For backward compatibility: if entity has the template attribute, use it without bounds
        # Otherwise use the mapped attribute name with bounds
        will_use_mapped = True
        target_attr = actual_attr
        
        # Try to set via get_state if it exists and returns dict
        uses_state = False
        if hasattr(entity, "get_state") and callable(entity.get_state):
            try:
                state = entity.get_state()
                if isinstance(state, dict):
                    # In state dict, always use mapped name and apply bounds
                    if apply_bounds:
                        if min_val is not None and value < min_val:
                            logging.debug(f"Clamping {actual_attr} value {value} to minimum {min_val}")
                            value = min_val
                        if max_val is not None and value > max_val:
                            logging.debug(f"Clamping {actual_attr} value {value} to maximum {max_val}")
                            value = max_val
                    state[actual_attr] = value
                    uses_state = True
            except (AttributeError, TypeError, KeyError) as e:
                # If state-based access fails, fall back to direct attribute setting below.
                logging.debug(
                    "Failed to update state via get_state() for %s on %s; "
                    "falling back to direct attribute update: %s",
                    actual_attr,
                    entity,
                    e,
                )
                pass
        
        # Set directly if not using state
        if not uses_state:
            try:
                # Backward compatibility: if entity has the template attribute, use it
                if actual_attr != template_attr and hasattr(entity, template_attr):
                    target_attr = template_attr
                    will_use_mapped = False
                
                # Only apply bounds if using mapped attribute name
                if apply_bounds and will_use_mapped:
                    if min_val is not None and value < min_val:
                        logging.debug(f"Clamping {actual_attr} value {value} to minimum {min_val}")
                        value = min_val
                    if max_val is not None and value > max_val:
                        logging.debug(f"Clamping {actual_attr} value {value} to maximum {max_val}")
                        value = max_val
                
                setattr(entity, target_attr, value)
            except (AttributeError, TypeError) as e:
                logging.error(f"Failed to set attribute {target_attr} on {entity}: {e}")
                return False
        
        return True
    
    @classmethod
    def get_supported_attributes(cls) -> Dict[str, str]:
        """
        Get a dictionary of all supported template attributes and their mappings.
        
        Returns:
            Dictionary mapping template attribute names to actual attribute names
        """
        return {template: mapping[0] for template, mapping in cls.ATTRIBUTE_MAP.items()}
    
    @classmethod
    def is_bounded_attribute(cls, template_attr: str) -> bool:
        """
        Check if an attribute has bounds defined.
        
        Args:
            template_attr: The template attribute name
            
        Returns:
            True if the attribute has min/max bounds
        """
        _, min_val, max_val, _ = cls.map_attribute(template_attr)
        return min_val is not None or max_val is not None
