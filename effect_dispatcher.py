"""
Effect Dispatcher - Central system for applying event effects.

This module provides a single entry point for applying effects to the game state,
handling all effect types consistently and safely.
"""

from typing import Any, Dict, List, Optional
import logging

from effect_schema import EffectV2, EffectType, OperatorType


class EffectDispatcher:
    """
    Central dispatcher for applying effects to game entities.
    
    This class provides a single, consistent interface for applying all types
    of effects, with proper error handling and logging.
    """
    
    def __init__(self, graph_manager=None):
        """
        Initialize the effect dispatcher.
        
        Args:
            graph_manager: Optional GraphManager instance for relationship effects
        """
        self.graph_manager = graph_manager
        self.applied_effects_log = []  # Track applied effects for debugging
    
    def apply_effect(
        self,
        effect: EffectV2,
        event: Any,
        context: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Apply a single effect to game entities.
        
        This is the main entry point for effect application. It routes the effect
        to the appropriate handler based on effect type.
        
        Args:
            effect: The effect to apply
            event: The event triggering this effect
            context: Optional context dictionary with additional data
            
        Returns:
            bool: True if effect was applied successfully, False otherwise
            
        Example:
            dispatcher = EffectDispatcher(graph_manager)
            effect = EffectV2(
                type=EffectType.ATTRIBUTE_CHANGE,
                targets=["participants"],
                attribute="happiness",
                change_value=10
            )
            success = dispatcher.apply_effect(effect, event)
        """
        try:
            # Validate the effect
            effect.validate()
            
            # Route to appropriate handler based on effect type
            if effect.type == EffectType.ATTRIBUTE_CHANGE:
                return self._apply_attribute_change(effect, event, context)
            elif effect.type == EffectType.RELATIONSHIP_CHANGE:
                return self._apply_relationship_change(effect, event, context)
            elif effect.type == EffectType.LOCATION_CHANGE:
                return self._apply_location_change(effect, event, context)
            elif effect.type == EffectType.WORLD_STATE_CHANGE:
                return self._apply_world_state_change(effect, event, context)
            else:
                logging.error(f"Unknown effect type: {effect.type}")
                return False
                
        except Exception as e:
            logging.error(f"Error applying effect {effect.attribute}: {e}")
            return False
    
    def _apply_attribute_change(
        self,
        effect: EffectV2,
        event: Any,
        context: Optional[Dict[str, Any]]
    ) -> bool:
        """Apply an attribute change effect to target entities."""
        applied_count = 0
        
        for target_spec in effect.targets:
            entities = self._resolve_targets(target_spec, event, context)
            
            for entity in entities:
                # Check conditions
                if not effect.should_apply(entity):
                    logging.debug(
                        f"Effect conditions not met for {entity}, skipping"
                    )
                    continue
                
                # Apply the attribute change
                if self._modify_entity_attribute(
                    entity,
                    effect.attribute,
                    effect.change_value,
                    effect.operator
                ):
                    applied_count += 1
                    
                    # Apply chained effects if specified
                    if effect.chain:
                        self._apply_chained_attributes(
                            entity,
                            effect.chain,
                            effect.change_value,
                            effect.operator
                        )
        
        if applied_count > 0:
            self._log_effect_application(effect, event, applied_count)
            return True
        
        return False
    
    def _apply_relationship_change(
        self,
        effect: EffectV2,
        event: Any,
        context: Optional[Dict[str, Any]]
    ) -> bool:
        """Apply a relationship change effect between entities."""
        if not self.graph_manager:
            logging.warning("Cannot apply relationship effect: no graph_manager")
            return False
        
        applied_count = 0
        
        # Get participants or specified targets
        entities = []
        for target_spec in effect.targets:
            entities.extend(self._resolve_targets(target_spec, event, context))
        
        # Apply relationship changes between all pairs
        for i, entity1 in enumerate(entities):
            for entity2 in entities[i + 1:]:
                try:
                    # Check if edge exists
                    if self.graph_manager.G.has_edge(entity1, entity2):
                        # Update the relationship
                        self.graph_manager.update_character_character_edge(
                            entity1,
                            entity2,
                            impact_factor=1,
                            impact_value=effect.change_value,
                        )
                        applied_count += 1
                    else:
                        logging.debug(
                            f"No relationship edge between {entity1} and {entity2}"
                        )
                except Exception as e:
                    logging.error(
                        f"Error updating relationship between {entity1} and {entity2}: {e}"
                    )
        
        if applied_count > 0:
            self._log_effect_application(effect, event, applied_count)
            return True
        
        return False
    
    def _apply_location_change(
        self,
        effect: EffectV2,
        event: Any,
        context: Optional[Dict[str, Any]]
    ) -> bool:
        """Apply a change to location attributes."""
        applied_count = 0
        
        for target_spec in effect.targets:
            # For location effects, "location" refers to event.location
            if target_spec == "location" and hasattr(event, 'location') and event.location:
                location = event.location
                
                if effect.should_apply(location):
                    if self._modify_entity_attribute(
                        location,
                        effect.attribute,
                        effect.change_value,
                        effect.operator
                    ):
                        applied_count += 1
            else:
                # Try to resolve other location targets
                locations = self._resolve_targets(target_spec, event, context)
                for location in locations:
                    if effect.should_apply(location):
                        if self._modify_entity_attribute(
                            location,
                            effect.attribute,
                            effect.change_value,
                            effect.operator
                        ):
                            applied_count += 1
        
        if applied_count > 0:
            self._log_effect_application(effect, event, applied_count)
            return True
        
        return False
    
    def _apply_world_state_change(
        self,
        effect: EffectV2,
        event: Any,
        context: Optional[Dict[str, Any]]
    ) -> bool:
        """Apply a change to global world state."""
        # World state changes typically affect a global state object
        # This would integrate with a WorldState or similar system
        
        if context and "world_state" in context:
            world_state = context["world_state"]
            
            if self._modify_entity_attribute(
                world_state,
                effect.attribute,
                effect.change_value,
                effect.operator
            ):
                self._log_effect_application(effect, event, 1)
                return True
        else:
            logging.warning(
                "World state change effect applied but no world_state in context"
            )
        
        return False
    
    def _resolve_targets(
        self,
        target_spec: str,
        event: Any,
        context: Optional[Dict[str, Any]]
    ) -> List[Any]:
        """
        Resolve a target specification to actual entities.
        
        Args:
            target_spec: Target specification string ("participants", "location", etc.)
            event: The event context
            context: Additional context
            
        Returns:
            List of resolved entities
        """
        entities = []
        
        if target_spec == "participants":
            # Get event participants
            if hasattr(event, 'participants'):
                entities = event.participants
        
        elif target_spec == "location":
            # Get event location
            if hasattr(event, 'location') and event.location:
                entities = [event.location]
        
        elif target_spec == "world":
            # Get world state if available
            if context and "world_state" in context:
                entities = [context["world_state"]]
        
        else:
            # Try to find target by name in graph
            if self.graph_manager:
                node = self.graph_manager.get_node(target_spec)
                if node:
                    entities = [node]
        
        return entities
    
    def _modify_entity_attribute(
        self,
        entity: Any,
        attribute: str,
        change_value: Any,
        operator: OperatorType
    ) -> bool:
        """
        Modify an attribute on an entity using the specified operator.
        
        Args:
            entity: The entity to modify
            attribute: The attribute name
            change_value: The value to apply
            operator: The operator to use (add, subtract, set, etc.)
            
        Returns:
            bool: True if modification was successful
        """
        try:
            # Get current value
            current_value = None
            uses_state = False
            
            # Try get_state if it exists and looks like a real method
            if hasattr(entity, "get_state") and callable(entity.get_state):
                try:
                    state = entity.get_state()
                    # Verify state is dict-like
                    if isinstance(state, dict):
                        current_value = state.get(attribute, 0)
                        uses_state = True
                except Exception:
                    # If get_state fails, try direct attribute access
                    pass
            
            if not uses_state and current_value is None:
                if hasattr(entity, attribute):
                    try:
                        current_value = getattr(entity, attribute)
                    except Exception:
                        # If attribute exists but can't be retrieved, use 0
                        current_value = 0
                else:
                    # Attribute doesn't exist, create it with default value
                    current_value = 0
            
            # Apply operator
            if operator == OperatorType.ADD:
                new_value = current_value + change_value
            elif operator == OperatorType.SUBTRACT:
                new_value = current_value - change_value
            elif operator == OperatorType.MULTIPLY:
                new_value = current_value * change_value
            elif operator == OperatorType.SET:
                new_value = change_value
            elif operator == OperatorType.MIN:
                new_value = min(current_value, change_value)
            elif operator == OperatorType.MAX:
                new_value = max(current_value, change_value)
            else:
                logging.warning(f"Unknown operator {operator}, using ADD")
                new_value = current_value + change_value
            
            # Set new value
            if uses_state:
                state[attribute] = new_value
            else:
                setattr(entity, attribute, new_value)
            
            logging.debug(
                f"Modified {entity} {attribute}: {current_value} -> {new_value} (operator: {operator.value})"
            )
            return True
            
        except Exception as e:
            logging.error(f"Error modifying attribute {attribute} on {entity}: {e}")
            return False
    
    def _apply_chained_attributes(
        self,
        entity: Any,
        chain: List[str],
        change_value: Any,
        operator: OperatorType
    ):
        """Apply the same change to chained attributes."""
        for chained_attr in chain:
            self._modify_entity_attribute(
                entity,
                chained_attr,
                change_value,
                operator
            )
    
    def _log_effect_application(
        self,
        effect: EffectV2,
        event: Any,
        count: int
    ):
        """Log an applied effect for debugging and auditing."""
        log_entry = {
            "effect_type": effect.type.value,
            "attribute": effect.attribute,
            "change_value": effect.change_value,
            "operator": effect.operator.value,
            "event": event.name if hasattr(event, 'name') else str(event),
            "entities_affected": count
        }
        self.applied_effects_log.append(log_entry)
        
        logging.info(
            f"Applied {effect.type.value} effect '{effect.attribute}' "
            f"to {count} entities from event '{log_entry['event']}'"
        )
    
    def get_applied_effects_summary(self) -> Dict[str, Any]:
        """
        Get a summary of all applied effects.
        
        Returns:
            Dictionary with effect statistics
        """
        summary = {
            "total_effects": len(self.applied_effects_log),
            "by_type": {},
            "by_attribute": {},
        }
        
        for log_entry in self.applied_effects_log:
            effect_type = log_entry["effect_type"]
            attribute = log_entry["attribute"]
            
            summary["by_type"][effect_type] = summary["by_type"].get(effect_type, 0) + 1
            summary["by_attribute"][attribute] = summary["by_attribute"].get(attribute, 0) + 1
        
        return summary
    
    def clear_log(self):
        """Clear the applied effects log."""
        self.applied_effects_log = []


# Convenience function for standalone usage
def apply_effect(
    effect: EffectV2,
    event: Any,
    graph_manager=None,
    context: Optional[Dict[str, Any]] = None
) -> bool:
    """
    Convenience function to apply an effect without managing a dispatcher instance.
    
    Args:
        effect: The effect to apply
        event: The event triggering this effect
        graph_manager: Optional GraphManager instance
        context: Optional context dictionary
        
    Returns:
        bool: True if effect was applied successfully
    """
    dispatcher = EffectDispatcher(graph_manager)
    return dispatcher.apply_effect(effect, event, context)
