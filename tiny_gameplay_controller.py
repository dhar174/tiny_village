import pygame
import random
import logging
import traceback
import json
import os
import datetime # Added for time-based achievement
from datetime import timedelta

# Constants for speed control
MAX_SPEED = 5.0
MIN_SPEED = 0.1
SPEED_STEP = 0.1

# UI Layout Constants
ACHIEVEMENT_SPACING = 25
ACHIEVEMENT_LINE_SPACING = 18

WEATHER_ENERGY_EFFECTS = {
    'rainy': 0.5,
    # 'snowy': 1.0, # easy to add more
    # Add other weather types here
}
WEATHER_UI_MESSAGES = {
    'rainy': ("Rainfall is tiring the villagers.", (180, 180, 220)),
    # 'snowy': ("Snow is exhausting the villagers.", (220, 220, 255)),
}
from typing import Dict, List, Any, Union, Optional
from tiny_strategy_manager import StrategyManager
from tiny_event_handler import EventHandler, Event
from tiny_types import GraphManager
from tiny_map_controller import MapController

class UIPanel:
    """Base class for UI panels in the modular UI system."""
    
    def __init__(self, name: str, position: tuple = (0, 0), size: tuple = None, visible: bool = True):
        self.name = name
        self.position = position  # (x, y)
        self.size = size  # (width, height) - None means auto-size
        self.visible = visible
        self.background_color = None
        self.border_color = None
        self.padding = 5
        
    def render(self, screen: pygame.Surface, controller, fonts: Dict[str, pygame.font.Font]) -> int:
        """
        Render the panel to the screen.
        
        Args:
            screen: The pygame surface to render to
            controller: The GameplayController instance for accessing game data
            fonts: Dictionary of font objects by size/type
            
        Returns:
            int: The height of the rendered panel
        """
        if not self.visible:
            return 0
            
        return self._render_content(screen, controller, fonts)
    
    def _render_content(self, screen: pygame.Surface, controller, fonts: Dict[str, pygame.font.Font]) -> int:
        """Override this method in subclasses to implement specific panel rendering."""
        return 0
    
    def toggle_visibility(self):
        """Toggle panel visibility."""
        self.visible = not self.visible


class CharacterInfoPanel(UIPanel):
    """Panel for displaying character count and basic info."""
    
    def _render_content(self, screen: pygame.Surface, controller, fonts: Dict[str, pygame.font.Font]) -> int:
        font = fonts.get('normal', pygame.font.Font(None, 24))
        x, y = self.position
        
        # Character count
        char_count_text = font.render(f"Characters: {len(controller.characters)}", True, (255, 255, 255))
        screen.blit(char_count_text, (x, y))
        
        return char_count_text.get_height() + self.padding


class GameStatusPanel(UIPanel):
    """Panel for displaying game status (pause, time, speed)."""
    
    def _render_content(self, screen: pygame.Surface, controller, fonts: Dict[str, pygame.font.Font]) -> int:
        font = fonts.get('normal', pygame.font.Font(None, 24))
        small_font = fonts.get('small', pygame.font.Font(None, 18))
        x, y = self.position
        current_y = y
        
        # Pause status
        if getattr(controller, "paused", False):
            pause_text = font.render("PAUSED", True, (255, 255, 0))
            screen.blit(pause_text, (screen.get_width() - 100, 10))
        
        # Game time
        if hasattr(controller, "gametime_manager") and controller.gametime_manager:
            try:
                game_time = controller.gametime_manager.get_calendar().get_game_time_string()
                time_text = small_font.render(f"Time: {game_time}", True, (255, 255, 255))
                screen.blit(time_text, (x, current_y))
                current_y += time_text.get_height() + 2

            except (AttributeError, TypeError, ValueError) as e:
                logging.error(f"Error rendering game time: {e}")
                pass
        
        # Speed indicator with caching
        try:
            if controller._last_time_scale_factor != controller.time_scale_factor or controller._cached_speed_text is None:
                controller._cached_speed_text = small_font.render(
                    f"Speed: {controller.time_scale_factor:.1f}x", True, (255, 255, 255)
                )
                controller._last_time_scale_factor = controller.time_scale_factor
            
            if controller._cached_speed_text:
                screen.blit(controller._cached_speed_text, (x, current_y))
                current_y += controller._cached_speed_text.get_height() + 2
        except (AttributeError, TypeError, ValueError) as e:
            pass  # Skip if speed rendering fails
        
        return current_y - y


class WeatherPanel(UIPanel):
    """Panel for displaying weather information."""
    
    def _render_content(self, screen: pygame.Surface, controller, fonts: Dict[str, pygame.font.Font]) -> int:
        small_font = fonts.get('small', pygame.font.Font(None, 18))
        tiny_font = fonts.get('tiny', pygame.font.Font(None, 16))
        x, y = self.position
        current_y = y
        
        if hasattr(controller, "weather_system"):
            weather_text = small_font.render(
                f"Weather: {controller.weather_system['current_weather']} {controller.weather_system['temperature']}°C",
                True, (200, 220, 255)
            )
            screen.blit(weather_text, (x, current_y))
            current_y += weather_text.get_height() + 2
            
            # Weather effects message
            current_weather = None
            if isinstance(controller.weather_system, dict):
                current_weather = controller.weather_system.get('current_weather')
            if current_weather in WEATHER_UI_MESSAGES:
                message, color = WEATHER_UI_MESSAGES[current_weather]
                weather_effect_text = tiny_font.render(message, True, color)
                screen.blit(weather_effect_text, (x, current_y))
                current_y += weather_effect_text.get_height() + 2
        
        return current_y - y


class StatsPanel(UIPanel):
    """Panel for displaying game statistics and analytics."""
    
    def _render_content(self, screen: pygame.Surface, controller, fonts: Dict[str, pygame.font.Font]) -> int:
        tiny_font = fonts.get('tiny', pygame.font.Font(None, 16))
        x, y = self.position
        current_y = y
        
        # Game statistics
        stats = controller.game_statistics
        stats_text = tiny_font.render(
            f"Actions: {stats['actions_executed']} | Failed: {stats['actions_failed']} | Recovered: {stats['errors_recovered']}",
            True, (180, 180, 180)
        )
        screen.blit(stats_text, (x, current_y))
        current_y += stats_text.get_height() + 2
        
        # Action analytics
        if hasattr(controller, "action_resolver"):
            try:
                analytics = controller.action_resolver.get_action_analytics()
                analytics_text = tiny_font.render(
                    f"Success Rate: {analytics['success_rate']:.1%} | Cache: {analytics['cache_size']}",
                    True, (150, 150, 150)
                )
                screen.blit(analytics_text, (x, current_y))
                current_y += analytics_text.get_height() + 2

            except Exception as e:
                logging.error(f"Error in action analytics rendering: {e}")
                logging.error(traceback.format_exc())
        
        # System health
        if hasattr(controller, "recovery_manager"):
            try:
                system_status = controller.recovery_manager.get_system_status()
                healthy_systems = sum(1 for status in system_status.values() if status == "healthy")
                total_systems = len(system_status)
                
                health_color = (
                    (0, 255, 0) if healthy_systems == total_systems
                    else (255, 255, 0) if healthy_systems > total_systems // 2
                    else (255, 0, 0)
                )
                health_text = tiny_font.render(
                    f"Systems: {healthy_systems}/{total_systems} healthy",
                    True, health_color
                )
                screen.blit(health_text, (x, current_y))
                current_y += health_text.get_height() + 2

            except Exception as e:
                logging.error(f"Error while rendering system health: {e}")
        
        return current_y - y


class AchievementPanel(UIPanel):
    """Panel for displaying achievements."""
    
    def _render_content(self, screen: pygame.Surface, controller, fonts: Dict[str, pygame.font.Font]) -> int:
        tiny_font = fonts.get('tiny', pygame.font.Font(None, 16))
        x, y = self.position
        current_y = y
        
        # First week survived achievement
        try:
            survived_week = controller.global_achievements.get("village_milestones", {}).get("first_week_survived", False)
            status_text = "Yes" if survived_week else "No"
            achievement_render = tiny_font.render(
                f"First Week Survived: {status_text}", True, (220, 220, 180)
            )
            screen.blit(achievement_render, (x, current_y))
            current_y += achievement_render.get_height() + 2

        except Exception as e:
            logging.error(f"Error while loading achievement panel: {e}")
            pass
        
        # All achievements
        milestones = controller.global_achievements.get("village_milestones", {})
        if milestones:
            header = tiny_font.render("Achievements:", True, (240, 240, 200))
            screen.blit(header, (x, current_y))
            current_y += header.get_height() + 5
            
            for key, achieved in milestones.items():
                title = key.replace("_", " ").title()
                status = "✓" if achieved else "✗"
                color = (180, 220, 180) if achieved else (200, 180, 180)
                text = tiny_font.render(f"{status} {title}", True, color)
                screen.blit(text, (x, current_y))
                current_y += text.get_height() + 2
        
        return current_y - y


class SelectedCharacterPanel(UIPanel):
    """Panel for displaying selected character information."""
    
    def _render_content(self, screen: pygame.Surface, controller, fonts: Dict[str, pygame.font.Font]) -> int:
        small_font = fonts.get('small', pygame.font.Font(None, 18))
        tiny_font = fonts.get('tiny', pygame.font.Font(None, 16))
        x, y = self.position
        current_y = y
        
        if (hasattr(controller.map_controller, "selected_character") 
            and controller.map_controller.selected_character):
            
            char = controller.map_controller.selected_character
            char_info = [
                f"Selected: {char.name}",
                f"Job: {getattr(char, 'job', 'Unknown')}",
                f"Energy: {getattr(char, 'energy', 0)}",
                f"Health: {getattr(char, 'health_status', 0)}",
            ]
            
            # Add social and quest info
            if hasattr(char, "uuid") and hasattr(controller, "social_networks"):
                try:
                    relationships = controller.social_networks["relationships"].get(char.uuid, {})
                    avg_relationship = (
                        sum(relationships.values()) / len(relationships)
                        if relationships else 50
                    )
                    char_info.append(f"Social: {avg_relationship:.0f}")

                except Exception as e:
                    logging.error(f"Error accessing social_networks while rendering selected character panel: {e}")
                    pass
            
            if hasattr(char, "uuid") and hasattr(controller, "quest_system"):
                try:
                    active_quests = len(controller.quest_system["active_quests"].get(char.uuid, []))
                    completed_quests = len(controller.quest_system["completed_quests"].get(char.uuid, []))
                    char_info.append(f"Quests: {active_quests} active, {completed_quests} done")

                except Exception as e:
                    logging.error(f"Error loading quest system while rendering selected character panel: {e}")
                    
                    pass
            
            # Render character info
            for info in char_info:
                info_text = small_font.render(info, True, (255, 255, 0))
                screen.blit(info_text, (x, current_y))
                current_y += info_text.get_height() + 2
            
            # Character achievements
            try:
                if hasattr(char, 'achievements') and char.achievements:
                    ach_header_text = small_font.render("Achievements:", True, (220, 220, 180))
                    screen.blit(ach_header_text, (x, current_y))
                    current_y += ach_header_text.get_height() + 2
                    
                    for achievement_id in char.achievements:
                        display_name = achievement_id.replace("_", " ").title()
                        ach_text = tiny_font.render(f"- {display_name}", True, (200, 200, 150))
                        screen.blit(ach_text, (x + 5, current_y))  # Indent slightly
                        current_y += ach_text.get_height() + 2

            except Exception as e:
                logging.error(f"Error loading achievements while rendering selected character panel: {e}")
                pass
        
        return current_y - y


class InstructionsPanel(UIPanel):
    """Panel for displaying game instructions."""
    
    def _render_content(self, screen: pygame.Surface, controller, fonts: Dict[str, pygame.font.Font]) -> int:
        tiny_font = fonts.get('tiny', pygame.font.Font(None, 16))
        x, y = self.position
        
        instructions = [
            "Click characters to select them",
            "Click buildings to interact",
            "SPACE to pause/unpause",
            "R to reset characters",
            "S to save game (basic)",
            "L to load game (basic)",
            "F to show feature status",
            "ESC to quit",
        ]
        
        for i, instruction in enumerate(instructions):
            inst_text = tiny_font.render(instruction, True, (200, 200, 200))
            screen.blit(inst_text, (x, y + i * 15))
        
        return len(instructions) * 15

""" 
This script integrates with the game loop, applying decisions from the strategy manager to the game state.
5. Gameplay Execution
Where it happens: gameplay_controller.py
What happens: The gameplay controller applies the decided plan to the game state, triggering animations, interactions, and state changes in the game. 

"""

# Set up logging for better error tracking
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ActionResolver:
    """
    Handles conversion between different action formats and execution.
    Provides robust action resolution with validation and fallback mechanisms.
    """

    def __init__(self, action_system=None, graph_manager=None):
        self.action_system = action_system
        self.graph_manager = graph_manager
        self.action_cache = {}  # Cache for performance optimization
        self.execution_history = []  # Track action execution for analytics
        self.fallback_actions = {
            "default_rest": {"name": "Rest", "energy_cost": -10, "satisfaction": 5},
            "default_idle": {"name": "Idle", "energy_cost": 0, "satisfaction": 1},
            "emergency_recover": {
                "name": "Emergency Recovery",
                "energy_cost": -20,
                "satisfaction": 10,
            },
        }

    def validate_action_preconditions(self, action, character) -> bool:
        """Validate that action preconditions are met."""
        try:
            if hasattr(action, "preconditions") and action.preconditions:
                for condition, value in action.preconditions.items():
                    if hasattr(character, condition):
                        if getattr(character, condition) < value:
                            return False
            return True
        except Exception as e:
            logger.warning(f"Error validating preconditions: {e}")
            return False

    def predict_action_effects(self, action, character) -> Dict[str, Any]:
        """Predict the effects of an action before execution."""
        predicted_effects = {
            "energy_change": 0,
            "satisfaction_change": 0,
            "health_change": 0,
            "success_probability": 0.8,  # Default success rate
        }

        try:
            if hasattr(action, "effects"):
                for effect in action.effects:
                    if "attribute" in effect and "change_value" in effect:
                        attr = effect["attribute"]
                        change = effect["change_value"]
                        if attr == "energy":
                            predicted_effects["energy_change"] = change
                        elif attr == "current_satisfaction":
                            predicted_effects["satisfaction_change"] = change
                        elif attr == "health_status":
                            predicted_effects["health_change"] = change

            # Adjust success probability based on character state
            if hasattr(character, "energy") and character.energy < 20:
                predicted_effects[
                    "success_probability"
                ] *= 0.7  # Lower success when tired

        except Exception as e:
            logger.warning(f"Error predicting action effects: {e}")

        return predicted_effects

    def resolve_action(
        self, action_data: Union[Dict, Any], character=None
    ) -> Optional[Any]:
        """Convert action data to executable action object with validation and caching."""
        try:
            # Create cache key for performance
            cache_key = self._generate_cache_key(action_data, character)
            if cache_key in self.action_cache:
                logger.debug(f"Using cached action for {cache_key}")
                return self.action_cache[cache_key]

            # If it's already an Action object with execute method
            if hasattr(action_data, "execute"):
                if self.validate_action_preconditions(action_data, character):
                    self.action_cache[cache_key] = action_data
                    return action_data
                else:
                    logger.warning(f"Action preconditions not met: {action_data}")
                    return self.get_fallback_action(character)

            # If it's a dictionary, try to convert to Action
            if isinstance(action_data, dict):
                resolved_action = self._dict_to_action(action_data, character)
                if resolved_action:
                    self.action_cache[cache_key] = resolved_action
                return resolved_action

            # If it's a string (action name), try to resolve
            if isinstance(action_data, str):
                resolved_action = self._resolve_by_name(action_data, character)
                if resolved_action:
                    self.action_cache[cache_key] = resolved_action
                return resolved_action

            logger.warning(f"Unknown action format: {type(action_data)}")
            return self.get_fallback_action(character)

        except Exception as e:
            logger.error(f"Error resolving action {action_data}: {e}")
            return self.get_fallback_action(character)

    def _generate_cache_key(self, action_data, character):
        """Generate a cache key for action data."""
        try:
            char_id = getattr(character, "uuid", "unknown") if character else "none"
            if isinstance(action_data, dict):
                return f"dict_{action_data.get('name', 'unknown')}_{char_id}"
            elif isinstance(action_data, str):
                return f"str_{action_data}_{char_id}"
            elif hasattr(action_data, "name"):
                return f"obj_{action_data.name}_{char_id}"
            else:
                return f"unknown_{type(action_data).__name__}_{char_id}"
        except:
            return f"fallback_{id(action_data)}_{id(character)}"

    def track_action_execution(self, action, character, success: bool):
        """Track action execution for analytics and improvement."""
        try:
            execution_record = {
                "timestamp": pygame.time.get_ticks(),
                "action_name": getattr(action, "name", str(action)),
                "character_id": (
                    getattr(character, "uuid", "unknown") if character else None
                ),
                "character_name": (
                    getattr(character, "name", "unknown") if character else None
                ),
                "success": success,
                "character_energy_before": (
                    getattr(character, "energy", None) if character else None
                ),
            }

            self.execution_history.append(execution_record)

            # Keep history limited to last 1000 actions
            if len(self.execution_history) > 1000:
                self.execution_history = self.execution_history[-1000:]

        except Exception as e:
            logger.warning(f"Error tracking action execution: {e}")

    def get_action_analytics(self) -> Dict[str, Any]:
        """Get analytics about action execution performance."""
        try:
            if not self.execution_history:
                return {"total_actions": 0, "success_rate": 0.0}

            total_actions = len(self.execution_history)
            successful_actions = sum(
                1 for record in self.execution_history if record["success"]
            )
            success_rate = (
                successful_actions / total_actions if total_actions > 0 else 0.0
            )

            # Action type analysis
            action_stats = {}
            for record in self.execution_history:
                action_name = record["action_name"]
                if action_name not in action_stats:
                    action_stats[action_name] = {"count": 0, "successes": 0}
                action_stats[action_name]["count"] += 1
                if record["success"]:
                    action_stats[action_name]["successes"] += 1

            return {
                "total_actions": total_actions,
                "success_rate": success_rate,
                "action_breakdown": action_stats,
                "cache_size": len(self.action_cache),
            }

        except Exception as e:
            logger.error(f"Error generating action analytics: {e}")
            return {"error": str(e)}


    def _dict_to_action(self, action_dict: Dict, character=None):
        """Convert dictionary action to Action object."""
        try:
            from actions import Action

            # Extract action information
            name = action_dict.get("name", "Unknown Action")
            energy_cost = action_dict.get("energy_cost", 0)
            satisfaction = action_dict.get("satisfaction", 0)

            # Create basic effects based on dictionary
            effects = []
            if energy_cost != 0:
                effects.append(
                    {
                        "targets": ["initiator"],
                        "attribute": "energy",
                        "change_value": -abs(
                            energy_cost
                        ),  # Energy cost is always negative
                    }
                )
            if satisfaction != 0:
                effects.append(
                    {
                        "targets": ["initiator"],
                        "attribute": "current_satisfaction",
                        "change_value": satisfaction,
                    }
                )

            # Create simple Action object
            action = Action(
                name=name,
                preconditions={},  # No preconditions for dict actions
                effects=effects,
                cost=max(1, abs(energy_cost)),
                initiator=character,
                default_target_is_initiator=True,
                graph_manager=self.graph_manager,
            )

            return action

        except Exception as e:
            logger.error(f"Error converting dict to action: {e}")
            return None

    def _resolve_by_name(self, action_name: str, character=None):
        """Resolve action by name using action system."""
        try:
            if self.action_system:
                # Try to generate action using action system
                actions = self.action_system.generate_actions(character)
                for action in actions:
                    if (
                        hasattr(action, "name")
                        and action.name.lower() == action_name.lower()
                    ):
                        return action

            # Fallback: create simple action
            return self._dict_to_action(
                {"name": action_name, "energy_cost": 5, "satisfaction": 2}, character
            )

        except Exception as e:
            logger.error(f"Error resolving action by name {action_name}: {e}")
            return None

    def get_fallback_action(self, character=None):
        """Get a safe fallback action when others fail."""
        return self._dict_to_action(self.fallback_actions["default_rest"], character)


class SystemRecoveryManager:
    """Manages system recovery and fallback strategies."""

    def __init__(self, gameplay_controller):
        self.gameplay_controller = gameplay_controller
        self.recovery_strategies = {}
        self.recovery_attempts = {}
        self.max_recovery_attempts = 3
        self.setup_recovery_strategies()

    def setup_recovery_strategies(self):
        """Define recovery strategies for different system failures."""
        self.recovery_strategies = {
            "strategy_manager": self._recover_strategy_manager,
            "graph_manager": self._recover_graph_manager,
            "event_handler": self._recover_event_handler,
            "action_system": self._recover_action_system,
            "map_controller": self._recover_map_controller,
            "character_system": self._recover_character_system,
            "storytelling_system": self._recover_storytelling_system,
        }

    def attempt_recovery(self, system_name: str) -> bool:
        """Attempt to recover a failed system with intelligent retry logic."""
        try:
            if system_name not in self.recovery_strategies:
                logger.error(f"No recovery strategy for system: {system_name}")
                return False

            # Check recovery attempt limits
            attempts = self.recovery_attempts.get(system_name, 0)
            if attempts >= self.max_recovery_attempts:
                logger.error(f"Max recovery attempts reached for {system_name}")
                return False

            logger.info(
                f"Attempting recovery for {system_name} (attempt {attempts + 1})"
            )

            # Increment attempt counter
            self.recovery_attempts[system_name] = attempts + 1

            # Execute recovery strategy
            success = self.recovery_strategies[system_name]()

            if success:
                logger.info(f"Successfully recovered {system_name}")
                # Reset attempt counter on success
                self.recovery_attempts[system_name] = 0
                return True
            else:
                logger.warning(f"Recovery attempt failed for {system_name}")
                return False

        except Exception as e:
            logger.error(f"Error during recovery attempt for {system_name}: {e}")
            return False

    def _recover_strategy_manager(self) -> bool:
        """Recover the strategy manager system."""
        try:
            if not self.gameplay_controller.strategy_manager:
                self.gameplay_controller.strategy_manager = StrategyManager()
                return True
            return True
        except Exception as e:
            logger.error(f"Strategy manager recovery failed: {e}")
            return False

    def _recover_graph_manager(self) -> bool:
        """Recover the graph manager system."""
        try:
            if not self.gameplay_controller.graph_manager:
                from tiny_graph_manager import GraphManager as ActualGraphManager

                self.gameplay_controller.graph_manager = ActualGraphManager()
                return True
            return True
        except Exception as e:
            logger.error(f"Graph manager recovery failed: {e}")
            return False

    def _recover_event_handler(self) -> bool:
        """Recover the event handler system."""
        try:
            if (
                not self.gameplay_controller.event_handler
                and self.gameplay_controller.graph_manager
            ):
                from tiny_storytelling_engine import StorytellingEventHandler
                self.gameplay_controller.event_handler = StorytellingEventHandler(
                    self.gameplay_controller.graph_manager
                )
                
                # Wire to storytelling system if available
                if hasattr(self.gameplay_controller, 'storytelling_system') and self.gameplay_controller.storytelling_system:
                    self.gameplay_controller.event_handler.storytelling_system = self.gameplay_controller.storytelling_system
                
                return True
            return True
        except Exception as e:
            logger.error(f"Event handler recovery failed: {e}")
            return False

    def _recover_action_system(self) -> bool:
        """Recover the action system."""
        try:
            if not getattr(self.gameplay_controller, "action_system", None):
                from actions import ActionSystem

                self.gameplay_controller.action_system = ActionSystem()
                self.gameplay_controller.action_system.setup_actions()
                self.gameplay_controller.action_resolver = ActionResolver(
                    self.gameplay_controller.action_system
                )
                return True
            return True
        except Exception as e:
            logger.error(f"Action system recovery failed: {e}")
            # Fallback to basic action resolver
            self.gameplay_controller.action_resolver = ActionResolver()
            return False

    def _recover_map_controller(self) -> bool:
        """Recover the map controller system."""
        try:
            if not self.gameplay_controller.map_controller:
                map_config = self.gameplay_controller.config.get("map", {})
                self.gameplay_controller.map_controller = MapController(
                    map_config.get("image_path", "assets/default_map.png"),
                    map_data={
                        "width": map_config.get("width", 100),
                        "height": map_config.get("height", 100),
                        "buildings": self.gameplay_controller._get_default_buildings(
                            map_config
                        ),
                    },
                )
                return True
            return True
        except Exception as e:
            logger.error(f"Map controller recovery failed: {e}")
            return False

    def _recover_character_system(self) -> bool:
        """Recover the character system."""
        try:
            if not self.gameplay_controller.characters:
                self.gameplay_controller._create_fallback_characters()
                return len(self.gameplay_controller.characters) > 0
            return True
        except Exception as e:
            logger.error(f"Character system recovery failed: {e}")
            return False

    def _recover_storytelling_system(self) -> bool:
        """Recover the storytelling system."""
        try:
            if not self.gameplay_controller.storytelling_system and self.gameplay_controller.event_handler:
                from tiny_storytelling_system import StorytellingSystem
                self.gameplay_controller.storytelling_system = StorytellingSystem(
                    self.gameplay_controller.event_handler
                )
                
                # Wire the event handler to forward events to storytelling system
                if hasattr(self.gameplay_controller.event_handler, 'storytelling_system'):
                    self.gameplay_controller.event_handler.storytelling_system = self.gameplay_controller.storytelling_system
                
                return True
            return True
        except Exception as e:
            logger.error(f"Storytelling system recovery failed: {e}")
            return False

    def get_system_status(self) -> Dict[str, str]:
        """Get current status of all systems."""
        status = {}
        gc = self.gameplay_controller

        status["strategy_manager"] = "healthy" if gc.strategy_manager else "failed"
        status["graph_manager"] = "healthy" if gc.graph_manager else "failed"
        status["event_handler"] = "healthy" if gc.event_handler else "failed"
        status["action_system"] = (
            "healthy" if getattr(gc, "action_system", None) else "failed"
        )
        status["map_controller"] = "healthy" if gc.map_controller else "failed"
        status["characters"] = "healthy" if gc.characters else "failed"
        status["storytelling_system"] = "healthy" if getattr(gc, "storytelling_system", None) else "failed"

        return status



class GameplayController:
    def __init__(
        self, graph_manager: GraphManager = None, config: Dict[str, Any] = None
    ):
        """Initialize the gameplay controller with improved error handling and configuration."""
        self.config = config or {}
        self.initialization_errors = []
        self.running = True
        self.paused = False
        self.time_scale_factor = 1.0 # Added for time scaling
        self._cached_speed_text = None
        self._last_time_scale_factor = None


        # Initialize recovery manager
        self.recovery_manager = SystemRecoveryManager(self)

        # Initialize core systems with error handling and recovery

        try:
            self.strategy_manager = StrategyManager()
        except Exception as e:
            logger.error(f"Failed to initialize StrategyManager: {e}")
            self.strategy_manager = None
            self.initialization_errors.append("StrategyManager initialization failed")
            # Attempt immediate recovery
            self.recovery_manager.attempt_recovery("strategy_manager")


        # Initialize graph manager if not provided
        if graph_manager is None:
            try:
                from tiny_graph_manager import GraphManager as ActualGraphManager

                self.graph_manager = ActualGraphManager()
            except Exception as e:
                logger.error(f"Failed to initialize GraphManager: {e}")
                self.graph_manager = None
                self.initialization_errors.append("GraphManager initialization failed")
                self.recovery_manager.attempt_recovery("graph_manager")

        else:
            self.graph_manager = graph_manager

        # Initialize event handler (using narrative-aware handler)
        try:
            from tiny_storytelling_engine import StorytellingEventHandler
            self.event_handler = (
                StorytellingEventHandler(self.graph_manager) if self.graph_manager else None
            )
        except Exception as e:
            logger.error(f"Failed to initialize StorytellingEventHandler: {e}")
            self.event_handler = None
            self.initialization_errors.append("StorytellingEventHandler initialization failed")
            self.recovery_manager.attempt_recovery("event_handler")

        # Initialize storytelling system and wire to event handler
        try:
            from tiny_storytelling_system import StorytellingSystem
            self.storytelling_system = StorytellingSystem(self.event_handler)
            
            # Wire the event handler to forward events to the storytelling system
            if hasattr(self.event_handler, 'storytelling_system'):
                self.event_handler.storytelling_system = self.storytelling_system
            
            logger.info("Storytelling system initialized and wired to event handler")
        except Exception as e:
            logger.error(f"Failed to initialize StorytellingSystem: {e}")
            self.storytelling_system = None
            self.initialization_errors.append("StorytellingSystem initialization failed")

        # Initialize pygame with error handling
        try:
            pygame.init()
            screen_width = self.config.get("screen_width", 800)
            screen_height = self.config.get("screen_height", 600)
            self.screen = pygame.display.set_mode((screen_width, screen_height))
            pygame.display.set_caption("Tiny Village")
            self.clock = pygame.time.Clock()
        except Exception as e:
            logger.error(f"Failed to initialize pygame: {e}")
            self.screen = None
            self.clock = None
            self.initialization_errors.append("Pygame initialization failed")

        # Initialize the Map Controller with dynamic configuration
        try:
            map_config = self.config.get("map", {})
            self.map_controller = MapController(
                map_config.get("image_path", "assets/default_map.png"),
                map_data={
                    "width": map_config.get("width", 100),
                    "height": map_config.get("height", 100),
                    "buildings": self._get_default_buildings(map_config),
                },
            )
        except Exception as e:
            logger.error(f"Failed to initialize MapController: {e}")
            self.map_controller = None
            self.initialization_errors.append("MapController initialization failed")

        # Initialize ActionResolver
        self.action_resolver = None

        # Initialize other game systems
        self.characters = {}
        self.events = []
        self.game_statistics = {
            "actions_executed": 0,
            "actions_failed": 0,
            "characters_created": 0,
            "errors_recovered": 0,
        }
        # Instance-level dictionary to track achievements for this game session.
        # Initialized here during the instantiation of GameplayController.
        self.global_achievements = {
            "village_milestones": {
                "first_character_created": False,
                "five_characters_active": False,
                "successful_harvest": False,
                "trade_established": False,
                "first_week_survived": False,
            },
            "social_achievements": {
                "first_friendship": False,
                "community_event": False,
                "conflict_resolved": False,
            },
            "economic_achievements": {
                "first_transaction": False,
                "wealthy_villager": False,
                "market_established": False,
            },
        }

        # Initialize all game systems
        self.initialize_game_systems()

        # Setup user-driven configuration
        self.setup_user_driven_configuration()

        # Initialize feature systems
        self.implement_achievement_system()
        self.implement_weather_system()
        self.implement_social_network_system()
        self.implement_quest_system()

        # Initialize world events for emergent storytelling
        self.initialize_world_events()
        # Initialize modular UI system
        self._init_ui_system()

        # Log initialization status
        if self.initialization_errors:
            logger.warning(
                f"Initialization completed with {len(self.initialization_errors)} errors: {self.initialization_errors}"
            )
        else:
            logger.info("GameplayController initialized successfully")

    def _init_ui_system(self):
        """Initialize the modular UI panel system."""
        try:
            # Create UI panels with positions
            self.ui_panels = {
                'character_info': CharacterInfoPanel('character_info', position=(10, 10)),
                'game_status': GameStatusPanel('game_status', position=(10, 35)),
                'weather': WeatherPanel('weather', position=(10, 120)),
                'stats': StatsPanel('stats', position=(10, 180)),
                'achievements': AchievementPanel('achievements', position=(10, 280)),
                'selected_character': SelectedCharacterPanel('selected_character', position=(10, 400)),
                'instructions': InstructionsPanel('instructions', position=(10, None))  # Position set dynamically
            }
            
            # Create font dictionary for consistent font usage
            self.ui_fonts = {
                'normal': pygame.font.Font(None, 24),
                'small': pygame.font.Font(None, 18),
                'tiny': pygame.font.Font(None, 16)
            }
            
            logger.info("Modular UI system initialized")
            
        except Exception as e:
            logger.error(f"Error initializing UI system: {e}")
            # Fallback to empty panels dict
            self.ui_panels = {}
            self.ui_fonts = {
                'normal': pygame.font.Font(None, 24),
                'small': pygame.font.Font(None, 18),
                'tiny': pygame.font.Font(None, 16)
            }

    def _get_default_buildings(self, map_config: Dict) -> List[Dict]:
        """Get buildings configuration with dynamic loading support."""

        # Try to load buildings from external file first
        buildings_file = map_config.get("buildings_file")
        if buildings_file:
            try:
                buildings = self._load_buildings_from_file(buildings_file)
                if buildings:
                    logger.info(
                        f"Loaded {len(buildings)} buildings from {buildings_file}"
                    )
                    return buildings
            except Exception as e:
                logger.warning(f"Failed to load buildings from {buildings_file}: {e}")

        # Fallback to default buildings
        default_buildings = [
            {
                "name": "Town Hall",
                "rect": pygame.Rect(100, 150, 50, 50),
                "type": "civic",
            },
            {
                "name": "Market",
                "rect": pygame.Rect(200, 100, 40, 40),
                "type": "commercial",
            },
            {"name": "Tavern", "rect": pygame.Rect(300, 200, 45, 45), "type": "social"},
            {
                "name": "Blacksmith",
                "rect": pygame.Rect(150, 300, 35, 35),
                "type": "crafting",
            },
            {
                "name": "Farm",
                "rect": pygame.Rect(400, 350, 60, 40),
                "type": "agricultural",
            },
            {
                "name": "School",
                "rect": pygame.Rect(500, 120, 45, 35),
                "type": "educational",
            },
        ]

        # Allow custom buildings from config
        custom_buildings = map_config.get("buildings", [])
        if custom_buildings:
            # Convert custom building data to pygame.Rect if needed
            for building in custom_buildings:
                if "rect" not in building and "x" in building:
                    building["rect"] = pygame.Rect(
                        building["x"],
                        building["y"],
                        building.get("width", 40),
                        building.get("height", 40),
                    )
                # Add default type if missing
                if "type" not in building:
                    building["type"] = "generic"
            return custom_buildings

        return default_buildings

    def _load_buildings_from_file(self, filepath: str) -> List[Dict]:
        """Load building configuration from external file."""
        try:
            import json
            import os

            if not os.path.exists(filepath):
                logger.warning(f"Buildings file not found: {filepath}")
                return []

            with open(filepath, "r") as f:
                data = json.load(f)

            buildings = []
            for building_data in data.get("buildings", []):
                # Convert to required format
                building = {
                    "name": building_data.get("name", "Unknown Building"),
                    "type": building_data.get("type", "generic"),
                    "rect": pygame.Rect(
                        building_data.get("x", 0),
                        building_data.get("y", 0),
                        building_data.get("width", 40),
                        building_data.get("height", 40),
                    ),
                }

                # Add any additional properties
                for key, value in building_data.items():
                    if key not in ["x", "y", "width", "height"]:
                        building[key] = value

                buildings.append(building)

            return buildings

        except Exception as e:
            logger.error(f"Error loading buildings from file {filepath}: {e}")
            return []

    def initialize_game_systems(self):
        """Initialize all game systems and create characters with improved error handling."""
        system_init_errors = []

        try:
            # Import required systems with error handling
            required_modules = {
                "Character": "tiny_characters",
                "Location": "tiny_locations",
                "ActionSystem": "actions",
                "GameTimeManager": "tiny_time_manager",
                "GameCalendar": "tiny_time_manager",
                "ItemInventory": "tiny_items",
                "ItemObject": "tiny_items",
                "FoodItem": "tiny_items",
            }

            imported_modules = {}
            for class_name, module_name in required_modules.items():
                try:
                    module = __import__(module_name, fromlist=[class_name])
                    imported_modules[class_name] = getattr(module, class_name)
                except Exception as e:
                    logger.error(
                        f"Failed to import {class_name} from {module_name}: {e}"
                    )
                    system_init_errors.append(f"Import {class_name} failed")

            # Initialize core systems with fallbacks
            if "ActionSystem" in imported_modules:
                try:
                    self.action_system = imported_modules["ActionSystem"](graph_manager=self.graph_manager)
                    self.action_system.setup_actions()
                    self.action_resolver = ActionResolver(action_system=self.action_system, graph_manager=self.graph_manager)
                    logger.info("Action system initialized successfully")
                except Exception as e:
                    logger.error(f"ActionSystem initialization failed: {e}")
                    self.action_system = None
                    self.action_resolver = (
                        ActionResolver(graph_manager=self.graph_manager)
                    )  # Fallback without action system but with graph_manager
                    system_init_errors.append("ActionSystem setup failed")
            else:
                self.action_system = None
                self.action_resolver = ActionResolver(graph_manager=self.graph_manager)

            # Initialize time management with fallback
            if (
                "GameCalendar" in imported_modules
                and "GameTimeManager" in imported_modules
            ):
                try:
                    calendar = imported_modules["GameCalendar"]()
                    self.gametime_manager = imported_modules["GameTimeManager"](
                        calendar
                    )
                    logger.info("Time management system initialized")
                except Exception as e:
                    logger.error(f"Time management initialization failed: {e}")
                    self.gametime_manager = None
                    system_init_errors.append("Time management failed")
            else:
                self.gametime_manager = None

            # Initialize animation system with fallback
            try:
                from tiny_animation_system import get_animation_system

                self.animation_system = get_animation_system()
                logger.info("Animation system initialized")
            except Exception as e:
                logger.warning(f"Animation system initialization failed: {e}")
                self.animation_system = None
                # Not adding to errors as animation is optional

            # Create characters with improved error handling
            character_creation_config = self.config.get("characters", {})
            self.characters = {}

            if self._can_create_characters(imported_modules):
                try:
                    sample_characters = self._create_sample_characters(
                        imported_modules, character_creation_config
                    )
                    self._register_characters(sample_characters)
                    self.game_statistics["characters_created"] = len(self.characters)
                    logger.info(
                        f"Successfully created {len(self.characters)} characters"
                    )
                except Exception as e:
                    logger.error(f"Character creation failed: {e}")
                    system_init_errors.append("Character creation failed")
                    # Create minimal fallback characters
                    self._create_fallback_characters()
            else:
                logger.warning("Cannot create characters due to missing dependencies")
                self._create_fallback_characters()

            # Initialize events system
            self.events = []

            # Log system initialization results
            if system_init_errors:
                logger.warning(
                    f"Game systems initialized with {len(system_init_errors)} errors: {system_init_errors}"
                )
                self.game_statistics["errors_recovered"] = len(system_init_errors)
            else:
                logger.info("All game systems initialized successfully")

        except Exception as e:
            logger.error(f"Critical error initializing game systems: {e}")
            logger.error(traceback.format_exc())
            # Ensure minimal fallback state
            self.characters = {}
            self.events = []
            if not hasattr(self, "action_resolver"):
                self.action_resolver = ActionResolver()

    def _can_create_characters(self, imported_modules: Dict) -> bool:
        """Check if we have the minimum required modules to create characters."""
        required_for_characters = ["Character", "Location", "ItemInventory", "FoodItem"]
        return all(module in imported_modules for module in required_for_characters)

    def _create_fallback_characters(self):
        """Create minimal fallback characters when full character creation fails."""
        try:
            logger.info("Creating fallback characters")
            # Create simple character data without complex dependencies
            fallback_character_data = {
                "name": "Fallback Villager",
                "position": pygame.math.Vector2(400, 300),
                "color": (100, 150, 200),
                "uuid": "fallback_001",
                "energy": 80,
                "health_status": 90,
                "path": [],
            }

            # Create a simple character dict (not a full Character object)
            fallback_char = type("FallbackCharacter", (), fallback_character_data)()

            # Add to systems if they exist
            if self.map_controller:
                self.map_controller.characters[fallback_char.uuid] = fallback_char
            self.characters[fallback_char.uuid] = fallback_char

            logger.info("Fallback character created successfully")
        except Exception as e:
            logger.error(f"Even fallback character creation failed: {e}")
            # Ensure characters dict exists even if empty
            self.characters = {}

    def _create_sample_characters(self, imported_modules: Dict, config: Dict):
        """Create sample characters with dynamic configuration and external file support."""
        Character = imported_modules["Character"]
        Location = imported_modules["Location"]
        ItemInventory = imported_modules["ItemInventory"]
        FoodItem = imported_modules["FoodItem"]

        characters = []

        # Get character configuration
        character_count = config.get("count", 4)
        use_custom_data = config.get("use_custom", False)
        characters_file = config.get("characters_file")

        # Try to load characters from external file first
        character_data_list = None
        if characters_file:
            try:
                character_data_list = self._load_characters_from_file(characters_file)
                if character_data_list:
                    logger.info(
                        f"Loaded {len(character_data_list)} characters from {characters_file}"
                    )
            except Exception as e:
                logger.warning(f"Failed to load characters from {characters_file}: {e}")

        # Fallback to config or default data
        if not character_data_list:
            if use_custom_data and "character_data" in config:
                character_data_list = config["character_data"]
            else:
                # Enhanced default character data with more variety
                character_data_list = [
                    {
                        "name": "Alice Cooper",
                        "age": 28,
                        "pronouns": "she/her",
                        "job": "Baker",
                        "recent_event": "Opened new bakery",
                        "long_term_goal": "Expand business to neighboring towns",
                        "personality": {"extraversion": 75, "conscientiousness": 85},
                        "specialties": ["baking", "customer_service"],
                    },
                    {
                        "name": "Bob Wilson",
                        "age": 35,
                        "pronouns": "he/him",
                        "job": "Blacksmith",
                        "recent_event": "Crafted special sword for mayor",
                        "long_term_goal": "Master legendary crafting techniques",
                        "personality": {"conscientiousness": 90, "openness": 60},
                        "specialties": ["metalworking", "tool_making"],
                    },
                    {
                        "name": "Charlie Green",
                        "age": 42,
                        "pronouns": "they/them",
                        "job": "Farmer",
                        "recent_event": "Harvested record crop",
                        "long_term_goal": "Develop sustainable farming practices",
                        "personality": {"conscientiousness": 80, "agreeableness": 85},
                        "specialties": ["agriculture", "animal_care"],
                    },
                    {
                        "name": "Diana Stone",
                        "age": 31,
                        "pronouns": "she/her",
                        "job": "Teacher",
                        "recent_event": "Student won regional competition",
                        "long_term_goal": "Build the village's first library",
                        "personality": {"openness": 90, "agreeableness": 80},
                        "specialties": ["education", "research"],
                    },
                    {
                        "name": "Erik Thornfield",
                        "age": 27,
                        "pronouns": "he/him",
                        "job": "Guard",
                        "recent_event": "Prevented bandit attack",
                        "long_term_goal": "Establish village militia",
                        "personality": {"conscientiousness": 85, "neuroticism": 30},
                        "specialties": ["combat", "security"],
                    },
                    {
                        "name": "Fiona Rivers",
                        "age": 39,
                        "pronouns": "she/her",
                        "job": "Healer",
                        "recent_event": "Cured mysterious illness",
                        "long_term_goal": "Open medical school",
                        "personality": {"agreeableness": 90, "conscientiousness": 80},
                        "specialties": ["medicine", "herbalism"],
                    },
                ]

        # Limit character creation to requested count
        character_data_list = character_data_list[:character_count]

        for char_data in character_data_list:
            try:
                character = self._create_single_character(char_data, imported_modules)
                if character:
                    characters.append(character)
                    logger.debug(f"Created character: {char_data['name']}")
            except Exception as e:
                logger.error(
                    f"Error creating character {char_data.get('name', 'Unknown')}: {e}"
                )
                continue

        return characters

    def _load_characters_from_file(self, filepath: str) -> List[Dict]:
        """Load character configuration from external file."""
        try:
            import json
            import os

            if not os.path.exists(filepath):
                logger.warning(f"Characters file not found: {filepath}")
                return []

            with open(filepath, "r") as f:
                data = json.load(f)

            characters_data = data.get("characters", [])

            # Validate and sanitize character data
            validated_characters = []
            for char_data in characters_data:
                if "name" in char_data:  # Minimum requirement
                    # Set defaults for missing fields
                    char_data.setdefault("age", random.randint(18, 65))
                    char_data.setdefault("pronouns", "they/them")
                    char_data.setdefault("job", "Villager")
                    char_data.setdefault("recent_event", "Recently joined the village")
                    char_data.setdefault("long_term_goal", "Live a fulfilling life")
                    char_data.setdefault("personality", {})
                    char_data.setdefault("specialties", [])

                    validated_characters.append(char_data)
                else:
                    logger.warning(f"Skipping invalid character data: {char_data}")

            return validated_characters

        except Exception as e:
            logger.error(f"Error loading characters from file {filepath}: {e}")
            return []

    def setup_user_driven_configuration(self):
        """Setup user-driven configuration options for dynamic gameplay setup."""
        try:
            # Create default configuration files if they don't exist
            self._create_default_config_files()

            # Check for user preference file
            user_prefs_file = self.config.get(
                "user_preferences_file", "user_preferences.json"
            )
            if os.path.exists(user_prefs_file):
                try:
                    with open(user_prefs_file, "r") as f:
                        user_prefs = json.load(f)

                    # Apply user preferences to configuration
                    self.config.update(user_prefs)
                    logger.info(f"Applied user preferences from {user_prefs_file}")

                except Exception as e:
                    logger.warning(f"Error loading user preferences: {e}")

        except Exception as e:
            logger.error(f"Error setting up user-driven configuration: {e}")

    def _create_default_config_files(self):
        """Create default configuration files for user customization."""
        try:
            import json
            import os

            # Create characters template file
            characters_template = {
                "characters": [
                    {
                        "name": "Custom Character 1",
                        "age": 25,
                        "pronouns": "they/them",
                        "job": "Artisan",
                        "recent_event": "Moved to the village",
                        "long_term_goal": "Master their craft",
                        "personality": {
                            "extraversion": 60,
                            "openness": 80,
                            "conscientiousness": 70,
                            "agreeableness": 75,
                            "neuroticism": 40,
                        },
                        "specialties": ["crafting", "creativity"],
                        "inventory_items": [
                            {"name": "Tools", "type": "equipment", "quantity": 1},
                            {"name": "Bread", "type": "food", "quantity": 2},
                        ],
                    }
                ]
            }

            characters_file = "custom_characters.json"
            if not os.path.exists(characters_file):
                with open(characters_file, "w") as f:
                    json.dump(characters_template, f, indent=2)
                logger.info(f"Created character template file: {characters_file}")

            # Create buildings template file
            buildings_template = {
                "buildings": [
                    {
                        "name": "Custom Building",
                        "type": "special",
                        "x": 250,
                        "y": 250,
                        "width": 50,
                        "height": 50,
                        "description": "A custom building for special purposes",
                    }
                ]
            }

            buildings_file = "custom_buildings.json"
            if not os.path.exists(buildings_file):
                with open(buildings_file, "w") as f:
                    json.dump(buildings_template, f, indent=2)
                logger.info(f"Created buildings template file: {buildings_file}")

            # Create user preferences template
            user_prefs_template = {
                "screen_width": 1024,
                "screen_height": 768,
                "target_fps": 60,
                "characters": {
                    "count": 6,
                    "use_custom": True,
                    "characters_file": "custom_characters.json",
                },
                "map": {"buildings_file": "custom_buildings.json"},
                "render": {"background_color": [20, 50, 80], "vsync": True},
            }

            prefs_file = "user_preferences.json"
            if not os.path.exists(prefs_file):
                with open(prefs_file, "w") as f:
                    json.dump(user_prefs_template, f, indent=2)
                logger.info(f"Created user preferences template: {prefs_file}")

        except Exception as e:
            logger.error(f"Error creating default config files: {e}")

    def _create_single_character(self, char_data: Dict, imported_modules: Dict):
        """Create a single character with error handling."""
        Character = imported_modules["Character"]
        Location = imported_modules["Location"]
        ItemInventory = imported_modules["ItemInventory"]
        FoodItem = imported_modules["FoodItem"]

        try:
            # Create basic inventory with error handling
            inventory_items = char_data.get(
                "inventory_items",
                [
                    {
                        "name": "Bread",
                        "type": "food",
                        "quantity": 2,
                        "weight": 1,
                        "value": 1,
                        "description": "Fresh bread",
                        "calories": 200,
                    },
                    {
                        "name": "Apple",
                        "type": "food",
                        "quantity": 1,
                        "weight": 0.5,
                        "value": 2,
                        "description": "Red apple",
                        "calories": 80,
                    },
                ],
            )

            food_items = []
            for item_data in inventory_items:
                try:
                    food_item = FoodItem(
                        name=item_data["name"],
                        description=item_data.get("description", "A food item"),
                        value=item_data.get("value", 1),
                        perishable=item_data.get("perishable", False),
                        weight=item_data.get("weight", 1),
                        quantity=item_data.get("quantity", 1),
                        action_system=(
                            self.action_system
                            if self.action_system
                            else imported_modules.get("ActionSystem", lambda: None)()
                        ),
                        calories=item_data.get("calories", 100),
                    )
                    food_items.append(food_item)
                except Exception as e:
                    logger.warning(
                        f"Failed to create food item {item_data.get('name', 'Unknown')}: {e}"
                    )

            inventory = ItemInventory(food_items=food_items)

            # Create location for character with error handling
            location_name = f"{char_data['name']}'s Home"
            try:
                location = Location(
                    location_name,
                    random.randint(0, 100),
                    random.randint(0, 100),
                    1,
                    1,
                    self.action_system,
                )
            except Exception as e:
                logger.warning(
                    f"Failed to create location for {char_data['name']}: {e}"
                )
                location = None

            # Generate personality traits with defaults
            personality_config = char_data.get("personality", {})
            personality_traits = {
                "extraversion": personality_config.get(
                    "extraversion", random.randint(30, 80)
                ),
                "openness": personality_config.get("openness", random.randint(40, 90)),
                "conscientiousness": personality_config.get(
                    "conscientiousness", random.randint(50, 90)
                ),
                "agreeableness": personality_config.get(
                    "agreeableness", random.randint(40, 85)
                ),
                "neuroticism": personality_config.get(
                    "neuroticism", random.randint(10, 50)
                ),
            }

            # Create character with comprehensive error handling
            character_kwargs = {
                "name": char_data["name"],
                "age": char_data.get("age", random.randint(18, 65)),
                "pronouns": char_data.get("pronouns", "they/them"),
                "job": char_data.get("job", "Villager"),
                "health_status": char_data.get(
                    "health_status", random.randint(80, 100)
                ),
                "hunger_level": char_data.get("hunger_level", random.randint(20, 50)),
                "wealth_money": char_data.get("wealth_money", random.randint(50, 200)),
                "mental_health": char_data.get("mental_health", random.randint(70, 90)),
                "social_wellbeing": char_data.get(
                    "social_wellbeing", random.randint(60, 90)
                ),
                "job_performance": char_data.get(
                    "job_performance", random.randint(70, 95)
                ),
                "community": char_data.get("community", random.randint(50, 80)),
                "recent_event": char_data.get(
                    "recent_event", "Recently joined the village"
                ),
                "long_term_goal": char_data.get(
                    "long_term_goal", "Live a fulfilling life"
                ),
                "inventory": inventory,  # Pass the ItemInventory object we created
                "personality_traits": personality_traits,
                "action_system": self.action_system,
                "gametime_manager": self.gametime_manager,
                "location": location,
                "graph_manager": self.graph_manager,
                "energy": char_data.get("energy", random.randint(60, 100)),
            }

            character = Character(**character_kwargs)
            return character

        except Exception as e:
            logger.error(
                f"Failed to create character {char_data.get('name', 'Unknown')}: {e}"
            )
            logger.error(traceback.format_exc())
            return None

    def _register_characters(self, characters: List):
        """Register characters with game systems."""
        for character in characters:
            try:
                # Add character as a node in the graph
                if self.graph_manager:
                    self.graph_manager.add_character_node(character)

                # Add character to map controller with random starting position
                screen_width = self.config.get("screen_width", 800)
                screen_height = self.config.get("screen_height", 600)

                start_x = random.randint(50, screen_width - 50)
                start_y = random.randint(50, screen_height - 50)
                character.position = pygame.math.Vector2(start_x, start_y)
                character.color = (
                    random.randint(100, 255),
                    random.randint(100, 255),
                    random.randint(100, 255),
                )
                character.path = []

                # Register with map controller and local registry
                if self.map_controller:
                    self.map_controller.characters[character.uuid] = character
                self.characters[character.uuid] = character

            except Exception as e:
                logger.error(f"Error registering character {character.name}: {e}")
                continue

    def game_loop(self):
        """Main game loop with configurable frame rate and performance monitoring."""
        # TODO: Add performance profiling and optimization
        # TODO: Add frame rate adjustment based on performance
        # TODO: Add game state persistence and checkpointing
        # TODO: Add network synchronization for multiplayer
        # TODO: Add mod system integration
        # TODO: Add automated testing hooks
        # TODO: Add real-time configuration updates

        target_fps = self.config.get("target_fps", 60)

        while self.running:
            dt = self.clock.tick(target_fps) / 1000.0 * self.time_scale_factor  # Frame time in seconds with time scale factor
            # TODO: Add performance monitoring
            # frame_start_time = time.time()

            self.handle_events()
            self.update_game_state(dt)
            self.render()

            # TODO: Add frame time analysis and optimization suggestions
            # frame_end_time = time.time()
            # self._analyze_frame_performance(frame_end_time - frame_start_time)

    def handle_events(self):
        """Handle pygame events and user input with improved interaction support."""
        # TODO: Implement more sophisticated input handling
        # TODO: Add keyboard shortcuts configuration
        # TODO: Add mouse gesture recognition
        # TODO: Add multi-touch support for mobile
        # TODO: Add gamepad/controller support
        # TODO: Add accessibility features (screen reader, high contrast mode)
        # TODO: Add customizable key bindings
        # TODO: Add input recording and playback for testing

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN:
                self._handle_keydown(event)
            elif event.type == pygame.MOUSEBUTTONDOWN:
                # TODO: Add mouse interaction handling
                # TODO: Add right-click context menus
                pass
            elif event.type == pygame.MOUSEWHEEL:
                # TODO: Add zoom functionality
                # TODO: Add scroll-based UI navigation
                pass
            else:
                # Pass events to the Map Controller for handling
                if self.map_controller:
                    try:
                        self.map_controller.handle_event(event)
                    except Exception as e:
                        logger.warning(f"Error handling event in map controller: {e}")

    def _handle_keydown(self, event):
        """Handle keyboard input with configurable key bindings and new features."""
        # Load key bindings from configuration
        key_bindings = self.config.get(
            "key_bindings",
            {
                "quit": [pygame.K_ESCAPE],
                "pause": [pygame.K_SPACE],
                "reset": [pygame.K_r],
                "save": [pygame.K_s],
                "load": [pygame.K_l],
                "help": [pygame.K_h, pygame.K_F1],
                "debug": [pygame.K_F3],
                "fullscreen": [pygame.K_F11],
                "feature_status": [pygame.K_f],
                "system_recovery": [pygame.K_F5],
                "analytics": [pygame.K_a],
                "increase_speed": [pygame.K_PAGEUP], # Added for time scaling
                "decrease_speed": [pygame.K_PAGEDOWN], # Added for time scaling
                "minimap": [pygame.K_m], # Added for mini-map toggle
                "overview": [pygame.K_o], # Added for overview mode toggle
            },
        )

        if event.key in key_bindings.get("quit", [pygame.K_ESCAPE]):
            self.running = False
        elif event.key in key_bindings.get("pause", [pygame.K_SPACE]):
            # Pause/unpause simulation
            self.paused = not getattr(self, "paused", False)
            logger.info(f"Game {'paused' if self.paused else 'unpaused'}")
        elif event.key in key_bindings.get("reset", [pygame.K_r]):
            # Reset/regenerate characters
            self._reset_characters()
        elif event.key in key_bindings.get("save", [pygame.K_s]):
            # Save game functionality
            save_path = "saves/quicksave.json"
            if self.save_game_state(save_path):
                logger.info(f"Game saved to {save_path}")
            else:
                logger.error("Failed to save game")
        elif event.key in key_bindings.get("load", [pygame.K_l]):
            # Load game functionality
            save_path = "saves/quicksave.json"
            if self.load_game_state(save_path):
                logger.info(f"Game loaded from {save_path}")
            else:
                logger.error("Failed to load game")
        elif event.key in key_bindings.get("help", []):
            # Show help overlay
            self._show_help_info()
        elif event.key in key_bindings.get("debug", []):
            # Toggle debug information display
            self._toggle_debug_mode()
        elif event.key in key_bindings.get("fullscreen", []):
            # Toggle fullscreen mode
            self._toggle_fullscreen()
        elif event.key in key_bindings.get("feature_status", [pygame.K_f]):
            # Toggle feature status overlay
            self._show_feature_status = not getattr(self, "_show_feature_status", False)
            logger.info(
                f"Feature status overlay {'shown' if self._show_feature_status else 'hidden'}"
            )
        elif event.key in key_bindings.get("system_recovery", [pygame.K_F5]):
            # Force system recovery attempt
            self._force_system_recovery()
        elif event.key in key_bindings.get("analytics", [pygame.K_a]):
            # Show analytics information
            self._show_analytics_info()
        elif event.key in key_bindings.get("increase_speed", []):
            self.time_scale_factor = min(MAX_SPEED, self.time_scale_factor + SPEED_STEP)
            logger.info(f"Time scale set to: {self.time_scale_factor:.1f}x")
            self._cached_speed_text = None # Invalidate cache on change

        # For decreasing speed
        elif event.key in key_bindings.get("decrease_speed", []): # Ensure this key binding exists or add it
            self.time_scale_factor = max(MIN_SPEED, self.time_scale_factor - SPEED_STEP)
            logger.info(f"Time scale set to: {self.time_scale_factor:.1f}x")
            self._cached_speed_text = None # Invalidate cache on change
        elif event.key in key_bindings.get("minimap", [pygame.K_m]):
            # Toggle mini-map mode
            self._minimap_mode = not getattr(self, "_minimap_mode", False)
            logger.info(f"Mini-map mode {'enabled' if self._minimap_mode else 'disabled'}")
        elif event.key in key_bindings.get("overview", [pygame.K_o]):
            # Toggle overview mode
            self._overview_mode = not getattr(self, "_overview_mode", False)
            logger.info(f"Overview mode {'enabled' if self._overview_mode else 'disabled'}")

    def _show_help_info(self):
        """Display help information."""
        try:
            logger.info("=== TINY VILLAGE HELP ===")
            logger.info("Controls:")
            logger.info("  SPACE - Pause/Unpause")
            logger.info("  R - Reset characters")
            logger.info("  S - Save game")
            logger.info("  L - Load game")
            logger.info("  F - Show feature status")
            logger.info("  F5 - Force system recovery")
            logger.info("  A - Show analytics")
            logger.info("  M - Toggle mini-map")
            logger.info("  O - Toggle overview mode")
            logger.info("  ESC - Quit")
            logger.info("Features:")
            logger.info("  - Character AI with goals and actions")
            logger.info("  - Basic quest system")
            logger.info("  - Weather simulation")
            logger.info("  - Social relationship tracking")
            logger.info("  - Achievement system")
            logger.info("  - Automatic error recovery")
        except Exception as e:
            logger.error(f"Error showing help: {e}")

    def _toggle_debug_mode(self):
        """Toggle debug information display."""
        try:
            self._debug_mode = not getattr(self, "_debug_mode", False)
            if self._debug_mode:
                logger.setLevel(logging.DEBUG)
                logger.info("Debug mode enabled")
            else:
                logger.setLevel(logging.INFO)
                logger.info("Debug mode disabled")
        except Exception as e:
            logger.error(f"Error toggling debug mode: {e}")

    def _toggle_fullscreen(self):
        """Toggle fullscreen mode."""
        try:
            # This is a basic implementation - can be enhanced
            flags = pygame.display.get_surface().get_flags()
            if flags & pygame.FULLSCREEN:
                # Switch to windowed
                screen_width = self.config.get("screen_width", 800)
                screen_height = self.config.get("screen_height", 600)
                self.screen = pygame.display.set_mode((screen_width, screen_height))
                logger.info("Switched to windowed mode")
            else:
                # Switch to fullscreen
                self.screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
                logger.info("Switched to fullscreen mode")
        except Exception as e:
            logger.error(f"Error toggling fullscreen: {e}")

    def _force_system_recovery(self):
        """Force system recovery attempt for all failed systems."""
        try:
            logger.info("Forcing system recovery for all systems...")
            if hasattr(self, "recovery_manager"):
                system_status = self.recovery_manager.get_system_status()
                failed_systems = [
                    name for name, status in system_status.items() if status == "failed"
                ]

                if failed_systems:
                    logger.info(
                        f"Attempting recovery for failed systems: {failed_systems}"
                    )
                    recovery_results = {}
                    for system_name in failed_systems:
                        success = self.recovery_manager.attempt_recovery(system_name)
                        recovery_results[system_name] = (
                            "SUCCESS" if success else "FAILED"
                        )

                    logger.info(f"Recovery results: {recovery_results}")
                else:
                    logger.info("No failed systems detected")
            else:
                logger.warning("Recovery manager not available")
        except Exception as e:
            logger.error(f"Error during forced system recovery: {e}")

    def _show_analytics_info(self):
        """Display analytics information."""
        try:
            logger.info("=== ANALYTICS REPORT ===")

            # Game statistics
            stats = self.game_statistics
            logger.info(f"Game Statistics:")
            logger.info(f"  Actions executed: {stats['actions_executed']}")
            logger.info(f"  Actions failed: {stats['actions_failed']}")
            logger.info(f"  Characters created: {stats['characters_created']}")
            logger.info(f"  Errors recovered: {stats['errors_recovered']}")

            # Action analytics
            if hasattr(self, "action_resolver"):
                analytics = self.action_resolver.get_action_analytics()
                logger.info(f"Action System:")
                logger.info(f"  Success rate: {analytics['success_rate']:.1%}")
                logger.info(f"  Cache size: {analytics['cache_size']}")

                if "action_breakdown" in analytics:
                    logger.info("  Action breakdown:")
                    for action_name, action_stats in analytics[
                        "action_breakdown"
                    ].items():
                        success_rate = (
                            action_stats["successes"] / action_stats["count"]
                            if action_stats["count"] > 0
                            else 0
                        )
                        logger.info(
                            f"    {action_name}: {action_stats['successes']}/{action_stats['count']} ({success_rate:.1%})"
                        )

            # System health
            if hasattr(self, "recovery_manager"):
                system_status = self.recovery_manager.get_system_status()
                healthy_count = sum(
                    1 for status in system_status.values() if status == "healthy"
                )
                logger.info(
                    f"System Health: {healthy_count}/{len(system_status)} systems healthy"
                )

                for system_name, status in system_status.items():
                    logger.info(f"  {system_name}: {status}")

            # Feature status
            feature_status = self.get_feature_implementation_status()
            implemented_features = sum(
                1
                for status in feature_status.values()
                if status in ["BASIC_IMPLEMENTED", "FULLY_IMPLEMENTED"]
            )
            logger.info(
                f"Features: {implemented_features}/{len(feature_status)} implemented"
            )

        except Exception as e:
            logger.error(f"Error showing analytics: {e}")

    def _reset_characters(self):
        """Reset and regenerate characters with improved error handling."""
        try:
            logger.info("Starting character reset...")

            # Clear existing characters safely
            if self.map_controller and hasattr(self.map_controller, "characters"):
                self.map_controller.characters.clear()
            self.characters.clear()

            # Get required modules for character creation
            try:
                required_modules = {
                    "Character": __import__(
                        "tiny_characters", fromlist=["Character"]
                    ).Character,
                    "Location": __import__(
                        "tiny_locations", fromlist=["Location"]
                    ).Location,
                    "ItemInventory": __import__(
                        "tiny_items", fromlist=["ItemInventory"]
                    ).ItemInventory,
                    "FoodItem": __import__(
                        "tiny_items", fromlist=["FoodItem"]
                    ).FoodItem,
                }
            except Exception as e:
                logger.error(
                    f"Failed to import required modules for character reset: {e}"
                )
                self._create_fallback_characters()
                return

            # Create new characters
            character_config = self.config.get("characters", {})
            if self._can_create_characters(required_modules):
                try:
                    sample_characters = self._create_sample_characters(
                        required_modules, character_config
                    )
                    self._register_characters(sample_characters)

                    logger.info(
                        f"Reset complete - {len(self.characters)} new characters created"
                    )
                    self.game_statistics["characters_created"] += len(self.characters)

                except Exception as e:
                    logger.error(f"Error during character creation in reset: {e}")
                    self._create_fallback_characters()
            else:
                logger.warning(
                    "Cannot create full characters during reset, using fallback"
                )
                self._create_fallback_characters()

        except Exception as e:
            logger.error(f"Critical error resetting characters: {e}")
            logger.error(traceback.format_exc())
            # Ensure we have at least empty character dict
            self.characters = {}
            if self.map_controller and hasattr(self.map_controller, "characters"):
                self.map_controller.characters = {}

    def update_game_state(self, dt):
        """
        Update all game systems with delta time, improved error handling, and automatic recovery.
        
        This method now includes the integrated event-driven strategy functionality that was
        previously separated in the legacy update method. This provides a unified update
        process that handles:
        
        1. Event-driven strategy updates (integrated from legacy method)
           - Event checking via event handler
           - Strategy updates via strategy manager
           - Decision application
        2. Core system updates
           - Map controller updates
           - Character AI and decision making
           - Time management
           - Animation system
        3. Event processing and feature system updates
        4. Automatic system recovery
        
        Args:
            dt (float): Delta time in seconds since last update
        """
        # Check if game is paused
        if getattr(self, "paused", False):
            return  # Skip all updates when paused

        update_errors = []
        systems_to_recover = []

        # Integrated event-driven strategy update (from legacy update method)
        try:
            # Check for new events if event handler exists
            events = []
            if self.event_handler:
                try:
                    events = self.event_handler.check_events()
                except Exception as e:
                    logger.warning(f"Error checking events: {e}")
                    update_errors.append("Event checking failed")

            # Update strategy based on events if strategy manager exists
            decisions = []
            if self.strategy_manager:
                try:
                    decisions = self.strategy_manager.update_strategy(events if events else [])
                except Exception as e:
                    logger.warning(f"Error updating strategy: {e}")
                    update_errors.append("Strategy update failed")

            # Apply decisions to game state
            for decision in decisions:
                try:
                    # Pass None as game_state since update_game_state doesn't have access to it
                    # The decision application logic will use the controller's internal state
                    self.apply_decision(decision, None)
                except Exception as e:
                    logger.error(f"Error applying decision: {e}")
                    update_errors.append(f"Decision application failed")
        except Exception as e:
            logger.error(f"Error in event-driven strategy update: {e}")
            update_errors.append("Event-driven strategy update failed")

        # Update the map controller (handles character movement and pathfinding)
        if self.map_controller:
            try:
                self.map_controller.update(dt)
            except Exception as e:
                logger.error(f"Error updating map controller: {e}")
                update_errors.append("Map controller update failed")
                systems_to_recover.append("map_controller")
        elif not self.map_controller:
            systems_to_recover.append("map_controller")

        # Update character AI and decision making
        for character_id, character in list(self.characters.items()):
            try:
                success = self._update_character(character, dt)
                if not success:
                    update_errors.append(
                        f"Character {character.name if hasattr(character, 'name') else character_id} update failed"
                    )
            except Exception as e:
                logger.error(f"Error updating character {character_id}: {e}")
                update_errors.append(f"Character {character_id} update failed")
                continue

        # Check if we have no characters and attempt recovery
        if not self.characters:
            systems_to_recover.append("character_system")

        # Update time manager
        if hasattr(self, "gametime_manager") and self.gametime_manager:
            try:
                # Process any scheduled behaviors
                behaviors = self.gametime_manager.get_scheduled_behaviors()
                for behavior in behaviors:
                    try:
                        behavior.check_calendar()
                    except Exception as e:
                        logger.warning(f"Error processing scheduled behavior: {e}")
            except Exception as e:
                logger.error(f"Error updating time manager: {e}")
                update_errors.append("Time manager update failed")

        # Update animation system
        if hasattr(self, "animation_system") and self.animation_system:
            try:
                self.animation_system.update(dt)
            except Exception as e:
                logger.warning(f"Error updating animation system: {e}")
                # Animation errors are not critical

        # Process events using the EventHandler system and drive strategy
        try:
            if self.event_handler:
                self._process_events_and_update_strategy(dt)
            elif hasattr(self, "events") and self.events:
                # Fallback to basic event processing if no EventHandler
                self._process_pending_events()
        except Exception as e:
            logger.error(f"Error processing events and strategy: {e}")
            update_errors.append("Event processing failed")
            systems_to_recover.append("event_handler")

        # Update feature systems
        try:
            self._update_feature_systems(dt)
        except Exception as e:
            logger.warning(f"Error updating feature systems: {e}")

        # Attempt automatic recovery for failed systems
        if systems_to_recover:
            logger.info(
                f"Attempting automatic recovery for systems: {systems_to_recover}"
            )
            for system_name in systems_to_recover:
                try:
                    if self.recovery_manager.attempt_recovery(system_name):
                        logger.info(f"Successfully recovered {system_name}")
                        if update_errors and system_name in str(update_errors):
                            self.game_statistics["errors_recovered"] += 1
                except Exception as e:
                    logger.error(f"Recovery failed for {system_name}: {e}")

        # Log update errors if any occurred
        if update_errors:
            logger.warning(
                f"Game state update completed with {len(update_errors)} errors"
            )
            self.game_statistics["errors_recovered"] += len(
                [e for e in update_errors if "recovered" in e]
            )

    def _update_feature_systems(self, dt):
        """Update all implemented feature systems."""
        try:
            # Update weather system
            if hasattr(self, "weather_system"):
                self.implement_weather_system()  # This updates weather state

            # Update social networks (relationship decay/growth)
            if hasattr(self, "social_networks"):
                self._update_social_relationships(dt)

            # Update quest system (time-based quest events)
            if hasattr(self, "quest_system"):
                self._update_quest_timers(dt)

        except Exception as e:
            logger.warning(f"Error updating feature systems: {e}")

    def _update_social_relationships(self, dt):
        """Update social relationships over time."""
        try:
            if not hasattr(self, "social_networks"):
                return

            # Slow relationship decay/growth over time
            for char_id, relationships in self.social_networks["relationships"].items():
                for other_id, strength in relationships.items():
                    # Very slow decay towards neutral (50)
                    if strength > 50:
                        relationships[other_id] = max(50, strength - 0.1 * dt)
                    elif strength < 50:
                        relationships[other_id] = min(50, strength + 0.1 * dt)

        except Exception as e:
            logger.warning(f"Error updating social relationships: {e}")

    def _update_quest_timers(self, dt):
        """Update quest-related timers and generate new quests."""
        try:
            if not hasattr(self, "quest_system"):
                return

            current_time = pygame.time.get_ticks()

            # Generate new quests for characters with no active quests
            for char_id in self.characters.keys():
                if char_id in self.quest_system["active_quests"]:
                    active_quests = self.quest_system["active_quests"][char_id]

                    # Remove expired quests (older than 5 minutes)
                    active_quests[:] = [
                        q
                        for q in active_quests
                        if current_time - q.get("assigned_time", 0) < 300000
                    ]

                    # Assign new quest if character has none
                    if (
                        len(active_quests) == 0 and random.random() < 0.1
                    ):  # 10% chance per update
                        template = random.choice(self.quest_system["quest_templates"])
                        quest = {
                            "id": f"{char_id}_{current_time}",
                            "name": template["name"],
                            "description": template["description"],
                            "type": template["type"],
                            "progress": 0,
                            "target": 100,
                            "assigned_time": current_time,
                        }
                        active_quests.append(quest)
                        logger.debug(
                            f"Assigned new quest to {self.characters[char_id].name}: {quest['name']}"
                        )

        except Exception as e:
            logger.warning(f"Error updating quest timers: {e}")

    def _update_character(self, character, dt) -> bool:
        """Update a single character with comprehensive error handling."""
        try:
            # Check if character has required methods
            if not hasattr(character, "name"):
                logger.warning(f"Character missing name attribute")
                return False

            # Update character's memory and decision making
            memory_success = self._update_character_memory(character)
            goals_success = self._update_character_goals(character)
            actions_success = self._execute_character_actions(character)

            # Apply weather effects
            weather = getattr(self, "weather_system", None)
            if weather:
                current_weather = weather.get('current_weather')
                energy_decrease = WEATHER_ENERGY_EFFECTS.get(current_weather, 0) * dt
                if energy_decrease and hasattr(character, 'energy'):
                    original_energy = character.energy
                    character.energy = max(0, character.energy - energy_decrease)
                    if original_energy > character.energy:
                        logger.debug(f"{current_weather.title()} weather decreased {character.name}'s energy by {energy_decrease:.2f} to {character.energy:.2f}.")

            # Character update is successful if at least one component works
            return memory_success or goals_success or actions_success

        except Exception as e:
            logger.error(
                f"Critical error updating character {getattr(character, 'name', 'Unknown')}: {e}"
            )
            return False

    def _update_character_memory(self, character) -> bool:
        """Update character memory with error handling."""
        try:
            if hasattr(character, "recall_recent_memories"):
                character.recall_recent_memories()
                return True
        except Exception as e:
            logger.warning(f"Error updating memory for {character.name}: {e}")
        return False

    def _update_character_goals(self, character) -> bool:
        """Update character goals with error handling."""
        try:
            if hasattr(character, "evaluate_goals"):
                goals = character.evaluate_goals()
                return goals is not None
        except Exception as e:
            logger.warning(f"Error evaluating goals for {character.name}: {e}")
        return False

    def _execute_character_actions(self, character) -> bool:
        """Execute character actions with error handling and optional LLM decision-making."""
        try:
            if not self.strategy_manager:
                return False

            # Check if character should use LLM decision-making
            use_llm_decisions = getattr(character, 'use_llm_decisions', False)
            
            if use_llm_decisions:
                # Use comprehensive LLM-based decision making via process_character_turn
                return self.process_character_turn(character)
            else:
                # Use traditional strategy manager approach
                try:
                    actions = self.strategy_manager.get_daily_actions(character)
                    if actions:
                        # Execute first action (simplified approach)
                        action = actions[0]
                        if hasattr(action, 'execute'):
                            return action.execute()
                        return True
                except Exception as e:
                    logger.warning(f"Error with traditional action execution for {character.name}: {e}")
                    return False
                    
        except Exception as e:
            logger.warning(f"Error executing actions for {character.name}: {e}")
        return False

    def process_character_turn(self, character) -> bool:
        """
        Process a character's turn using LLM-driven decision making.
        
        This method integrates:
        1. PromptBuilder to create contextual decision prompts
        2. GOAP system for intelligent action planning  
        3. LLM communication via TinyBrainIO
        4. OutputInterpreter to parse LLM responses
        5. Action execution and tracking
        
        Args:
            character: Character object to process turn for
            
        Returns:
            bool: True if turn processed successfully, False otherwise
        """
        try:
            from tiny_prompt_builder import PromptBuilder
            from tiny_brain_io import TinyBrainIO
            from tiny_output_interpreter import OutputInterpreter
            from tiny_goap_system import GOAPPlanner
            
            logger.info(f"Processing LLM-driven turn for {character.name}")
            
            # Step 1: Initialize LLM components with error handling
            prompt_builder = None
            brain_io = None
            output_interpreter = None
            goap_planner = None
            
            try:
                # Initialize PromptBuilder with memory manager if available
                memory_manager = getattr(self, 'memory_manager', None)
                prompt_builder = PromptBuilder(character, memory_manager)
                
                # Initialize output interpreter
                output_interpreter = OutputInterpreter(self.graph_manager)
                
                # Initialize GOAP planner
                goap_planner = GOAPPlanner(self.graph_manager)
                
                # Initialize LLM brain - use fallback if not available
                try:
                    brain_io = TinyBrainIO("alexredna/TinyLlama-1.1B-Chat-v1.0-reasoning-v2")
                except Exception as e:
                    logger.warning(f"Could not initialize LLM brain: {e}. Using fallback decision making.")
                    brain_io = None
                    
            except ImportError as e:
                logger.warning(f"LLM components not available: {e}. Falling back to basic actions.")
                return self._execute_fallback_character_action(character)
                
            # Step 2: Get available actions from strategy manager and GOAP
            potential_actions = []
            try:
                if self.strategy_manager:
                    # Get actions from strategy manager
                    strategy_actions = self.strategy_manager.get_daily_actions(character)
                    potential_actions.extend(strategy_actions or [])
                    
                # Get additional actions from GOAP planner if available
                if goap_planner:
                    goap_actions = goap_planner.get_available_actions(character)
                    potential_actions.extend(goap_actions or [])
                    
                # Ensure we have at least some basic actions
                if not potential_actions:
                    potential_actions = self._get_basic_fallback_actions(character)
                    
            except Exception as e:
                logger.warning(f"Error getting potential actions: {e}")
                potential_actions = self._get_basic_fallback_actions(character)
            
            # Step 3: Generate decision prompt
            try:
                # Get current context
                current_time = getattr(self, 'current_time', 'morning')
                weather_system = getattr(self, 'weather_system', {})
                current_weather = weather_system.get('current_weather', 'clear')
                
                # Create action choices for prompt
                action_choices = []
                for i, action in enumerate(potential_actions[:5]):  # Limit to top 5
                    action_name = getattr(action, 'name', str(action))
                    action_cost = getattr(action, 'cost', 1.0)
                    action_choices.append(f"{i+1}. {action_name} (Cost: {action_cost:.1f})")
                
                # Generate the decision prompt
                if prompt_builder:
                    prompt = prompt_builder.generate_decision_prompt(
                        time=current_time,
                        weather=current_weather,
                        action_choices=action_choices,
                        include_conversation_context=True,
                        include_few_shot_examples=True,
                        include_memory_integration=True,
                        output_format="json"
                    )
                else:
                    # Fallback basic prompt
                    prompt = self._generate_basic_decision_prompt(character, action_choices, current_time, current_weather)
                    
            except Exception as e:
                logger.warning(f"Error generating decision prompt: {e}")
                prompt = self._generate_basic_decision_prompt(character, [], current_time, 'clear')
            
            # Step 4: Get LLM response
            llm_response = None
            if brain_io and prompt:
                try:
                    logger.debug(f"Sending prompt to LLM for {character.name}")
                    llm_response = brain_io.generate_text(prompt, max_tokens=150, temperature=0.7)
                    logger.debug(f"LLM response for {character.name}: {llm_response[:100]}...")
                except Exception as e:
                    logger.warning(f"Error getting LLM response: {e}")
                    llm_response = None
            
            # Step 5: Parse LLM response and select action
            selected_actions = []
            if llm_response and output_interpreter:
                try:
                    # Parse the LLM response and get actions
                    selected_actions = output_interpreter.interpret_response(
                        llm_response, character, potential_actions
                    )
                    logger.info(f"LLM selected {len(selected_actions)} actions for {character.name}")
                except Exception as e:
                    logger.warning(f"Error interpreting LLM response: {e}")
                    selected_actions = []
            
            # Step 6: Fallback to GOAP or default if LLM failed
            if not selected_actions and goap_planner:
                try:
                    # Use GOAP to plan actions toward character's goals
                    character_goals = []
                    if hasattr(character, 'evaluate_goals'):
                        goal_queue = character.evaluate_goals()
                        if goal_queue:
                            # Convert goal queue to goal objects
                            for utility_score, goal in goal_queue[:1]:  # Take top goal
                                character_goals.append(goal)
                    
                    if character_goals:
                        goap_plan = goap_planner.plan_for_character(character, character_goals[0])
                        if goap_plan:
                            selected_actions = goap_plan[:1]  # Take first action from plan
                            logger.info(f"GOAP selected actions for {character.name}")
                except Exception as e:
                    logger.warning(f"Error with GOAP planning: {e}")
            
            # Step 7: Final fallback to first potential action
            if not selected_actions and potential_actions:
                selected_actions = [potential_actions[0]]
                logger.info(f"Using fallback action for {character.name}")
            
            # Step 8: Execute selected actions
            execution_success = False
            if selected_actions:
                for action in selected_actions:
                    try:
                        if hasattr(action, 'execute'):
                            success = action.execute()
                            if success:
                                execution_success = True
                                logger.debug(f"Successfully executed {getattr(action, 'name', 'action')} for {character.name}")
                                
                                # Record conversation turn for learning
                                if prompt_builder and llm_response:
                                    prompt_builder.record_conversation_turn(
                                        prompt=prompt,
                                        response=llm_response,
                                        action_taken=getattr(action, 'name', 'unknown'),
                                        outcome="success" if success else "failure"
                                    )
                                break
                            else:
                                logger.warning(f"Action {getattr(action, 'name', 'action')} failed for {character.name}")
                        else:
                            logger.warning(f"Action has no execute method: {action}")
                    except Exception as e:
                        logger.warning(f"Error executing action for {character.name}: {e}")
                        continue
            
            # Step 9: Update statistics
            if execution_success:
                self.game_statistics["actions_executed"] += 1
            else:
                self.game_statistics["actions_failed"] += 1
                
            return execution_success
            
        except Exception as e:
            logger.error(f"Critical error in process_character_turn for {getattr(character, 'name', 'Unknown')}: {e}")
            # Final fallback to basic action
            return self._execute_fallback_character_action(character)
    
    def _execute_fallback_character_action(self, character) -> bool:
        """Execute a simple fallback action when LLM decision making fails."""
        try:
            # Try to get actions from strategy manager
            if self.strategy_manager:
                actions = self.strategy_manager.get_daily_actions(character)
                if actions and len(actions) > 0:
                    action = actions[0]
                    if hasattr(action, 'execute'):
                        return action.execute()
            
            # Create a basic rest action as ultimate fallback
            from actions import Action
            rest_action = Action(
                name="Rest",
                preconditions={},
                effects=[{"attribute": "energy", "change_value": 5}],
                cost=0
            )
            return rest_action.execute() if hasattr(rest_action, 'execute') else True
            
        except Exception as e:
            logger.warning(f"Even fallback action failed for {character.name}: {e}")
            return False
    
    def _get_basic_fallback_actions(self, character):
        """Get basic fallback actions when no other actions are available."""
        try:
            from actions import Action, NoOpAction, SleepAction
            
            actions = []
            
            # Create basic actions based on character needs
            if hasattr(character, 'energy') and character.energy < 50:
                sleep_action = SleepAction(duration=8, initiator_id=getattr(character, 'id', character.name))
                actions.append(sleep_action)
            
            if hasattr(character, 'hunger_level') and character.hunger_level > 7:
                from actions import EatAction
                eat_action = EatAction(item_name="food", initiator_id=getattr(character, 'id', character.name))
                actions.append(eat_action)
            
            # Always add a no-op action as final fallback
            noop_action = NoOpAction(initiator_id=getattr(character, 'id', character.name))
            actions.append(noop_action)
            
            return actions
            
        except Exception as e:
            logger.warning(f"Error creating fallback actions: {e}")
            # Return empty list if we can't even create basic actions
            return []
    
    def _generate_basic_decision_prompt(self, character, action_choices, current_time, current_weather):
        """Generate a basic decision prompt when PromptBuilder is not available."""
        try:
            prompt = f"""You are {character.name}. It's {current_time} and the weather is {current_weather}.
            
Current status:
- Energy: {getattr(character, 'energy', 50)}/100
- Hunger: {getattr(character, 'hunger_level', 5)}/10
- Health: {getattr(character, 'health_status', 75)}/100

Available actions:
"""
            
            if action_choices:
                for choice in action_choices:
                    prompt += f"{choice}\n"
            else:
                prompt += "1. Rest (Cost: 0.0)\n2. Look around (Cost: 0.1)\n"
            
            prompt += "\nChoose an action by responding with just the number (e.g., '1')."
            
            return prompt
            
        except Exception as e:
            logger.warning(f"Error generating basic prompt: {e}")
            return "Choose an action: 1. Rest"
