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

# Constants for UI elements
# MINIMAP_SIZE = 150  # Size of the minimap window

# UI Layout Constants
ACHIEVEMENT_SPACING = 25
ACHIEVEMENT_LINE_SPACING = 18
PANEL_SPACING = 8
INSTRUCTIONS_BOTTOM_MARGIN = 150
MINIMAP_SIZE = 120
DEFAULT_COLOR = (100, 150, 200)

# Notification priorities for the event system
NOTIFICATION_PRIORITIES = {
    'CRITICAL': 'high',
    'WARNING': 'medium', 
    'INFO': 'normal',
    'DEBUG': 'low'
}

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
                cache_size = analytics.get('cache_size', 0)
                analytics_text = tiny_font.render(
                    f"Success Rate: {analytics.get('success_rate', 0):.1%} | Cache: {cache_size}",
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
    """Panel for displaying selected character information with enhanced needs tracking."""
    
    def _render_content(self, screen: pygame.Surface, controller, fonts: Dict[str, pygame.font.Font]) -> int:
        small_font = fonts.get('small', pygame.font.Font(None, 18))
        tiny_font = fonts.get('tiny', pygame.font.Font(None, 16))
        x, y = self.position
        current_y = y
        
        if (hasattr(controller.map_controller, "selected_character") 
            and controller.map_controller.selected_character):
            
            char = controller.map_controller.selected_character
            
            # Character basic info
            name_text = small_font.render(f"Selected: {char.name}", True, (255, 255, 0))
            screen.blit(name_text, (x, current_y))
            current_y += name_text.get_height() + 2
            
            job_text = small_font.render(f"Job: {getattr(char, 'job', 'Unknown')}", True, (255, 255, 255))
            screen.blit(job_text, (x, current_y))
            current_y += job_text.get_height() + 2
            
            # Enhanced Needs Display
            current_y += 5  # Add spacing
            needs_header = small_font.render("Crucial Needs:", True, (200, 255, 200))
            screen.blit(needs_header, (x, current_y))
            current_y += needs_header.get_height() + 2
            
            # Energy with color coding
            energy = getattr(char, 'energy', 0)
            energy_color = (0, 255, 0) if energy > 60 else (255, 255, 0) if energy > 30 else (255, 100, 100)
            energy_text = tiny_font.render(f"  Energy: {energy}/100", True, energy_color)
            screen.blit(energy_text, (x, current_y))
            current_y += energy_text.get_height() + 1
            
            # Hunger (simulated based on energy for now)
            hunger = max(0, 100 - energy - HUNGER_OFFSET)  # Simple hunger simulation
            hunger_color = (0, 255, 0) if hunger < 30 else (255, 255, 0) if hunger < 60 else (255, 100, 100)
            hunger_text = tiny_font.render(f"  Hunger: {hunger}/100", True, hunger_color)
            screen.blit(hunger_text, (x, current_y))
            current_y += hunger_text.get_height() + 1
            
            # Health with color coding
            health = getattr(char, 'health_status', 0)
            health_color = (0, 255, 0) if health > 70 else (255, 255, 0) if health > 40 else (255, 100, 100)
            health_text = tiny_font.render(f"  Health: {health}/100", True, health_color)
            screen.blit(health_text, (x, current_y))
            current_y += health_text.get_height() + 3
            
            # Current Goal and Action
            goal_header = small_font.render("Current Status:", True, (200, 200, 255))
            screen.blit(goal_header, (x, current_y))
            current_y += goal_header.get_height() + 2
            
            # Try to get current goal from various sources
            current_goal = self._get_character_goal(char, controller)
            goal_text = tiny_font.render(f"  Goal: {current_goal}", True, (220, 220, 220))
            screen.blit(goal_text, (x, current_y))
            current_y += goal_text.get_height() + 1
            
            # Try to get current action
            current_action = self._get_character_action(char, controller)
            action_text = tiny_font.render(f"  Action: {current_action}", True, (220, 220, 220))
            screen.blit(action_text, (x, current_y))
            current_y += action_text.get_height() + 3
            
            # Social and quest info (condensed)
            if hasattr(char, "uuid") and hasattr(controller, "social_networks"):
                try:
                    relationships = controller.social_networks["relationships"].get(char.uuid, {})
                    avg_relationship = (
                        sum(relationships.values()) / len(relationships)
                        if relationships else 50
                    )
                    social_text = tiny_font.render(f"Social: {avg_relationship:.0f}/100", True, (180, 180, 180))
                    screen.blit(social_text, (x, current_y))
                    current_y += social_text.get_height() + 1

                except Exception as e:
                    logging.error(f"Error accessing social_networks while rendering selected character panel: {e}")
            
            if hasattr(char, "uuid") and hasattr(controller, "quest_system"):
                try:
                    active_quests = len(controller.quest_system["active_quests"].get(char.uuid, []))
                    completed_quests = len(controller.quest_system["completed_quests"].get(char.uuid, []))
                    quest_text = tiny_font.render(f"Quests: {active_quests} active, {completed_quests} done", True, (180, 180, 180))
                    screen.blit(quest_text, (x, current_y))
                    current_y += quest_text.get_height() + 1

                except Exception as e:
                    logging.error(f"Error loading quest system while rendering selected character panel: {e}")
            
            # Character achievements (condensed)
            try:
                if hasattr(char, 'achievements') and char.achievements:
                    ach_text = tiny_font.render(f"Achievements: {len(char.achievements)}", True, (200, 200, 150))
                    screen.blit(ach_text, (x, current_y))
                    current_y += ach_text.get_height() + 1

            except Exception as e:
                logging.error(f"Error loading achievements while rendering selected character panel: {e}")
        
        return current_y - y
    
    def _get_character_goal(self, char, controller):
        """Get character's current primary goal."""
        try:
            # Try to get goal from various sources
            if hasattr(char, 'current_goal'):
                return str(char.current_goal)
            elif hasattr(char, 'long_term_goal'):
                return str(char.long_term_goal)
            elif hasattr(controller, 'quest_system') and hasattr(char, 'uuid'):
                active_quests = controller.quest_system.get("active_quests", {}).get(char.uuid, [])
                if active_quests:
                    return active_quests[0].get('name', 'Complete Quest')
            
            # Fallback based on needs
            energy = getattr(char, 'energy', 50)
            if energy < 30:
                return "Rest and recover energy"
            elif energy < 50:
                return "Find food and rest"
            else:
                return "Work and socialize"
                
        except Exception as e:
            return "Living peacefully"
    
    def _get_character_action(self, char, controller):
        """Get character's current action."""
        try:
            # Try to get current action from various sources
            if hasattr(char, 'current_action'):
                return str(char.current_action)
            elif hasattr(char, 'last_action'):
                return f"Last: {char.last_action}"
            elif hasattr(controller, 'action_resolver'):
                # Try to infer from recent action history
                history = getattr(controller.action_resolver, 'execution_history', [])
                char_actions = [h for h in history if h.get('character_id') == getattr(char, 'uuid', '')]
                if char_actions:
                    return char_actions[-1].get('action_name', 'Unknown action')
            
            # Fallback based on character state
            energy = getattr(char, 'energy', 50)
            if energy < 20:
                return "Resting"
            elif energy < 40:
                return "Looking for food"
            else:
                return "Working"
                
        except Exception as e:
            return "Idle"


class VillageOverviewPanel(UIPanel):
    """Panel for displaying village-wide information."""
    
    def _render_content(self, screen: pygame.Surface, controller, fonts: Dict[str, pygame.font.Font]) -> int:
        small_font = fonts.get('small', pygame.font.Font(None, 18))
        tiny_font = fonts.get('tiny', pygame.font.Font(None, 16))
        x, y = self.position
        current_y = y
        
        # Header
        header_text = small_font.render("Village Overview", True, (255, 200, 100))
        screen.blit(header_text, (x, current_y))
        current_y += header_text.get_height() + 3
        
        # Population info
        total_chars = len(controller.characters)
        pop_text = tiny_font.render(f"Population: {total_chars}", True, (200, 200, 200))
        screen.blit(pop_text, (x, current_y))
        current_y += pop_text.get_height() + 1
        
        # Homeless count (simulated for now)
        homeless_count = self._calculate_homeless(controller)
        homeless_color = (0, 255, 0) if homeless_count == 0 else (255, 255, 0) if homeless_count < 3 else (255, 100, 100)
        homeless_text = tiny_font.render(f"Homeless: {homeless_count}", True, homeless_color)
        screen.blit(homeless_text, (x, current_y))
        current_y += homeless_text.get_height() + 1
        
        # General mood
        village_mood = self._calculate_village_mood(controller)
        mood_color = (0, 255, 0) if village_mood > 70 else (255, 255, 0) if village_mood > 40 else (255, 100, 100)
        mood_text = tiny_font.render(f"General Mood: {village_mood}/100", True, mood_color)
        screen.blit(mood_text, (x, current_y))
        current_y += mood_text.get_height() + 1
        
        # Active major events
        current_y += 3
        events_header = tiny_font.render("Active Events:", True, (200, 255, 200))
        screen.blit(events_header, (x, current_y))
        current_y += events_header.get_height() + 1
        
        active_events = self._get_active_events(controller)
        if active_events:
            for event_name in active_events[:3]:  # Show max 3 events
                event_text = tiny_font.render(f"• {event_name}", True, (220, 220, 180))
                screen.blit(event_text, (x + 5, current_y))
                current_y += event_text.get_height() + 1
        else:
            no_events_text = tiny_font.render("  No major events", True, (150, 150, 150))
            screen.blit(no_events_text, (x, current_y))
            current_y += no_events_text.get_height() + 1
        
        return current_y - y
    
    def _calculate_homeless(self, controller):
        """Calculate number of homeless villagers."""
        try:
            # For now, simulate based on building capacity vs population
            if hasattr(controller, 'map_controller') and controller.map_controller:
                buildings = getattr(controller.map_controller, 'buildings', [])
                houses = [b for b in buildings if 'house' in str(b.get('name', '')).lower() or b.get('type') == 'residential']
                house_capacity = len(houses) * 2  # Assume 2 people per house
                return max(0, len(controller.characters) - house_capacity)
            return 0
        except:
            return 0
    
    def _calculate_village_mood(self, controller):
        """Calculate overall village mood."""
        try:
            if not controller.characters:
                return 50
            
            total_mood = 0
            count = 0
            
            for char in controller.characters.values():
                # Base mood on energy and health
                energy = getattr(char, 'energy', 50)
                health = getattr(char, 'health_status', 50)
                char_mood = (energy + health) / 2
                
                # Factor in social relationships if available
                if hasattr(controller, 'social_networks') and hasattr(char, 'uuid'):
                    try:
                        relationships = controller.social_networks.get('relationships', {}).get(char.uuid, {})
                        if relationships:
                            avg_relationship = sum(relationships.values()) / len(relationships)
                            char_mood = (char_mood + avg_relationship) / 2
                    except:
                        pass
                
                total_mood += char_mood
                count += 1
            
            return int(total_mood / count) if count > 0 else 50
        except:
            return 50
    
    def _get_active_events(self, controller):
        """Get list of active major events."""
        try:
            events = []
            
            # Check weather events
            if hasattr(controller, 'weather_system'):
                weather = controller.weather_system.get('current_weather', 'clear')
                if weather != 'clear':
                    events.append(f"{weather.title()} weather")
            
            # Check storytelling events
            if hasattr(controller, 'storytelling_system') and controller.storytelling_system:
                try:
                    current_stories = controller.storytelling_system.get_current_stories()
                    if current_stories and 'active_stories' in current_stories:
                        for story in current_stories['active_stories'][:2]:  # Max 2 stories
                            events.append(story.get('title', 'Story Event'))
                except:
                    pass
            
            # Check for system events
            if hasattr(controller, 'events') and controller.events:
                recent_events = controller.events[-3:]  # Last 3 events
                for event in recent_events:
                    if hasattr(event, 'name'):
                        events.append(event.name)
                    elif isinstance(event, dict):
                        events.append(event.get('name', 'Unknown Event'))
            
            return events
        except:
            return []


class EventNotificationPanel(UIPanel):
    """Panel for displaying important event notifications."""
    
    def __init__(self, name: str, position: tuple = (0, 0), size: tuple = None, visible: bool = True):
        super().__init__(name, position, size, visible)
        self.notification_queue = []
        self.max_notifications = 3
        self.notification_timeout = self.DEFAULT_NOTIFICATION_TIMEOUT
    
    # Class-level constant for default notification timeout
    DEFAULT_NOTIFICATION_TIMEOUT = 5000  # 5 seconds in milliseconds
    def add_notification(self, message: str, priority: str = "normal"):
        """Add a notification to the queue."""
        import pygame
        notification = {
            'message': message,
            'priority': priority,
            'timestamp': pygame.time.get_ticks(),
            'color': self._get_priority_color(priority)
        }
        
        self.notification_queue.append(notification)
        
        # Keep only the most recent notifications
        if len(self.notification_queue) > self.max_notifications:
            self.notification_queue.pop(0)
    
    def _get_priority_color(self, priority: str):
        """Get color based on notification priority."""
        colors = {
            'high': (255, 100, 100),    # Red
            'medium': (255, 255, 100),  # Yellow  
            'normal': (200, 200, 255),  # Light blue
            'low': (150, 150, 150)      # Gray
        }
        return colors.get(priority, colors['normal'])
    
    def _render_content(self, screen: pygame.Surface, controller, fonts: Dict[str, pygame.font.Font]) -> int:
        import pygame
        tiny_font = fonts.get('tiny', pygame.font.Font(None, 16))
        x, y = self.position
        current_y = y
        
        # Remove expired notifications
        current_time = pygame.time.get_ticks()
        self.notification_queue = [
            notif for notif in self.notification_queue
            if current_time - notif['timestamp'] < self.notification_timeout
        ]
        
        if not self.notification_queue:
            return 0
        
        # Header
        header_text = tiny_font.render("Events", True, (255, 200, 100))
        screen.blit(header_text, (x, current_y))
        current_y += header_text.get_height() + 2
        
        # Render notifications
        for notification in self.notification_queue:
            # Calculate fade based on age
            age = current_time - notification['timestamp']
            alpha = max(100, 255 - int((age / self.notification_timeout) * 155))
            
            # Create surface with alpha for fade effect
            notif_surface = pygame.Surface((300, 15), pygame.SRCALPHA)
            color = (*notification['color'], alpha)
            
            notif_text = tiny_font.render(f"• {notification['message']}", True, notification['color'])
            screen.blit(notif_text, (x, current_y))
            current_y += notif_text.get_height() + 1
        
        return current_y - y


class TimeControlPanel(UIPanel):
    """Panel for time control UI buttons."""
    
    def __init__(self, name: str, position: tuple = (0, 0), size: tuple = None, visible: bool = True):
        super().__init__(name, position, size, visible)
        self.button_width = 60
        self.button_height = 20
        self.button_spacing = 5
    
    def _render_content(self, screen: pygame.Surface, controller, fonts: Dict[str, pygame.font.Font]) -> int:
        tiny_font = fonts.get('tiny', pygame.font.Font(None, 16))
        x, y = self.position
        current_y = y
        
        # Header
        header_text = tiny_font.render("Time Controls", True, (200, 255, 200))
        screen.blit(header_text, (x, current_y))
        current_y += header_text.get_height() + 3
        
        # Time control buttons
        buttons = [
            ('Pause', 0.0, (255, 100, 100)),
            ('Normal', 1.0, (100, 255, 100)),
            ('Fast', 2.0, (100, 200, 255)),
            ('Faster', 3.0, (255, 200, 100))
        ]
        
        current_speed = getattr(controller, 'time_scale_factor', 1.0)
        
        for i, (label, speed, color) in enumerate(buttons):
            button_x = x + i * (self.button_width + self.button_spacing)
            button_rect = pygame.Rect(button_x, current_y, self.button_width, self.button_height)
            
            # Highlight current speed
            if abs(current_speed - speed) < 0.1:
                pygame.draw.rect(screen, (100, 100, 100), button_rect)
                pygame.draw.rect(screen, color, button_rect, 2)
            else:
                pygame.draw.rect(screen, (50, 50, 50), button_rect)
                pygame.draw.rect(screen, color, button_rect, 1)
            
            # Button text
            text_surface = tiny_font.render(label, True, color)
            text_rect = text_surface.get_rect(center=button_rect.center)
            screen.blit(text_surface, text_rect)
        
        current_y += self.button_height + 5
        
        # Current speed display
        speed_text = tiny_font.render(f"Current: {current_speed:.1f}x", True, (200, 200, 200))
        screen.blit(speed_text, (x, current_y))
        current_y += speed_text.get_height()
        
        return current_y - y
    
    def handle_click(self, position, controller):
        """Handle clicks on time control buttons."""
        x, y = self.position
        button_y = y + 18  # Adjust for header
        
        if button_y <= position[1] <= button_y + self.button_height:
            buttons = [0.0, 1.0, 2.0, 3.0]  # Speed values
            
            for i, speed in enumerate(buttons):
                button_x = x + i * (self.button_width + self.button_spacing)
                if button_x <= position[0] <= button_x + self.button_width:
                    controller.time_scale_factor = speed
                    if speed == 0.0:
                        controller.paused = True
                    else:
                        controller.paused = False
                    return True
        return False


class InstructionsPanel(UIPanel):
    """Panel for displaying enhanced game instructions and help."""
    
    def __init__(self, name: str, position: tuple = (0, 0), size: tuple = None, visible: bool = True):
        super().__init__(name, position, size, visible)
        self.help_mode = 'basic'  # 'basic', 'advanced', 'tutorial'
        self.tutorial_step = 0
    
    def _render_content(self, screen: pygame.Surface, controller, fonts: Dict[str, pygame.font.Font]) -> int:
        tiny_font = fonts.get('tiny', pygame.font.Font(None, 16))
        x, y = self.position
        current_y = y
        
        if self.help_mode == 'tutorial':
            return self._render_tutorial(screen, controller, fonts, x, current_y)
        elif self.help_mode == 'advanced':
            return self._render_advanced_help(screen, controller, fonts, x, current_y)
        else:
            return self._render_basic_help(screen, controller, fonts, x, current_y)
    
    def _render_basic_help(self, screen, controller, fonts, x, y):
        """Render basic help instructions."""
        tiny_font = fonts.get('tiny', pygame.font.Font(None, 16))
        current_y = y
        
        # Header
        header_text = tiny_font.render("Controls & Help", True, (255, 255, 100))
        screen.blit(header_text, (x, current_y))
        current_y += header_text.get_height() + 2
        
        instructions = [
            "MOUSE:",
            "• Click characters to select",
            "• Click buildings to interact",
            "• Right-click for context menu",
            "",
            "KEYBOARD:",
            "• SPACE - pause/unpause",
            "• R - reset characters",
            "• S - save game",
            "• L - load game", 
            "• F - feature status",
            "• M - toggle mini-map",
            "• O - overview mode",
            "• H - cycle help modes",
            "• ESC - quit",
            "",
            "Press H for advanced help"
        ]
        
        for instruction in instructions:
            if instruction == "":
                current_y += 3  # Spacing for empty lines
                continue
            
            color = (200, 255, 200) if instruction.endswith(":") else (200, 200, 200)
            inst_text = tiny_font.render(instruction, True, color)
            screen.blit(inst_text, (x, current_y))
            current_y += inst_text.get_height() + 1
        
        return current_y - y
    
    def _render_advanced_help(self, screen, controller, fonts, x, y):
        """Render advanced help information."""
        tiny_font = fonts.get('tiny', pygame.font.Font(None, 16))
        current_y = y
        
        # Header
        header_text = tiny_font.render("Advanced Help", True, (255, 255, 100))
        screen.blit(header_text, (x, current_y))
        current_y += header_text.get_height() + 2
        
        advanced_info = [
            "CHARACTER NEEDS:",
            "• Energy: Rest when low",
            "• Hunger: Find food regularly", 
            "• Health: Seek healing if needed",
            "• Social: Interact with others",
            "",
            "VILLAGE MANAGEMENT:",
            "• Monitor homeless count",
            "• Watch village mood",
            "• Respond to events",
            "",
            "BUILDINGS:",
            "• Houses: Provide shelter",
            "• Market: Trade resources", 
            "• Tavern: Social gathering",
            "• Farm: Food production",
            "",
            "Press H for tutorial mode"
        ]
        
        for info in advanced_info:
            if info == "":
                current_y += 3
                continue
            
            color = (200, 255, 200) if info.endswith(":") else (200, 200, 200)
            info_text = tiny_font.render(info, True, color)
            screen.blit(info_text, (x, current_y))
            current_y += info_text.get_height() + 1
        
        return current_y - y
    
    def _render_tutorial(self, screen, controller, fonts, x, y):
        """Render interactive tutorial."""
        tiny_font = fonts.get('tiny', pygame.font.Font(None, 16))
        current_y = y
        
        # Header
        header_text = tiny_font.render(f"Tutorial Step {self.tutorial_step + 1}", True, (255, 255, 100))
        screen.blit(header_text, (x, current_y))
        current_y += header_text.get_height() + 2
        
        tutorial_steps = [
            [
                "Welcome to Tiny Village!",
                "",
                "This is your village with",
                "characters living their lives.",
                "",
                "Click a character to select",
                "them and see their info.",
                "",
                "Press H to continue..."
            ],
            [
                "Character Information",
                "",
                "Selected characters show:",
                "• Energy and hunger levels",
                "• Current goals and actions", 
                "• Social relationships",
                "",
                "Keep characters happy by",
                "meeting their needs!",
                "",
                "Press H to continue..."
            ],
            [
                "Village Overview",
                "",
                "The village panel shows:",
                "• Total population",
                "• Homeless count",
                "• General mood",
                "• Active events",
                "",
                "A happy village is a",
                "thriving village!",
                "",
                "Press H for basic help"
            ]
        ]
        
        if self.tutorial_step < len(tutorial_steps):
            for line in tutorial_steps[self.tutorial_step]:
                if line == "":
                    current_y += 3
                    continue
                
                color = (255, 255, 100) if line.endswith("!") else (200, 200, 200)
                line_text = tiny_font.render(line, True, color)
                screen.blit(line_text, (x, current_y))
                current_y += line_text.get_height() + 1
        
        return current_y - y
    
    def cycle_help_mode(self):
        """Cycle through help modes."""
        if self.help_mode == 'basic':
            self.help_mode = 'advanced'
        elif self.help_mode == 'advanced':
            self.help_mode = 'tutorial'
            self.tutorial_step = 0
        else:
            if self.tutorial_step < 2:
                self.tutorial_step += 1
            else:
                self.help_mode = 'basic'


class BuildingInteractionPanel(UIPanel):
    """Panel for displaying building interaction prompts."""
    
    def __init__(self, name: str, position: tuple = (0, 0), size: tuple = None, visible: bool = False):
        super().__init__(name, position, size, visible)
        self.selected_building = None
        self.interaction_timeout = 0
    
    def show_building_interaction(self, building, mouse_pos):
        """Show interaction options for a building."""
        import pygame
        self.selected_building = building
        self.position = (mouse_pos[0] + 10, mouse_pos[1] - 50)  # Position near cursor
        self.visible = True
        self.interaction_timeout = pygame.time.get_ticks() + 5000  # 5 second timeout
    
    def _render_content(self, screen: pygame.Surface, controller, fonts: Dict[str, pygame.font.Font]) -> int:
        import pygame
        if not self.selected_building or pygame.time.get_ticks() > self.interaction_timeout:
            self.visible = False
            return 0
        
        tiny_font = fonts.get('tiny', pygame.font.Font(None, 16))
        small_font = fonts.get('small', pygame.font.Font(None, 18))
        x, y = self.position
        current_y = y
        
        # Background
        panel_width = 150
        panel_height = 80
        panel_rect = pygame.Rect(x, y, panel_width, panel_height)
        pygame.draw.rect(screen, (20, 20, 20), panel_rect)
        pygame.draw.rect(screen, (100, 100, 100), panel_rect, 2)
        
        # Building name
        building_name = self.selected_building.get('name', 'Unknown Building')
        name_text = small_font.render(building_name, True, (255, 255, 100))
        screen.blit(name_text, (x + 5, current_y + 5))
        current_y += name_text.get_height() + 8
        
        # Available actions
        actions = self._get_building_actions(self.selected_building)
        for action in actions:
            action_text = tiny_font.render(f"• {action}", True, (200, 200, 200))
            screen.blit(action_text, (x + 8, current_y))
            current_y += action_text.get_height() + 2
        
        return panel_height
    
    def _get_building_actions(self, building):
        """Get available actions for a building."""
        BUILDING_ACTIONS = {
            'residential': ['Enter home', 'Rest inside', 'Visit resident'],
            'commercial': ['Browse goods', 'Trade items', 'Meet merchants'],
            'social': ['Get a drink', 'Socialize', 'Listen to stories'],
            'agricultural': ['Help with crops', 'Gather food', 'Learn farming'],
            'crafting': ['Commission item', 'Learn crafting', 'Repair tools'],
            'educational': ['Attend class', 'Study books', 'Teach others'],
        }
        DEFAULT_ACTIONS = ['Examine', 'Enter building', 'Look around']
        
        building_type = building.get('type', 'generic')
        building_name = building.get('name', '').lower()
        
        # Match building type or keywords in name
        for key, actions in BUILDING_ACTIONS.items():
            if key in building_type or key in building_name:
                return actions[:3]  # Limit to 3 actions
        
        return DEFAULT_ACTIONS[:3]  # Fallback to default actions

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
                'time_controls': TimeControlPanel('time_controls', position=(10, 120)),
                'weather': WeatherPanel('weather', position=(10, 170)),
                'village_overview': VillageOverviewPanel('village_overview', position=(10, 220)),
                'stats': StatsPanel('stats', position=(10, 320)),
                'achievements': AchievementPanel('achievements', position=(10, 380)),
                'selected_character': SelectedCharacterPanel('selected_character', position=(10, 480)),
                'event_notifications': EventNotificationPanel('event_notifications', position=(400, 10)),
                'building_interaction': BuildingInteractionPanel('building_interaction', position=(0, 0), visible=False),
                'instructions': InstructionsPanel('instructions', position=(10, None))  # Position set dynamically
            }
            
            # Create font dictionary for consistent font usage
            self.ui_fonts = {
                'normal': pygame.font.Font(None, 24),
                'small': pygame.font.Font(None, 18),
                'tiny': pygame.font.Font(None, 16)
            }
            
            logger.info("Enhanced modular UI system initialized")
            
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
                self._handle_mouse_click(event)
            elif event.type == pygame.MOUSEWHEEL:
                self._handle_mouse_wheel(event)
            else:
                # Pass events to the Map Controller for handling
                if self.map_controller:
                    try:
                        self.map_controller.handle_event(event)
                    except Exception as e:
                        logger.warning(f"Error handling event in map controller: {e}")
    
    def _handle_mouse_click(self, event):
        """Enhanced mouse click handling for UI and game interactions."""
        try:
            mouse_pos = event.pos
            
            # Check UI panel clicks first
            if self.handle_ui_click(mouse_pos):
                return
            
            # Handle building interactions on right-click
            if event.button == 3:  # Right click
                if self.map_controller:
                    building = self._find_building_at_position(mouse_pos)
                    if building:
                        self.show_building_interaction(building, mouse_pos)
                        return
            
            # Handle character/building selection on left-click
            if event.button == 1:  # Left click
                # Hide building interaction panel
                if hasattr(self, 'ui_panels') and 'building_interaction' in self.ui_panels:
                    self.ui_panels['building_interaction'].visible = False
                
                # Pass to map controller for character/building selection
                if self.map_controller:
                    self.map_controller.handle_event(event)
                    
        except Exception as e:
            logger.error(f"Error handling mouse click: {e}")
    
    def _handle_mouse_wheel(self, event):
        """Handle mouse wheel for zoom and navigation."""
        try:
            # TODO: Implement zoom functionality
            # For now, use wheel for time scale adjustment
            if event.y > 0:  # Scroll up
                self.time_scale_factor = min(5.0, self.time_scale_factor + 0.2)
            elif event.y < 0:  # Scroll down
                self.time_scale_factor = max(0.1, self.time_scale_factor - 0.2)
            
            # Provide feedback
            self.add_event_notification(f"Speed: {self.time_scale_factor:.1f}x", "normal")
            
        except Exception as e:
            logger.error(f"Error handling mouse wheel: {e}")
    
    def _find_building_at_position(self, position):
        """Find building at the given screen position."""
        try:
            if not self.map_controller or not hasattr(self.map_controller, 'map_data'):
                return None
            
            buildings = self.map_controller.map_data.get('buildings', [])
            for building in buildings:
                if 'rect' in building and building['rect'].collidepoint(position):
                    return building
            return None
        except Exception as e:
            logger.error(f"Error finding building at position: {e}")
            return None

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
        elif event.key in key_bindings.get("help", [pygame.K_h, pygame.K_F1]):
            # Cycle help modes
            self.cycle_help_mode()
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

        # Robust event-driven strategy update using EventHandler
        try:
            self._process_events_and_drive_strategy(update_errors)
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

        # Note: Event processing and strategy update is now handled above in _process_events_and_drive_strategy
        # No need for separate event processing calls

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
                            result = action.execute(target=character, initiator=character)
                            if result:
                                # Update character state after successful action
                                self._update_character_state_after_action(character, action)

                                # Track successful execution
                                self.action_resolver.track_action_execution(
                                    action, character, True
                                )

                                # Update quest progress if applicable
                                self._update_quest_progress(character, action)
                                
                                # Provide feedback for successful action
                                action_name = getattr(action, 'name', str(action))
                                character_name = getattr(character, 'name', 'Character')
                                self.provide_action_feedback(action_name, True, character_name)

                                return True
                            else:
                                logger.warning(f"Action {action.name} execution returned False")
                                self.action_resolver.track_action_execution(
                                    action, character, False
                                )
                                
                                # Provide feedback for failed action
                                action_name = getattr(action, 'name', str(action))
                                character_name = getattr(character, 'name', 'Character')
                                self.provide_action_feedback(action_name, False, character_name)
                                return False
                        else:
                            logger.warning(f"Action {action} has no execute method")
                            self.action_resolver.track_action_execution(action, character, False)
                            return False
                    else:
                        logger.warning(f"No actions available for {character.name}")
                        return False
                except Exception as e:
                    logger.error(f"Error executing action: {e}")
                    # Try fallback action
                    fallback_success = self._execute_fallback_action(character)
                    if hasattr(locals(), 'action'):
                        self.action_resolver.track_action_execution(
                            action, character, fallback_success
                        )
                    return fallback_success

        except Exception as e:
            logger.error(f"Critical error executing character actions: {e}")
            fallback_success = self._execute_fallback_action(character)
            return fallback_success

    def _execute_fallback_action(self, character) -> bool:
        """Execute a safe fallback action when normal actions fail."""
        try:
            fallback_action = self.action_resolver.get_fallback_action(character)
            if fallback_action and hasattr(fallback_action, "execute"):
                result = fallback_action.execute(target=character, initiator=character)
                if result:
                    logger.info(f"Fallback action executed for {character.name}")
                    return True

            # If even fallback fails, just update character energy minimally
            if hasattr(character, "energy"):
                character.energy = max(
                    0, character.energy - 1
                )  # Minimal energy cost for existing
            return True

        except Exception as e:
            logger.error(f"Even fallback action failed for {character.name}: {e}")
            return False

    def _update_character_state_after_action(self, character, action):
        """Update character state and related systems after action execution with comprehensive tracking."""
        try:
            # Track state before updates for comparison (if needed for future features)
            # initial_state = self._capture_character_state(character)

            # The following direct call to graph_manager.update_character_state is removed.
            # Action.execute() is now responsible for updating the GraphManager based on action effects.
            # The method update_character_state was also found to not exist in GraphManager.

            # Update memory system - record the action as a memory
            # This is controller-level logic, managing how actions translate to memories.
            if hasattr(character, "add_memory"):
                try:
                    memory_text = (
                        f"Performed action: {getattr(action, 'name', str(action))}"
                    )
                    character.add_memory(memory_text)
                    return True
                except Exception as e:
                    logger.warning(f"Error adding memory for {character.name}: {e}")
                    return False
            
            return True  # Success if no memory system available
                    
        except Exception as e:
            logger.warning(f"Error updating character state after action for {character.name}: {e}")
            return False

    def _update_quest_progress(self, character, action):
        """Update quest progress based on action execution."""
        try:
            if not hasattr(self, "quest_system"):
                return
            
            char_id = getattr(character, 'uuid', getattr(character, 'id', character.name))
            if char_id not in self.quest_system.get("active_quests", {}):
                return
            
            active_quests = self.quest_system["active_quests"][char_id]
            action_name = getattr(action, 'name', str(action))
            
            # Update quest progress based on action type
            for quest in active_quests:
                quest_type = quest.get('type', '')
                if quest_type == 'collection' and 'gather' in action_name.lower():
                    quest['progress'] = min(100, quest.get('progress', 0) + 10)
                elif quest_type == 'social' and 'talk' in action_name.lower():
                    quest['progress'] = min(100, quest.get('progress', 0) + 15)
                elif quest_type == 'skill' and 'work' in action_name.lower():
                    quest['progress'] = min(100, quest.get('progress', 0) + 5)
                
                # Complete quest if progress reaches target
                if quest.get('progress', 0) >= quest.get('target', 100):
                    # Move to completed quests
                    completed_quests = self.quest_system.get("completed_quests", {})
                    if char_id not in completed_quests:
                        completed_quests[char_id] = []
                    completed_quests[char_id].append(quest)
                    active_quests.remove(quest)
                    logger.info(f"Quest '{quest['name']}' completed by {character.name}")
                    
        except Exception as e:
            logger.warning(f"Error updating quest progress for {character.name}: {e}")

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

            # For now, use fallback action since full LLM integration is complex
            # This can be expanded later with full LLM processing logic
            return self._execute_fallback_character_action(character)
                
        except Exception as e:
            logger.error(f"Critical error in process_character_turn for {getattr(character, 'name', 'Unknown')}: {e}")
            # Final fallback to basic action
            return self._execute_fallback_character_action(character)

    def _update_social_networks_from_event(self, event_name):
        """Update social networks based on social events."""
        try:
            if hasattr(self, 'social_networks'):
                # Strengthen relationships for participants in social events
                for char_id, relationships in self.social_networks.get('relationships', {}).items():
                    for other_id in relationships:
                        # Small boost to all relationships after community events
                        current_strength = relationships[other_id]
                        relationships[other_id] = min(100, current_strength + 2)
                        
        except Exception as e:
            logger.warning(f"Error updating social networks from event: {e}")

    def _update_economic_state_from_event(self, event_name):
        """Update economic state based on economic events."""
        try:
            # Update character wealth and village economic activity
            for character in self.characters.values():
                if hasattr(character, 'wealth_money'):
                    if 'market' in event_name.lower():
                        # Market events boost everyone's wealth slightly
                        character.wealth_money += random.randint(1, 5)
                    elif 'trade' in event_name.lower():
                        # Trade events have bigger economic impact
                        character.wealth_money += random.randint(5, 15)
                        
        except Exception as e:
            logger.warning(f"Error updating economic state from event: {e}")

    def _apply_strategy_to_character(self, character, strategy):
        """Apply a strategy decision to a character."""
        try:
            # If strategy is a single action, execute it
            if hasattr(strategy, 'execute'):
                success = self._execute_single_action(character, strategy)
                logger.debug(f"Applied strategy action {strategy.name} to {character.name}: {'success' if success else 'failed'}")
                
            # If strategy is a list of actions, execute them
            elif isinstance(strategy, list):
                for action in strategy:
                    if hasattr(action, 'execute'):
                        success = self._execute_single_action(character, action)
                        logger.debug(f"Applied strategy action {action.name} to {character.name}: {'success' if success else 'failed'}")
                        
            # If strategy is a decision object with actions
            elif hasattr(strategy, 'actions'):
                for action in strategy.actions:
                    success = self._execute_single_action(character, action)
                    logger.debug(f"Applied strategy action from decision to {character.name}: {'success' if success else 'failed'}")
                    
        except Exception as e:
            logger.warning(f"Error applying strategy to character {character.name}: {e}")

    def render(self):
        """Render all game elements with configurable quality and effects."""
        # TODO: Add render quality settings (low, medium, high, ultra)
        # TODO: Add dynamic resolution scaling based on performance
        # TODO: Add anti-aliasing options
        # TODO: Add post-processing effects (bloom, shadows, etc.)
        # TODO: Add level-of-detail (LOD) system for distant objects
        # TODO: Add particle effects system
        # TODO: Add lighting and shadow system
        # TODO: Add weather and atmospheric effects
        # TODO: Add screenshot and video recording functionality
        # TODO: Add VR/AR rendering support

        # Get render configuration
        render_config = self.config.get("render", {})
        background_color = render_config.get("background_color", (0, 0, 0))
        enable_vsync = render_config.get("vsync", True)

        # Clear the screen with configurable background
        self.screen.fill(background_color)

        # Check if overview mode is active
        if getattr(self, "_overview_mode", False):
            # Render overview mode instead of normal view
            self._render_overview()
        else:
            # Render the map and game world normally
            if self.map_controller:
                try:
                    self.map_controller.render(self.screen)
                except Exception as e:
                    logger.error(f"Error rendering map: {e}")
                    # TODO: Add fallback rendering for when map fails

            # Render UI elements
            self._render_ui()

        # TODO: Add render effect layers (lighting, particles, post-processing)

        # Flip the display to show the updated frame
        if enable_vsync:
            pygame.display.flip()
        else:
            pygame.display.update()
    def _render_ui(self):
        """Render user interface elements using the modular panel system."""
        try:
            # Use modular UI system if available
            if hasattr(self, 'ui_panels') and self.ui_panels:
                self._render_modular_ui()
            else:
                # Fallback to legacy rendering
                self._render_legacy_ui()
                
        except Exception as e:
            # Ultimate fallback to minimal UI
            self._render_minimal_ui()
    
    def _render_modular_ui(self):
        """Render UI using the modular panel system."""
        current_y = 10
        
        # Render left panels in order
        left_panel_order = ['character_info', 'game_status', 'time_controls', 'weather', 'village_overview', 'stats', 'achievements', 'selected_character']
        
        for panel_name in left_panel_order:
            panel = self.ui_panels.get(panel_name)
            if panel and panel.visible:
                # Update panel position if needed
                if panel.position[1] != current_y and getattr(panel, 'auto_position', True):
                    panel.position = (panel.position[0], current_y)
                
                # Render panel and update y position
                height = panel.render(self.screen, self, self.ui_fonts)
                current_y += height + PANEL_SPACING  # Add spacing between panels
        
        # Render right-side panels (notifications, building interactions)
        right_panels = ['event_notifications', 'building_interaction']
        for panel_name in right_panels:
            panel = self.ui_panels.get(panel_name)
            if panel and panel.visible:
                panel.render(self.screen, self, self.ui_fonts)
        
        # Render instructions at bottom
        instructions_panel = self.ui_panels.get('instructions')
        if instructions_panel and instructions_panel.visible:
            # Position instructions at bottom of screen
            instructions_y = self.screen.get_height() - INSTRUCTIONS_BOTTOM_MARGIN  # Reserve space for instructions
            instructions_panel.position = (10, instructions_y)
            instructions_panel.render(self.screen, self, self.ui_fonts)
        
        # Show feature status overlay if enabled
        if getattr(self, "_show_feature_status", False):
            self._render_feature_status_overlay()
    
    def _render_legacy_ui(self):
        """Legacy UI rendering method for fallback."""
        try:

            # TODO: Implement modular UI system with panels
            # TODO: Add character relationship visualization
            # TODO: Add village statistics dashboard
            # TODO: Add interactive building information panels
            # TODO: Add mini-map and overview mode - IMPLEMENTED: Toggle mini-map with 'M' key and overview mode with 'O' key
            # TODO: Add save/load game functionality UI
            # TODO: Add settings and configuration panels
            # TODO: Add help and tutorial overlays
            # TODO: Add drag-and-drop interaction hints
            # TODO: Add notification system for important events

            # Create font for UI text
            font = pygame.font.Font(None, 24)
            small_font = pygame.font.Font(None, 18)
            tiny_font = pygame.font.Font(None, 16)

            # Render character count and basic info
            char_count_text = font.render(
                f"Characters: {len(self.characters)}", True, (255, 255, 255)
            )
            self.screen.blit(char_count_text, (10, 10))

            # Render pause status
            if getattr(self, "paused", False):
                pause_text = font.render("PAUSED", True, (255, 255, 0))
                self.screen.blit(pause_text, (self.screen.get_width() - 100, 10))

            # Show basic error message
            error_text = small_font.render("Using legacy UI (panels unavailable)", True, (255, 200, 0))
            self.screen.blit(error_text, (10, 40))
            
        except Exception as e:
            logger.error(f"Error in legacy UI rendering: {e}")
            # Display minimal error state
            try:
                font = pygame.font.Font(None, 24)
                error_text = font.render("UI Error - Basic mode active", True, (255, 0, 0))
                self.screen.blit(error_text, (10, 10))
            except:
                pass  # If even basic rendering fails, just continue
    
    def _render_minimal_ui(self):
        """Ultimate fallback UI rendering when all else fails."""
        try:
            font = pygame.font.Font(None, 24)
            text = font.render("Tiny Village (Safe Mode)", True, (255, 255, 255))
            self.screen.blit(text, (10, 10))
        except Exception as e:
            logger.error(f"Even minimal UI rendering failed: {e}")

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
            
            # Simple energy restoration as ultimate fallback
            if hasattr(character, 'energy'):
                character.energy = min(100, character.energy + 5)
                logger.debug(f"Applied basic rest to {character.name}")
                return True
            
            return True  # Success by default
            
        except Exception as e:
            logger.warning(f"Even fallback action failed for {character.name}: {e}")
            return False
    
    def _get_basic_fallback_actions(self, character):
        """Get basic fallback actions when no other actions are available."""
        try:
            # Return simple action descriptions for fallback processing
            actions = []
            
            # Create basic actions based on character needs
            if hasattr(character, 'energy') and character.energy < 50:
                actions.append({"name": "Rest", "cost": 0, "energy_gain": 10})
            
            if hasattr(character, 'hunger_level') and character.hunger_level > 7:
                actions.append({"name": "Eat", "cost": 0, "hunger_reduction": 3})
            
            # Always add a basic action as final fallback
            actions.append({"name": "Idle", "cost": 0, "energy_cost": 1})
            
            return actions
            
        except Exception as e:
            logger.warning(f"Error creating fallback actions: {e}")
            # Return empty list if we can't even create basic actions
            return []

    def _process_events_and_drive_strategy(self, update_errors):
        """
        Robust event processing and strategy driving using EventHandler.check_events().
        
        This method replaces the previous insufficient _process_pending_events and
        consolidates all event-driven strategy logic into a single, comprehensive approach.
        
        Args:
            update_errors (list): List to append any errors encountered during processing
        """
        if not self.event_handler:
            # Fallback to basic event processing for legacy compatibility
            self._process_basic_events_fallback(update_errors)
            return

        try:
            # Step 1: Check for events using EventHandler - this is the primary driver
            events = []
            try:
                events = self.event_handler.check_events()
                logger.debug(f"EventHandler found {len(events)} events to process")
            except Exception as e:
                logger.warning(f"Error checking events via EventHandler: {e}")
                update_errors.append("Event checking failed")
                return

            # Step 2: Process events if any were found
            if events:
                try:
                    # Let EventHandler process the events and their effects
                    event_results = self.event_handler.process_events()
                    logger.debug(f"EventHandler processed events: {len(event_results.get('processed_events', []))} successful")
                    
                    # Handle event processing results
                    if event_results.get('failed_events'):
                        logger.warning(f"Some events failed processing: {event_results['failed_events']}")
                        update_errors.append(f"Event processing failures: {len(event_results['failed_events'])}")
                        
                except Exception as e:
                    logger.warning(f"Error processing events: {e}")
                    update_errors.append("Event processing failed")

            # Step 3: Update strategy based on events (whether processed or not)
            strategy_result = None
            if self.strategy_manager:
                try:
                    # Strategy manager should receive all events to make informed decisions
                    strategy_result = self.strategy_manager.update_strategy(events)
                    logger.debug(f"StrategyManager generated strategy result: {type(strategy_result)}")
                except Exception as e:
                    logger.warning(f"Error updating strategy based on events: {e}")
                    update_errors.append("Strategy update failed")

            # Step 4: Apply strategic result to game state
            self._apply_strategy_result(strategy_result, update_errors)
            
            # Step 5: Handle cascading events and dynamic event generation
            self._handle_cascading_and_dynamic_events(events, update_errors)
            
        except Exception as e:
            logger.error(f"Critical error in event-driven strategy processing: {e}")
            update_errors.append("Event-driven strategy system failure")

    def _apply_strategy_result(self, strategy_result, update_errors):
        """Apply strategy result from the strategy manager, handling different return types."""
        if strategy_result is None:
            return
            
        try:
            # Handle different types of strategy results
            if isinstance(strategy_result, list):
                # List of decisions - apply each one
                for i, decision in enumerate(strategy_result):
                    try:
                        if decision:
                            self.apply_decision(decision, None)
                            logger.debug(f"Applied strategic decision {i+1}/{len(strategy_result)}")
                        else:
                            logger.warning(f"Received empty decision at index {i}")
                    except Exception as e:
                        logger.error(f"Error applying strategic decision {i}: {e}")
                        update_errors.append(f"Decision application failed (decision {i})")
                        continue
            
            elif hasattr(strategy_result, 'execute'):
                # Single action - execute it directly
                try:
                    success = strategy_result.execute()
                    if success:
                        logger.debug(f"Successfully executed strategy action: {strategy_result.name}")
                        self.game_statistics["actions_executed"] += 1
                    else:
                        logger.warning(f"Strategy action execution failed: {strategy_result.name}")
                        self.game_statistics["actions_failed"] += 1
                        update_errors.append("Strategy action execution failed")
                        
                    # Track action execution for analytics
                    if hasattr(self, 'action_resolver'):
                        self.action_resolver.track_action_execution(strategy_result, None, success)
                        
                except Exception as e:
                    logger.error(f"Error executing strategy action: {e}")
                    update_errors.append("Strategy action execution error")
            
            elif isinstance(strategy_result, dict):
                # Dictionary decision - apply it as a single decision
                try:
                    self.apply_decision(strategy_result, None)
                    logger.debug("Applied dictionary-based strategic decision")
                except Exception as e:
                    logger.error(f"Error applying dictionary decision: {e}")
                    update_errors.append("Dictionary decision application failed")
            
            else:
                # Unknown type - log warning but don't fail
                logger.warning(f"Unknown strategy result type: {type(strategy_result)}. Skipping application.")
                
        except Exception as e:
            logger.error(f"Critical error applying strategy result: {e}")
            update_errors.append("Strategy result application failure")

    def _apply_strategic_decisions(self, decisions, update_errors):
        """
        DEPRECATED: Use _apply_strategy_result instead.
        Apply strategic decisions generated by the strategy manager.
        """
        logger.warning("_apply_strategic_decisions is deprecated. Use _apply_strategy_result instead.")
        self._apply_strategy_result(decisions, update_errors)

    def _handle_cascading_and_dynamic_events(self, events, update_errors):
        """Handle cascading events and generate new dynamic events based on current state."""
        try:
            # Process any cascading events that were triggered
            if self.event_handler and hasattr(self.event_handler, 'process_cascading_queue'):
                cascading_processed = self.event_handler.process_cascading_queue()
                if cascading_processed:
                    logger.info(f"Processed {len(cascading_processed)} cascading events")

            # Generate dynamic events based on current world state
            if self.event_handler and hasattr(self.event_handler, 'generate_dynamic_events'):
                world_state = self._get_current_world_state()
                dynamic_events = self.event_handler.generate_dynamic_events(
                    world_state, 
                    list(self.characters.values()) if self.characters else None
                )
                if dynamic_events:
                    logger.info(f"Generated {len(dynamic_events)} dynamic events")
                    
        except Exception as e:
            logger.warning(f"Error handling cascading/dynamic events: {e}")
            update_errors.append("Cascading event processing failed")

    def _get_current_world_state(self):
        """Get current world state for dynamic event generation."""
        try:
            if not self.characters:
                return {"average_wealth": 50, "average_relationships": 50, "average_health": 75}
                
            # Calculate averages for world state analysis
            total_chars = len(self.characters)
            avg_wealth = sum(getattr(char, 'wealth_money', 50) for char in self.characters.values()) / total_chars
            avg_health = sum(getattr(char, 'health_status', 75) for char in self.characters.values()) / total_chars
            
            # Calculate average relationships if social networks exist
            avg_relationships = 50
            if hasattr(self, 'social_networks') and self.social_networks.get('relationships'):
                relationship_values = []
                for char_relationships in self.social_networks['relationships'].values():
                    relationship_values.extend(char_relationships.values())
                if relationship_values:
                    avg_relationships = sum(relationship_values) / len(relationship_values)
            
            return {
                "average_wealth": avg_wealth,
                "average_relationships": avg_relationships,
                "average_health": avg_health,
                "population": total_chars,
                "time": pygame.time.get_ticks() if 'pygame' in globals() else 0
            }
            
        except Exception as e:
            logger.warning(f"Error calculating world state: {e}")
            return {"average_wealth": 50, "average_relationships": 50, "average_health": 75}

    def _process_basic_events_fallback(self, update_errors):
        """
        Fallback event processing when EventHandler is not available.
        This is a much improved version of the old _process_pending_events.
        """
        try:
            if not hasattr(self, "events") or not self.events:
                return
                
            logger.info("Using fallback event processing (EventHandler not available)")
            events_to_remove = []
            
            for event in self.events:
                try:
                    # Basic event processing that actually drives strategy
                    logger.debug(f"Processing fallback event: {event}")
                    
                    # Try to trigger strategy update even for basic events
                    if self.strategy_manager:
                        try:
                            # Convert basic event to a format strategy manager can understand
                            event_for_strategy = {
                                'type': getattr(event, 'type', 'general'),
                                'name': getattr(event, 'name', str(event)),
                                'importance': getattr(event, 'importance', 5)
                            }
                            strategy_result = self.strategy_manager.update_strategy([event_for_strategy])
                            self._apply_strategy_result(strategy_result, update_errors)
                        except Exception as e:
                            logger.warning(f"Error applying strategy for basic event: {e}")
                    
                    events_to_remove.append(event)
                    
                except Exception as e:
                    logger.warning(f"Error processing basic event: {e}")
                    events_to_remove.append(event)  # Remove problematic events
            
            # Remove processed events
            for event in events_to_remove:
                if event in self.events:
                    self.events.remove(event)
                    
            if events_to_remove:
                logger.debug(f"Processed {len(events_to_remove)} basic events")
                    
        except Exception as e:
            logger.error(f"Error in fallback event processing: {e}")
            update_errors.append("Fallback event processing failed")

    def _process_pending_events(self):
        """
        DEPRECATED: This method has been replaced by _process_events_and_drive_strategy.
        Kept for backward compatibility but now delegates to the new robust implementation.
        """
        logger.warning("_process_pending_events is deprecated. Use _process_events_and_drive_strategy instead.")
        update_errors = []
        self._process_basic_events_fallback(update_errors)
        if update_errors:
            logger.warning(f"Deprecated _process_pending_events completed with errors: {update_errors}")

    def _process_events_and_update_strategy(self, dt):
        """
        DEPRECATED: This method has been replaced by _process_events_and_drive_strategy.
        Kept for backward compatibility but functionality is now integrated into update_game_state.
        """
        logger.warning("_process_events_and_update_strategy is deprecated. Event processing is now integrated into update_game_state.")
        # No operation - functionality moved to _process_events_and_drive_strategy

    def apply_decision(self, decision, game_state):
        """Apply a strategic decision to the game state."""
        try:
            if not decision:
                return

            # Apply the decision based on its type
            decision_type = decision.get("type", "unknown")
            
            if decision_type == "character_action":
                character_id = decision.get("character_id")
                action = decision.get("action")
                
                if character_id in self.characters and action:
                    character = self.characters[character_id]
                    resolved_action = self.action_resolver.resolve_action(action, character)
                    
                    if resolved_action and hasattr(resolved_action, 'execute'):
                        success = resolved_action.execute()
                        if success:
                            self.game_statistics["actions_executed"] += 1
                        else:
                            self.game_statistics["actions_failed"] += 1
                        
                        # Track the action execution for analytics
                        self.action_resolver.track_action_execution(resolved_action, character, success)
                        
            elif decision_type == "event_response":
                # Handle event-based decisions
                event_id = decision.get("event_id")
                response = decision.get("response")
                logger.info(f"Responding to event {event_id}: {response}")
                
            else:
                logger.warning(f"Unknown decision type: {decision_type}")
                
        except Exception as e:
            logger.error(f"Error applying decision: {e}")

    def render(self):
        """Render the game with modular UI system and error handling."""
        if not self.screen:
            return

        try:
            # Get background color from config
            bg_color = self.config.get("render", {}).get("background_color", [20, 50, 80])
            self.screen.fill(bg_color)

            # Render map controller (characters, buildings, etc.)
            if self.map_controller:
                try:
                    self.map_controller.render(self.screen)
                except Exception as e:
                    logger.warning(f"Error rendering map controller: {e}")

            # Render modular UI panels
            self._render_ui_panels()

            # Render feature status overlay if enabled
            if getattr(self, "_show_feature_status", False):
                self._render_feature_status_overlay()

            pygame.display.flip()

        except Exception as e:
            logger.error(f"Error during rendering: {e}")

    def _render_ui_panels(self):
        """Render all visible UI panels using the modular system."""
        try:
            current_y = 10
            
            for panel_name, panel in self.ui_panels.items():
                if panel.visible:
                    try:
                        # Update panel position for stacking
                        if panel_name == 'instructions':
                            # Position instructions at bottom
                            panel.position = (10, self.screen.get_height() - 150)
                        elif panel.position[1] is None:
                            panel.position = (panel.position[0], current_y)
                        
                        # Render panel and get height
                        panel_height = panel.render(self.screen, self, self.ui_fonts)
                        
                        # Update current_y for next panel (only for left-side panels)
                        if panel.position[0] <= 20:  # Left-side panels
                            current_y += panel_height + 10
                            
                    except Exception as e:
                        logger.warning(f"Error rendering UI panel {panel_name}: {e}")
                        
        except Exception as e:
            logger.error(f"Error rendering UI panels: {e}")

    def _render_feature_status_overlay(self):
        """Render feature status overlay."""
        try:
            font = self.ui_fonts.get('small', pygame.font.Font(None, 18))
            feature_status = self.get_feature_implementation_status()
            
            overlay_x = self.screen.get_width() - 300
            overlay_y = 50
            
            # Background for overlay
            overlay_rect = pygame.Rect(overlay_x - 10, overlay_y - 10, 290, len(feature_status) * 20 + 20)
            overlay_surface = pygame.Surface((overlay_rect.width, overlay_rect.height))
            overlay_surface.set_alpha(180)
            overlay_surface.fill((0, 0, 0))
            self.screen.blit(overlay_surface, overlay_rect)
            
            # Feature status text
            for i, (feature, status) in enumerate(feature_status.items()):
                color = {
                    "NOT_STARTED": (255, 100, 100),
                    "STUB_IMPLEMENTED": (255, 255, 100),  
                    "BASIC_IMPLEMENTED": (100, 255, 100),
                    "FULLY_IMPLEMENTED": (100, 255, 200)
                }.get(status, (255, 255, 255))
                
                text = font.render(f"{feature}: {status}", True, color)
                self.screen.blit(text, (overlay_x, overlay_y + i * 20))
                
        except Exception as e:
            logger.error(f"Error rendering feature status overlay: {e}")

    def run(self):
        """Main run method to start the game loop."""
        try:
            logger.info("Starting Tiny Village...")
            logger.info(f"Initialized with {len(self.characters)} characters")
            
            if self.initialization_errors:
                logger.warning(f"Started with {len(self.initialization_errors)} initialization errors")
            
            self.game_loop()
            
        except Exception as e:
            logger.error(f"Critical error in main run loop: {e}")
            logger.error(traceback.format_exc())
        finally:
            logger.info("Tiny Village shutting down...")

    def save_game_state(self, filepath: str) -> bool:
        """Save current game state to file."""
        try:
            import json
            import os
            
            # Create saves directory if it doesn't exist
            save_dir = os.path.dirname(filepath)
            if save_dir and not os.path.exists(save_dir):
                os.makedirs(save_dir)
            
            # Collect saveable game state
            game_state = {
                "timestamp": pygame.time.get_ticks(),
                "characters": {},
                "achievements": self.global_achievements,
                "statistics": self.game_statistics,
                "weather": getattr(self, "weather_system", {}),
                "quest_system": getattr(self, "quest_system", {}),
                "social_networks": getattr(self, "social_networks", {})
            }
            
            # Save character data
            for char_id, character in self.characters.items():
                try:
                    char_data = {
                        "name": getattr(character, "name", "Unknown"),
                        "energy": getattr(character, "energy", 50),
                        "health_status": getattr(character, "health_status", 75),
                        "position": {
                            "x": character.position.x if hasattr(character, "position") else 0,
                            "y": character.position.y if hasattr(character, "position") else 0
                        },
                        "job": getattr(character, "job", "Villager")
                    }
                    game_state["characters"][char_id] = char_data
                except Exception as e:
                    logger.warning(f"Error saving character {char_id}: {e}")
            
            # Write to file
            with open(filepath, 'w') as f:
                json.dump(game_state, f, indent=2)
            
            logger.info(f"Game state saved to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving game state: {e}")
            return False

    def load_game_state(self, filepath: str) -> bool:
        """Load game state from file."""
        try:
            import json
            import os
            
            if not os.path.exists(filepath):
                logger.warning(f"Save file not found: {filepath}")
                return False
            
            with open(filepath, 'r') as f:
                game_state = json.load(f)
            
            # Restore achievements
            if "achievements" in game_state:
                self.global_achievements.update(game_state["achievements"])
            
            # Restore statistics  
            if "statistics" in game_state:
                self.game_statistics.update(game_state["statistics"])
                
            # Restore weather
            if "weather" in game_state:
                self.weather_system = game_state["weather"]
                
            # Restore quest system
            if "quest_system" in game_state:
                self.quest_system = game_state["quest_system"]
                
            # Restore social networks
            if "social_networks" in game_state:
                self.social_networks = game_state["social_networks"]
            
            # Note: Character restoration is more complex and would require
            # full character recreation, which is beyond basic save/load
            
            logger.info(f"Game state loaded from {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading game state: {e}")
            return False

    def implement_achievement_system(self):
        """Implement basic achievement tracking system."""
        try:
            # Achievement checking logic
            current_time = pygame.time.get_ticks()
            
            # First character created achievement
            if len(self.characters) > 0:
                self.global_achievements["village_milestones"]["first_character_created"] = True
            
            # Five characters active achievement
            if len(self.characters) >= 5:
                self.global_achievements["village_milestones"]["five_characters_active"] = True
            
            # First week survived (based on game ticks - roughly 7 real minutes)
            if current_time > 420000:  # 7 minutes in milliseconds
                self.global_achievements["village_milestones"]["first_week_survived"] = True
                
        except Exception as e:
            logger.error(f"Error in achievement system: {e}")

    def implement_weather_system(self):
        """Implement basic weather simulation."""
        try:
            if not hasattr(self, "weather_system"):
                self.weather_system = {
                    "current_weather": "clear",
                    "temperature": 20,
                    "last_change": pygame.time.get_ticks()
                }
            
            current_time = pygame.time.get_ticks()
            
            # Change weather every 2 minutes (120,000 ms)
            if current_time - self.weather_system["last_change"] > 120000:
                weather_options = ["clear", "cloudy", "rainy"]
                self.weather_system["current_weather"] = random.choice(weather_options)
                self.weather_system["temperature"] = random.randint(10, 30)
                self.weather_system["last_change"] = current_time
                
        except Exception as e:
            logger.error(f"Error in weather system: {e}")

    def implement_social_network_system(self):
        """Implement basic social relationship tracking."""
        try:
            if not hasattr(self, "social_networks"):
                self.social_networks = {
                    "relationships": {},
                    "last_update": pygame.time.get_ticks()
                }
            
            # Initialize relationships for all characters
            for char_id in self.characters.keys():
                if char_id not in self.social_networks["relationships"]:
                    self.social_networks["relationships"][char_id] = {}
                    
                    # Create relationships with other characters
                    for other_id in self.characters.keys():
                        if other_id != char_id:
                            # Random initial relationship strength (30-70)
                            self.social_networks["relationships"][char_id][other_id] = random.randint(30, 70)
                            
        except Exception as e:
            logger.error(f"Error in social network system: {e}")

    def implement_quest_system(self):
        """Implement basic quest and goal system."""
        try:
            if not hasattr(self, "quest_system"):
                self.quest_system = {
                    "active_quests": {},
                    "completed_quests": {},
                    "quest_templates": [
                        {
                            "name": "Gather Resources", 
                            "description": "Collect materials for the village",
                            "type": "collection"
                        },
                        {
                            "name": "Social Interaction",
                            "description": "Talk to other villagers", 
                            "type": "social"
                        },
                        {
                            "name": "Skill Development",
                            "description": "Improve your abilities",
                            "type": "skill"
                        }
                    ]
                }
            
            # Initialize quest tracking for all characters
            for char_id in self.characters.keys():
                if char_id not in self.quest_system["active_quests"]:
                    self.quest_system["active_quests"][char_id] = []
                if char_id not in self.quest_system["completed_quests"]:
                    self.quest_system["completed_quests"][char_id] = []
                    
        except Exception as e:
            logger.error(f"Error in quest system: {e}")

    def initialize_world_events(self):
        """Initialize world events for emergent storytelling."""
        try:
            # Initialize basic event system
            if not hasattr(self, "world_events"):
                self.world_events = {
                    "event_queue": [],
                    "last_event_time": pygame.time.get_ticks(),
                    "event_templates": [
                        {"type": "weather_change", "description": "Weather patterns shift"},
                        {"type": "visitor_arrival", "description": "A stranger visits the village"},
                        {"type": "resource_discovery", "description": "New resources are discovered"},
                        {"type": "festival", "description": "The village celebrates a festival"}
                    ]
                }
                
        except Exception as e:
            logger.error(f"Error initializing world events: {e}")

    def add_event_notification(self, message: str, priority: str = "normal"):
        """Add an event notification to the UI."""
        try:
            if hasattr(self, 'ui_panels') and 'event_notifications' in self.ui_panels:
                self.ui_panels['event_notifications'].add_notification(message, priority)
                logger.info(f"Added event notification: {message} (priority: {priority})")
        except Exception as e:
            logger.error(f"Error adding event notification: {e}")
    
    def show_building_interaction(self, building, mouse_pos):
        """Show building interaction prompts."""
        try:
            if hasattr(self, 'ui_panels') and 'building_interaction' in self.ui_panels:
                self.ui_panels['building_interaction'].show_building_interaction(building, mouse_pos)
        except Exception as e:
            logger.error(f"Error showing building interaction: {e}")
    
    def handle_ui_click(self, position):
        """Handle clicks on UI elements."""
        try:
            # Check time control panel
            if hasattr(self, 'ui_panels') and 'time_controls' in self.ui_panels:
                time_panel = self.ui_panels['time_controls']
                if time_panel.visible and time_panel.handle_click(position, self):
                    return True
            
            # Check other clickable UI elements here
            return False
        except Exception as e:
            logger.error(f"Error handling UI click: {e}")
            return False
    
    def cycle_help_mode(self):
        """Cycle through help modes in the instructions panel."""
        try:
            if hasattr(self, 'ui_panels') and 'instructions' in self.ui_panels:
                self.ui_panels['instructions'].cycle_help_mode()
        except Exception as e:
            logger.error(f"Error cycling help mode: {e}")
    
    def provide_action_feedback(self, action_name: str, success: bool, character_name: str = None):
        """Provide visual feedback for actions taken."""
        try:
            if success:
                if character_name:
                    message = f"{character_name} completed: {action_name}"
                else:
                    message = f"Action completed: {action_name}"
                self.add_event_notification(message, "normal")
            else:
                if character_name:
                    message = f"{character_name} failed: {action_name}"
                else:
                    message = f"Action failed: {action_name}"
                self.add_event_notification(message, "medium")
        except Exception as e:
            logger.error(f"Error providing action feedback: {e}")
    
    def notify_major_event(self, event_name: str, description: str = None):
        """Notify about major village events."""
        try:
            if description:
                message = f"{event_name}: {description}"
            else:
                message = event_name
            self.add_event_notification(message, "high")
            logger.info(f"Major event notification: {message}")
        except Exception as e:
            logger.error(f"Error notifying major event: {e}")

    def get_feature_implementation_status(self) -> Dict[str, str]:
        """
        Report the implementation status of all planned features.
        
        Returns a dictionary with feature names and their implementation status:
        - NOT_STARTED: Feature not yet implemented
        - STUB_IMPLEMENTED: Basic structure in place, core functionality missing
        - BASIC_IMPLEMENTED: Core functionality working, needs enhancement
        - FULLY_IMPLEMENTED: Feature complete and polished
        """
        return {
            "save_load_system": "BASIC_IMPLEMENTED",
            "achievement_system": "BASIC_IMPLEMENTED",
            "weather_system": "STUB_IMPLEMENTED",
            "social_network_system": "STUB_IMPLEMENTED", 
            "quest_system": "STUB_IMPLEMENTED",
            "skill_progression": "BASIC_IMPLEMENTED",
            "reputation_system": "BASIC_IMPLEMENTED",
            "economic_simulation": "STUB_IMPLEMENTED",
            "event_driven_storytelling": "BASIC_IMPLEMENTED",
            "character_status_display": "FULLY_IMPLEMENTED",  # NEW: Enhanced character needs tracking
            "village_overview": "FULLY_IMPLEMENTED",  # NEW: Village-wide information panel
            "interaction_prompts": "BASIC_IMPLEMENTED",  # NEW: Building interaction prompts
            "feedback_system": "FULLY_IMPLEMENTED",  # NEW: Visual/textual feedback for actions
            "time_controls": "FULLY_IMPLEMENTED",  # NEW: UI buttons for time control
            "event_notifications": "FULLY_IMPLEMENTED",  # NEW: Event notification system
            "enhanced_help_system": "FULLY_IMPLEMENTED",  # NEW: Improved help with tutorial modes
            "mouse_interactions": "BASIC_IMPLEMENTED",  # NEW: Enhanced mouse handling with right-click
            "mod_system": "NOT_STARTED",
            "multiplayer_support": "NOT_STARTED",
            "advanced_ai_behaviors": "IMPLEMENTED",
            "procedural_content_generation": "NOT_STARTED",
            "advanced_graphics_effects": "NOT_STARTED",
            "sound_and_music_system": "NOT_STARTED",
            "accessibility_features": "NOT_STARTED",
            "performance_optimization": "NOT_STARTED",
            "automated_testing": "NOT_STARTED",
            "configuration_ui": "NOT_STARTED",
        }

    def get_current_stories(self) -> Dict[str, Any]:
        """Get current story state and narratives from the storytelling system."""
        if not self.storytelling_system:
            return {
                "error": "Storytelling system not available",
                "feature_status": "NOT_AVAILABLE"
            }
        
        try:
            return self.storytelling_system.get_current_stories()
        except Exception as e:
            logger.error(f"Error getting current stories: {e}")
            return {
                "error": str(e),
                "feature_status": "ERROR"
            }

    def get_story_summary(self, days_back: int = 7) -> str:
        """Get a summary of recent story developments."""
        if not self.storytelling_system:
            return "Storytelling system not available."
        
        try:
            return self.storytelling_system.generate_story_summary(days_back)
        except Exception as e:
            logger.error(f"Error generating story summary: {e}")
            return f"Error generating story summary: {e}"

    def get_character_stories(self, character_name: str) -> Dict[str, Any]:
        """Get a character's involvement in current stories."""
        if not self.storytelling_system:
            return {
                "error": "Storytelling system not available",
                "character": character_name
            }
        
        try:
            return self.storytelling_system.get_character_story_involvement(character_name)
        except Exception as e:
            logger.error(f"Error getting character stories for {character_name}: {e}")
            return {
                "error": str(e),
                "character": character_name
            }


if __name__ == "__main__":
    game_controller = GameplayController()
    game_controller.run()
    pygame.quit()
