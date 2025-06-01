import unittest
import pygame # Pygame will be needed for font rendering and potentially other UI elements.
from tiny_gameplay_controller import GameplayController, MIN_SPEED, MAX_SPEED, SPEED_STEP # Import constants too

# Minimal mock for GraphManager if needed by GameplayController constructor
class MockGraphManager:
    pass

class TestGameplayController(unittest.TestCase):

    def setUp(self):
        # Basic Pygame setup needed for font rendering.
        # This avoids full game initialization if possible.
        pygame.init()
        # It's good practice to set a display mode, even if it's small and not shown,
        # as some Pygame modules (like font) might depend on it.
        try:
            pygame.display.set_mode((1, 1))
        except pygame.error as e:
            print(f"Skipping display.set_mode in test setup (possibly headless environment): {e}")


        # Mock GraphManager if GameplayController requires it.
        # Adjust based on actual GameplayController constructor requirements.
        self.mock_graph_manager = MockGraphManager()
        # Provide a minimal config if necessary
        self.config = {
            "screen_width": 800,
            "screen_height": 600,
            "map": {
                "image_path": "assets/default_map.png", # Dummy path, map won't be loaded
                "width": 100,
                "height": 100,
                "buildings_file": None
            },
             "characters": {
                "count": 0 # Avoid character creation for these tests
            },
            "key_bindings": { # Add if controller accesses this during init for speed keys
                "increase_speed": [pygame.K_PAGEUP],
                "decrease_speed": [pygame.K_PAGEDOWN],
            }
        }
        self.controller = GameplayController(graph_manager=self.mock_graph_manager, config=self.config)
        # Ensure a screen is available for rendering, even if it's a dummy one for tests.
        if not self.controller.screen:
            self.controller.screen = pygame.Surface((self.config["screen_width"], self.config["screen_height"]))


    def tearDown(self):
        pygame.quit()

    def test_global_achievements_initialization(self):
        """Test that global_achievements is initialized correctly."""
        self.assertIsNotNone(self.controller.global_achievements)
        self.assertIsInstance(self.controller.global_achievements, dict)
        self.assertIn("village_milestones", self.controller.global_achievements)
        self.assertIn("social_achievements", self.controller.global_achievements)
        self.assertIn("economic_achievements", self.controller.global_achievements)
        self.assertIsInstance(self.controller.global_achievements["village_milestones"], dict)

    def test_speed_text_caching(self):
        """Test the caching mechanism for the speed text UI element."""
        # Initial render
        self.controller._render_ui() # Call private method for testing specific UI part
        initial_cached_surface = self.controller._cached_speed_text
        self.assertIsNotNone(initial_cached_surface, "Speed text should be cached on first render.")

        # Call render_ui again without changing time_scale_factor
        self.controller._render_ui()
        second_cached_surface = self.controller._cached_speed_text
        self.assertIs(initial_cached_surface, second_cached_surface,
                        "Cached surface should be the same object if time_scale_factor is unchanged.")

        # Change time_scale_factor (ensure it's a different value)
        original_speed = self.controller.time_scale_factor
        new_speed = original_speed + SPEED_STEP
        if new_speed > MAX_SPEED: # ensure new_speed is valid and different
            new_speed = original_speed - SPEED_STEP
            if new_speed < MIN_SPEED: # if original was MAX_SPEED
                 new_speed = MIN_SPEED if MAX_SPEED == MIN_SPEED else (MIN_SPEED + MAX_SPEED) / 2


        self.controller.time_scale_factor = new_speed
        # Manually invalidate cache as _handle_keydown would do, or rely on _render_ui's check
        # In this specific test, we want to check the _render_ui internal caching logic,
        # so we don't invalidate it here. The next test checks _handle_keydown's invalidation.
        # self.controller._cached_speed_text = None

        self.controller._render_ui()
        third_cached_surface = self.controller._cached_speed_text
        self.assertIsNotNone(third_cached_surface, "Speed text should be re-rendered and cached.")
        self.assertIsNot(initial_cached_surface, third_cached_surface,
                         "Cached surface should be a new object if time_scale_factor changed.")

        # Test that setting the speed to the same value (after it was changed) still uses cache if not invalidated
        # This requires _last_time_scale_factor to be correctly updated
        self.controller.time_scale_factor = new_speed # Set to same new_speed
        self.controller._render_ui() # Render with new_speed (should use cache from previous render at new_speed)
        fourth_cached_surface = self.controller._cached_speed_text
        self.assertIs(third_cached_surface, fourth_cached_surface,
                        "Cached surface should remain the same if time_scale_factor is set to the same value it was just changed to.")


    def test_speed_text_cache_invalidation_via_handle_keydown(self):
        """Test that cache is invalidated when speed changes via _handle_keydown."""
        # Initial render and cache
        self.controller._render_ui()
        initial_cached_surface = self.controller._cached_speed_text
        self.assertIsNotNone(initial_cached_surface)

        # Simulate key press to increase speed
        # Need to find the actual key, not just the string 'increase_speed'
        increase_key = self.controller.config.get("key_bindings", {}).get("increase_speed", [pygame.K_PAGEUP])[0]
        mock_event_increase = pygame.event.Event(pygame.KEYDOWN, key=increase_key)

        original_speed = self.controller.time_scale_factor
        self.controller._handle_keydown(mock_event_increase) # This should invalidate the cache (_cached_speed_text = None)

        # Speed must have changed for cache to be different
        self.assertNotEqual(original_speed, self.controller.time_scale_factor, "Speed should change on key press.")

        self.controller._render_ui() # This should re-render due to invalidated cache
        new_cached_surface_increase = self.controller._cached_speed_text
        self.assertIsNotNone(new_cached_surface_increase)
        self.assertIsNot(initial_cached_surface, new_cached_surface_increase,
                         "Cache should be different after speed increase.")

        # Simulate key press to decrease speed
        decrease_key = self.controller.config.get("key_bindings", {}).get("decrease_speed", [pygame.K_PAGEDOWN])[0]
        mock_event_decrease = pygame.event.Event(pygame.KEYDOWN, key=decrease_key)

        current_speed_before_decrease = self.controller.time_scale_factor
        self.controller._handle_keydown(mock_event_decrease)

        self.assertNotEqual(current_speed_before_decrease, self.controller.time_scale_factor, "Speed should change on key press for decrease.")

        self.controller._render_ui()
        new_cached_surface_decrease = self.controller._cached_speed_text
        self.assertIsNotNone(new_cached_surface_decrease)
        self.assertIsNot(new_cached_surface_increase, new_cached_surface_decrease,
                         "Cache should be different after speed decrease.")

if __name__ == '__main__':
    unittest.main()
