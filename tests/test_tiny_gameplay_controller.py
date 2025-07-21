import unittest
import os
import pygame # Pygame will be needed for font rendering and potentially other UI elements.
from tiny_gameplay_controller import GameplayController, MIN_SPEED, MAX_SPEED, SPEED_STEP # Import constants too

from unittest.mock import MagicMock, patch # Ensure MagicMock and patch are imported

# Minimal mock for GraphManager if needed by GameplayController constructor
# class MockGraphManager: # Replaced by MagicMock
#     pass

class MockCharacter: # Simple mock for Character
    def __init__(self, name="Test Char"):
        self.name = name
        self.uuid = f"{name}_uuid"
        # Add any other attributes needed by the controller or actions during tests
        self.energy = 100
        # Make add_memory a proper mock method
        self.add_memory = MagicMock()


class TestGameplayController(unittest.TestCase):

    def setUp(self):
        pygame.init()
        try:
            pygame.display.set_mode((1, 1))
        except pygame.error as e:
            print(f"Skipping display.set_mode in test setup (possibly headless environment): {e}")

        self.mock_graph_manager = MagicMock() # Use MagicMock for GraphManager

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
                "minimap": [pygame.K_m],
                "overview": [pygame.K_o],
            }
        }
        self.controller = GameplayController(graph_manager=self.mock_graph_manager, config=self.config)

        # Mock ActionResolver and its methods
        self.mock_action_resolver = MagicMock()
        self.controller.action_resolver = self.mock_action_resolver # Inject mock

        # Mock Action and Character for relevant tests
        self.mock_action = MagicMock()
        self.mock_action.name = "TestAction"
        self.mock_character = MockCharacter() # Use our simple mock

        # Ensure a screen is available for rendering, even if it's a dummy one for tests.
        if not self.controller.screen: # Should be set by GameplayController now
            self.controller.screen = pygame.Surface((self.config["screen_width"], self.config["screen_height"]))


    def tearDown(self):
        pygame.quit()

    @patch('tiny_gameplay_controller.GameplayController._update_character_state_after_action') # Mock this method
    def test_execute_single_action_calls_action_execute(self, mock_update_state_after_action):
        # Configure the mock ActionResolver to return our mock_action
        self.mock_action_resolver.resolve_action.return_value = self.mock_action
        # Configure the mock_action's execute method to return True (successful execution)
        self.mock_action.execute = MagicMock(return_value=True)

        # Mock validate_action_preconditions to return True
        self.mock_action_resolver.validate_action_preconditions = MagicMock(return_value=True)
        # Mock predict_action_effects
        self.mock_action_resolver.predict_action_effects = MagicMock(return_value={})


        mock_action_data = {"name": "TestActionData"} # Dummy action data

        # Call the method under test
        result = self.controller._execute_single_action(self.mock_character, mock_action_data)

        self.assertTrue(result) # Action execution should be successful

        # Assert that action_resolver.resolve_action was called
        self.mock_action_resolver.resolve_action.assert_called_once_with(mock_action_data, self.mock_character)

        # Assert that the action's execute method was called
        # The action.execute in actions.py now takes character and graph_manager
        self.mock_action.execute.assert_called_once_with(target=self.mock_character, initiator=self.mock_character)

        # Assert that _update_character_state_after_action was called
        mock_update_state_after_action.assert_called_once_with(self.mock_character, self.mock_action)

    def test_update_character_state_no_redundant_graph_call(self):
        # We need a fresh mock for graph_manager to check its calls for this specific test
        specific_test_graph_manager = MagicMock()
        self.controller.graph_manager = specific_test_graph_manager # Override controller's GM

        # Mock other methods called by _update_character_state_after_action to isolate the test
        self.controller._update_character_skills = MagicMock()
        self.controller._update_social_consequences = MagicMock()
        self.controller._update_economic_state = MagicMock()
        self.controller._generate_action_events = MagicMock()
        self.controller._check_achievements = MagicMock()
        self.controller._update_reputation = MagicMock()
        self.controller._track_state_changes = MagicMock()
        # self.mock_character.add_memory is already a mock method from MockCharacter

        # Call the method under test
        self.controller._update_character_state_after_action(self.mock_character, self.mock_action)

        # Assert that the (now removed) graph_manager.update_character_state was NOT called.
        # If update_character_state was a method on the mock, we'd use assert_not_called().
        # Since it's not expected to exist, we can check that no such attribute was accessed *if* it wasn't a MagicMock.
        # With MagicMock, it auto-creates methods on access. So, the correct check is:
        specific_test_graph_manager.update_character_state.assert_not_called()

        # Verify other methods (controller-level logic) are still called
        self.mock_character.add_memory.assert_called() # Called if character has add_memory
        self.controller._update_character_skills.assert_called_once_with(self.mock_character, self.mock_action)
        # ... (add assertions for other helper methods if their call is mandatory)


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
    def test_minimap_toggle(self):
        """Test that mini-map mode can be toggled correctly."""
        # Initially mini-map should be disabled
        self.assertFalse(getattr(self.controller, "_minimap_mode", False))
        
        # Simulate 'M' key press to enable mini-map
        mock_event = MagicMock()
        mock_event.key = pygame.K_m
        
        self.controller._handle_keydown(mock_event)
        
        # Mini-map should now be enabled
        self.assertTrue(getattr(self.controller, "_minimap_mode", False))
        
        # Press 'M' again to disable mini-map
        self.controller._handle_keydown(mock_event)
        
        # Mini-map should now be disabled
        self.assertFalse(getattr(self.controller, "_minimap_mode", False))

    def test_overview_mode_toggle(self):
        """Test that overview mode can be toggled correctly."""
        # Initially overview mode should be disabled
        self.assertFalse(getattr(self.controller, "_overview_mode", False))
        
        # Simulate 'O' key press to enable overview mode
        mock_event = MagicMock()
        mock_event.key = pygame.K_o
        
        self.controller._handle_keydown(mock_event)
        
        # Overview mode should now be enabled
        self.assertTrue(getattr(self.controller, "_overview_mode", False))
        
        # Press 'O' again to disable overview mode
        self.controller._handle_keydown(mock_event)
        
        # Overview mode should now be disabled
        self.assertFalse(getattr(self.controller, "_overview_mode", False))

    def test_render_minimap_no_errors(self):
        """Test that mini-map rendering doesn't crash when enabled."""
        # Enable mini-map mode
        self.controller._minimap_mode = True
        
        # Import and create a real MapController for integration testing
        from tiny_map_controller import MapController
        
        # Create test map data
        test_map_data = {
            'width': 100,
            'height': 100,
            'buildings': [
                {'name': 'Test Building', 'rect': pygame.Rect(25, 25, 20, 20)}
            ]
        }
        
        # Create test map image
        test_map_image_path = 'assets/test_minimap.png'
        test_surface = pygame.Surface((100, 100))
        test_surface.fill((0, 128, 0))  # Green background
        pygame.image.save(test_surface, test_map_image_path)
        
        # Use real MapController instead of mock for integration testing
        real_map_controller = MapController(test_map_image_path, test_map_data)
        self.controller.map_controller = real_map_controller
        
        # Should not raise any exceptions
        try:
            self.controller._render_minimap()
        except Exception as e:
            self.fail(f"Mini-map rendering failed with error: {e}")
        
        # Clean up test image
        import os
        if os.path.exists(test_map_image_path):
            os.remove(test_map_image_path)

    def test_render_overview_no_errors(self):
        """Test that overview mode rendering doesn't crash when enabled."""
        # Enable overview mode
        self.controller._overview_mode = True
        
        # Import and create a real MapController for integration testing
        from tiny_map_controller import MapController
        
        # Create test map data with more comprehensive data for overview testing
        test_map_data = {
            'width': 100,
            'height': 100,
            'buildings': [
                {'name': 'Town Hall', 'rect': pygame.Rect(30, 30, 25, 25)},
                {'name': 'Market', 'rect': pygame.Rect(60, 60, 15, 15)}
            ]
        }
        
        # Create test map image
        test_map_image_path = 'assets/test_overview.png'
        test_surface = pygame.Surface((100, 100))
        test_surface.fill((0, 100, 200))  # Blue background
        pygame.image.save(test_surface, test_map_image_path)
        
        # Use real MapController instead of mock for integration testing
        real_map_controller = MapController(test_map_image_path, test_map_data)
        self.controller.map_controller = real_map_controller
        
        # Should not raise any exceptions
        try:
            self.controller._render_overview()
        except Exception as e:
            self.fail(f"Overview rendering failed with error: {e}")
        
        # Clean up test image
        import os
        if os.path.exists(test_map_image_path):
            os.remove(test_map_image_path)

    def test_map_controller_integration(self):
        """Test integration with real MapController to ensure actual functionality works."""
        # Import and create a real MapController for integration testing
        from tiny_map_controller import MapController
        
        # Create comprehensive test map data
        test_map_data = {
            'width': 50,
            'height': 50,
            'buildings': [
                {'name': 'Building A', 'rect': pygame.Rect(10, 10, 5, 5)},
                {'name': 'Building B', 'rect': pygame.Rect(30, 30, 8, 8)}
            ]
        }
        
        # Create test map image
        test_map_image_path = 'assets/test_integration.png'
        test_surface = pygame.Surface((50, 50))
        test_surface.fill((255, 255, 0))  # Yellow background
        pygame.image.save(test_surface, test_map_image_path)
        
        # Use real MapController for comprehensive integration testing
        real_map_controller = MapController(test_map_image_path, test_map_data)
        
        # Test that real MapController has proper functionality
        self.assertIsNotNone(real_map_controller.map_image, "Map image should be loaded")
        self.assertEqual(real_map_controller.map_data['width'], 50, "Map width should be set correctly")
        self.assertEqual(len(real_map_controller.map_data['buildings']), 2, "Should have 2 buildings")
        self.assertIsInstance(real_map_controller.characters, dict, "Characters should be a dictionary")
        self.assertIsNotNone(real_map_controller.pathfinder, "Pathfinder should be initialized")
        
        # Test pathfinding functionality (this would fail with a mock)
        start_pos = (0, 0)
        goal_pos = (5, 5)
        path = real_map_controller.find_path_cached(start_pos, goal_pos)
        self.assertIsInstance(path, list, "Path should be a list")
        
        # Test dynamic obstacle management (unique to real implementation)
        obstacle_pos = (20, 20)
        real_map_controller.add_dynamic_obstacle(obstacle_pos)
        self.assertIn(obstacle_pos, real_map_controller.dynamic_obstacles, "Obstacle should be added")
        
        real_map_controller.remove_dynamic_obstacle(obstacle_pos)
        self.assertNotIn(obstacle_pos, real_map_controller.dynamic_obstacles, "Obstacle should be removed")
        
        # Test with GameplayController
        self.controller.map_controller = real_map_controller
        
        # Test that minimap and overview work with real data
        self.controller._minimap_mode = True
        try:
            self.controller._render_minimap()
        except Exception as e:
            self.fail(f"Minimap rendering with real MapController failed: {e}")
        
        self.controller._overview_mode = True
        try:
            self.controller._render_overview()
        except Exception as e:
            self.fail(f"Overview rendering with real MapController failed: {e}")
        
        # Clean up test image
        import os
        if os.path.exists(test_map_image_path):
            os.remove(test_map_image_path)

    def test_map_controller_bug_detection(self):
        """Test that demonstrates real MapController bugs would be caught (unlike with mocks)."""
        # Import and create a real MapController for integration testing
        from tiny_map_controller import MapController
        
        # Create test map data that would expose pathfinding issues
        test_map_data = {
            'width': 10,
            'height': 10,
            'buildings': [
                {'name': 'Obstacle', 'rect': pygame.Rect(3, 3, 4, 4)},  # Large obstacle
            ]
        }
        
        # Create test map image
        test_map_image_path = 'assets/test_bug_detection.png'
        test_surface = pygame.Surface((10, 10))
        test_surface.fill((128, 64, 0))  # Brown background
        pygame.image.save(test_surface, test_map_image_path)
        
        # Use real MapController - this will catch actual implementation bugs
        real_map_controller = MapController(test_map_image_path, test_map_data)
        
        # Test pathfinding around obstacles (a mock wouldn't catch pathfinding bugs)
        start_pos = (0, 0)
        goal_pos = (9, 9)
        path = real_map_controller.find_path_cached(start_pos, goal_pos)
        
        # Verify path is valid and avoids obstacles
        self.assertIsInstance(path, list, "Path should be a list")
        if path:  # If a path exists
            # Verify path doesn't go through obstacles
            for step in path:
                x, y = step
                for building in test_map_data['buildings']:
                    rect = building['rect']
                    self.assertFalse(rect.collidepoint(x, y), 
                                   f"Path step {step} should not be inside obstacle {building['name']}")
            
            # Verify path starts and ends correctly
            self.assertEqual(path[0], start_pos, "Path should start at start position")
            self.assertEqual(path[-1], goal_pos, "Path should end at goal position")
        
        # Test dynamic obstacle functionality (mocks wouldn't test this)
        obstacle_pos = (8, 8)
        real_map_controller.add_dynamic_obstacle(obstacle_pos)
        
        # Pathfinding should now avoid the dynamic obstacle
        new_path = real_map_controller.find_path_cached(start_pos, goal_pos)
        if new_path:
            self.assertNotIn(obstacle_pos, new_path, "Path should avoid dynamic obstacles")
        
        # Test cache invalidation (complex behavior that mocks wouldn't catch)
        initial_cache_size = len(real_map_controller.path_cache)
        real_map_controller.invalidate_path_cache()
        post_invalidation_cache_size = len(real_map_controller.path_cache)
        self.assertEqual(post_invalidation_cache_size, 0, "Cache should be empty after invalidation")
        
        # Clean up test image
        import os
        if os.path.exists(test_map_image_path):
            os.remove(test_map_image_path)


if __name__ == '__main__':
    unittest.main()
