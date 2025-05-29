# tiny_map_controller.py
import pygame
import heapq
import time
import logging
from typing import Dict, List, Tuple, Optional, Set
from functools import lru_cache


class MapController:
    def __init__(self, map_image_path, map_data):
        self.map_image = pygame.image.load(map_image_path)  # Load map image
        self.map_data = map_data  # Metadata about map features
        self.characters = {}  # Dictionary of characters currently on the map
        self.selected_character = None  # For user interactions
        self.pathfinder = EnhancedAStarPathfinder(self.map_data)  # Enhanced pathfinding system
        self.dynamic_obstacles = set()  # Dynamic obstacles that can change
        self.obstacle_update_time = 0  # Track when obstacles were last updated
        self.path_cache = {}  # Cache for computed paths
        self.cache_timeout = 5.0  # Cache timeout in seconds

    def add_dynamic_obstacle(self, position: Tuple[int, int]):
        """Add a dynamic obstacle that can be updated in real-time"""
        self.dynamic_obstacles.add(position)
        self.pathfinder.add_dynamic_obstacle(position)
        self.invalidate_path_cache()
        self.obstacle_update_time = time.time()

    def remove_dynamic_obstacle(self, position: Tuple[int, int]):
        """Remove a dynamic obstacle"""
        self.dynamic_obstacles.discard(position)
        self.pathfinder.remove_dynamic_obstacle(position)
        self.invalidate_path_cache()
        self.obstacle_update_time = time.time()

    def invalidate_path_cache(self):
        """Clear the path cache when obstacles change"""
        self.path_cache.clear()

    def find_path_cached(self, start: Tuple[int, int], goal: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Find path with caching for better performance"""
        cache_key = (start, goal)
        current_time = time.time()

        # Check if we have a valid cached path
        if cache_key in self.path_cache:
            cached_path, cache_time = self.path_cache[cache_key]
            if current_time - cache_time < self.cache_timeout and cache_time > self.obstacle_update_time:
                return cached_path

        # Compute new path
        path = self.pathfinder.find_path(start, goal)
        self.path_cache[cache_key] = (path, current_time)

        return path

    def render(self, surface):
        # Render the map image
        surface.blit(self.map_image, (0, 0))

        # Render buildings on the map
        for building in self.map_data["buildings"]:
            pygame.draw.rect(surface, (150, 150, 150), building["rect"])

        # Render characters on the map
        for character in self.characters.values():
            pygame.draw.circle(surface, character.color, character.position, 5)

        # Render selected character indicator
        if self.selected_character:
            pygame.draw.circle(
                surface, (255, 0, 0), self.selected_character.position, 10, 2
            )

    def update(self, dt):
        # Update each character's position on the map
        for char_id, character in self.characters.items():
            self.update_character_position(char_id, dt)

    def handle_event(self, event):
        # Handle events like mouse clicks or key presses
        if event.type == pygame.MOUSEBUTTONDOWN:
            self.handle_click(event.pos)

    def handle_click(self, position):
        # Determine what is at the clicked position
        char_id = self.is_character(position)
        if char_id:
            self.select_character(char_id)
        elif self.is_building(position):
            self.enter_building(position)

    def update_character_position(self, character_id, dt):
        # Update character positions based on pathfinding
        character = self.characters[character_id]
        if character.path:
            next_node = character.path[0]
            direction = pygame.math.Vector2(next_node) - character.position
            if direction.length() < character.speed * dt:
                character.position = next_node
                character.path.pop(0)
            else:
                character.position += direction.normalize() * character.speed * dt

    def is_character(self, position):
        # Check if a character is at the clicked position
        for char_id, char_info in self.characters.items():
            if char_info.position.distance_to(pygame.math.Vector2(position)) < 10:
                return char_id
        return None

    def select_character(self, char_id):
        # Select a character when clicked
        self.selected_character = self.characters[char_id]
        print(f"Selected {self.selected_character.name}")

    def is_building(self, position):
        # Check if a building is at the clicked position
        for building in self.map_data["buildings"]:
            if building["rect"].collidepoint(position):
                return building
        return None

    def enter_building(self, position):
        # Enter a building and interact with it
        building = self.is_building(position)
        if building:
            print(f"Entering {building['name']}")


class AStarPathfinder:
    def __init__(self, map_data):
        self.map_data = map_data
        self.grid = self.create_grid(map_data)

    def create_grid(self, map_data):
        # Create a grid for pathfinding based on map data
        grid = []
        for y in range(map_data["height"]):
            row = []
            for x in range(map_data["width"]):
                row.append(0)  # 0 for walkable, 1 for non-walkable
            grid.append(row)

        for building in map_data["buildings"]:
            for y in range(building["rect"].top, building["rect"].bottom):
                for x in range(building["rect"].left, building["rect"].right):
                    grid[y][x] = 1

        return grid

    def find_path(self, start, goal):
        # Implement A* algorithm to find path from start to goal
        open_set = []
        heapq.heappush(open_set, (0, start))
        came_from = {}
        g_score = {start: 0}
        f_score = {start: self.heuristic(start, goal)}

        while open_set:
            _, current = heapq.heappop(open_set)

            if current == goal:
                return self.reconstruct_path(came_from, current)

            for neighbor in self.get_neighbors(current):
                tentative_g_score = g_score[current] + 1
                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = g_score[neighbor] + self.heuristic(
                        neighbor, goal
                    )
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))

        return []

    def heuristic(self, a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def get_neighbors(self, node):
        neighbors = [
            (node[0] + 1, node[1]),
            (node[0] - 1, node[1]),
            (node[0], node[1] + 1),
            (node[0], node[1] - 1),
        ]
        return [n for n in neighbors if self.is_walkable(n)]

    def is_walkable(self, node):
        x, y = node
        return (
            0 <= x < len(self.grid[0])
            and 0 <= y < len(self.grid)
            and self.grid[y][x] == 0
        )

    def reconstruct_path(self, came_from, current):
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        path.reverse()
        return path


class EnhancedAStarPathfinder:
    def __init__(self, map_data):
        self.map_data = map_data
        self.grid = self.create_grid(map_data)
        self.dynamic_obstacles = set()
        self.movement_costs = {}  # Different terrain movement costs
        self.path_smoothing = True  # Enable path smoothing

    def create_grid(self, map_data):
        """Create a grid for pathfinding with terrain costs"""
        grid = []
        for y in range(map_data["height"]):
            row = []
            for x in range(map_data["width"]):
                # Default terrain cost (0 = walkable, higher numbers = more expensive)
                terrain_cost = map_data.get("terrain", {}).get((x, y), 1)
                row.append(terrain_cost)
            grid.append(row)

        # Mark buildings as non-walkable
        for building in map_data["buildings"]:
            for y in range(building["rect"].top, building["rect"].bottom):
                for x in range(building["rect"].left, building["rect"].right):
                    if 0 <= y < len(grid) and 0 <= x < len(grid[0]):
                        grid[y][x] = float('inf')  # Impassable

        return grid

    def add_dynamic_obstacle(self, position: Tuple[int, int]):
        """Add a dynamic obstacle"""
        self.dynamic_obstacles.add(position)

    def remove_dynamic_obstacle(self, position: Tuple[int, int]):
        """Remove a dynamic obstacle"""
        self.dynamic_obstacles.discard(position)

    def find_path(self, start: Tuple[int, int], goal: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Enhanced A* algorithm with dynamic obstacles and jump point search optimization"""
        if not self.is_walkable(start) or not self.is_walkable(goal):
            logging.warning(f"Invalid start {start} or goal {goal} for pathfinding")
            return []

        # Use Jump Point Search for better performance on open areas
        if self.should_use_jps(start, goal):
            return self.jump_point_search(start, goal)
        else:
            return self.standard_astar(start, goal)

    def should_use_jps(self, start: Tuple[int, int], goal: Tuple[int, int]) -> bool:
        """Determine if Jump Point Search should be used"""
        # Use JPS for longer distances with fewer obstacles
        distance = abs(start[0] - goal[0]) + abs(start[1] - goal[1])
        obstacle_density = len(self.dynamic_obstacles) / (self.map_data["width"] * self.map_data["height"])
        return distance > 20 and obstacle_density < 0.1

    def jump_point_search(self, start: Tuple[int, int], goal: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Jump Point Search implementation for faster pathfinding"""
        open_set = []
        heapq.heappush(open_set, (0, start))
        came_from = {}
        g_score = {start: 0}
        f_score = {start: self.heuristic(start, goal)}

        directions = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

        while open_set:
            _, current = heapq.heappop(open_set)

            if current == goal:
                path = self.reconstruct_path(came_from, current)
                return self.smooth_path(path) if self.path_smoothing else path

            # Find jump points in all directions
            for dx, dy in directions:
                jump_point = self.jump(current, (dx, dy), goal)
                if jump_point:
                    new_cost = g_score[current] + self.heuristic(current, jump_point)

                    if jump_point not in g_score or new_cost < g_score[jump_point]:
                        came_from[jump_point] = current
                        g_score[jump_point] = new_cost
                        f_score[jump_point] = new_cost + self.heuristic(jump_point, goal)
                        heapq.heappush(open_set, (f_score[jump_point], jump_point))

        return []

    def jump(self, current: Tuple[int, int], direction: Tuple[int, int], goal: Tuple[int, int]) -> Optional[Tuple[int, int]]:
        """Jump function for Jump Point Search"""
        x, y = current
        dx, dy = direction

        next_x, next_y = x + dx, y + dy

        if not self.is_walkable((next_x, next_y)):
            return None

        if (next_x, next_y) == goal:
            return (next_x, next_y)

        # Check for forced neighbors (indicating a jump point)
        if self.has_forced_neighbors((next_x, next_y), direction):
            return (next_x, next_y)

        # Diagonal movement
        if dx != 0 and dy != 0:
            # Check horizontal and vertical jumps
            if self.jump((next_x, next_y), (dx, 0), goal) or self.jump((next_x, next_y), (0, dy), goal):
                return (next_x, next_y)

        # Continue jumping in the same direction
        return self.jump((next_x, next_y), direction, goal)

    def has_forced_neighbors(self, position: Tuple[int, int], direction: Tuple[int, int]) -> bool:
        """Check if a position has forced neighbors (jump point condition)"""
        x, y = position
        dx, dy = direction

        if dx == 0 or dy == 0:  # Straight movement
            if dx == 0:  # Vertical movement
                return (not self.is_walkable((x - 1, y - dy)) and self.is_walkable((x - 1, y))) or \
                       (not self.is_walkable((x + 1, y - dy)) and self.is_walkable((x + 1, y)))
            else:  # Horizontal movement
                return (not self.is_walkable((x - dx, y - 1)) and self.is_walkable((x, y - 1))) or \
                       (not self.is_walkable((x - dx, y + 1)) and self.is_walkable((x, y + 1)))
        else:  # Diagonal movement
            return (not self.is_walkable((x - dx, y)) and self.is_walkable((x - dx, y + dy))) or \
                   (not self.is_walkable((x, y - dy)) and self.is_walkable((x + dx, y - dy)))

    def standard_astar(self, start: Tuple[int, int], goal: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Standard A* algorithm with terrain costs"""
        open_set = []
        heapq.heappush(open_set, (0, start))
        came_from = {}
        g_score = {start: 0}
        f_score = {start: self.heuristic(start, goal)}

        while open_set:
            _, current = heapq.heappop(open_set)

            if current == goal:
                path = self.reconstruct_path(came_from, current)
                return self.smooth_path(path) if self.path_smoothing else path

            for neighbor in self.get_neighbors(current):
                # Calculate movement cost including terrain
                movement_cost = self.get_movement_cost(current, neighbor)
                tentative_g_score = g_score[current] + movement_cost

                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = g_score[neighbor] + self.heuristic(neighbor, goal)
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))

        return []

    def get_movement_cost(self, from_pos: Tuple[int, int], to_pos: Tuple[int, int]) -> float:
        """Calculate movement cost between two positions"""
        base_cost = 1.0

        # Diagonal movement costs more
        if abs(from_pos[0] - to_pos[0]) + abs(from_pos[1] - to_pos[1]) == 2:
            base_cost = 1.414  # sqrt(2)

        # Add terrain cost
        x, y = to_pos
        if 0 <= y < len(self.grid) and 0 <= x < len(self.grid[0]):
            terrain_cost = self.grid[y][x]
            if terrain_cost == float('inf'):
                return float('inf')
            base_cost *= terrain_cost

        return base_cost

    def smooth_path(self, path: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """Smooth the path by removing unnecessary waypoints"""
        if len(path) <= 2:
            return path

        smoothed = [path[0]]
        i = 0

        while i < len(path) - 1:
            j = len(path) - 1

            # Find the farthest visible point
            while j > i + 1:
                if self.line_of_sight(path[i], path[j]):
                    break
                j -= 1

            smoothed.append(path[j])
            i = j

        return smoothed

    def line_of_sight(self, start: Tuple[int, int], end: Tuple[int, int]) -> bool:
        """Check if there's a clear line of sight between two points"""
        x0, y0 = start
        x1, y1 = end

        # Bresenham's line algorithm
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)

        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1

        err = dx - dy

        while True:
            if not self.is_walkable((x0, y0)):
                return False

            if x0 == x1 and y0 == y1:
                break

            e2 = 2 * err

            if e2 > -dy:
                err -= dy
                x0 += sx

            if e2 < dx:
                err += dx
                y0 += sy

        return True

    def heuristic(self, a: Tuple[int, int], b: Tuple[int, int]) -> float:
        """Improved heuristic function (octile distance for 8-directional movement)"""
        dx = abs(a[0] - b[0])
        dy = abs(a[1] - b[1])
        return max(dx, dy) + (1.414 - 1) * min(dx, dy)

    def get_neighbors(self, node: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Get 8-directional neighbors"""
        x, y = node
        neighbors = []

        # 8-directional movement
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue

                new_x, new_y = x + dx, y + dy
                if self.is_walkable((new_x, new_y)):
                    # Check for corner cutting in diagonal movement
                    if dx != 0 and dy != 0:
                        if not self.is_walkable((x + dx, y)) or not self.is_walkable((x, y + dy)):
                            continue
                    neighbors.append((new_x, new_y))

        return neighbors

    def is_walkable(self, node: Tuple[int, int]) -> bool:
        """Check if a node is walkable considering dynamic obstacles"""
        x, y = node

        # Check bounds
        if not (0 <= x < len(self.grid[0]) and 0 <= y < len(self.grid)):
            return False

        # Check static obstacles
        if self.grid[y][x] == float('inf'):
            return False

        # Check dynamic obstacles
        if (x, y) in self.dynamic_obstacles:
            return False

        return True

    def reconstruct_path(self, came_from: Dict, current: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Reconstruct path from came_from dictionary"""
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        path.reverse()
        return path


# Example usage
if __name__ == "__main__":
    pygame.init()
    screen = pygame.display.set_mode((800, 600))
    clock = pygame.time.Clock()
    running = True

    # Initialize the Map Controller
    map_controller = MapController(
        "path_to_map_image.png",
        map_data={
            "width": 100,
            "height": 100,
            "buildings": [
                {"name": "Town Hall", "rect": pygame.Rect(100, 150, 50, 50)},
                # Add more buildings with their positions and sizes
            ],
        },
    )

    # Main game loop
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            else:
                map_controller.handle_event(event)

        map_controller.update(clock.tick() / 1000.0)
        map_controller.render(screen)
        pygame.display.flip()

    pygame.quit()
