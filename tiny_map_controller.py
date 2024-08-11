# tiny_map_controller.py
import pygame
import heapq


class MapController:
    def __init__(self, map_image_path, map_data):
        self.map_image = pygame.image.load(map_image_path)  # Load map image
        self.map_data = map_data  # Metadata about map features
        self.characters = {}  # Dictionary of characters currently on the map
        self.selected_character = None  # For user interactions
        self.pathfinder = AStarPathfinder(self.map_data)  # Pathfinding system

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
