class Location:
    def __init__(self, x, y, width, height):
        self.x = x
        self.y = y
        self.width = width
        self.height = height

    def contains_point(self, point_x, point_y):
        return (self.x <= point_x < self.x + self.width) and (
            self.y <= point_y < self.y + self.height
        )

    def overlaps(self, other):
        return (
            self.x < other.x + other.width
            and self.x + self.width > other.x
            and self.y < other.y + other.height
            and self.y + self.height > other.y
        )

    def move(self, delta_x, delta_y):
        self.x += delta_x
        self.y += delta_y

    def resize(self, new_width, new_height):
        self.width = new_width
        self.height = new_height

    def __repr__(self):
        return f"Location(x={self.x}, y={self.y}, width={self.width}, height={self.height})"


class LocationManager:
    def __init__(self):
        self.locations = []

    def add_location(self, location):
        self.locations.append(location)

    def find_locations_containing_point(self, point_x, point_y):
        return [loc for loc in self.locations if loc.contains_point(point_x, point_y)]

    def find_overlapping_locations(self, location):
        return [loc for loc in self.locations if loc.overlaps(location)]

    def move_location(self, location, delta_x, delta_y):
        if location in self.locations:
            location.move(delta_x, delta_y)

    def resize_location(self, location, new_width, new_height):
        if location in self.locations:
            location.resize(new_width, new_height)
