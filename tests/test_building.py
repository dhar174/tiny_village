import unittest
from tiny_buildings import Building, House

class TestBuilding(unittest.TestCase):
    def setUp(self):
        self.building = House("Test House", 100, 50, 30, "123 Test St", 2, 3, 2, 20, 0)
        print(f"Name: {self.building.name} \n Height: {self.building.height} \n Width: {self.building.width} \n Length: {self.building.length} \n Address: {self.building.address} \n Stories: {self.building.stories} \n Bedrooms: {self.building.bedrooms} \n Bathrooms: {self.building.bathrooms} \n Initial Beauty Value: {self.building.beauty_value} \n Price Value: {self.building.price}")

    def test_init(self):
        self.assertEqual(self.building.name, "Test House")
        self.assertEqual(self.building.height, 100)
        self.assertEqual(self.building.width, 50)
        self.assertEqual(self.building.length, 30)
        self.assertEqual(self.building.address, "123 Test St")
        self.assertEqual(self.building.stories, 2)
        self.assertEqual(self.building.bedrooms, 3)
        self.assertEqual(self.building.bathrooms, 2)
        self.assertEqual(self.building.area_val, 3000)

    def test_calculate_shelter_value(self):
        expected_shelter_value = 1
        expected_shelter_value += min(round(self.building.area_val / 1000), 5)
        if self.building.bedrooms > 1:
            expected_shelter_value += 1
        if self.building.bathrooms > 1:
            expected_shelter_value += 1
        if self.building.stories > 1:
            expected_shelter_value += 1
        if self.building.bedrooms > 2:
            expected_shelter_value += 1
        if self.building.bathrooms > 2:
            expected_shelter_value += 1
        if self.building.stories > 2:
            expected_shelter_value += 1

        self.assertEqual(self.building.shelter_value, expected_shelter_value)

    def test_set_beauty_value(self):
        # Assuming the set_beauty_value method is based on some properties of the building
        # Replace with the actual calculation logic
        expected_beauty_value = 20  # This is the initial_beauty_value passed in the __init__ method
        self.assertEqual(self.building.beauty_value, expected_beauty_value)

    def test_calculate_price(self):
        # Assuming the calculate_price method is based on some properties of the building
        # Replace with the actual calculation logic
        expected_price = 10 * (((8 * 20)/100) * 3000)
        self.assertEqual(self.building.price, expected_price)


class TestBuilding(unittest.TestCase):
    def setUp(self):
        self.building = House("Test Mansion", 100, 100, 100, "123 Mansion St", 4, 8, 8, 20, 0)
        print("\n\n\n")
        print(f"Name: {self.building.name} \n Area: {self.building.area_val} \n Width: {self.building.width} \n Length: {self.building.length} \n Address: {self.building.address} \n Stories: {self.building.stories} \n Bedrooms: {self.building.bedrooms} \n Bathrooms: {self.building.bathrooms} \n Initial Beauty Value: {self.building.beauty_value} \n Price Value: {self.building.price}")

    def test_init(self):
        self.assertEqual(self.building.name, "Test Mansion")
        self.assertEqual(self.building.height, 100)
        self.assertEqual(self.building.width, 100)
        self.assertEqual(self.building.length, 100)
        self.assertEqual(self.building.address, "123 Mansion St")
        self.assertEqual(self.building.stories, 4)
        self.assertEqual(self.building.bedrooms, 8)
        self.assertEqual(self.building.bathrooms, 8)
        self.assertEqual(self.building.area_val, 40000)

if __name__ == '__main__':
    unittest.main()