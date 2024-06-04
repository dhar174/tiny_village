import unittest

from regex import P
from tiny_characters import CreateCharacter, Character
import tiny_buildings as tb

class TestCreateCharacter(unittest.TestCase):
    def setUp(self):
        self.character = None

    # def test_create_new_character_manual(self):
    #     # Test creating a character manually
    #     self.character = CreateCharacter().create_new_character(
    #         mode="manual",
    #         name="John Cusack",
    #         age=25,
    #         pronouns="he/him",
    #         job="engineer",
    #         wealth_money=1000,
    #         mental_health=8,
    #         social_wellbeing=7,
    #         job_performance=90,
    #         home="hovel",
    #         recent_event="Won a coding competition",
    #         long_term_goal="Start a tech company"
    #     )
    #     # Add assertions here to check if the character was created correctly
    #     for key, val in self.character.to_dict().items():
    #         print(key, val)
    #         print("\n")
    #     print("\n")
    #     assert isinstance(self.character, Character)
    #     assert self.character.get_name() == "John Cusack"
    #     assert self.character.get_age() == 25
    #     assert self.character.pronouns == "he/him"
    #     assert self.character.get_job().get_job_name() == "engineer"
    #     assert self.character.wealth_money == 1000
    #     assert self.character.mental_health == 8
    #     assert self.character.social_wellbeing == 7
    #     assert self.character.job_performance == 90
    #     assert isinstance(self.character.get_home(), tb.House)
    #     assert self.character.get_home().get_name().lower() == "hovel"
    #     assert self.character.recent_event == "Won a coding competition"  
    #     assert self.character.long_term_goal == "Start a tech company"

        

    def test_create_new_character_auto(self):
        # Test creating a character automatically
        self.character = CreateCharacter().create_new_character()
        # Add assertions here to check if the character was created correctly
        assert self.character.name != "John Doe"
        for key, val in self.character.to_dict().items():
            print(key, val)
        print("\n")
        for key, _, val in self.character.get_motives().items():
            print(key, val)


if __name__ == '__main__':
    unittest.main()