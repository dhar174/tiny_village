import sys
import unittest
from types import ModuleType
from unittest.mock import Mock, patch

# Create stub modules to satisfy imports in tiny_prompt_builder
stub_tc = ModuleType('tiny_characters')
stub_attr = ModuleType('attr')

class MockMotive:
    """Mock class to simulate a Character's motive with a score."""
    def __init__(self, score=50):
        self.score = score
    
    def __call__(self):
        return self.score

class MockMotives:
    """Mock class to simulate PersonalMotives with all required motive methods."""
    def __init__(self):
        # Create mock motives with default scores
        self._motives = {
            'hunger': MockMotive(40),
            'health': MockMotive(60), 
            'wealth': MockMotive(30),
            'mental_health': MockMotive(70),
            'social_wellbeing': MockMotive(50),
            'happiness': MockMotive(55),
            'shelter': MockMotive(80),
            'stability': MockMotive(45),
            'luxury': MockMotive(20),
            'hope': MockMotive(65),
            'success': MockMotive(35),
            'control': MockMotive(40),
            'job_performance': MockMotive(60),
            'beauty': MockMotive(50),
            'community': MockMotive(55),
            'material_goods': MockMotive(25),
        }
    
    def get_hunger_motive(self): return self._motives['hunger']
    def get_health_motive(self): return self._motives['health']
    def get_wealth_motive(self): return self._motives['wealth']
    def get_mental_health_motive(self): return self._motives['mental_health']
    def get_social_wellbeing_motive(self): return self._motives['social_wellbeing']
    def get_happiness_motive(self): return self._motives['happiness']
    def get_shelter_motive(self): return self._motives['shelter']
    def get_stability_motive(self): return self._motives['stability']
    def get_luxury_motive(self): return self._motives['luxury']
    def get_hope_motive(self): return self._motives['hope']
    def get_success_motive(self): return self._motives['success']
    def get_control_motive(self): return self._motives['control']
    def get_job_performance_motive(self): return self._motives['job_performance']
    def get_beauty_motive(self): return self._motives['beauty']
    def get_community_motive(self): return self._motives['community']
    def get_material_goods_motive(self): return self._motives['material_goods']

class MockInventory:
    """Mock class to simulate ItemInventory."""
    def __init__(self):
        self.items = []

class MockCharacter:
    """
    Mock Character class that provides the same interface as the real Character class.
    This ensures that tests properly validate integration with PromptBuilder.
    """
    def __init__(self):
        # Basic attributes that MockCharacter had before
        self.name = "Eve"
        self.job = "Farmer"
        self.recent_event = "outbreak"
        self.wealth_money = 10
        self.health_status = 7
        self.hunger_level = 5
        self.energy = 5
        self.mental_health = 6
        self.social_wellbeing = 6
        self.long_term_goal = "become successful farmer"
        
        # Additional attributes that PromptBuilder expects
        self.motives = MockMotives()
        self.inventory = MockInventory()
        
        # Additional attributes to match Character interface
        self.happiness = 5
        self.shelter = 8
        self.stability = 4
        self.luxury = 2
        self.hope = 6
        self.success = 3
        self.control = 4
        self.job_performance = 6
        self.beauty = 5
        self.community = 5
        self.material_goods = 2
        self.friendship_grid = {}
    
    # Getter methods that PromptBuilder expects
    def get_name(self): return self.name
    def get_health_status(self): return self.health_status
    def get_hunger_level(self): return self.hunger_level
    def get_wealth(self): return self.wealth_money
    def get_wealth_money(self): return self.wealth_money
    def get_mental_health(self): return self.mental_health
    def get_social_wellbeing(self): return self.social_wellbeing
    def get_motives(self): return self.motives
    def get_long_term_goal(self): return self.long_term_goal
    def get_inventory(self): return self.inventory
    def get_happiness(self): return self.happiness
    def get_shelter(self): return self.shelter
    def get_stability(self): return self.stability
    def get_luxury(self): return self.luxury
    def get_hope(self): return self.hope
    def get_success(self): return self.success
    def get_control(self): return self.control
    def get_job_performance(self): return self.job_performance
    def get_beauty(self): return self.beauty
    def get_community(self): return self.community
    def get_material_goods(self): return self.material_goods
    def get_friendship_grid(self): return self.friendship_grid
    
    # Additional methods that may be needed
    def evaluate_goals(self): return []
    def to_dict(self): return {"name": self.name, "job": self.job}

# Set the Character class to our proper mock instead of object
stub_tc.Character = MockCharacter

class CrisisPromptTests(unittest.TestCase):
    def test_prompt_contains_description_and_assistant_cue(self):
        with patch.dict(sys.modules, {"tiny_characters": stub_tc, "attr": stub_attr}):
            import tiny_prompt_builder
            tiny_prompt_builder.descriptors.event_recent.setdefault("default", [""])
            tiny_prompt_builder.descriptors.financial_situation.setdefault("default", [""])
            PromptBuilder = tiny_prompt_builder.PromptBuilder
            char = MockCharacter()
            builder = PromptBuilder(char)
            prompt = builder.generate_crisis_response_prompt("barn fire", urgency="high")
        self.assertIn("barn fire", prompt)
        self.assertTrue(prompt.strip().endswith("<|assistant|>"))

if __name__ == "__main__":
    unittest.main()
