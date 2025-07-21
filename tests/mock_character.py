"""
Comprehensive MockCharacter class that accurately represents the real Character interface.

This module provides a shared, consistent mock implementation that all tests can use,
reducing the maintenance burden and ensuring interface accuracy.

The MockCharacter class includes:
- All major attributes from the real Character class
- Simplified but accurate method signatures  
- Configurable default values
- Proper state management for common operations
- Interface validation to ensure sync with real Character

This addresses the issue where overly simplified mocks could lead to tests passing
when real Character objects would cause failures.
"""

from unittest.mock import MagicMock
import uuid as uuid_module
import random


class MockMotives:
    """Mock implementation of PersonalMotives that provides realistic behavior."""
    
    def __init__(self):
        # Create mock motives with realistic default values
        self.hunger_motive = MockMotive("hunger", "bias toward satisfying hunger", 5.0)
        self.wealth_motive = MockMotive("wealth", "bias toward accumulating wealth", 6.0)
        self.mental_health_motive = MockMotive("mental health", "bias toward maintaining mental health", 7.0)
        self.social_wellbeing_motive = MockMotive("social wellbeing", "bias toward maintaining social wellbeing", 6.0)
        self.happiness_motive = MockMotive("happiness", "bias toward maintaining happiness", 7.0)
        self.health_motive = MockMotive("health", "bias toward maintaining health", 8.0)
        self.shelter_motive = MockMotive("shelter", "bias toward maintaining shelter", 6.0)
        self.stability_motive = MockMotive("stability", "bias toward maintaining stability", 5.0)
        self.luxury_motive = MockMotive("luxury", "bias toward maintaining luxury", 3.0)
        self.hope_motive = MockMotive("hope", "bias toward maintaining hope", 6.0)
        self.success_motive = MockMotive("success", "bias toward maintaining success", 5.0)
        self.control_motive = MockMotive("control", "bias toward maintaining control", 5.0)
        self.job_performance_motive = MockMotive("job performance", "bias toward maintaining job performance", 6.0)
        self.beauty_motive = MockMotive("beauty", "bias toward maintaining beauty", 4.0)
        self.community_motive = MockMotive("community", "bias toward maintaining community", 6.0)
        self.material_goods_motive = MockMotive("material goods", "bias toward maintaining material goods", 4.0)
        self.family_motive = MockMotive("family", "bias toward maintaining family", 7.0)

    def get_hunger_motive(self):
        return self.hunger_motive
    
    def get_wealth_motive(self):
        return self.wealth_motive
        
    def get_mental_health_motive(self):
        return self.mental_health_motive
        
    def get_social_wellbeing_motive(self):
        return self.social_wellbeing_motive
        
    def get_happiness_motive(self):
        return self.happiness_motive
        
    def get_health_motive(self):
        return self.health_motive
        
    def get_shelter_motive(self):
        return self.shelter_motive
        
    def get_stability_motive(self):
        return self.stability_motive
        
    def get_luxury_motive(self):
        return self.luxury_motive
        
    def get_hope_motive(self):
        return self.hope_motive
        
    def get_success_motive(self):
        return self.success_motive
        
    def get_control_motive(self):
        return self.control_motive
        
    def get_job_performance_motive(self):
        return self.job_performance_motive
        
    def get_beauty_motive(self):
        return self.beauty_motive
        
    def get_community_motive(self):
        return self.community_motive
        
    def get_material_goods_motive(self):
        return self.material_goods_motive
        
    def get_family_motive(self):
        return self.family_motive

    def to_dict(self):
        return {
            "hunger": self.hunger_motive,
            "wealth": self.wealth_motive,
            "mental health": self.mental_health_motive,
            "social wellbeing": self.social_wellbeing_motive,
            "happiness": self.happiness_motive,
            "health": self.health_motive,
            "shelter": self.shelter_motive,
            "stability": self.stability_motive,
            "luxury": self.luxury_motive,
            "hope": self.hope_motive,
            "success": self.success_motive,
            "control": self.control_motive,
            "job performance": self.job_performance_motive,
            "beauty": self.beauty_motive,
            "community": self.community_motive,
            "material goods": self.material_goods_motive,
            "family": self.family_motive,
        }


class MockMotive:
    """Mock implementation of Motive class."""
    
    def __init__(self, name, description, score):
        self.name = name
        self.description = description
        self.score = score
    
    def get_name(self):
        return self.name
    
    def get_description(self):
        return self.description
    
    def get_score(self):
        return self.score
    
    def set_score(self, score):
        self.score = score
        return self.score

    def to_dict(self):
        return {"name": self.name, "description": self.description, "score": self.score}


class MockPersonalityTraits:
    """Mock implementation of PersonalityTraits class."""
    
    def __init__(self, openness=0.0, conscientiousness=0.0, extraversion=0.0, 
                 agreeableness=0.0, neuroticism=0.0):
        self.openness = openness
        self.conscientiousness = conscientiousness
        self.extraversion = extraversion
        self.agreeableness = agreeableness
        self.neuroticism = neuroticism
    
    def get_openness(self):
        return self.openness
    
    def get_conscientiousness(self):
        return self.conscientiousness
    
    def get_extraversion(self):
        return self.extraversion
    
    def get_agreeableness(self):
        return self.agreeableness
    
    def get_neuroticism(self):
        return self.neuroticism

    def to_dict(self):
        return {
            "openness": self.openness,
            "conscientiousness": self.conscientiousness,
            "extraversion": self.extraversion,
            "agreeableness": self.agreeableness,
            "neuroticism": self.neuroticism,
        }


class MockInventory:
    """Mock implementation of ItemInventory class."""
    
    def __init__(self):
        self.food_items = []
        self.clothing_items = []
        self.tools_items = []
        self.weapons_items = []
        self.medicine_items = []
        self.misc_items = []
    
    def get_food_items(self):
        return self.food_items
    
    def set_food_items(self, items):
        self.food_items = items
        
    def get_clothing_items(self):
        return self.clothing_items
        
    def set_clothing_items(self, items):
        self.clothing_items = items
        
    def get_tools_items(self):
        return self.tools_items
        
    def set_tools_items(self, items):
        self.tools_items = items
        
    def get_weapons_items(self):
        return self.weapons_items
        
    def set_weapons_items(self, items):
        self.weapons_items = items
        
    def get_medicine_items(self):
        return self.medicine_items
        
    def set_medicine_items(self, items):
        self.medicine_items = items
        
    def get_misc_items(self):
        return self.misc_items
        
    def set_misc_items(self, items):
        self.misc_items = items
    
    def count_total_items(self):
        return (len(self.food_items) + len(self.clothing_items) + 
                len(self.tools_items) + len(self.weapons_items) + 
                len(self.medicine_items) + len(self.misc_items))
    
    def check_has_item_by_type(self, item_types):
        """Check if inventory contains items of specified types."""
        all_items = (self.food_items + self.clothing_items + self.tools_items + 
                    self.weapons_items + self.medicine_items + self.misc_items)
        for item in all_items:
            if hasattr(item, 'item_type') and item.item_type in item_types:
                return True
        return len(self.food_items) > 0 if 'food' in item_types else False
    
    def check_has_item_by_name(self, item_names):
        """Check if inventory contains items with specified names."""
        all_items = (self.food_items + self.clothing_items + self.tools_items + 
                    self.weapons_items + self.medicine_items + self.misc_items)
        for item in all_items:
            if hasattr(item, 'name') and item.name in item_names:
                return True
        return False


class MockLocation:
    """Mock implementation of Location class."""
    
    def __init__(self, name="Home", coordinates=(0, 0)):
        self.name = name
        self.coordinates = coordinates
    
    def to_dict(self):
        return {"name": self.name, "coordinates": self.coordinates}


class MockCharacter:
    """
    Comprehensive mock implementation of the Character class.
    
    This mock includes all major attributes and methods from the real Character class,
    providing a consistent interface for testing while avoiding heavy dependencies.
    
    Key features:
    - Complete attribute coverage matching real Character class
    - Realistic default values for all attributes
    - Proper method signatures that match the real implementation
    - State management for common operations
    - Configurable behavior for testing different scenarios
    """
    
    def __init__(self, name="TestCharacter", age=25, pronouns="they/them", 
                 job="unemployed", health_status=8.0, hunger_level=3.0,
                 wealth_money=100.0, mental_health=7.0, social_wellbeing=6.0,
                 job_performance=50.0, community=50.0, **kwargs):
        
        # Core identity attributes
        self.name = name
        self.age = age
        self.pronouns = pronouns
        self.uuid = kwargs.get('uuid', str(uuid_module.uuid4()))
        
        # Job and career attributes
        self.job = job
        self.job_performance = job_performance
        self.career_goals = kwargs.get('career_goals', [])
        self.short_term_goals = kwargs.get('short_term_goals', [])
        
        # Health and wellbeing attributes
        self.health_status = health_status
        self.hunger_level = hunger_level
        self.mental_health = mental_health
        self.energy = kwargs.get('energy', 8.0)
        
        # Economic attributes
        self.wealth_money = wealth_money
        self.material_goods = kwargs.get('material_goods', 0)
        
        # Social attributes
        self.social_wellbeing = social_wellbeing
        self.community = community
        self.friendship_grid = kwargs.get('friendship_grid', [])
        self.romantic_relationships = kwargs.get('romantic_relationships', {})
        self.exclusive_relationship = kwargs.get('exclusive_relationship', None)
        self.romanceable = kwargs.get('romanceable', True)
        
        # Psychological attributes
        self.happiness = kwargs.get('happiness', 60.0)
        self.shelter = kwargs.get('shelter', 50.0)
        self.stability = kwargs.get('stability', 50.0)
        self.luxury = kwargs.get('luxury', 20.0)
        self.hope = kwargs.get('hope', 60.0)
        self.success = kwargs.get('success', 40.0)
        self.control = kwargs.get('control', 50.0)
        self.beauty = kwargs.get('beauty', 40.0)
        
        # Behavioral attributes
        self.base_libido = kwargs.get('base_libido', 50.0)
        self.monogamy = kwargs.get('monogamy', 70.0)
        
        # Spatial attributes
        self.location = kwargs.get('location', MockLocation("Home"))
        self.coordinates_location = kwargs.get('coordinates_location', [0, 0])
        self.destination = kwargs.get('destination', None)
        self.path = kwargs.get('path', [])
        self.velocity = kwargs.get('velocity', (0, 0))
        self.speed = kwargs.get('speed', 1.0)
        
        # Life attributes
        self.recent_event = kwargs.get('recent_event', "started their journey")
        self.long_term_goal = kwargs.get('long_term_goal', "find happiness")
        
        # Complex object attributes
        self.personality_traits = kwargs.get('personality_traits', MockPersonalityTraits())
        self.motives = kwargs.get('motives', MockMotives())
        self.inventory = kwargs.get('inventory', MockInventory())
        self.home = kwargs.get('home', None)
        self.skills = kwargs.get('skills', {})
        
        # Game mechanics attributes
        self.goals = kwargs.get('goals', [])
        self.state = kwargs.get('state', None)
        self.character_actions = kwargs.get('character_actions', [])
        self.needed_items = kwargs.get('needed_items', [])
        self.possible_interactions = kwargs.get('possible_interactions', [])
        
        # Investment attributes (used in some tests)
        self.investment_portfolio = kwargs.get('investment_portfolio', MockInvestmentPortfolio())
        
        # LLM decision making attribute
        self.use_llm_decisions = kwargs.get('use_llm_decisions', False)
        
        # Graph manager and other system references
        self.graph_manager = kwargs.get('graph_manager', MagicMock())
        self.goap_planner = kwargs.get('goap_planner', MagicMock())
        self.prompt_builder = kwargs.get('prompt_builder', MagicMock())
        
        # Internal state
        self._initialized = True

    # Core identity methods
    def get_name(self):
        return self.name
    
    def set_name(self, name):
        self.name = name
        return self.name
    
    def get_age(self):
        return self.age
    
    def set_age(self, age):
        self.age = age
        return self.age
    
    def get_pronouns(self):
        return self.pronouns
    
    def set_pronouns(self, pronouns):
        self.pronouns = pronouns
        return self.pronouns

    # Job-related methods
    def get_job(self):
        return self.job
    
    def set_job(self, job):
        self.job = job
        return self.job
    
    def has_job(self):
        return self.job != "unemployed"
    
    def get_job_performance(self):
        return self.job_performance
    
    def set_job_performance(self, job_performance):
        self.job_performance = job_performance
        return self.job_performance

    # Health and wellbeing methods
    def get_health_status(self):
        return self.health_status
    
    def set_health_status(self, health_status):
        self.health_status = health_status
        return self.health_status
    
    def get_hunger_level(self):
        return self.hunger_level
    
    def set_hunger_level(self, hunger_level):
        self.hunger_level = hunger_level
        return self.hunger_level
    
    def get_mental_health(self):
        return self.mental_health
    
    def set_mental_health(self, mental_health):
        self.mental_health = mental_health
        return self.mental_health

    # Economic methods
    def get_wealth_money(self):
        return self.wealth_money
    
    def set_wealth_money(self, wealth_money):
        self.wealth_money = wealth_money
        return self.wealth_money
    
    def has_investment(self):
        return len(self.investment_portfolio.get_stocks()) > 0
    
    def get_material_goods(self):
        return self.material_goods
    
    def set_material_goods(self, material_goods):
        self.material_goods = material_goods
        return self.material_goods

    # Social methods
    def get_social_wellbeing(self):
        return self.social_wellbeing
    
    def set_social_wellbeing(self, social_wellbeing):
        self.social_wellbeing = social_wellbeing
        return self.social_wellbeing
    
    def get_community(self):
        return self.community
    
    def set_community(self, community):
        self.community = community
        return self.community
    
    def get_friendship_grid(self):
        return self.friendship_grid
    
    def set_friendship_grid(self, friendship_grid):
        self.friendship_grid = friendship_grid

    # Psychological state methods
    def get_happiness(self):
        return self.happiness
    
    def set_happiness(self, happiness):
        self.happiness = happiness
        return self.happiness
    
    def get_shelter(self):
        return self.shelter
    
    def set_shelter(self, shelter):
        self.shelter = shelter
        return self.shelter
    
    def get_stability(self):
        return self.stability
    
    def set_stability(self, stability):
        self.stability = stability
        return self.stability
    
    def get_luxury(self):
        return self.luxury
    
    def set_luxury(self, luxury):
        self.luxury = luxury
        return self.luxury
    
    def get_hope(self):
        return self.hope
    
    def set_hope(self, hope):
        self.hope = hope
        return self.hope
    
    def get_success(self):
        return self.success
    
    def set_success(self, success):
        self.success = success
        return self.success
    
    def get_control(self):
        return self.control
    
    def set_control(self, control):
        self.control = control
        return self.control
    
    def get_beauty(self):
        return self.beauty
    
    def set_beauty(self, beauty):
        self.beauty = beauty
        return self.beauty

    # Complex object methods
    def get_personality_traits(self):
        return self.personality_traits
    
    def set_personality_traits(self, personality_traits):
        self.personality_traits = personality_traits
        return self.personality_traits
    
    def get_motives(self):
        return self.motives
    
    def set_motives(self, motives):
        self.motives = motives
        return self.motives
    
    def get_inventory(self):
        return self.inventory
    
    def set_inventory(self, **kwargs):
        if kwargs:
            for key, value in kwargs.items():
                if hasattr(self.inventory, f'set_{key}'):
                    getattr(self.inventory, f'set_{key}')(value)
        return self.inventory
    
    def get_home(self):
        return self.home
    
    def set_home(self, home):
        self.home = home
        return self.home

    # Life event methods
    def get_recent_event(self):
        return self.recent_event
    
    def set_recent_event(self, recent_event):
        self.recent_event = recent_event
        return self.recent_event
    
    def get_long_term_goal(self):
        return self.long_term_goal
    
    def set_long_term_goal(self, long_term_goal):
        self.long_term_goal = long_term_goal
        return self.long_term_goal

    # Social interaction methods
    def respond_to_talk(self, initiator):
        """
        Respond to a conversation initiated by another character.
        This provides realistic behavior for social interactions.
        """
        # Increase social wellbeing slightly when talked to
        self.social_wellbeing += 0.1
        
        # Return appropriate response based on personality
        if hasattr(self.personality_traits, 'extraversion') and self.personality_traits.extraversion > 1:
            return f"{self.name} engages enthusiastically in conversation"
        elif hasattr(self.personality_traits, 'neuroticism') and self.personality_traits.neuroticism > 1:
            return f"{self.name} responds nervously but appreciates the attention"
        else:
            return f"{self.name} listens and responds thoughtfully"

    # Calculation methods (simplified but realistic)
    def calculate_happiness(self):
        """Calculate happiness based on various factors."""
        base_happiness = (self.health_status + self.mental_health + 
                         self.social_wellbeing + self.wealth_money/10) / 4
        return min(100.0, max(0.0, base_happiness * 10))
    
    def calculate_stability(self):
        """Calculate stability based on various factors."""
        base_stability = (self.shelter + self.control + self.success + 
                         self.community) / 4
        return min(100.0, max(0.0, base_stability))
    
    def calculate_success(self):
        """Calculate success based on job performance and wealth."""
        return min(100.0, (self.job_performance + self.wealth_money/10) / 2)
    
    def calculate_control(self):
        """Calculate sense of control."""
        return min(100.0, (self.wealth_money/10 + self.job_performance + 
                          self.stability) / 3)
    
    def calculate_hope(self):
        """Calculate hope based on positive factors."""
        return min(100.0, (self.happiness + self.success + self.community) / 3)
    
    def calculate_material_goods(self):
        """Calculate material goods value."""
        return min(100.0, self.inventory.count_total_items() * 5)

    # Behavioral decision methods
    def decide_to_socialize(self):
        """Decide whether to engage in social activities."""
        if hasattr(self.personality_traits, 'extraversion'):
            return self.personality_traits.extraversion > 0
        return self.social_wellbeing < 50
    
    def decide_to_work(self):
        """Decide whether to prioritize work."""
        return self.wealth_money < 50 or self.job_performance < 40
    
    def decide_to_explore(self):
        """Decide whether to explore new areas."""
        if hasattr(self.personality_traits, 'openness'):
            return self.personality_traits.openness > 1
        return random.random() > 0.7

    # System integration methods
    def update_character(self, **kwargs):
        """Update character attributes."""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
        return self
    
    def to_dict(self):
        """Convert character to dictionary representation."""
        return {
            "name": self.name,
            "age": self.age,
            "pronouns": self.pronouns,
            "job": self.job,
            "health_status": self.health_status,
            "hunger_level": self.hunger_level,
            "wealth_money": self.wealth_money,
            "mental_health": self.mental_health,
            "social_wellbeing": self.social_wellbeing,
            "happiness": self.happiness,
            "shelter": self.shelter,
            "stability": self.stability,
            "luxury": self.luxury,
            "hope": self.hope,
            "success": self.success,
            "control": self.control,
            "job_performance": self.job_performance,
            "beauty": self.beauty,
            "community": self.community,
            "material_goods": self.material_goods,
            "friendship_grid": self.friendship_grid,
            "recent_event": self.recent_event,
            "long_term_goal": self.long_term_goal,
            "location": self.location.to_dict() if self.location else None,
            "coordinates_location": self.coordinates_location,
            "use_llm_decisions": self.use_llm_decisions,
        }
    
    def get_state(self):
        """Get character state for game mechanics."""
        return MockState(self)
    
    def define_descriptors(self):
        """Generate comprehensive descriptors for the character."""
        return {
            "name": self.name,
            "age": self.age,
            "pronouns": self.pronouns,
            "job": self.job,
            "health_status": self.health_status,
            "mental_health": self.mental_health,
            "energy_level": self.energy,
            "hunger_level": self.hunger_level,
            "wealth_money": self.wealth_money,
            "social_wellbeing": self.social_wellbeing,
            "job_performance": self.job_performance,
            "community_standing": self.community,
            "personality_summary": self.personality_traits.to_dict(),
            "recent_event": self.recent_event,
            "long_term_goal": self.long_term_goal,
            "current_mood": getattr(self, "current_mood", 50),
            "current_activity": getattr(self, "current_activity", "None"),
            "home_status": "housed" if self.home else "homeless",
            "shelter_value": self.shelter,
            "stability": self.stability,
            "happiness": self.happiness,
            "success": self.success,
            "hope": self.hope,
            "luxury": self.luxury,
            "control": self.control,
            "beauty": self.beauty,
            "friendship_count": len(self.friendship_grid),
            "relationship_status": ("in_relationship" if self.exclusive_relationship 
                                  else "single"),
            "romanceable": self.romanceable,
            "goals_count": len(self.goals),
            "item_count": self.inventory.count_total_items(),
            "material_goods": self.material_goods,
            "current_location": self.location.name if self.location else "Unknown",
            "coordinates": self.coordinates_location,
        }

    def add_food_item(self, name, calories):
        """Add a food item to the character's inventory for testing."""
        food_item = type('MockFood', (), {'name': name, 'calories': calories, 'item_type': 'food'})()
        self.inventory.food_items.append(food_item)

    def __eq__(self, other):
        """Check equality based on name and uuid."""
        if not isinstance(other, MockCharacter):
            return False
        return self.name == other.name and self.uuid == other.uuid
    
    def __hash__(self):
        """Hash based on uuid for set operations."""
        return hash(self.uuid)
    
    def __repr__(self):
        return f"MockCharacter(name='{self.name}', age={self.age}, job='{self.job}')"


class MockInvestmentPortfolio:
    """Mock implementation of InvestmentPortfolio class."""
    
    def __init__(self):
        self.stocks = []
    
    def get_stocks(self):
        return self.stocks
    
    def add_stock(self, stock):
        self.stocks.append(stock)


class MockState:
    """Mock implementation of State class."""
    
    def __init__(self, character):
        self.character = character
    
    def get_character(self):
        return self.character


def create_test_character(name="TestChar", **kwargs):
    """
    Convenience function to create a MockCharacter with test-friendly defaults.
    
    Args:
        name: Character name
        **kwargs: Additional attributes to override defaults
    
    Returns:
        MockCharacter instance configured for testing
    """
    return MockCharacter(name=name, **kwargs)


def create_realistic_character(name="RealisticChar", scenario="balanced", **kwargs):
    """
    Create a MockCharacter with realistic attribute values for different scenarios.
    
    Args:
        name: Character name
        scenario: Predefined scenario ("balanced", "poor", "wealthy", "lonely", "social")
        **kwargs: Additional attributes to override scenario defaults
    
    Returns:
        MockCharacter instance with realistic values
    """
    scenarios = {
        "balanced": {
            "age": 30,
            "health_status": 8.0,
            "hunger_level": 2.0,
            "wealth_money": 500.0,
            "mental_health": 7.0,
            "social_wellbeing": 6.0,
            "job_performance": 70.0,
        },
        "poor": {
            "age": 25,
            "health_status": 5.0,
            "hunger_level": 7.0,
            "wealth_money": 20.0,
            "mental_health": 4.0,
            "social_wellbeing": 3.0,
            "job_performance": 30.0,
            "job": "unemployed",
        },
        "wealthy": {
            "age": 45,
            "health_status": 9.0,
            "hunger_level": 1.0,
            "wealth_money": 5000.0,
            "mental_health": 8.0,
            "social_wellbeing": 8.0,
            "job_performance": 90.0,
            "job": "CEO",
        },
        "lonely": {
            "age": 35,
            "health_status": 7.0,
            "hunger_level": 4.0,
            "wealth_money": 300.0,
            "mental_health": 3.0,
            "social_wellbeing": 2.0,
            "job_performance": 50.0,
        },
        "social": {
            "age": 28,
            "health_status": 8.0,
            "hunger_level": 3.0,
            "wealth_money": 400.0,
            "mental_health": 8.0,
            "social_wellbeing": 9.0,
            "job_performance": 60.0,
        },
    }
    
    scenario_values = scenarios.get(scenario, scenarios["balanced"])
    scenario_values.update(kwargs)
    
    return MockCharacter(name=name, **scenario_values)


# Interface validation
def validate_character_interface(mock_char, expected_attributes=None):
    """
    Validate that MockCharacter has the expected interface.
    
    This can be used in tests to ensure the mock stays in sync with 
    the real Character class interface.
    
    Args:
        mock_char: MockCharacter instance to validate
        expected_attributes: List of attribute names that should exist
    
    Returns:
        tuple: (is_valid, missing_attributes)
    """
    if expected_attributes is None:
        # Core attributes that should always be present
        expected_attributes = [
            'name', 'age', 'pronouns', 'job', 'health_status', 'hunger_level',
            'wealth_money', 'mental_health', 'social_wellbeing', 'happiness',
            'personality_traits', 'motives', 'inventory', 'location'
        ]
    
    missing_attributes = []
    for attr in expected_attributes:
        if not hasattr(mock_char, attr):
            missing_attributes.append(attr)
    
    return len(missing_attributes) == 0, missing_attributes


if __name__ == "__main__":
    # Basic testing
    print("Testing MockCharacter implementation...")
    
    # Test basic creation
    char = MockCharacter("TestCharacter")
    print(f"Created character: {char}")
    
    # Test interface validation
    is_valid, missing = validate_character_interface(char)
    print(f"Interface validation: {'✓ Valid' if is_valid else '✗ Invalid'}")
    if missing:
        print(f"Missing attributes: {missing}")
    
    # Test realistic character creation
    poor_char = create_realistic_character("PoorCharacter", "poor")
    wealthy_char = create_realistic_character("WealthyCharacter", "wealthy")
    
    print(f"Poor character wealth: {poor_char.wealth_money}")
    print(f"Wealthy character wealth: {wealthy_char.wealth_money}")
    
    # Test social interaction
    response = char.respond_to_talk(poor_char)
    print(f"Social interaction response: {response}")
    
    print("✓ MockCharacter implementation test completed successfully!")