#!/usr/bin/env python3
"""
Simplified Character class for demonstrating the correct testing approach.

This shows what the test would look like when testing a REAL Character implementation
without complex dependencies.
"""

class SimplePersonalityTraits:
    def __init__(self, neuroticism=50, extraversion=50):
        self.neuroticism = neuroticism
        self.extraversion = extraversion

class SimpleCharacter:
    """
    Simplified Character implementation with location evaluation methods.
    
    This demonstrates what the real Character class should have for location-based AI.
    """
    
    def __init__(self, name, age=25, energy=10, social_wellbeing=8):
        self.name = name
        self.age = age
        self.energy = energy
        self.social_wellbeing = social_wellbeing
        self.wealth_money = 50
        self.personality_traits = SimplePersonalityTraits()
        
    def evaluate_location_for_visit(self, building):
        """
        REAL implementation of location evaluation logic.
        
        This is what the actual Character class should have.
        """
        if not hasattr(building, 'get_location'):
            return 0
            
        location = building.get_location()
        score = 50  # Base score
        
        # Factor in security based on personality
        security = location.security
        if hasattr(self.personality_traits, 'neuroticism'):
            neuroticism = self.personality_traits.neuroticism
            if neuroticism > 70:
                score += (security - 5) * 3  # Bonus for secure locations
            elif neuroticism < 30:
                score -= (security - 5) * 1  # Small penalty for overly secure locations
                
        # Factor in popularity based on extraversion
        popularity = location.popularity
        if hasattr(self.personality_traits, 'extraversion'):
            extraversion = self.personality_traits.extraversion
            if extraversion > 70:
                score += (popularity - 5) * 3  # Extraverts like popular places
            elif extraversion < 30:
                score -= (popularity - 5) * 2  # Introverts avoid crowded places
                
        # Factor in activities based on current needs
        activities = location.activities_available
        
        # Rest activities are valuable when energy is low
        if self.energy < 5:
            if any('rest' in activity or 'sleep' in activity for activity in activities):
                score += 20
                
        # Social activities are valuable for social wellbeing
        if hasattr(self, 'social_wellbeing') and self.social_wellbeing < 5:
            if any('visit' in activity or 'social' in activity for activity in activities):
                score += 15
                
        return max(0, min(100, score))
    
    def find_suitable_locations(self, available_buildings, min_score=60):
        """Find locations suitable for this character"""
        evaluated_locations = []
        
        for building in available_buildings:
            score = self.evaluate_location_for_visit(building)
            if score >= min_score:
                evaluated_locations.append((building, score))
                
        # Sort by score (highest first)
        evaluated_locations.sort(key=lambda x: x[1], reverse=True)
        return evaluated_locations
    
    def make_location_decision(self, available_buildings):
        """Make a location decision based on character state and personality"""
        if not available_buildings:
            return None, "No locations available"
            
        # Determine primary motivation
        if self.energy < 3:
            motivation = 'rest'
        elif self.social_wellbeing < 4:
            motivation = 'social'
        else:
            motivation = 'general'
            
        # Find the best location
        suitable_locations = self.find_suitable_locations(available_buildings, min_score=40)
        if suitable_locations:
            chosen_building, score = suitable_locations[0]
            return chosen_building, {"motivation": motivation, "score": score}
        else:
            return None, {"motivation": motivation, "score": 0}