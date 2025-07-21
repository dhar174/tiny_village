"""
Social Model for Tiny Village

This module contains the SocialModel class which handles all social simulation logic
including relationship calculations, romance compatibility, social influence, and
character interaction dynamics.

The SocialModel operates on WorldState (graph data) provided as a dependency,
maintaining separation of concerns from the GraphManager.
"""

import copy
import math
import time
from datetime import datetime
from functools import lru_cache
import importlib
import numpy as np


# Utility functions for social calculations
@lru_cache(maxsize=1024)
def cached_sigmoid_motive_scale_approx_optimized(
    motive_value, max_value=10.0, steepness=0.01
):
    motive_value = max(0, motive_value)
    mid_point = max_value / 2
    x = steepness * (motive_value - mid_point)
    sigmoid_value = 0.5 * (x / (1 + abs(x)) + 1)
    return min(sigmoid_value * max_value, max_value)


@lru_cache(maxsize=1024)
def cached_sigmoid_relationship_scale_approx_optimized(
    days_known, max_days=1000, steepness=0.01
):
    days_known = max(0, days_known)
    mid_point = max_days / 2
    x = steepness * (days_known - mid_point)
    sigmoid_value = 0.5 * (x / (1 + abs(x)) + 1)
    return min(sigmoid_value * 100, 100)


@lru_cache(maxsize=1024)
def tanh_scaling_raw(x, data_max, data_min, data_avg, data_std):
    a = data_std
    centered_value = x - data_avg
    scaled_value = math.tanh(centered_value / a)
    return scaled_value


def calculate_relationship_type(
    char1, char2, emotional_impact, interaction_frequency, strength, trust, historical
):
    """
    Determine the relationship type using a more complex model with weighted and interacting factors.
    """
    # Calculate the base relationship type based on emotional impact
    if emotional_impact > 0.5:
        base_type = "positive"
    elif emotional_impact < -0.5:
        base_type = "negative"
    else:
        base_type = "neutral"

    # Adjust the relationship type based on interaction frequency
    if interaction_frequency > 0.5:
        if base_type == "positive":
            final_type = "significant"
        elif base_type == "negative":
            final_type = "strained"
        else:
            final_type = "casual"

    # Adjust the relationship type based on strength and trust
    if strength > 0.5:
        if base_type == "positive":
            final_type = "close"
        elif base_type == "negative":
            final_type = "hostile"
        else:
            final_type = "professional"

    if trust > 0.5:
        if base_type == "positive":
            final_type = "trusted"
        elif base_type == "negative":
            final_type = "distrusted"
        else:
            final_type = "acquaintance"

    # Adjust the relationship type based on historical interactions
    if historical > 0.5:
        if base_type == "positive":
            final_type = "loyal"
        elif base_type == "negative":
            final_type = "grudge"
        else:
            final_type = "familiar"

    if char1.job.location == char2.job.location:
        if base_type == "positive":
            final_type = "colleague"
        elif base_type == "negative":
            final_type = "professional rival"
        else:
            final_type = "associate"

    if char1.home == char2.home:
        if base_type == "positive":
            final_type = "roommate"
        elif base_type == "negative":
            final_type = "adversary"
        else:
            final_type = "live-in neighbor"

    return final_type


def update_emotional_impact(
    impact_value,
    interaction_type,
    historical,
    emotional_impact,
    trust,
    impact_rating,
    base_decay_rate=0.01,
):
    """Update emotional impact with decay over time."""
    current_impact = emotional_impact
    stability_factor = historical / 100
    impact_value *= (
        1 - stability_factor
    )  # Adjust impact value based on historical interactions

    if interaction_type == "positive":
        current_impact += impact_value * (1 + trust)
    elif interaction_type == "negative":
        current_impact -= impact_value * (1 - trust)
    else:  # Neutral interaction
        current_impact *= 0.95  # Slight decay for neutral interaction

    emotional_impact = max(-1, min(1, current_impact))
    last_interaction_time = datetime.now().timestamp()

    return emotional_impact


def update_trust(
    trust_increment=0.01,
    interaction_type="proximity",
    betrayal=False,
    impact_rating=1,
    historical=50,
    personality_traits={},
    trust=0.5,
):
    stability_factor = historical / 100
    if betrayal:
        trust -= trust_increment * (1 - stability_factor)
    elif interaction_type == "cooperative":
        trust += (
            trust_increment
            * (1 + personality_traits.get("agreeableness", 0.5))
            * stability_factor
        )
    elif interaction_type == "proximity":
        trust += trust_increment * 0.5 * stability_factor
    trust = max(0, min(1, trust))
    return trust


class SocialModel:
    """
    SocialModel handles all social simulation logic including relationship calculations,
    romance compatibility, social influence, and character interaction dynamics.
    
    This class operates on WorldState (graph data) provided as a dependency and is 
    designed to be independent of the GraphManager's internal structure.
    """
    
    def __init__(self, world_state=None):
        """
        Initialize the SocialModel with a WorldState dependency.
        
        Args:
            world_state: The graph/world state object containing characters, relationships, etc.
        """
        self.world_state = world_state
        
    def set_world_state(self, world_state):
        """Set or update the world state dependency."""
        self.world_state = world_state
        
    def calculate_dynamic_weights(self, historical):
        """
        Calculate dynamic weights for personality traits based on relationship history.
        
        Args:
            historical: Historical value representing relationship length/maturity
            
        Returns:
            dict: Dictionary of weights for different personality traits
        """
        initial_stage = historical < 20
        middle_stage = 20 <= historical < 60
        mature_stage = historical >= 60

        if initial_stage:
            return {
                "openness": 0.2,
                "extraversion": 0.25,
                "conscientiousness": 0.15,
                "agreeableness": 0.25,
                "neuroticism": 0.05,
                "openness_conscientiousness_interaction": 0.05,
                "extraversion_agreeableness_interaction": 0.05,
                "neuroticism_stabilization": 0.1,
            }
        elif middle_stage:
            return {
                "openness": 0.15,
                "extraversion": 0.2,
                "conscientiousness": 0.2,
                "agreeableness": 0.2,
                "neuroticism": 0.1,
                "openness_conscientiousness_interaction": 0.05,
                "extraversion_agreeableness_interaction": 0.05,
                "neuroticism_stabilization": 0.05,
            }
        elif mature_stage:
            return {
                "openness": 0.1,
                "extraversion": 0.15,
                "conscientiousness": 0.25,
                "agreeableness": 0.2,
                "neuroticism": 0.1,
                "openness_conscientiousness_interaction": 0.1,
                "extraversion_agreeableness_interaction": 0.05,
                "neuroticism_stabilization": 0.05,
            }
        return None

    def calculate_romance_compatibility(self, char1, char2, historical):
        """
        Calculate romantic compatibility between two characters based on personality traits.
        
        Args:
            char1: First character object
            char2: Second character object  
            historical: Historical value representing relationship length
            
        Returns:
            float: Compatibility score between 0.0 and 1.0
        """
        # Calculate compatibility components
        openness = char1.personality_traits.get_openness()
        extraversion = char1.personality_traits.get_extraversion()
        conscientiousness = char1.personality_traits.get_conscientiousness()
        agreeableness = char1.personality_traits.get_agreeableness()
        neuroticism = char1.personality_traits.get_neuroticism()
        partner_agreeableness = char2.personality_traits.get_agreeableness()
        partner_conscientiousness = char2.personality_traits.get_conscientiousness()
        partner_extraversion = char2.personality_traits.get_extraversion()
        partner_neuroticism = char2.personality_traits.get_neuroticism()
        partner_openness = char2.personality_traits.get_openness()

        openness_compat = 1 - abs(openness - partner_openness) / 8
        extraversion_compat = 1 - abs(extraversion - partner_extraversion) / 8
        conscientiousness_compat = (
            1 - abs(conscientiousness - partner_conscientiousness) / 8
        )
        agreeableness_compat = 1 - abs(agreeableness - partner_agreeableness) / 8
        neuroticism_compat = 1 - abs(neuroticism - partner_neuroticism) / 8

        # Interaction terms
        openness_conscientiousness_interaction = (
            (openness + partner_openness)
            / 2
            * (1 - abs(conscientiousness - partner_conscientiousness) / 8)
        )
        extraversion_agreeableness_interaction = (
            (extraversion + partner_extraversion)
            / 2
            * (1 - abs(agreeableness - partner_agreeableness) / 8)
        )
        neuroticism_stabilization = (1 - abs(neuroticism - partner_neuroticism) / 8) * (
            1 - (neuroticism + partner_neuroticism) / 16
        )

        # Weights for each trait's influence on compatibility
        weights = self.calculate_dynamic_weights(historical=historical)

        # Calculate weighted compatibility score
        compatibility_score = (
            openness_compat * weights["openness"]
            + extraversion_compat * weights["extraversion"]
            + conscientiousness_compat * weights["conscientiousness"]
            + agreeableness_compat * weights["agreeableness"]
            + neuroticism_compat * weights["neuroticism"]
            + openness_conscientiousness_interaction
            * weights["openness_conscientiousness_interaction"]
            + extraversion_agreeableness_interaction
            * weights["extraversion_agreeableness_interaction"]
            + neuroticism_stabilization * weights["neuroticism_stabilization"]
        )

        return max(0.0, min(1.0, compatibility_score))

    def calculate_romance_interest(
        self,
        char1,
        char2,
        romance_compat,
        romance_value,
        relationship_type,
        strength,
        historical,
        trust,
        interaction_frequency,
        emotional_impact,
    ):
        """
        Calculate romance interest between two characters based on various factors.
        
        Args:
            char1: First character object
            char2: Second character object
            romance_compat: Romance compatibility score
            romance_value: Current romance value
            relationship_type: Type of relationship
            strength: Relationship strength
            historical: Historical relationship data
            trust: Trust level between characters
            interaction_frequency: How often characters interact
            emotional_impact: Emotional impact of the relationship
            
        Returns:
            float: Romance interest score
        """
        # Get motive values and normalize them
        wealth_motive = char1.get_motives().get_wealth_motive().score
        wealth_motive = cached_sigmoid_motive_scale_approx_optimized(wealth_motive)
        partner_wealth_motive = char2.get_motives().get_wealth_motive().score
        partner_wealth_motive = cached_sigmoid_motive_scale_approx_optimized(
            partner_wealth_motive
        )

        family_motive = char1.get_motives().get_family_motive().score
        family_motive = cached_sigmoid_motive_scale_approx_optimized(family_motive)
        partner_family_motive = char2.get_motives().get_family_motive().score
        partner_family_motive = cached_sigmoid_motive_scale_approx_optimized(
            partner_family_motive
        )

        # Get and normalize wealth values if world_state is available
        wealth_money = char1.wealth_money
        partner_wealth_money = char2.wealth_money
        
        if self.world_state and hasattr(self.world_state, 'get_maximum_attribute_value'):
            # Normalize wealth using world state statistics
            wealth_money = tanh_scaling_raw(
                wealth_money,
                data_max=(
                    self.world_state.get_maximum_attribute_value("wealth_money")
                    if self.world_state.get_maximum_attribute_value("wealth_money")
                    else 100000
                ),
                data_min=0,
                data_avg=(
                    self.world_state.get_average_attribute_value("wealth_money")
                    if self.world_state.get_average_attribute_value("wealth_money")
                    else 50000
                ),
                data_std=(
                    self.world_state.get_stddev_attribute_value("wealth_money")
                    if self.world_state.get_stddev_attribute_value("wealth_money")
                    else 25000
                ),
            )

            partner_wealth_money = tanh_scaling_raw(
                partner_wealth_money,
                data_max=(
                    self.world_state.get_maximum_attribute_value("wealth_money")
                    if self.world_state.get_maximum_attribute_value("wealth_money")
                    else 100000
                ),
                data_min=0,
                data_avg=(
                    self.world_state.get_average_attribute_value("wealth_money")
                    if self.world_state.get_average_attribute_value("wealth_money")
                    else 50000
                ),
                data_std=(
                    self.world_state.get_stddev_attribute_value("wealth_money")
                    if self.world_state.get_stddev_attribute_value("wealth_money")
                    else 25000
                ),
            )
        
        wealth_differenceab = abs(wealth_money - partner_wealth_money)

        beauty_motive = char1.get_motives().get_beauty_motive().score
        beauty_motive = cached_sigmoid_motive_scale_approx_optimized(beauty_motive)

        partner_beauty_motive = char2.get_motives().get_beauty_motive().score
        partner_beauty_motive = cached_sigmoid_motive_scale_approx_optimized(
            partner_beauty_motive
        )

        # Get other motives
        luxury_motive = char1.get_motives().get_luxury_motive().score
        luxury_motive = cached_sigmoid_motive_scale_approx_optimized(luxury_motive)
        partner_luxury_motive = char2.get_motives().get_luxury_motive().score
        partner_luxury_motive = cached_sigmoid_motive_scale_approx_optimized(
            partner_luxury_motive
        )

        stability_motive = char1.get_motives().get_stability_motive().score
        stability_motive = cached_sigmoid_motive_scale_approx_optimized(
            stability_motive
        )
        partner_stability_motive = char2.get_motives().get_stability_motive().score
        partner_stability_motive = cached_sigmoid_motive_scale_approx_optimized(
            partner_stability_motive
        )

        control_motive = char1.get_motives().get_control_motive().score
        control_motive = cached_sigmoid_motive_scale_approx_optimized(control_motive)
        partner_control_motive = char2.get_motives().get_control_motive().score
        partner_control_motive = cached_sigmoid_motive_scale_approx_optimized(
            partner_control_motive
        )

        # Calculate attraction factors
        partner_agreeableness = char2.personality_traits.get_agreeableness()
        factor_a1 = (control_motive - char2.get_control()) + partner_agreeableness
        factor_a2 = (partner_control_motive - char1.get_control()) + char1.personality_traits.get_agreeableness()

        factor_b1 = (wealth_motive * partner_wealth_money) - (
            abs(partner_luxury_motive - luxury_motive) * 0.1
        )
        factor_b2 = (partner_wealth_motive * wealth_money) - (
            abs(luxury_motive - partner_luxury_motive) * 0.1
        )

        # Calculate basic libido and attraction
        base_libido = char1.get_base_libido()
        libido = (
            base_libido
            + ((romance_value / 10) * romance_compat)
            + char2.beauty
            + char1.energy
            + char1.get_motives().get_family_motive().score
        )
        
        # Normalize libido if world_state is available
        if self.world_state and hasattr(self.world_state, 'get_maximum_attribute_value'):
            base_libido = tanh_scaling_raw(
                base_libido,
                data_max=(
                    self.world_state.get_maximum_attribute_value("base_libido")
                    if self.world_state.get_maximum_attribute_value("base_libido")
                    else 100
                ),
                data_min=0,
                data_avg=(
                    self.world_state.get_average_attribute_value("base_libido")
                    if self.world_state.get_average_attribute_value("base_libido")
                    else 50
                ),
                data_std=(
                    self.world_state.get_stddev_attribute_value("base_libido")
                    if self.world_state.get_stddev_attribute_value("base_libido")
                    else 25
                ),
            )

        # Calculate various compatibility differences
        age_difference = abs(char1.age - char2.age)
        beauty_difference = abs(char1.beauty - char2.beauty)
        control_difference = abs(char1.get_control() - char2.get_control())
        stability_difference = abs(char1.stability - char2.stability)
        agreeableness_difference = abs(char1.personality_traits.get_agreeableness() - char2.personality_traits.get_agreeableness())
        success_difference = abs(char1.success - char2.success)
        shelter_difference = abs(char1.shelter - char2.shelter)
        luxury_difference = abs(char1.luxury - char2.luxury)
        monogamy_difference = abs(char1.monogamy - char2.monogamy)

        # Return a simplified romance interest calculation
        # This is a basic implementation - could be expanded with more sophisticated logic
        base_interest = romance_compat * 0.4
        motive_compatibility = (family_motive + partner_family_motive) / 20 * 0.3
        wealth_factor = (1 - wealth_differenceab) * 0.2
        personality_factor = (1 - agreeableness_difference / 10) * 0.1
        
        romance_interest = base_interest + motive_compatibility + wealth_factor + personality_factor
        return max(0.0, min(1.0, romance_interest))

    def calculate_social_influence(
        self,
        character,
        influence_attributes=None,
        memory_topic=None,
        memory_weight=0.5,
        decision_context=None,
        influence_factors=None,
    ):
        """
        Calculate social influence on a character's decisions based on relationships.
        
        Args:
            character: The character node/object to calculate influence for
            influence_attributes: List of edge attributes and weights to consider
            memory_topic: Topic to query from memories
            memory_weight: Weight for memory influence
            decision_context: Context for the decision being influenced
            influence_factors: Custom influence factors for different relationship types
            
        Returns:
            float: Weighted influence score affecting character's decisions
        """
        if not self.world_state:
            return 0
            
        total_influence = 0
        relationships = self._get_neighbors_with_attributes(character, type="character")
        memory_influence = 0
        
        if influence_factors is None:
            influence_factors = {
                "friend": {
                    "weight": 1,
                    "attributes": {
                        "trust": 1,
                        "historical": 1,
                        "interaction_frequency": 1,
                        "emotional_impact": 1,
                    },
                },
                "enemy": {
                    "weight": -1,
                    "attributes": {
                        "trust": -1,
                        "conflict": 1,
                        "historical": 0.4,
                        "interaction_frequency": 0.4,
                        "emotional_impact": -1,
                    },
                },
                "neutral": {"weight": 0, "attributes": {"trust": 0}},
            }
            
        if influence_attributes is None:
            influence_attributes = [("trust", 1)]

        # Calculate social influence based on current relationships
        for charnode in relationships:
            if not self._has_edge(character, charnode):
                continue
                
            relationship_type = self._get_edge_attribute(character, charnode, "relationship_type")
            attributes = self._get_edge_attributes(character, charnode)
            
            if relationship_type in influence_factors:
                relationship_factor = influence_factors[relationship_type]
                relationship_weight = relationship_factor["weight"]
                
                # Calculate attribute-based influence
                attribute_influence = 0
                for attr_name, attr_weight in influence_attributes:
                    if attr_name in attributes:
                        attr_value = attributes[attr_name]
                        if attr_name in relationship_factor["attributes"]:
                            factor_weight = relationship_factor["attributes"][attr_name]
                            attribute_influence += attr_value * attr_weight * factor_weight
                
                total_influence += relationship_weight * attribute_influence

        return total_influence

    def retrieve_characters_relationships(self, character):
        """
        Retrieve all relationships of a given character.
        
        Args:
            character: The character node identifier
            
        Returns:
            dict: Dictionary with each connected character and relationship details
        """
        if not self.world_state:
            return {}
            
        relationships = {}
        neighbors = self._get_neighbors(character)
        
        for neighbor in neighbors:
            if self._has_edge(character, neighbor):
                relationships[neighbor] = self._get_edge_attributes(character, neighbor)
                
        return relationships

    def update_relationship_status(self, char1, char2, update_info):
        """
        Update relationship status between two characters.
        
        Args:
            char1: First character node identifier
            char2: Second character node identifier  
            update_info: Dictionary with attributes to update
        """
        if not self.world_state or not self._has_edge(char1, char2):
            return
            
        for key, value in update_info.items():
            current_value = self._get_edge_attribute(char1, char2, key)
            if current_value is not None:
                new_value = current_value + value
            else:
                new_value = value
            self._set_edge_attribute(char1, char2, key, new_value)

    def analyze_character_relationships(self, character_id):
        """
        Analyze relationships for a specific character.
        
        Args:
            character_id: The character to analyze relationships for
            
        Returns:
            dict: Analysis of the character's relationships
        """
        if not self.world_state:
            return {}
            
        relationships = self.retrieve_characters_relationships(character_id)
        analysis = {
            "total_relationships": len(relationships),
            "relationship_types": {},
            "average_trust": 0,
            "average_emotional_impact": 0,
            "strongest_relationships": [],
            "weakest_relationships": []
        }
        
        if not relationships:
            return analysis
            
        trust_values = []
        emotional_values = []
        strength_values = []
        
        for neighbor, attrs in relationships.items():
            rel_type = attrs.get("relationship_type", "unknown")
            analysis["relationship_types"][rel_type] = analysis["relationship_types"].get(rel_type, 0) + 1
            
            trust = attrs.get("trust", 0)
            emotional = attrs.get("emotional", 0)
            strength = attrs.get("strength", 0)
            
            trust_values.append(trust)
            emotional_values.append(emotional)
            strength_values.append((neighbor, strength))
            
        # Calculate averages
        if trust_values:
            analysis["average_trust"] = sum(trust_values) / len(trust_values)
        if emotional_values:
            analysis["average_emotional_impact"] = sum(emotional_values) / len(emotional_values)
            
        # Find strongest and weakest relationships
        strength_values.sort(key=lambda x: x[1], reverse=True)
        analysis["strongest_relationships"] = strength_values[:3]
        analysis["weakest_relationships"] = strength_values[-3:]
        
        return analysis

    def analyze_relationship_health(self, char1, char2):
        """
        Analyze the health of a relationship between two characters.
        
        Args:
            char1: First character
            char2: Second character
            
        Returns:
            dict: Relationship health analysis
        """
        if not self.world_state or not self._has_edge(char1, char2):
            return {"health_score": 0, "status": "no_relationship"}
            
        attrs = self._get_edge_attributes(char1, char2)
        
        trust = attrs.get("trust", 0)
        emotional = attrs.get("emotional", 0)  
        strength = attrs.get("strength", 0)
        historical = attrs.get("historical", 0)
        interaction_frequency = attrs.get("interaction_frequency", 0)
        
        # Calculate health score
        health_components = [
            trust * 0.3,
            max(0, emotional) * 0.2,  # Only positive emotional impact counts towards health
            strength * 0.2,
            min(historical / 100, 1) * 0.15,  # Normalize historical
            interaction_frequency * 0.15
        ]
        
        health_score = sum(health_components)
        
        # Determine status
        if health_score >= 0.8:
            status = "excellent"
        elif health_score >= 0.6:
            status = "good"
        elif health_score >= 0.4:
            status = "fair"
        elif health_score >= 0.2:
            status = "poor"
        else:
            status = "critical"
            
        return {
            "health_score": health_score,
            "status": status,
            "components": {
                "trust": trust,
                "emotional_impact": emotional,
                "strength": strength, 
                "historical": historical,
                "interaction_frequency": interaction_frequency
            }
        }

    def evaluate_relationship_strength(self, char1, char2):
        """
        Evaluate the overall strength of a relationship.
        
        Args:
            char1: First character
            char2: Second character
            
        Returns:
            float: Relationship strength score
        """
        if not self.world_state or not self._has_edge(char1, char2):
            return 0
            
        attrs = self._get_edge_attributes(char1, char2)
        
        trust = attrs.get("trust", 0)
        emotional = attrs.get("emotional", 0)
        historical = attrs.get("historical", 0)
        interaction_frequency = attrs.get("interaction_frequency", 0)
        
        # Weight different factors for relationship strength
        strength = (
            trust * 0.4 +
            abs(emotional) * 0.3 +  # Both positive and negative emotions contribute to strength
            min(historical / 100, 1) * 0.2 +
            interaction_frequency * 0.1
        )
        
        return min(1.0, strength)

    # Helper methods to interface with world_state
    def _get_neighbors(self, node):
        """Get neighbors of a node from world_state."""
        try:
            if hasattr(self.world_state, 'neighbors'):
                return list(self.world_state.neighbors(node))
            elif hasattr(self.world_state, 'G'):
                return list(self.world_state.G.neighbors(node))
        except:
            # Node doesn't exist or has no neighbors
            pass
        return []
        
    def _get_neighbors_with_attributes(self, node, type=None):
        """Get neighbors with specific attributes."""
        neighbors = self._get_neighbors(node)
        if type:
            # Filter by node type if specified
            filtered = []
            for neighbor in neighbors:
                node_attrs = self._get_node_attributes(neighbor)
                if node_attrs.get("type") == type:
                    filtered.append(neighbor)
            return filtered
        return neighbors
        
    def _has_edge(self, node1, node2):
        """Check if edge exists between two nodes."""
        if hasattr(self.world_state, 'has_edge'):
            return self.world_state.has_edge(node1, node2)
        elif hasattr(self.world_state, 'G'):
            return self.world_state.G.has_edge(node1, node2)
        return False
        
    def _get_edge_attributes(self, node1, node2):
        """Get all edge attributes between two nodes."""
        if hasattr(self.world_state, 'get_edge_data'):
            data = self.world_state.get_edge_data(node1, node2)
            return data if data else {}
        elif hasattr(self.world_state, 'G'):
            if self.world_state.G.has_edge(node1, node2):
                return dict(self.world_state.G[node1][node2])
        return {}
        
    def _get_edge_attribute(self, node1, node2, attr_name):
        """Get specific edge attribute."""
        attrs = self._get_edge_attributes(node1, node2)
        return attrs.get(attr_name)
        
    def _set_edge_attribute(self, node1, node2, attr_name, value):
        """Set specific edge attribute."""
        if hasattr(self.world_state, 'G') and self.world_state.G.has_edge(node1, node2):
            self.world_state.G[node1][node2][attr_name] = value
            
    def _get_node_attributes(self, node):
        """Get all node attributes."""
        if hasattr(self.world_state, 'nodes'):
            return self.world_state.nodes.get(node, {})
        elif hasattr(self.world_state, 'G'):
            return self.world_state.G.nodes.get(node, {})
        return {}