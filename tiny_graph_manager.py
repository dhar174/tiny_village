import heapq
from math import inf
import networkx as nx
from numpy import add
from pkg_resources import add_activation_listener
from tiny_characters import Character
from tiny_locations import Location
from tiny_event_handler import EventHandler
from actions import Action
from tiny_items import ItemObject
import tiny_memories

""" Graph Construction
Defining Nodes:
Characters: Each character in the game will be a node. This includes not only playable characters but also non-playable characters (NPCs).
Locations: Places in the game such as homes, workplaces, public spaces, and event locations.
Events: Significant events that can affect character decisions and relationships, like festivals, job openings, etc.
Interests and Activities: Specific hobbies or tasks that characters can engage in, impacting their skills and relationships.
Defining Edges:
Relationships Between Characters: These edges represent different types of relationships (friends, family, colleagues, antagonists) and their strength or status.
Character-Location Relationships: Connections between characters and locations (e.g., owner, frequent visitor, employee).
Character-Event Relationships: How characters are involved or affected by events (participant, organizer, bystander).
Character-Interest Relationships: Links between characters and their interests or activities, which could influence their skills and social connections.
Edge Attributes:
Strength of Relationship: Quantitative measure of how strong a relationship is, which can affect decision-making.
Nature of Interaction: Attributes that describe the type of interaction (positive, negative, neutral).
Frequency of Interaction: How often characters interact at a location or through an event, impacting the character's routine and decisions. 


1. Nodes Categorization
Nodes represent entities within the game world. Each type of node will have specific attributes based on its role and interactions.

Character Nodes
Attributes: Name, age, job, current status (e.g., happiness, energy level), long-term aspirations, skills.
Purpose: Represent the player and NPC dynamics within the game.
Location Nodes
Attributes: Name, type (e.g., cafe, park, workplace), popularity, activities available.
Purpose: Represent places characters can visit or interact with.
Event Nodes
Attributes: Name, type (e.g., festival, job fair), date, significance.
Purpose: Represent scheduled or random occurrences that characters can participate in.
Object Nodes
Attributes: Name, type (e.g., book, tool), value, usability.
Purpose: Represent items that characters can own, use, or interact with.
Activity Nodes
Attributes: Name, type (e.g., exercise, study), related skill, satisfaction level.
Purpose: Represent actions characters can undertake for personal development, leisure, or job-related tasks.
2. Edges Categorization
Edges represent relationships or interactions between nodes, with attributes that describe the nature and dynamics of these relationships.

Character-Character Edges
Attributes: Relationship type (e.g., friend, family, colleague), strength, history.
Purpose: Represent social and professional dynamics between characters.
Character-Location Edges
Attributes: Frequency of visits, last visit, favorite activities.
Purpose: Indicate how often and why characters visit certain locations.
Character-Event Edges
Attributes: Participation status, role (e.g., organizer, attendee), impact.
Purpose: Reflect characters' involvement in events and their effects.
Character-Object Edges
Attributes: Ownership status, usage frequency, sentimental value.
Purpose: Represent characters' possessions and how they use or value them.
Character-Activity Edges
Attributes: Engagement level, skill improvement, recentness.
Purpose: Show how characters engage in various activities and their effects.
3. Attributes Definition
Each node and edge will have attributes that need to be quantitatively or qualitatively defined to enable effective graph analyses.

Quantitative Attributes: Numeric values (e.g., strength of relationships, frequency of visits) that can be easily measured and calculated.
Qualitative Attributes: Descriptive characteristics (e.g., type of relationship, role in an event) that provide context for decision-making processes.
4. Graph Dynamics
The graph should be dynamic, allowing for updates based on characters' actions and game events. This involves:

Object-Object Edges
Attributes: Compatibility (e.g., items that are parts of a set), combinability (e.g., items that can be used together like ingredients in a recipe), conflict (e.g., items that negate each other’s effects).
Purpose: These edges can represent how different objects can interact or be used together, enhancing gameplay depth.
Object-Activity Edges
Attributes: Necessity (e.g., tools required for an activity), enhancement (e.g., items that improve the effectiveness or outcomes of an activity), obstruction (e.g., items that hinder an activity).
Purpose: Reflect how objects are utilized in activities, influencing choices in gameplay related to skill development or task completion.
Activity-Activity Edges
Attributes: Synergy (e.g., activities that enhance each other, like studying then practicing a skill), conflict (e.g., activities that interfere with each other, like noisy and quiet activities happening concurrently), dependency (e.g., prerequisite activities).
Purpose: Indicate relationships between activities that might affect scheduling, planning, or the strategic development of skills and character growth.
Location-Activity Edges
Attributes: Suitability (e.g., appropriateness of a location for certain activities, such as a park for jogging), popularity (e.g., how popular an activity is at a location, which could affect social interactions or congestion), exclusivity (e.g., activities only available at specific locations).
Purpose: Show how locations support or are associated with different activities, influencing character decisions about where to go for various tasks.
Location-Location Edges
Attributes: Proximity (e.g., how close locations are to each other), connectivity (e.g., ease of traveling between locations, such as direct bus routes), rivalry (e.g., locations competing for the same visitors or resources).
Purpose: These edges can help model travel decisions, area popularity, and strategic choices regarding where characters spend their time.
Implementing These Interactions in the Graph
To effectively use these interactions within the game, you would set up your graph database to store and update these relationships dynamically. Here’s how you might approach this:

Dynamic Interaction Updates: As characters use objects in various activities or visit different locations, update the object-activity and location-activity edges to reflect changes in necessity, enhancement, or suitability.
Real-Time Feedback: Use feedback from these interactions to adjust the attributes dynamically. For example, if an activity at a location becomes extremely popular, increase the congestion attribute, which might affect future decisions by characters to visit.
Complex Decision-Making: Integrate these detailed relationships into the decision-making algorithms for GOAP and utility evaluations. For example, consider object necessity and activity synergy when planning a character’s daily schedule or career path.

Adding/Removing Nodes and Edges: As new characters, locations, or items are introduced or removed.
Updating Attributes: As characters develop or relationships evolve.
By systematically defining these elements, we can ensure that the graph not only represents the complexity of the game world accurately but also supports the AI’s ability to make nuanced decisions. This structure will serve as the foundation for implementing the graph-based analyses and decision-making processes

3. GOAP System and Graph Analysis
Where it happens: goap_system.py and graph_manager.py
What happens: The GOAP planner uses the graph to analyze relationships and preferences, and formulates a plan consisting of a sequence of actions that maximize the character’s utility for the day.
"""

import networkx as nx


import networkx as nx
from networkx.algorithms import community


class GraphManager:
    def __init__(self):
        self.characters = []
        self.locations = []
        self.objects = []
        self.events = []
        self.activities = []
        self.jobs = []
        self.G = None
        self.graph = self.initialize_graph()

    def initialize_graph(self):
        self.G = (
            nx.MultiDiGraph()
        )  # Using MultiDiGraph for directional and multiple edges
        return self.G

    # Node Addition Methods
    def add_character_node(self, char):
        self.G.add_node(
            char.name,
            type="character",
            age=char.age,
            job=char.job,
            happiness=char.happiness,
            energy_level=char.energy_level,
            relationships={},  # Stores additional details about relationships
            emotional_state={},
            coordinate_location=char.coordinate_location,
        )

    def add_location_node(self, loc):
        self.G.add_node(
            loc.name,
            type="location",
            popularity=loc.popularity,
            activities_available=loc.activities_available,
            accessibility=loc.accessibility,
            safety_measures=loc.safety_measures,
            coordinate_location=loc.coordinate_location,
        )

    def add_event_node(self, event):
        self.G.add_node(
            event.name,
            type="event",
            date=event.date,
            importance=event.importance,
            impact=event.impact,
            required_items=event.required_items,
            coordinate_location=event.coordinate_location,
        )

    def add_object_node(self, obj):
        self.G.add_node(
            obj.name,
            type="object",
            value=obj.value,
            usability=obj.usability,
            sentimental_value=obj.sentimental_value,
            trade_value=obj.trade_value,
            scarcity=obj.scarcity,
            coordinate_location=obj.coordinate_location,
        )

    def add_activity_node(self, act):
        self.G.add_node(
            act.name,
            type="activity",
            related_skill=act.related_skill,
            satisfaction_level=act.satisfaction_level,
            necessary_tools=act.necessary_tools,
            conflict_activities=act.conflict_activities,
            dependency_activities=act.dependency_activities,
            coordinate_location=act.coordinate_location,
        )

    def add_job_node(self, job):
        self.G.add_node(
            job.name,
            type="job",
            required_skills=job.required_skills,
            location=job.location,
            salary=job.salary,
        )

    # Edge Addition Methods with Detailed Attributes
    # Character-Character
    def add_character_character_edge(
        self,
        char1,
        char2,
        relationship_type,
        strength,
        history,
        emotional_impact,
        interaction_frequency,
    ):
        self.G.add_edge(
            char1,
            char2,
            type="social",
            relationship_type=relationship_type,
            strength=strength,
            history=history,
            emotional_impact=emotional_impact,
            interaction_frequency=interaction_frequency,
        )

    # Character-Location
    def add_character_location_edge(
        self,
        char,
        loc,
        frequency_of_visits,
        last_visit,
        favorite_activities,
        ownership_status,
    ):
        self.G.add_edge(
            char,
            loc,
            type="visits",
            frequency_of_visits=frequency_of_visits,
            last_visit=last_visit.strftime("%Y-%m-%d"),
            favorite_activities=favorite_activities,
            ownership_status=ownership_status,
        )

    # Character-Item
    def add_character_object_edge(
        self,
        char,
        obj,
        ownership_status,
        usage_frequency,
        sentimental_value,
        last_used_time,
    ):
        self.G.add_edge(
            char,
            obj,
            type="ownership",
            ownership_status=ownership_status,
            usage_frequency=usage_frequency,
            sentimental_value=sentimental_value,
            last_used_time=last_used_time.strftime("%Y-%m-%d"),
        )

    # Character-Event
    def add_character_event_edge(
        self,
        char,
        event,
        participation_status,
        role,
        impact_on_character,
        emotional_outcome,
    ):
        self.G.add_edge(
            char,
            event,
            type="participation",
            participation_status=participation_status,
            role=role,
            impact_on_character=impact_on_character,
            emotional_outcome=emotional_outcome,
        )

    # Character-Activity
    def add_character_activity_edge(
        self,
        char,
        act,
        engagement_level,
        skill_improvement,
        activity_frequency,
        motivation,
    ):
        self.G.add_edge(
            char,
            act,
            type="engagement",
            engagement_level=engagement_level,
            skill_improvement=skill_improvement,
            activity_frequency=activity_frequency,
            motivation=motivation,
        )

        # Location-Location Edges

    def add_location_location_edge(
        self, loc1, loc2, proximity, connectivity, rivalry, trade_relations
    ):
        self.G.add_edge(
            loc1,
            loc2,
            type="connectivity",
            proximity=proximity,
            connectivity=connectivity,
            rivalry=rivalry,
            trade_relations=trade_relations,
        )

    # Location-Item Edges
    def add_location_item_edge(self, loc, obj, item_presence, item_relevance):
        self.G.add_edge(
            loc,
            obj,
            type="contains",
            item_presence=item_presence,
            item_relevance=item_relevance,
        )

    # Location-Event Edges
    def add_location_event_edge(
        self, loc, event, event_occurrence, location_role, capacity, preparation_level
    ):
        self.G.add_edge(
            loc,
            event,
            type="hosting_event",
            event_occurrence=event_occurrence,
            location_role=location_role,
            capacity=capacity,
            preparation_level=preparation_level,
        )

    # Location-Activity Edges
    def add_location_activity_edge(
        self, loc, act, activity_suitability, activity_popularity, exclusivity
    ):
        self.G.add_edge(
            loc,
            act,
            type="activity_location",
            activity_suitability=activity_suitability,
            activity_popularity=activity_popularity,
            exclusivity=exclusivity,
        )

    # Item-Item Edges
    def add_item_item_edge(self, obj1, obj2, compatibility, conflict, combinability):
        self.G.add_edge(
            obj1,
            obj2,
            type="item_interaction",
            compatibility=compatibility,
            conflict=conflict,
            combinability=combinability,
        )

    # Item-Activity Edges
    def add_item_activity_edge(self, obj, act, necessity, enhancement, obstruction):
        self.G.add_edge(
            obj,
            act,
            type="item_usage",
            necessity=necessity,
            enhancement=enhancement,
            obstruction=obstruction,
        )

    # Event-Activity Edges
    def add_event_activity_edge(
        self, event, act, activities_involved, activity_impact, activity_requirements
    ):
        self.G.add_edge(
            event,
            act,
            type="event_activity",
            activities_involved=activities_involved,
            activity_impact=activity_impact,
            activity_requirements=activity_requirements,
        )

    # Event-Item Edges
    def add_event_item_edge(
        self, event, obj, required_for_event, item_usage, item_impact
    ):
        self.G.add_edge(
            event,
            obj,
            type="event_item",
            required_for_event=required_for_event,
            item_usage=item_usage,
            item_impact=item_impact,
        )

    # Activity-Activity Edges
    def add_activity_activity_edge(self, act1, act2, synergy, conflict, dependency):
        self.G.add_edge(
            act1,
            act2,
            type="activity_relation",
            synergy=synergy,
            conflict=conflict,
            dependency=dependency,
        )

    # Additional Job-Related Edges
    # Character-Job Edges
    def add_character_job_edge(self, char, job, role, job_status, job_performance):
        self.G.add_edge(
            char,
            job,
            type="employment",
            role=role,
            job_status=job_status,
            job_performance=job_performance,
        )

    # Job-Location Edges
    def add_job_location_edge(self, job, loc, essential_for_job, location_dependence):
        self.G.add_edge(
            job,
            loc,
            type="job_location",
            essential_for_job=essential_for_job,
            location_dependence=location_dependence,
        )

    # Job-Activity Edges
    def add_job_activity_edge(
        self, job, act, activity_necessity, performance_enhancement
    ):
        self.G.add_edge(
            job,
            act,
            type="job_activity",
            activity_necessity=activity_necessity,
            performance_enhancement=performance_enhancement,
        )

    # Adding temporal, emotional, economic, historical, and security attributes dynamically
    def add_temporal_edge_attribute(self, node1, node2, temporal_data):
        self.G[node1][node2]["temporal"] = temporal_data

    def add_emotional_edge_attribute(self, node1, node2, emotional_data):
        self.G[node1][node2]["emotional"] = emotional_data

    def add_economic_edge_attribute(self, node1, node2, economic_data):
        self.G[node1][node2]["economic"] = economic_data

    def add_historical_edge_attribute(self, node1, node2, historical_data):
        self.G[node1][node2]["historical"] = historical_data

    def add_security_edge_attribute(self, node1, node2, security_data):
        self.G[node1][node2]["security"] = security_data

        # Enhanced Character-Character Edges

    def add_enhanced_character_character_edge(
        self, char1, char2, shared_experiences, mutual_relations
    ):
        self.G.add_edge(
            char1,
            char2,
            type="enhanced_social",
            shared_experiences=shared_experiences,
            mutual_relations=mutual_relations,
        )

    # Enhanced Character-Location Edges
    def add_enhanced_character_location_edge(
        self, char, loc, emotional_attachment, significant_events
    ):
        self.G.add_edge(
            char,
            loc,
            type="enhanced_visit",
            emotional_attachment=emotional_attachment,
            significant_events=significant_events,
        )

    # Enhanced Character-Item Edges
    def add_enhanced_character_item_edge(
        self, char, obj, items_exchanged, items_lost_found
    ):
        self.G.add_edge(
            char,
            obj,
            type="enhanced_ownership",
            items_exchanged=items_exchanged,
            items_lost_found=items_lost_found,
        )

    # Enhanced Character-Event Edges
    def add_enhanced_character_event_edge(self, char, event, anticipations, memories):
        self.G.add_edge(
            char,
            event,
            type="enhanced_participation",
            anticipations=anticipations,
            memories=memories,
        )

    # Enhanced Character-Activity Edges
    def add_enhanced_character_activity_edge(self, char, act, aversions, aspirations):
        self.G.add_edge(
            char,
            act,
            type="enhanced_engagement",
            aversions=aversions,
            aspirations=aspirations,
        )

    # Enhanced Location-Location Edges
    def add_enhanced_location_location_edge(
        self, loc1, loc2, historical_links, environmental_factors
    ):
        self.G.add_edge(
            loc1,
            loc2,
            type="enhanced_connectivity",
            historical_links=historical_links,
            environmental_factors=environmental_factors,
        )

    # Enhanced Location-Item Edges
    def add_enhanced_location_item_edge(self, loc, obj, items_history, symbolic_items):
        self.G.add_edge(
            loc,
            obj,
            type="enhanced_contains",
            items_history=items_history,
            symbolic_items=symbolic_items,
        )

    # Enhanced Location-Event Edges
    def add_enhanced_location_event_edge(
        self, loc, event, recurring_events, historic_impact
    ):
        self.G.add_edge(
            loc,
            event,
            type="enhanced_hosting_event",
            recurring_events=recurring_events,
            historic_impact=historic_impact,
        )

    # Enhanced Location-Activity Edges
    def add_enhanced_location_activity_edge(
        self, loc, act, prohibitions, historical_activities
    ):
        self.G.add_edge(
            loc,
            act,
            type="enhanced_activity_location",
            prohibitions=prohibitions,
            historical_activities=historical_activities,
        )

    # Enhanced Item-Item Edges
    def add_enhanced_item_item_edge(self, obj1, obj2, part_of_set, usage_combinations):
        self.G.add_edge(
            obj1,
            obj2,
            type="enhanced_item_interaction",
            part_of_set=part_of_set,
            usage_combinations=usage_combinations,
        )

    # Enhanced Item-Activity Edges
    def add_enhanced_item_activity_edge(
        self, obj, act, damage_risks, repair_opportunities
    ):
        self.G.add_edge(
            obj,
            act,
            type="enhanced_item_usage",
            damage_risks=damage_risks,
            repair_opportunities=repair_opportunities,
        )

    # Enhanced Event-Activity Edges
    def add_enhanced_event_activity_edge(
        self, event, act, preventions_triggers, traditional_activities
    ):
        self.G.add_edge(
            event,
            act,
            type="enhanced_event_activity",
            preventions_triggers=preventions_triggers,
            traditional_activities=traditional_activities,
        )

    # Enhanced Event-Item Edges
    def add_enhanced_event_item_edge(
        self, event, obj, event_triggers, traditional_uses
    ):
        self.G.add_edge(
            event,
            obj,
            type="enhanced_event_item",
            event_triggers=event_triggers,
            traditional_uses=traditional_uses,
        )

    # Enhanced Activity-Activity Edges
    def add_enhanced_activity_activity_edge(self, act1, act2, exclusivity, sequences):
        self.G.add_edge(
            act1,
            act2,
            type="enhanced_activity_relation",
            exclusivity=exclusivity,
            sequences=sequences,
        )

    def find_shortest_path(self, source, target):
        """
        Returns the shortest path between source and target nodes using Dijkstra's algorithm.

        Parameters:
            source (str): Node identifier for the source node.
            target (str): Node identifier for the target node.

        Returns:
            list or None: List of nodes representing the shortest path or None if no path exists.

        Usage example:
            path = graph_manager.find_shortest_path('char1', 'char2')
            if path:
                print("Path found:", path)
            else:
                print("No path exists between the characters.")
        """
        try:
            path = nx.shortest_path(self.G, source=source, target=target)
            return path
        except nx.NetworkXNoPath:
            return None

    def detect_communities(self):
        """
        Detects communities within the graph using the Louvain method for community detection.

        Returns:
            list of sets: A list where each set contains the nodes that form a community.

        Usage example:
            communities = graph_manager.detect_communities()
            print("Detected communities:", communities)
        """
        communities = community.louvain_communities(self.G, weight="weight")
        return communities

    def calculate_centrality(self):
        """
        Calculates and returns centrality measures for nodes in the graph, useful for identifying
        key influencers or central nodes within the network.

        Returns:
            dict: A dictionary where keys are node identifiers and values are centrality scores.

        Usage example:
            centrality = graph_manager.calculate_centrality()
            print("Centrality scores:", centrality)
        """
        centrality = nx.degree_centrality(self.G)
        return centrality

    def shortest_path_between_characters(self, char1, char2):
        """
        Find the most direct connection or interaction chain between two characters, which can be useful
        for understanding potential influences or conflicts.

        Parameters:
            char1 (str): Node identifier for the first character.
            char2 (str): Node identifier for the second character.

        Returns:
            list or None: List of characters forming the path or None if no path exists.

        Usage example:
            path = graph_manager.shortest_path_between_characters('char1', 'char3')
            print("Direct interaction chain:", path)
        """
        return self.find_shortest_path(char1, char2)

    def common_interests_cluster(self):
        """
        Identify clusters of characters that share common interests, which can be used to form groups
        or communities within the game.

        Returns:
            list of sets: Each set contains characters that share common interests.

        Usage example:
            interest_clusters = graph_manager.common_interests_cluster()
            print("Clusters based on common interests:", interest_clusters)
        """

        # Assuming 'interests' is a node attribute containing a set of interests for each character
        def shared_interests(node1, node2):
            return len(
                set(self.G.nodes[node1]["interests"])
                & set(self.G.nodes[node2]["interests"])
            )

        clusters = community.greedy_modularity_communities(
            self.G, weight=shared_interests
        )
        return clusters

    def most_influential_character(self):
        """
        Identify the character who has the most connections or the highest influence scores with others,
        which could be used to simulate social dynamics.

        Returns:
            str: Node identifier of the most influential character.

        Usage example:
            influencer = graph_manager.most_influential_character()
            print("Most influential character:", influencer)
        """
        centrality = nx.degree_centrality(self.G)
        most_influential = max(centrality, key=centrality.get)
        return most_influential

    def expanded_event_impact_analysis(self, event_node):
        """
        Analyze the broader range of impacts from an event, including long-term changes in relationships
        and character development.

        Parameters:
            event_node (str): Node identifier for the event.

        Returns:
            dict: A dictionary detailing the impacts on each character linked to the event.

        Usage example:
            impacts = graph_manager.expanded_event_impact_analysis('event1')
            print("Event impacts:", impacts)
        """
        impacts = {}
        for node in self.G.nodes:
            if self.G.has_edge(node, event_node):
                impacts[node] = {
                    "emotional": self.G[node][event_node].get("emotional", 0),
                    "historical": self.G[node][event_node].get("historical", 0),
                }
        return impacts

    def retrieve_characters_relationships(self, character):
        """
        Retrieves all relationships of a given character, showing the type and strength of each.

        Parameters:
            character (str): The character node identifier.

        Returns:
            dict: A dictionary with each connected character and details of the relationship.

        Usage example:
            relationships = graph_manager.retrieve_characters_relationships('char1')
            print("Character relationships:", relationships)
        """
        relationships = {
            neighbor: self.G[character][neighbor]
            for neighbor in self.G.neighbors(character)
        }
        return relationships

    def update_relationship_status(self, char1, char2, update_info):
        """
        Updates the relationship status between two characters, modifying attributes like emotional or historical data.

        Parameters:
            char1 (str): Node identifier for the first character.
            char2 (str): Node identifier for the second character.
            update_info (dict): A dictionary with the attributes to update.

        Usage example:
            update_info = {'emotional': 5, 'historical': 10}
            graph_manager.update_relationship_status('char1', 'char2', update_info)
        """
        if self.G.has_edge(char1, char2):
            for key, value in update_info.items():
                if key in self.G[char1][char2]:
                    self.G[char1][char2][key] += value
                else:
                    self.G[char1][char2][key] = value

    def analyze_location_popularity(self):
        """
        Analyzes and ranks locations based on visitation frequency.

        Returns:
            dict: A dictionary where each location node is mapped to its visit count.

        Usage example:
            location_popularity = graph_manager.analyze_location_popularity()
            print("Location popularity:", location_popularity)
        """
        location_visits = {
            node: self.G.nodes[node]["visit_count"]
            for node in self.G.nodes(data=True)
            if "location" in node
        }
        popular_locations = dict(
            sorted(location_visits.items(), key=lambda item: item[1], reverse=True)
        )
        return popular_locations

    def item_ownership_network(self):
        """
        Analyzes the network of item ownership and transactions between characters.

        Returns:
            dict: A dictionary representing the flow of items between characters.

        Usage example:
            ownership_network = graph_manager.item_ownership_network()
            print("Item ownership network:", ownership_network)
        """
        ownership_network = {}
        for u, v, data in self.G.edges(data=True):
            if data.get("type") == "ownership":
                if u not in ownership_network:
                    ownership_network[u] = {}
                ownership_network[u][v] = data
        return ownership_network

    def transfer_item_ownership(self, item, from_char, to_char):
        """
        Transfers ownership of an item from one character to another.

        Parameters:
            item (str): Node identifier for the item.
            from_char (str): Node identifier for the character transferring the item.
            to_char (str): Node identifier for the recipient character.

        Usage example:
            graph_manager.transfer_item_ownership('item1', 'char1', 'char2')
            print(f"{item} has been transferred from {from_char} to {to_char}.")
        """
        if self.G.has_edge(from_char, item):
            self.G.remove_edge(from_char, item)
            self.G.add_edge(to_char, item, type="ownership")
        else:
            raise ValueError(f"The item {item} is not owned by {from_char}.")

    def analyze_character_relationships(self, character_id):
        """
        Analyzes the relationships of a specific character based on historical interactions and emotional bonds.

        Parameters:
            character_id (str): Node identifier for the character.

        Returns:
            dict: A dictionary containing relationship details with other characters.

        Usage example:
            relationships = graph_manager.analyze_character_relationships('char1')
            print("Relationship details:", relationships)
        """
        relationships = {}
        for neighbor in self.G.neighbors(character_id):
            relationships[neighbor] = {
                "emotional": self.G[character_id][neighbor].get("emotional", 0),
                "historical": self.G[character_id][neighbor].get("historical", 0),
            }
        return relationships

    def location_popularity_analysis(self):
        """
        Determines the popularity of locations based on visitation data.

        Returns:
            dict: A dictionary where keys are location IDs and values are the number of visits.

        Usage example:
            location_popularity = graph_manager.location_popularity_analysis()
            print("Location popularity:", location_popularity)
        """
        popularity = {}
        for node in self.G.nodes(data=True):
            if node[1].get("type") == "location":
                popularity[node[0]] = sum(1 for _ in self.G.edges(node[0]))
        return popularity

    def track_item_ownership(self):
        """
        Tracks ownership and transaction history of items.

        Returns:
            dict: A dictionary where keys are item IDs and values are details of ownership and transaction history.

        Usage example:
            item_ownership = graph_manager.track_item_ownership()
            print("Item ownership details:", item_ownership)
        """
        ownership = {}
        for node in self.G.nodes(data=True):
            if node[1].get("type") == "item":
                ownership[node[0]] = {
                    "current_owner": self.G.nodes[node[0]].get("owner"),
                    "transaction_history": self.G.nodes[node[0]].get(
                        "transaction_history", []
                    ),
                }
        return ownership

    def predict_future_relationships(self, character_id):
        """
        Predicts future interactions and relationships based on historical data.

        Parameters:
            character_id (str): Node identifier for the character.

        Returns:
            dict: Predictions of future relationships.

        Usage example:
            future_relationships = graph_manager.predict_future_relationships('char2')
            print("Predicted future relationships:", future_relationships)
        """
        # Placeholder for a more complex predictive model
        return {
            "info": "This feature requires additional predictive modeling capabilities."
        }

    def update_node_attribute(self, node, attribute, value):
        """
        Updates an attribute for a node with a new value.

        Parameters:
            node (str): Node identifier.
            attribute (str): Attribute to update.
            value (any): New value for the attribute.

        Usage example:
            graph_manager.update_node_attribute('char1', 'mood', 'happy')
        """
        if node in self.G:
            self.G.nodes[node][attribute] = value
        else:
            raise ValueError("Node does not exist in the graph.")

    def find_all_paths(self, source, target, max_length=None):
        """
        Finds all paths between source and target nodes up to a maximum length.

        Parameters:
            source (str): Node identifier for the source.
            target (str): Node identifier for the target.
            max_length (int): Maximum number of edges in the path (optional).

        Returns:
            list of lists: A list containing all paths, each path is a list of nodes.

        Usage example:
            paths = graph_manager.find_all_paths('char1', 'char2', max_length=5)
            print("All paths up to length 5:", paths)
        """
        paths = list(
            nx.all_simple_paths(self.G, source=source, target=target, cutoff=max_length)
        )
        return paths

    def node_influence_spread(self, node, decay_function=lambda x: 1 / x):
        """
        Calculates the influence spread of a node over others in the graph, with influence decreasing
        over distance based on a decay function.

        Parameters:
            node (str): Node identifier.
            decay_function (function): A function that defines how influence decays with distance.

        Returns:
            dict: A dictionary with node identifiers as keys and influence scores as values.

        Usage example:
            influence = graph_manager.node_influence_spread('char1')
            print("Influence spread of char1:", influence)
        """
        influence = {}
        for target in self.G.nodes:
            if target != node:
                paths = self.find_all_paths(node, target)
                for path in paths:
                    distance = len(path) - 1  # edges are one less than nodes in path
                    if target not in influence:
                        influence[target] = 0
                    influence[target] += decay_function(distance)
        return influence

    def analyze_relationship_health(self, char1, char2):
        """
        Analyzes the health of the relationship between two characters based on attributes like emotional and historical values.

        Parameters:
            char1 (str): Node identifier for the first character.
            char2 (str): Node identifier for the second character.

        Returns:
            float: A score representing the relationship health.

        Usage example:
            health_score = graph_manager.analyze_relationship_health('char1', 'char2')
            print("Relationship health score:", health_score)
        """
        if self.G.has_edge(char1, char2):
            emotional = self.G[char1][char2].get("emotional", 0)
            historical = self.G[char1][char2].get("historical", 0)
            # Combine these attributes to form a health score, example below
            return emotional * 0.75 + historical * 0.25
        else:
            return 0  # No relationship exists

    def update_node_attribute(self, node, attribute, value):
        """
        Updates or adds an attribute to a node.

        Parameters:
            node (str): The node identifier.
            attribute (str): The attribute to update or add.
            value (any): The new value for the attribute.

        Usage example:
            graph_manager.update_node_attribute('char1', 'mood', 'happy')
        """
        self.G.nodes[node][attribute] = value

    def update_edge_attribute(self, node1, node2, attribute, value):
        """
        Updates or adds an attribute to an edge between two nodes.

        Parameters:
            node1 (str): The first node identifier.
            node2 (str): The second node identifier.
            attribute (str): The attribute to update or add.
            value (any): The new value for the attribute.

        Usage example:
            graph_manager.update_edge_attribute('char1', 'char2', 'trust', 75)
        """
        self.G[node1][node2][attribute] = value

    def evaluate_relationship_strength(self, char1, char2):
        """
        Evaluate the strength of the relationship between two characters based on edge attributes.

        Parameters:
            char1 (str): First character node identifier.
            char2 (str): Second character node identifier.

        Returns:
            int: The cumulative strength of the relationship.

        Usage example:
            strength = graph_manager.evaluate_relationship_strength('char1', 'char2')
            print("Relationship strength:", strength)
        """
        attributes = ["trust", "history", "interaction"]
        strength = sum(self.G[char1][char2].get(attr, 0) for attr in attributes)
        return strength

    def check_friendship_status(self, char1, char2):
        """
        Determines the friendship status between two characters based on their historical interactions and emotional connections.

        Parameters:
            char1 (str): Node identifier for the first character.
            char2 (str): Node identifier for the second character.

        Returns:
            str: Describes the friendship status ('friends', 'neutral', or 'enemies').

        Usage example:
            status = graph_manager.check_friendship_status('char1', 'char2')
            print("Friendship status:", status)
        """
        if self.G.has_edge(char1, char2):
            emotional = self.G[char1][char2].get("emotional", 0)
            historical = self.G[char1][char2].get("historical", 0)
            if emotional > 50 and historical > 5:
                return "friends"
            elif emotional < -50 or historical < -5:
                return "enemies"
            else:
                return "neutral"
        return "neutral"

    def character_location_frequency(self, char):
        """
        Returns a dictionary detailing the frequency of visits a character has made to various locations.

        Parameters:
            char (str): Node identifier for the character.

        Returns:
            dict: Keys are locations and values are counts of visits.

        Usage example:
            frequency = graph_manager.character_location_frequency('char1')
            print("Location visit frequency:", frequency)
        """
        frequency = {}
        for edge in self.G.edges(char, data=True):
            if edge[2]["type"] == "visit":
                location = edge[1]
                if location in frequency:
                    frequency[location] += 1
                else:
                    frequency[location] = 1
        return frequency

    def location_popularity(self, location):
        """
        Determines the popularity of a location based on visit frequency by characters.

        Parameters:
            location (str): Location node identifier.

        Returns:
            int: Number of visits to the location.

        Usage example:
            popularity = graph_manager.location_popularity('park')
            print("Popularity of the location:", popularity)
        """
        return self.G.nodes[location].get("visit_count", 0)

    def item_ownership_history(self, item):
        """
        Tracks the history of ownership for a specific item.

        Parameters:
            item (str): Item node identifier.

        Returns:
            list: History of characters who have owned the item.

        Usage example:
            history = graph_manager.item_ownership_history('sword')
            print("Ownership history of the item:", history)
        """
        return self.G.nodes[item].get("ownership_history", [])

    def can_interact_directly(self, char1, char2, conditions=None):
        """
        Determines if two characters can interact directly based on a set of specified conditions.

        Parameters:
            char1 (str): Node identifier for the first character.
            char2 (str): Node identifier for the second character.
            conditions (dict): A dictionary where keys are attributes or conditions to check and
                            values are the required values for these attributes or conditions.
                            Conditions can specify node or edge attributes.

        Returns:
            bool: True if all conditions are met for interaction, False otherwise.

        Usage example:
            conditions = {
                'proximity': True,
                'relationship_status': 'friendly',
                'mood': ['happy', 'neutral']  # Character must be either happy or neutral
            }
            if graph_manager.can_interact_directly('char1', 'char2', conditions):
                print("Characters can interact.")
            else:
                print("Characters cannot interact.")
        """
        if not self.G.has_edge(char1, char2):
            return False

        # Check edge-specific conditions
        for key, required_value in conditions.items():
            if key in self.G[char1][char2]:
                value = self.G[char1][char2][key]
                if isinstance(required_value, list) and value not in required_value:
                    return False
                elif not isinstance(required_value, list) and value != required_value:
                    return False

        # Check node-specific conditions for both characters
        for char in [char1, char2]:
            for key, required_value in conditions.items():
                if key in self.G.nodes[char]:
                    value = self.G.nodes[char][key]
                    if isinstance(required_value, list) and value not in required_value:
                        return False
                    elif (
                        not isinstance(required_value, list) and value != required_value
                    ):
                        return False

        return True

    def get_nearest_resource(
        self,
        character,
        resource_filter,
        attribute_name="weight",
        max_search_depth=None,
        default_attribute_value=float("inf"),
    ):
        """
        Enhanced function to find the nearest resource to a character using a custom filter function or attribute criteria,
        considering a configurable edge attribute and optional max search depth for optimized performance.

        Parameters:
            character (str): Node identifier for the character.
            resource_filter (function or dict): Custom filter function or dictionary of attribute-value pairs defining the resource.
            attribute_name (str): Name of the edge attribute to use for path calculations (default is 'weight').
            max_search_depth (int): Optional maximum depth to search in the graph.
            default_attribute_value (float or int): Default value for the edge attribute if not present (default is infinity).

        Returns:
            tuple: Identifier of the nearest resource and the distance, or (None, None) if no resource is found.
        """
        is_filter_function = callable(resource_filter)
        to_visit = [(0, character)]
        visited = set()
        nearest_resource = None
        shortest_distance = float("inf")

        while to_visit:
            dist, current_node = heapq.heappop(to_visit)
            if current_node in visited:
                continue
            visited.add(current_node)
            node_data = self.G.nodes[current_node]

            # Check resource criteria or filter
            if (is_filter_function and resource_filter(current_node, node_data)) or (
                not is_filter_function
                and all(node_data.get(k) == v for k, v in resource_filter.items())
            ):
                if dist < shortest_distance:
                    shortest_distance = dist
                    nearest_resource = (current_node, dist)

            # Process neighbors considering the specified edge attribute
            for neighbor in self.G.neighbors(current_node):
                if neighbor not in visited:
                    # Select the edge with the minimum attribute value
                    edge_data = min(
                        self.G[current_node][neighbor].values(),
                        key=lambda e: e.get(attribute_name, default_attribute_value),
                    )
                    edge_attribute_value = edge_data.get(
                        attribute_name, default_attribute_value
                    )

                    new_dist = dist + edge_attribute_value
                    if max_search_depth is None or new_dist <= max_search_depth:
                        heapq.heappush(to_visit, (new_dist, neighbor))

        return nearest_resource if nearest_resource else (None, None)

    def track_event_participation(self, char, event):
        """
        Tracks a character's participation in an event, updating the graph to reflect this.

        Parameters:
            char (str): Node identifier for the character.
            event (str): Node identifier for the event.

        Usage example:
            graph_manager.track_event_participation('char1', 'event1')
        """
        self.G.add_edge(char, event, type="participation")

    def check_safety_of_locations(self, loc):
        """
        Checks the safety of a specific location based on nearby threats and its security attributes.

        Parameters:
            loc (str): Location node identifier.

        Returns:
            float: Safety score of the location.

        Usage example:
            safety_score = graph_manager.check_safety_of_locations('location1')
            print("Safety score for location1:", safety_score)
        """
        threats = sum(data.get("threat_level", 0) for node, data in self.G[loc].items())
        security = self.G.nodes[loc].get("security", 1)
        return security / (1 + threats)

    def evaluate_trade_opportunities(self, char):
        """
        Evaluates trade opportunities for a character based on the surplus and demand across the graph.

        Parameters:
            char (str): Character node identifier.

        Returns:
            dict: A dictionary of potential trades, where keys are resource types and values are lists of potential trade partners.

        Usage example:
            trade_opportunities = graph_manager.evaluate_trade_opportunities('char1')
            print("Trade opportunities for char1:", trade_opportunities)
        """
        surplus = {
            res: qty for res, qty in self.G.nodes[char]["resources"].items() if qty > 10
        }
        opportunities = {}
        for res in surplus:
            interested = [
                node
                for node in self.G.nodes
                if self.G.nodes[node].get("needed_resources", {}).get(res, 0) > 0
            ]
            if interested:
                opportunities[res] = interested
        return opportunities

    def find_most_accessible_resources(self, char, resource_type):
        """
        Finds the most accessible resource nodes of a specific type for a given character, based on proximity and quantity.

        Parameters:
            char (str): Character node identifier.
            resource_type (str): Type of resource to find.

        Returns:
            list: Sorted list of resource nodes, starting with the most accessible.

        Usage example:
            resources = graph_manager.find_most_accessible_resources('char1', 'water')
            print("Accessible water resources for char1:", resources)
        """
        resources = [
            (res, data["quantity"] / nx.shortest_path_length(self.G, char, res))
            for res, data in self.G.nodes(data=True)
            if data.get("type") == resource_type
        ]
        return sorted(resources, key=lambda x: x[1], reverse=True)

    def get_neighbors_with_attributes(self, node, **attributes):
        """
        Retrieves neighbors of a given node that meet specified attribute criteria.

        Parameters:
            node (str): Node identifier.
            attributes (dict): Keyword arguments specifying the attribute values to match.

        Returns:
            list: A list of nodes (neighbors) that match the specified attributes.

        Usage example:
            allies = graph_manager.get_neighbors_with_attributes('char1', alignment='friendly')
            print("Friendly characters near char1:", allies)
        """
        matches = []
        for neighbor in self.G.neighbors(node):
            if all(
                self.G.nodes[neighbor].get(key) == value
                for key, value in attributes.items()
            ):
                matches.append(neighbor)
        return matches

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
        Calculates the social influence on a character's decisions, considering various edge attributes,
        and applies multiple decay functions based on interaction characteristics.

        Parameters:
            character (str): Node identifier for the character.
            influence_attributes (list of tuples): List of tuples specifying the edge attributes and their weight in the calculation.
            memory_topic (str): A topic to query from memories that might influence the decision.

        Returns:
            float: Weighted influence score affecting the character's decisions.
        """
        total_influence = 0
        relationships = self.G.edges(character, data=True)
        memory_influence = 0
        if influence_factors is None:
            influence_factors = {
                "friend": {"weight": 1, "attributes": {"trust": 1, "friendship": 1}},
                "enemy": {"weight": -1, "attributes": {"trust": -1, "conflict": 1}},
                "neutral": {"weight": 0, "attributes": {"trust": 0}},
            }
        if influence_attributes is None:
            influence_attributes = [("trust", 1)]

        # Query memories if a topic is provided

        relationship_weights = 0
        relationships = self.G.edges(character, data=True)

        # Calculate social influence based on current relationships

        for _, _, attributes in relationships:
            relationship_type = attributes.get("type", "neutral")
            if relationship_type in influence_factors:
                factor = influence_factors[relationship_type]
                relationship_weight = factor.get("weight", 1)
                attribute_score = sum(
                    attributes.get(attr, 0) * factor.get("attributes", {}).get(attr, 1)
                    for attr in factor.get("attributes", [])
                )
                total_influence += relationship_weight * attribute_score
                relationship_weights += abs(relationship_weight)

        # Normalize social influence if needed
        if relationship_weights > 0:
            total_influence /= relationship_weights
        memories = []
        # Integrate memories from 'tiny_memories.py'
        if memory_weight > 0:
            if memory_topic:
                memories.extend(self.query_memories(character, memory_topic))
                memory_influence = sum(
                    mem["sentiment"] * mem["relevance"] for mem in memories
                )
                memory_influence *= memory_weight  # Apply weighting to memory influence
                total_influence = (
                    1 - memory_weight
                ) * total_influence + memory_weight * memory_influence
            if decision_context:
                memories.extend(self.query_memories(character, decision_context))
            memory_influence = sum(
                mem["sentiment"] * mem["relevance"] for mem in memories
            )
            total_influence = (
                1 - memory_weight
            ) * total_influence + memory_weight * memory_influence

        # Apply decay functions to influence calculation
        for _, target, attributes in relationships:
            distance = nx.shortest_path_length(
                self.G, character, target
            )  # Calculates distance between nodes
            influence = 0
            for attr, weight in influence_attributes or []:
                attr_value = attributes.get(attr, 0)
                # Apply specific decay functions
                time_decay = self.time_based_decay(
                    attributes.get("time_since_last_interaction", 0)
                )
                frequency_decay = self.frequency_based_decay(
                    attributes.get("frequency_of_interaction", 0)
                )
                advanced_decay = self.advanced_decay(
                    distance, attributes
                )  # Applies advanced decay based on distance and other factors
                influence += (
                    attr_value * weight * time_decay * frequency_decay * advanced_decay
                )
            total_influence += influence

        total_influence += memory_influence
        # Normalize the total influence score
        if total_influence > 0:
            total_influence /= len(relationships) + (1 if memory_topic else 0)

        return total_influence

    # Including the provided decay functions here for completeness
    def time_based_decay(self, time_since):
        return 1 / (1 + time_since / 365)

    def frequency_based_decay(self, frequency):
        return 1 + (frequency / 12)

    def advanced_decay(self, distance, attributes):
        base_decay = 1 / (1 + distance)
        emotional_factor = (
            attributes.get("trust", 1) * 0.5 + attributes.get("friendship", 1) * 0.5
        )
        emotional_decay = base_decay * emotional_factor
        if "professional" in attributes:
            professional_factor = 1 + attributes["professional"] * 0.1
            emotional_decay *= professional_factor
        if "family" in attributes:
            family_factor = 1 + attributes["family"] * 0.2
            emotional_decay *= family_factor
        historical_factor = 1 + attributes.get("shared_history", 0) * 0.3
        emotional_decay *= historical_factor
        proximity_factor = 1 / (1 + 0.1 * (10 - attributes.get("proximity", 10)))
        emotional_decay *= proximity_factor
        return emotional_decay

    def query_memories(self, character, topic):
        """
        Queries memories related to a specific topic to determine their influence on the character's current decision.

        This is a placeholder and assumes integration with tiny_memories.py where actual memory querying would be implemented.

        Parameters:
            character (str): The character whose memories to query.
            topic (str): The topic to query about in the memories.

        Returns:
            float: Influence score from memories.
        """
        if topic:
            topic = tiny_memories.MemoryManager().search_memories(topic)
        return 0.5  # Example fixed return value

    # More edge methods for other types (Location-Location, Item-Item, etc.), and edge methods from previous parts

    # Additional edge methods with comprehensive attributes to cover all detailed aspects
    # Continued in the following sections for remaining edge types and additional attributes (Temporal, Emotional, Economic, Historical, Security)

    # Other helper methods as required for managing the graph

    def create_tiny_village_graph(
        self, characters, locations, events, objects, activities
    ):
        # Create a new graph
        G = self.G

        # Add character nodes
        # characters = ["Emma", "John", "Alice", "Bob"]
        for char in characters:
            self.add_character_node(char)

        # Adding location nodes
        # locations = ["Cafe", "Park", "Library"]
        for loc in locations:
            G.add_node(
                loc,
                type="location",
                popularity=5,
                activities_available=["Read", "Socialize"],
            )

        # Add event nodes
        # events = ["Festival", "Concert"]
        for event in events:
            G.add_node(event, type="event", date="2024-10-05")

        # Adding object nodes
        # objects = ["Book", "Laptop", "Coffee"]
        for obj in objects:
            G.add_node(obj, type="object", value=20, usability=True)

        # Adding activity nodes
        # activities = ["Read", "Write", "Jog"]
        for act in activities:
            G.add_node(
                act, type="activity", related_skill="Literature", satisfaction_level=7
            )

        return G

    def graph_analysis(character, event, graph, context):
        """
        Analyzes the implications of an event on a character's network using graph theory.

        Args:
            character (str): Name of the character whose situation is being analyzed.
            event (dict): Details of the current event or decision point.
            graph (Graph): A NetworkX graph representing relationships, influences, etc.
            context (str): Context of the analysis (e.g., 'daily', 'career', 'relationship').

        Returns:
            impacted_nodes (list): List of nodes (characters, locations, etc.) impacted by the event.
            action_recommendations (list): Recommended actions based on the analysis.
        """
        # Initialize a list to store nodes impacted by the event
        impacted_nodes = []
        action_recommendations = []

        # Identify nodes directly connected to the event
        if context == "career":
            # Career context might focus on professional connections and opportunities
            for node in graph.neighbors(character):
                if graph.edges[character, node]["type"] == "professional":
                    impacted_nodes.append(node)
                    if "opportunity" in graph.edges[character, node]:
                        action_recommendations.append(
                            {"action": "pursue_opportunity", "node": node}
                        )
        elif context == "relationship":
            # Relationship context focuses on personal connections
            for node in graph.neighbors(character):
                if graph.edges[character, node]["type"] == "personal":
                    impacted_nodes.append(node)
                    if "conflict" in graph.edges[character, node]:
                        action_recommendations.append(
                            {"action": "resolve_conflict", "node": node}
                        )
        elif context == "daily":
            # Daily context might consider locations and routines
            for node in graph.neighbors(character):
                if graph.edges[character, node]["type"] == "location":
                    impacted_nodes.append(node)
                    if "favorite" in graph.edges[character, node]:
                        action_recommendations.append({"action": "visit", "node": node})

        # Evaluate the broader impact using a breadth-first search or similar method
        # to explore secondary impacts
        for node in impacted_nodes:
            for secondary_node in graph.neighbors(node):
                if secondary_node not in impacted_nodes:  # Avoid cycles
                    impacted_nodes.append(secondary_node)
                    # Further analysis to add more sophisticated recommendations

        return impacted_nodes, action_recommendations

    def update_graph(graph, action, character, target=None):
        """
        Update the graph based on an action taken by a character.
        Args:
            graph (Graph): The networkx graph.
            action (str): The action taken, which will determine the update logic.
            character (str): The character who is performing the action.
            target (str, optional): The target of the action, if applicable.
        """
        if action == "Make Friend":
            if target and not graph.has_edge(character, target):
                graph.add_edge(character, target, type="friend", strength=0.1)
            elif target:
                # Strengthen existing friendship
                graph.edges[character, target]["strength"] += 0.1

    def get_character_state(self, character_name):
        """
        Fetches the current state of the character from the graph.

        Args:
            character_name (str): The name of the character.

        Returns:
            dict: The state of the character.
        """
        if character_name in self.graph.nodes:
            return self.graph.nodes[character_name]
        else:
            raise ValueError(f"No character named {character_name} in the graph.")

    def get_possible_actions(self, character_name):
        """
        Analyzes the graph to determine possible actions for the character.

        Args:
            character_name (str): The name of the character.

        Returns:
            list: A list of possible actions and their utilities.
        """
        if character_name not in self.graph.nodes:
            raise ValueError(f"No character named {character_name} in the graph.")

        possible_actions = []
        for neighbor in self.graph.neighbors(character_name):
            action = self.graph.edges[character_name, neighbor].get("action")
            utility = self.graph.edges[character_name, neighbor].get("utility")
            if action and utility:
                possible_actions.append({"name": action, "utility": utility})

        return possible_actions

    def get_strongest_relationships(graph, character):
        # Filter edges to find strongest relationships for a given character
        relationships = [
            (n, attrs["strength"])
            for n, attrs in graph[character].items()
            if attrs["type"] in ["friends", "family", "colleagues"]
        ]
        # Sort relationships by strength
        relationships.sort(key=lambda x: x[1], reverse=True)
        return relationships[:5]  # Return top 5 strongest relationships

    def get_favorite_locations(graph, character):
        # Get locations with highest frequency of visits
        locations = [
            (n, attrs["frequency"])
            for n, attrs in graph[character].items()
            if attrs["type"] == "frequent_visitor"
        ]
        locations.sort(key=lambda x: x[1], reverse=True)
        return locations[:3]  # Return top 3 locations

    def analyze_event_impact(graph, event_node):
        # Find all characters connected to an event and evaluate impact
        participants = [(n, attrs["type"]) for n, attrs in graph[event_node].items()]
        impact_analysis = {}
        for participant, relation_type in participants:
            impact = (
                "high"
                if relation_type == "organizer"
                else "medium" if relation_type == "participant" else "low"
            )
            impact_analysis[participant] = impact
        return impact_analysis

    def explore_career_opportunities(graph, character):
        # Examine professional connections and opportunities
        opportunities = []
        for node, attrs in graph[character].items():
            if attrs["type"] == "professional" and "opportunity" in attrs:
                opportunities.append((node, attrs["opportunity"]))
        return opportunities

    def analyze_daily_preferences(graph, character):
        # Fetch locations and activities with positive outcomes
        locations = []
        activities = []
        for node, attr in graph.nodes(data=True):
            if attr["type"] == "location":
                # Check historical data for positive experiences
                if (
                    graph[character][node]["experience"] > 7
                ):  # Assuming experience is rated out of 10
                    locations.append(node)
            elif attr["type"] == "activity":
                if graph[character][node]["satisfaction"] > 7:
                    activities.append(node)

        # Consider social factors, e.g., friends going to the same location
        for friend in [
            n
            for n, attrs in graph[character].items()
            if attrs["type"] == "friend" and attrs["strength"] > 0.5
        ]:
            for location in locations:
                if (
                    graph.has_edge(friend, location)
                    and graph[friend][location]["frequency"] > 3
                ):
                    activities.append((location, "social"))

        return locations, activities

    def analyze_career_impact(graph, character, job_offer):
        # Evaluate how the new job aligns with the character's career aspirations
        current_job_node = next(
            (n for n, attrs in graph.nodes(data=True) if attrs.get("current_job")), None
        )
        new_job_node = job_offer["job_node"]

        # Assess potential benefits based on professional network expansion
        potential_connections = set(graph.neighbors(new_job_node)) - set(
            graph.neighbors(character)
        )
        career_benefit = len(potential_connections)

        # Analyze past job roles for similar positions
        past_experience_benefit = 0
        for node, attrs in graph.nodes(data=True):
            if (
                attrs["type"] == "job"
                and graph[character][node]["satisfaction"]
                and attrs["industry"] == job_offer["industry"]
            ):
                past_experience_benefit += attrs["satisfaction"]

        return career_benefit, past_experience_benefit


""" 
Graph-Based Decision Making:
Use the graph to inform decisions dynamically. For example, if a character is planning their social interactions, the graph can provide current friends and their relationship strengths, which in turn informs the utility evaluation of different social actions.
Real-Time Adaptation
Feedback Loops:
Establish feedback loops where the outcome of actions affects future planning cycles by updating both the state and the graph.
After executing an action, the system should assess the actual outcomes versus expected and adjust the planning parameters if necessary.
Learning and Adaptation:
Integrate simple learning mechanisms where characters can adjust their preferences based on past outcomes (e.g., if attending certain events consistently leads to positive outcomes, increase the preference for similar future events). """


# Create the graph
class TinyVillageGraph:
    def __init__(self):
        self.graph_manager = GraphManager()
        self.graph = self.graph_manager.create_tiny_village_graph()

    def update_strategy(self, event):
        if event["event"] == "new_day":
            character_state = self.graph_manager.get_character_state("Emma")
            actions = self.graph_manager.get_possible_actions("Emma")
            plan = self.goap_planner.goap_planner("Emma", character_state, actions)
            return plan

    def goap_planner(self, character, state, actions):
        # Placeholder for GOAP algorithm logic
        # Sorts actions based on utility and state requirements
        return sorted(actions, key=lambda x: -x["utility"])

    def plan_daily_activities(self, character):
        # Define the goal for daily activities
        goal = {"satisfaction": max, "energy_usage": min}

        # Get potential actions from a dynamic or context-specific action generator
        actions = self.graph_manager.get_possible_actions(character)

        # Use the graph to analyze current relationships and preferences
        current_state = self.graph_manager.graph_analysis(
            self.graph, character, "daily"
        )

        # Plan the career steps using GOAP
        plan = self.goap_planner(character, goal, current_state, actions)

        # Evaluate the utility of each step in the plan
        utility_scores = self.graph_manager.evaluate_utility(actions, character)

        return plan, utility_scores

    def update_graph(self, action, character, target=None):
        self.graph_manager.update_graph(self.graph, action, character, target)
        return self.graph

    def get_strongest_relationships(self, character):
        return self.graph_manager.get_strongest_relationships(self.graph, character)

    def get_favorite_locations(self, character):
        return self.graph_manager.get_favorite_locations(self.graph, character)

    def analyze_event_impact(self, event_node):
        return self.graph_manager.analyze_event_impact(self.graph, event_node)

    def explore_career_opportunities(self, character):
        return self.graph_manager.explore_career_opportunities(self.graph, character)

    def analyze_daily_preferences(self, character):
        return self.graph_manager.analyze_daily_preferences(self.graph, character)

    def analyze_career_impact(self, character, job_offer):
        return self.graph_manager.analyze_career_impact(
            self.graph, character, job_offer
        )