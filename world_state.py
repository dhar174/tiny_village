"""
WorldState Module

This module contains the WorldState class which encapsulates the core graph storage
and basic CRUD operations for the tiny_village simulation.

The WorldState class is responsible for:
- Initializing and managing the networkx.DiGraph instance
- Providing basic Create, Read, Update, and Delete operations for nodes and edges
- Maintaining the graph data structure without business logic
"""

import networkx as nx
import logging
from typing import Dict, Any, Optional, List


class WorldState:
    """
    WorldState manages the core graph data structure for the tiny_village simulation.
    
    This class encapsulates the networkx.DiGraph and provides basic CRUD operations
    for nodes and edges without implementing business logic or complex analysis.
    """
    
    def __init__(self):
        """Initialize the WorldState with an empty graph."""
        self.graph = self._initialize_graph()
        
        # Dictionaries to store references to game objects by type
        self.characters = {}
        self.locations = {}
        self.objects = {}
        self.events = {}
        self.activities = {}
        self.jobs = {}
        self.stocks = {}
        
        # Map node types to their corresponding storage dictionaries
        self.type_to_dict_map = {
            "character": self.characters,
            "location": self.locations,
            "object": self.objects,
            "event": self.events,
            "activity": self.activities,
            "job": self.jobs,
            "stock": self.stocks,
        }
        
        logging.debug("WorldState initialized with empty graph")
    
    def _initialize_graph(self) -> nx.MultiDiGraph:
        """
        Initialize and return a new networkx MultiDiGraph.
        
        Returns:
            nx.MultiDiGraph: A new directed multigraph for storing world state
        """
        return nx.MultiDiGraph()
    
    # ===== Specific Node Addition Methods =====
    
    def add_character_node(self, character, **additional_attrs):
        """Add a character node with character-specific attributes."""
        attrs = {
            'age': getattr(character, 'age', None),
            'job': getattr(character, 'job', None),
            'happiness': getattr(character, 'happiness', None),
            'energy_level': getattr(character, 'energy', None),
            'relationships': {},
            'emotional_state': {},
            'coordinates_location': getattr(character, 'coordinates_location', None),
            'resources': getattr(character, 'inventory', None),
            'needed_resources': getattr(character, 'needed_items', None),
            'mood': getattr(character, 'current_mood', None),
            'wealth_money': getattr(character, 'wealth_money', None),
            'health_status': getattr(character, 'health_status', None),
            'hunger_level': getattr(character, 'hunger_level', None),
            'mental_health': getattr(character, 'mental_health', None),
            'social_wellbeing': getattr(character, 'social_wellbeing', None),
            'shelter': getattr(character, 'shelter', None),
            'has_investment': getattr(character, 'has_investment', None),
        }
        attrs.update(additional_attrs)
        self.add_node(character, "character", **attrs)
    
    def add_location_node(self, location, **additional_attrs):
        """Add a location node with location-specific attributes."""
        attrs = {
            'popularity': getattr(location, 'popularity', None),
            'activities_available': getattr(location, 'activities_available', None),
            'accessible': getattr(location, 'accessible', None),
            'security': getattr(location, 'security', None),
            'coordinates_location': getattr(location, 'coordinates_location', None),
            'threat_level': getattr(location, 'threat_level', None),
            'visit_count': getattr(location, 'visit_count', None),
        }
        attrs.update(additional_attrs)
        self.add_node(location, "location", **attrs)
    
    def add_event_node(self, event, **additional_attrs):
        """Add an event node with event-specific attributes."""
        attrs = {
            'event_type': getattr(event, 'type', None),
            'date': getattr(event, 'date', None),
            'importance': getattr(event, 'importance', None),
            'impact': getattr(event, 'impact', None),
            'required_items': getattr(event, 'required_items', None),
            'coordinates_location': getattr(event, 'coordinates_location', None),
        }
        attrs.update(additional_attrs)
        self.add_node(event, "event", **attrs)
    
    def add_object_node(self, obj, **additional_attrs):
        """Add an object node with object-specific attributes."""
        attrs = {
            'item_type': getattr(obj, 'item_type', None),
            'item_subtype': getattr(obj, 'item_subtype', None),
            'value': getattr(obj, 'value', None),
            'usability': getattr(obj, 'usability', None),
            'coordinates_location': getattr(obj, 'coordinates_location', None),
            'ownership_history': getattr(obj, 'ownership_history', None),
            'type_specific_attributes': getattr(obj, 'type_specific_attributes', {}),
        }
        attrs.update(additional_attrs)
        self.add_node(obj, "object", **attrs)
    
    def add_stock_node(self, stock, **additional_attrs):
        """Add a stock node with stock-specific attributes."""
        attrs = {
            'stock_type': getattr(stock, 'stock_type', None),
            'stock_description': getattr(stock, 'stock_description', None),
            'value': getattr(stock, 'value', None),
            'scarcity': getattr(stock, 'scarcity', None),
            'ownership_history': getattr(stock, 'ownership_history', None),
            'type_specific_attributes': getattr(stock, 'type_specific_attributes', {}),
        }
        attrs.update(additional_attrs)
        self.add_node(stock, "stock", **attrs)
    
    def add_activity_node(self, activity, **additional_attrs):
        """Add an activity node with activity-specific attributes."""
        attrs = {
            'related_skills': getattr(activity, 'related_skills', None),
            'required_items': getattr(activity, 'required_items', None),
            'preconditions': getattr(activity, 'preconditions', None),
            'effects': getattr(activity, 'effects', None),
        }
        attrs.update(additional_attrs)
        self.add_node(activity, "activity", **attrs)
    
    def add_job_node(self, job, **additional_attrs):
        """Add a job node with job-specific attributes."""
        attrs = {
            'required_skills': getattr(job, 'job_skills', None),
            'location': getattr(job, 'location', None),
            'salary': getattr(job, 'job_salary', None),
            'job_title': getattr(job, 'job_title', None),
            'available': getattr(job, 'available', None),
        }
        attrs.update(additional_attrs)
        self.add_node(job, "job", **attrs)
    
    def add_dict_of_nodes(self, nodes_dict):
        """Add multiple nodes from a dictionary organized by type."""
        for node_type, nodes in nodes_dict.items():
            if node_type == "characters":
                for char in nodes:
                    self.add_character_node(char)
            elif node_type == "locations":
                for loc in nodes:
                    self.add_location_node(loc)
            elif node_type == "events":
                for event in nodes:
                    self.add_event_node(event)
            elif node_type == "objects":
                for obj in nodes:
                    self.add_object_node(obj)
            elif node_type == "activities":
                for act in nodes:
                    self.add_activity_node(act)
            elif node_type == "jobs":
                for job in nodes:
                    self.add_job_node(job)
            elif node_type == "stocks":
                for stock in nodes:
                    self.add_stock_node(stock)

    # ===== Node CRUD Operations =====
    
    def add_node(self, node_obj: Any, node_type: str, **attributes) -> None:
        """
        Add a node to the graph with specified attributes.
        
        Args:
            node_obj: The object instance to add as a node
            node_type: Type of the node ('character', 'location', etc.)
            **attributes: Additional attributes to store with the node
        """
        # Store the object in the appropriate dictionary
        if hasattr(node_obj, 'name'):
            name = node_obj.name
        elif hasattr(node_obj, 'job_name'):
            name = node_obj.job_name
        else:
            name = str(node_obj)
            
        if node_type in self.type_to_dict_map:
            self.type_to_dict_map[node_type][name] = node_obj
        
        # Add the node to the graph with type and other attributes
        self.graph.add_node(node_obj, type=node_type, name=name, **attributes)
        
        logging.debug(f"Added {node_type} node: {name}")
    
    def remove_node(self, node_obj: Any) -> None:
        """
        Remove a node from the graph.
        
        Args:
            node_obj: The node object to remove
        """
        if self.graph.has_node(node_obj):
            # Remove from type-specific dictionary
            node_data = self.graph.nodes[node_obj]
            node_type = node_data.get('type')
            name = node_data.get('name')
            
            if node_type in self.type_to_dict_map and name in self.type_to_dict_map[node_type]:
                del self.type_to_dict_map[node_type][name]
            
            self.graph.remove_node(node_obj)
            logging.debug(f"Removed node: {name}")
        else:
            logging.warning(f"Attempted to remove non-existent node: {node_obj}")
    
    def has_node(self, node_obj: Any) -> bool:
        """
        Check if a node exists in the graph.
        
        Args:
            node_obj: The node object to check
            
        Returns:
            bool: True if the node exists, False otherwise
        """
        return self.graph.has_node(node_obj)
    
    def get_node_attributes(self, node_obj: Any) -> Dict[str, Any]:
        """
        Get all attributes for a node.
        
        Args:
            node_obj: The node object
            
        Returns:
            Dict[str, Any]: Dictionary of node attributes
        """
        if self.graph.has_node(node_obj):
            return dict(self.graph.nodes[node_obj])
        return {}
    
    def update_node_attribute(self, node_obj: Any, attribute: str, value: Any) -> None:
        """
        Update a single attribute of a node.
        
        Args:
            node_obj: The node object
            attribute: The attribute name to update
            value: The new value for the attribute
        """
        if self.graph.has_node(node_obj):
            self.graph.nodes[node_obj][attribute] = value
            logging.debug(f"Updated node attribute: {attribute} = {value}")
        else:
            raise ValueError(f"Node does not exist in the graph: {node_obj}")
    
    # ===== Edge CRUD Operations =====
    
    def add_edge(self, source: Any, target: Any, edge_type: str = None, **attributes) -> None:
        """
        Add an edge between two nodes.
        
        Args:
            source: Source node object
            target: Target node object
            edge_type: Type of the edge (optional)
            **attributes: Additional edge attributes
        """
        edge_attrs = attributes.copy()
        if edge_type:
            edge_attrs['type'] = edge_type
            
        self.graph.add_edge(source, target, **edge_attrs)
        logging.debug(f"Added edge: {source} -> {target} (type: {edge_type})")
    
    def remove_edge(self, source: Any, target: Any) -> None:
        """
        Remove an edge between two nodes.
        
        Args:
            source: Source node object
            target: Target node object
        """
        if self.graph.has_edge(source, target):
            self.graph.remove_edge(source, target)
            logging.debug(f"Removed edge: {source} -> {target}")
        else:
            logging.warning(f"Attempted to remove non-existent edge: {source} -> {target}")
    
    def has_edge(self, source: Any, target: Any) -> bool:
        """
        Check if an edge exists between two nodes.
        
        Args:
            source: Source node object
            target: Target node object
            
        Returns:
            bool: True if the edge exists, False otherwise
        """
        return self.graph.has_edge(source, target)
    
    def get_edge_attributes(self, source: Any, target: Any, key: int = 0) -> Dict[str, Any]:
        """
        Get all attributes for an edge.
        
        Args:
            source: Source node object
            target: Target node object
            key: Edge key for MultiDiGraph (default: 0)
            
        Returns:
            Dict[str, Any]: Dictionary of edge attributes
        """
        if self.graph.has_edge(source, target):
            return dict(self.graph.edges[source, target, key])
        return {}
    
    def update_edge_attribute(self, source: Any, target: Any, attribute: str, value: Any, key: int = 0) -> None:
        """
        Update a single attribute of an edge.
        
        Args:
            source: Source node object
            target: Target node object
            attribute: The attribute name to update
            value: The new value for the attribute
            key: Edge key for MultiDiGraph (default: 0)
        """
        if self.graph.has_edge(source, target):
            self.graph.edges[source, target, key][attribute] = value
            logging.debug(f"Updated edge attribute: {source} -> {target}, {attribute} = {value}")
        else:
            raise ValueError(f"Edge does not exist: {source} -> {target}")
    
    # ===== Graph Information Methods =====
    
    def get_nodes(self, node_type: str = None) -> List[Any]:
        """
        Get all nodes, optionally filtered by type.
        
        Args:
            node_type: Optional node type filter
            
        Returns:
            List[Any]: List of node objects
        """
        if node_type:
            return [node for node, data in self.graph.nodes(data=True) 
                   if data.get('type') == node_type]
        return list(self.graph.nodes())
    
    def get_edges(self, edge_type: str = None) -> List[tuple]:
        """
        Get all edges, optionally filtered by type.
        
        Args:
            edge_type: Optional edge type filter
            
        Returns:
            List[tuple]: List of (source, target) tuples
        """
        if edge_type:
            return [(u, v) for u, v, data in self.graph.edges(data=True) 
                   if data.get('type') == edge_type]
        return list(self.graph.edges())
    
    def get_neighbors(self, node_obj: Any) -> List[Any]:
        """
        Get all neighboring nodes.
        
        Args:
            node_obj: The node object
            
        Returns:
            List[Any]: List of neighboring node objects
        """
        if self.graph.has_node(node_obj):
            return list(self.graph.neighbors(node_obj))
        return []
    
    def get_node_count(self) -> int:
        """Get the total number of nodes in the graph."""
        return self.graph.number_of_nodes()
    
    def get_edge_count(self) -> int:
        """Get the total number of edges in the graph."""
        return self.graph.number_of_edges()
    
    # ===== Object Retrieval Methods =====
    
    def get_object_by_name(self, name: str, object_type: str = None) -> Optional[Any]:
        """
        Retrieve an object by name, optionally filtered by type.
        
        Args:
            name: The name of the object
            object_type: Optional type filter
            
        Returns:
            Optional[Any]: The object if found, None otherwise
        """
        if object_type and object_type in self.type_to_dict_map:
            return self.type_to_dict_map[object_type].get(name)
        
        # Search all dictionaries if no type specified
        for type_dict in self.type_to_dict_map.values():
            if name in type_dict:
                return type_dict[name]
        return None
    
    def get_all_objects_by_type(self, object_type: str) -> Dict[str, Any]:
        """
        Get all objects of a specific type.
        
        Args:
            object_type: The type of objects to retrieve
            
        Returns:
            Dict[str, Any]: Dictionary mapping names to objects
        """
        if object_type in self.type_to_dict_map:
            return self.type_to_dict_map[object_type].copy()
        return {}
    
    # ===== Utility Methods =====
    
    def clear(self) -> None:
        """Clear all nodes and edges from the graph."""
        self.graph.clear()
        for type_dict in self.type_to_dict_map.values():
            type_dict.clear()
        logging.debug("WorldState cleared")
    
    def get_graph_copy(self) -> nx.MultiDiGraph:
        """
        Get a copy of the underlying graph.
        
        Returns:
            nx.MultiDiGraph: A copy of the graph
        """
        return self.graph.copy()
    
    def __str__(self) -> str:
        """String representation of the WorldState."""
        return f"WorldState(nodes={self.get_node_count()}, edges={self.get_edge_count()})"
    
    def __repr__(self) -> str:
        """Detailed string representation of the WorldState."""
        return (f"WorldState(nodes={self.get_node_count()}, edges={self.get_edge_count()}, "
                f"characters={len(self.characters)}, locations={len(self.locations)})")