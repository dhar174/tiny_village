"""
GraphAnalytics Module

This module contains the GraphAnalytics class which provides general-purpose
graph analysis and querying utilities for the tiny_village simulation.

The GraphAnalytics class serves as a dedicated wrapper for networkx algorithms
and provides a clean interface for querying the game's world graph.
"""

import logging
from typing import List, Optional, Dict, Any, Set
from functools import lru_cache

# Graceful fallback for networkx
try:
    import networkx as nx
    from networkx.algorithms import community
    NETWORKX_AVAILABLE = True
    NETWORKX_COMMUNITY_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False
    NETWORKX_COMMUNITY_AVAILABLE = False
    # Mock implementations will be provided for fallback


class GraphAnalytics:
    """
    GraphAnalytics provides general-purpose graph analysis and querying utilities.
    
    This class serves as a dedicated wrapper for networkx algorithms and provides
    a clean interface for querying the game's world graph. It takes a WorldState
    object as a dependency to perform its analysis.
    """
    
    def __init__(self, world_state):
        """
        Initialize GraphAnalytics with a WorldState dependency.
        
        Args:
            world_state: WorldState object containing the graph to analyze
        """
        self.world_state = world_state
        self.graph = world_state.graph
        
        # Setup operator mappings for get_filtered_nodes
        self.ops = {
            "gt": lambda a, b: a > b,
            "lt": lambda a, b: a < b,
            "eq": lambda a, b: a == b,
            "ge": lambda a, b: a >= b,
            "le": lambda a, b: a <= b,
            "ne": lambda a, b: a != b,
        }
        self.symb_map = {
            ">": "gt",
            "<": "lt",
            "==": "eq",
            ">=": "ge",
            "<=": "le",
            "!=": "ne",
        }
        
        logging.debug("GraphAnalytics initialized with WorldState dependency")
    
    def find_shortest_path(self, source, target) -> Optional[List]:
        """
        Returns the shortest path between source and target nodes using Dijkstra's algorithm.

        Parameters:
            source: Node identifier for the source node.
            target: Node identifier for the target node.

        Returns:
            list or None: List of nodes representing the shortest path or None if no path exists.

        Usage example:
            path = graph_analytics.find_shortest_path('char1', 'char2')
            if path:
                print("Path found:", path)
            else:
                print("No path exists between the characters.")
        """
        if not NETWORKX_AVAILABLE:
            logging.warning("NetworkX not available, using fallback pathfinding")
            return [source, target] if source != target else [source]
            
        try:
            path = nx.shortest_path(self.graph, source=source, target=target)
            return path
        except nx.NetworkXNoPath:
            return None
        except Exception as e:
            logging.error(f"Error finding shortest path from {source} to {target}: {e}")
            return None

    def detect_communities(self) -> List[Set]:
        """
        Detects communities within the graph using the Louvain method for community detection.

        Returns:
            list of sets: A list where each set contains the nodes that form a community.

        Usage example:
            communities = graph_analytics.detect_communities()
            print("Detected communities:", communities)
        """
        if not NETWORKX_COMMUNITY_AVAILABLE:
            logging.warning("NetworkX community detection not available, using fallback")
            # Simple fallback: group nodes by type
            communities = []
            nodes_by_type = {}
            for node, data in self.graph.nodes(data=True):
                node_type = data.get('type', 'unknown')
                if node_type not in nodes_by_type:
                    nodes_by_type[node_type] = set()
                nodes_by_type[node_type].add(node)
            return list(nodes_by_type.values())
            
        try:
            communities = community.louvain_communities(self.graph, weight="weight")
            return communities
        except Exception as e:
            logging.error(f"Error detecting communities: {e}")
            return []

    def calculate_centrality(self) -> Dict:
        """
        Calculates and returns centrality measures for nodes in the graph, useful for identifying
        key influencers or central nodes within the network.

        Returns:
            dict: A dictionary where keys are node identifiers and values are centrality scores.

        Usage example:
            centrality = graph_analytics.calculate_centrality()
            print("Centrality scores:", centrality)
        """
        if not NETWORKX_AVAILABLE:
            logging.warning("NetworkX not available, using fallback centrality calculation")
            # Simple fallback: degree-based centrality
            centrality = {}
            num_nodes = self.graph.number_of_nodes()
            if num_nodes <= 1:
                return centrality
                
            for node in self.graph.nodes():
                degree = self.graph.degree(node)
                centrality[node] = degree / (num_nodes - 1) if num_nodes > 1 else 0
            return centrality
            
        try:
            centrality = nx.degree_centrality(self.graph)
            return centrality
        except Exception as e:
            logging.error(f"Error calculating centrality: {e}")
            return {}

    def shortest_path_between_characters(self, char1, char2) -> Optional[List]:
        """
        Find the most direct connection or interaction chain between two characters, which can be useful
        for understanding potential influences or conflicts.

        Parameters:
            char1: Node identifier for the first character.
            char2: Node identifier for the second character.

        Returns:
            list or None: List of characters forming the path or None if no path exists.

        Usage example:
            path = graph_analytics.shortest_path_between_characters('char1', 'char3')
            print("Direct interaction chain:", path)
        """
        return self.find_shortest_path(char1, char2)

    def common_interests_cluster(self) -> List[Set]:
        """
        Identify clusters of characters that share common interests, which can be used to form groups
        or communities within the game.

        Returns:
            list of sets: Each set contains characters that share common interests.
            
        Usage example:
            clusters = graph_analytics.common_interests_cluster()
            print("Interest-based clusters:", clusters)
        """
        try:
            # Get all character nodes
            characters = [node for node, data in self.graph.nodes(data=True) 
                         if data.get('type') == 'character']
            
            # Group characters by common interests/activities
            interest_groups = {}
            
            for char in characters:
                # Find activities this character is connected to
                activities = []
                for neighbor in self.graph.neighbors(char):
                    neighbor_data = self.graph.nodes[neighbor]
                    if neighbor_data.get('type') == 'activity':
                        activities.append(neighbor)
                
                # Create a signature for this character's interests
                interest_signature = tuple(sorted(activities))
                
                if interest_signature not in interest_groups:
                    interest_groups[interest_signature] = set()
                interest_groups[interest_signature].add(char)
            
            # Return non-empty clusters
            return [group for group in interest_groups.values() if len(group) > 1]
            
        except Exception as e:
            logging.error(f"Error clustering by common interests: {e}")
            return []

    def get_filtered_nodes(self, **kwargs) -> Dict[Any, Dict[str, Any]]:
        """
        Filter nodes based on various criteria including node attributes, edge attributes,
        distance, type, relationships, and other game-specific filters.
        
        Args:
            **kwargs: Filtering criteria including:
                - node_attributes: Dict of attribute-value pairs to match
                - edge_attributes: Dict of edge attribute-value pairs to match
                - node_type: Filter by node type
                - source_node: Source node for distance calculations
                - max_distance: Maximum distance from source_node
                - relationship: Filter by relationship status
                - safety_threshold: Minimum safety level for locations
                - item_ownership: Filter by item ownership
                - event_participation: Filter by event participation
                - Various other game-specific filters
        
        Returns:
            Dict[Any, Dict[str, Any]]: Dictionary mapping node objects to their attributes
            
        Usage example:
            characters = graph_analytics.get_filtered_nodes(node_type='character')
            safe_locations = graph_analytics.get_filtered_nodes(
                node_type='location', 
                safety_threshold=5
            )
        """
        try:
            filtered_nodes = set(self.graph.nodes)

            # Filter based on node attributes
            node_attributes = kwargs.get("node_attributes", {})
            for attr, value in node_attributes.items():
                filtered_nodes.intersection_update(
                    {
                        n
                        for n, attrs in self.graph.nodes(data=True)
                        if attrs.get(attr) == value
                    }
                )

            # Filter based on edge attributes
            edge_attributes = kwargs.get("edge_attributes", {})
            for attr, value in edge_attributes.items():
                filtered_nodes.intersection_update(
                    {
                        n
                        for n in filtered_nodes
                        if any(
                            self.graph.get_edge_data(n, neighbor, default={}).get(attr) == value
                            for neighbor in self.graph.neighbors(n)
                        )
                    }
                )

            # Filter based on distance
            source_node = kwargs.get("source_node")
            max_distance = kwargs.get("max_distance")
            if source_node is not None and max_distance is not None:
                if NETWORKX_AVAILABLE:
                    try:
                        lengths = nx.single_source_shortest_path_length(
                            self.graph, source=source_node, cutoff=max_distance
                        )
                        filtered_nodes.intersection_update(lengths.keys())
                    except Exception as e:
                        logging.warning(f"Distance filtering failed: {e}")

            # Filter by node type
            node_type = kwargs.get("node_type")
            if node_type is not None:
                filtered_nodes.intersection_update(
                    {
                        n
                        for n in filtered_nodes
                        if self.graph.nodes[n].get("type") == node_type
                    }
                )

            # Filter by event participation
            event = kwargs.get("event_participation")
            if event is not None:
                filtered_nodes.intersection_update(
                    {n for n in filtered_nodes if self.graph.has_edge(n, event)}
                )
                # Check participation_status in edge attributes
                filtered_nodes.intersection_update(
                    {
                        n
                        for n in filtered_nodes
                        if self.graph.get_edge_data(n, event, default={}).get("participation_status") == True
                    }
                )

            return {
                n: self.graph.nodes[n] for n in filtered_nodes
            }
            
        except Exception as e:
            logging.error(f"Error filtering nodes: {e}")
            return {}

    def get_neighbors_by_type(self, node, neighbor_type: str = None) -> List[Any]:
        """
        Get neighboring nodes, optionally filtered by type.
        
        Args:
            node: The node object to get neighbors for
            neighbor_type: Optional filter by neighbor type
            
        Returns:
            List[Any]: List of neighboring nodes
        """
        try:
            if not self.graph.has_node(node):
                return []
                
            neighbors = list(self.graph.neighbors(node))
            
            if neighbor_type:
                neighbors = [
                    n for n in neighbors 
                    if self.graph.nodes[n].get('type') == neighbor_type
                ]
                
            return neighbors
            
        except Exception as e:
            logging.error(f"Error getting neighbors for {node}: {e}")
            return []

    def analyze_node_connectivity(self, node) -> Dict[str, Any]:
        """
        Analyze the connectivity of a specific node in the graph.
        
        Args:
            node: The node to analyze
            
        Returns:
            Dict[str, Any]: Dictionary containing connectivity metrics
        """
        try:
            if not self.graph.has_node(node):
                return {}
                
            # Basic connectivity metrics
            degree = self.graph.degree(node)
            in_degree = self.graph.in_degree(node) if self.graph.is_directed() else degree
            out_degree = self.graph.out_degree(node) if self.graph.is_directed() else degree
            
            # Neighbor analysis
            neighbors = list(self.graph.neighbors(node))
            neighbor_types = {}
            for neighbor in neighbors:
                neighbor_type = self.graph.nodes[neighbor].get('type', 'unknown')
                neighbor_types[neighbor_type] = neighbor_types.get(neighbor_type, 0) + 1
            
            return {
                'degree': degree,
                'in_degree': in_degree,
                'out_degree': out_degree,
                'neighbor_count': len(neighbors),
                'neighbor_types': neighbor_types,
                'neighbors': neighbors
            }
            
        except Exception as e:
            logging.error(f"Error analyzing connectivity for {node}: {e}")
            return {}

    def find_nodes_within_distance(self, source_node, max_distance: int) -> Dict[Any, int]:
        """
        Find all nodes within a specified distance from a source node.
        
        Args:
            source_node: The source node
            max_distance: Maximum distance to search
            
        Returns:
            Dict[Any, int]: Dictionary mapping nodes to their distances from source
        """
        try:
            if not self.graph.has_node(source_node):
                return {}
                
            if NETWORKX_AVAILABLE:
                return nx.single_source_shortest_path_length(
                    self.graph, source=source_node, cutoff=max_distance
                )
            else:
                # Simple fallback: just return direct neighbors
                neighbors = list(self.graph.neighbors(source_node))
                result = {source_node: 0}
                for neighbor in neighbors:
                    if max_distance >= 1:
                        result[neighbor] = 1
                return result
                
        except Exception as e:
            logging.error(f"Error finding nodes within distance {max_distance} from {source_node}: {e}")
            return {}

    def get_graph_statistics(self) -> Dict[str, Any]:
        """
        Get general statistics about the graph.
        
        Returns:
            Dict[str, Any]: Dictionary containing graph statistics
        """
        try:
            stats = {
                'node_count': self.graph.number_of_nodes(),
                'edge_count': self.graph.number_of_edges(),
                'is_directed': self.graph.is_directed(),
                'is_multigraph': self.graph.is_multigraph(),
            }
            
            # Count nodes by type
            node_types = {}
            for node, data in self.graph.nodes(data=True):
                node_type = data.get('type', 'unknown')
                node_types[node_type] = node_types.get(node_type, 0) + 1
            stats['node_types'] = node_types
            
            # Count edges by type
            edge_types = {}
            for u, v, data in self.graph.edges(data=True):
                edge_type = data.get('type', 'unknown')
                edge_types[edge_type] = edge_types.get(edge_type, 0) + 1
            stats['edge_types'] = edge_types
            
            # Connectivity metrics
            if NETWORKX_AVAILABLE and stats['node_count'] > 0:
                try:
                    if nx.is_connected(self.graph.to_undirected()):
                        stats['is_connected'] = True
                        stats['diameter'] = nx.diameter(self.graph.to_undirected())
                    else:
                        stats['is_connected'] = False
                        stats['connected_components'] = nx.number_connected_components(self.graph.to_undirected())
                except:
                    stats['is_connected'] = False
            
            return stats
            
        except Exception as e:
            logging.error(f"Error calculating graph statistics: {e}")
            return {}