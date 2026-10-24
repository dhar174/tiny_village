import json, logging
logging.basicConfig(level=logging.ERROR)
from tiny_gameplay_controller import GameplayController
from tiny_graph_manager import GraphManager
from tiny_characters import Character
graph=GraphManager()
controller=GameplayController(graph, {'headless':True,'memory':{'mode':'basic'},'characters':{'count':3,'use_llm':False},'checkpoint':{'auto_enabled':False}})
characters=list(controller.characters.values())
assert all(isinstance(c,Character) and c._initialized and c.graph_manager is graph for c in characters)
import networkx as nx
try:
 a,b,c=characters
 graph.G.add_edge(a,b,edge_type='probe737')
 graph.G.add_edge(b,c,edge_type='probe737')
 analytics=graph.graph_analytics
 assert analytics.world_state.graph is graph.G
 before=(graph.G.number_of_nodes(),graph.G.number_of_edges())
 assert analytics.find_nodes_within_distance('absent-737',2)=={}
 zero=analytics.find_nodes_within_distance(a,0);assert zero=={a:0},zero
 one=analytics.find_nodes_within_distance(a,1);assert one[b]==1 and c not in one,one
 two=analytics.find_nodes_within_distance(a,2);assert two[c]==2,two
 assert before==(graph.G.number_of_nodes(),graph.G.number_of_edges())
 print(json.dumps({'real_characters':3,'shared_graph':True,'absent_zero_one_two_hop':'pass','graph_sizes_preserved':True,'networkx_available':True,'networkx_free_execution':'not verified'}))
finally: controller.close()
