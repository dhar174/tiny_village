import json, logging
logging.basicConfig(level=logging.ERROR)
from tiny_gameplay_controller import GameplayController
from tiny_graph_manager import GraphManager
from tiny_characters import Character
graph=GraphManager()
controller=GameplayController(graph, {'headless':True,'memory':{'mode':'basic'},'characters':{'count':3,'use_llm':False},'checkpoint':{'auto_enabled':False}})
characters=list(controller.characters.values())
assert all(isinstance(c,Character) and c._initialized and c.graph_manager is graph for c in characters)
from tiny_locations import Location
try:
 places=[Location('Isolated737-'+str(i),1000+i*20,1000,10,10,controller.action_system) for i in range(3)]
 a,b,c=places
 for place in places: graph.world_state.add_location_node(place)
 graph.world_state.add_edge(a,b,'probe737')
 graph.world_state.add_edge(b,c,'probe737')
 analytics=graph.graph_analytics
 assert analytics.world_state.graph is graph.G
 before=(graph.G.number_of_nodes(),graph.G.number_of_edges())
 assert analytics.find_nodes_within_distance('absent-737',2)=={}
 assert analytics.find_nodes_within_distance(a,0)=={a:0}
 one=analytics.find_nodes_within_distance(a,1);assert one=={a:0,b:1}
 two=analytics.find_nodes_within_distance(a,2);assert two=={a:0,b:1,c:2}
 assert before==(graph.G.number_of_nodes(),graph.G.number_of_edges())
 print(json.dumps({'real_locations':3,'shared_graph':True,'absent_zero_one_two_hop':'pass','graph_sizes_preserved':True,'networkx_available':True,'networkx_free_execution':'not verified','initial_probe_fixture_issue':'controller character graph already connected; isolated locations used without deleting existing edges'}))
finally: controller.close()
