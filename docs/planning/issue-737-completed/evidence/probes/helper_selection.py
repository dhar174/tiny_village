import json, logging
logging.basicConfig(level=logging.ERROR)
from tiny_gameplay_controller import GameplayController
from tiny_graph_manager import GraphManager
from tiny_characters import Character
graph=GraphManager()
controller=GameplayController(graph, {'headless':True,'memory':{'mode':'basic'},'characters':{'count':3,'use_llm':False},'checkpoint':{'auto_enabled':False}})
characters=list(controller.characters.values())
assert all(isinstance(c,Character) and c._initialized and c.graph_manager is graph for c in characters)
from llm_character_utils import enable_llm_decisions,enable_llm_for_characters,get_llm_enabled_characters
try:
 a,b,c=characters
 assert enable_llm_decisions(a,True) is a and a.use_llm_decisions
 assert not b.use_llm_decisions
 assert enable_llm_for_characters(characters)==characters
 selected=enable_llm_for_characters(characters,[a.name,c.name])
 assert selected==[a,c] and not b.use_llm_decisions
 assert get_llm_enabled_characters(characters)==[a,c]
 assert get_llm_enabled_characters([])==[]
 assert get_llm_enabled_characters([object()])==[]
 print(json.dumps({'real_characters':len(characters),'selection_and_independence_assertions':'pass','live_ai_exercised':False}))
finally: controller.close()
