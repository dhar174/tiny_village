import json, logging
logging.basicConfig(level=logging.ERROR)
from tiny_gameplay_controller import GameplayController
from tiny_graph_manager import GraphManager
from tiny_characters import Character
graph=GraphManager()
controller=GameplayController(graph, {'headless':True,'memory':{'mode':'basic'},'characters':{'count':3,'use_llm':False},'checkpoint':{'auto_enabled':False}})
characters=list(controller.characters.values())
assert all(isinstance(c,Character) and c._initialized and c.graph_manager is graph for c in characters)
from unittest.mock import patch
from tiny_prompt_builder import ContextManager,NeedsPriorities
try:
 records=[]
 class Capture(logging.Handler):
  def emit(self,record): records.append(record.getMessage())
 handler=Capture();logger=logging.getLogger('tiny_prompt_builder');logger.addHandler(handler)
 try:
  with patch.object(Character,'evaluate_goals',side_effect=RuntimeError('controlled goal failure')),patch.object(NeedsPriorities,'calculate_needs_priorities',side_effect=ValueError('controlled needs failure')):
   result=ContextManager(characters[0]).gather_goal_context()
 finally: logger.removeHandler(handler)
 print(json.dumps({'context':result,'diagnostics':records,'faults_injected':2}),flush=True)
 assert len(records)>=2, 'Goal/needs failures remain silent; see #808'
finally: controller.close()
