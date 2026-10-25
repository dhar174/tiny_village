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
from tiny_brain_io import TinyBrainIO
from llm_character_utils import configure_strategy_manager_for_llm
try:
 strategy=controller.strategy_manager
 cause=OSError('controlled unavailable model assets')
 raised=None
 with patch.object(TinyBrainIO,'load_model',side_effect=cause):
  try: configure_strategy_manager_for_llm(strategy)
  except Exception as error: raised=error
 observed={'requested':strategy.use_llm,'status':strategy.llm_status,'provider_present':strategy.brain_io is not None,'raised':type(raised).__name__ if raised else None,'cause_preserved':raised is cause}
 print(json.dumps(observed),flush=True)
 assert strategy.llm_status=='unavailable' and strategy.llm_load_error is cause, 'Helper failed to report actual unavailable capability; see #807'
finally: controller.close()
