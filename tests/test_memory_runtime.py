"""Real memory contracts in bounded, fresh interpreters."""

from tests.test_graph_initialization import FreshPythonTestCase


NO_AI_IMPORTS = '''
import importlib.abc, sys
blocked = {'pygame', 'pandas', 'scipy', 'sklearn', 'torch', 'faiss',
           'transformers', 'sentence_transformers', 'spacy', 'nltk',
           'rake_nltk', 'textblob', 'matplotlib', 'seaborn', 'tiny_brain_io',
           'tiny_sr_mapping'}
class NoAIImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in blocked:
            raise AssertionError('Unexpected optional import: ' + fullname)
sys.meta_path.insert(0, NoAIImports())
'''


class TestLazyMemory(FreshPythonTestCase):
    def test_import_does_not_initialize_ai(self):
        self.run_python(NO_AI_IMPORTS + '''
import tiny_memories as memory
assert memory.model is None
assert memory.nlp is None
assert memory.sentiment_analysis is None
assert not (blocked & sys.modules.keys())
''')

    def test_strategy_without_llm_does_not_import_ai(self):
        self.run_python(NO_AI_IMPORTS + '''
from tiny_graph_manager import GraphManager
from tiny_strategy_manager import StrategyManager
graph = GraphManager()
strategy = StrategyManager(False, None, graph)
assert strategy.graph_manager is graph
assert strategy.brain_io is None
assert not strategy.use_llm
assert not (blocked & sys.modules.keys())
''')
