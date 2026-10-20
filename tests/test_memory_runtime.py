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

    def test_requested_llm_remains_lazy_and_observable(self):
        self.run_python(NO_AI_IMPORTS + '''
from tiny_graph_manager import GraphManager
from tiny_strategy_manager import StrategyManager
strategy = StrategyManager(True, None, GraphManager())
assert strategy.use_llm
assert strategy.brain_io is None
assert strategy.llm_status == 'not_initialized'
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


class TestBasicMemory(FreshPythonTestCase):
    def test_real_records_retain_fields_and_evict_oldest(self):
        self.run_python(NO_AI_IMPORTS + '''
from datetime import datetime
from tiny_memories import Memory, MemoryManager
from tiny_time_manager import GameCalendar, GameTimeManager
manager = MemoryManager(GameTimeManager(GameCalendar()), mode='basic')
timestamp = datetime(2026, 10, 20, 12)
for i in range(52):
    record = Memory(f'Record {i}', timestamp, 7)
    manager.add_memory(record)
recent = list(manager.flat_access.get_recent_memories())
assert len(recent) == 50
assert recent[0].description == 'Record 2'
assert recent[-1] is record
assert all(m.creation_time == timestamp and m.importance_score == 7 for m in recent)
first, second = Memory('First'), Memory('Second')
assert first.creation_time is not second.creation_time
assert first.last_access_time == first.creation_time
assert manager.get_capabilities()['recording'] == 'ready'
assert manager.get_capabilities()['semantic_search'] == 'disabled'
assert not (blocked & sys.modules.keys())
''')

    def test_modes_and_semantic_operations_fail_explicitly(self):
        self.run_python(NO_AI_IMPORTS + '''
from tiny_memories import MemoryManager, MemoryFeatureUnavailableError
from tiny_time_manager import GameCalendar, GameTimeManager
clock = GameTimeManager(GameCalendar())
for options in ({'mode':'unknown'}, {'mode':'basic', 'index_load_filename':'old.index'}):
    try:
        MemoryManager(clock, **options)
    except ValueError:
        pass
    else:
        raise AssertionError('Invalid mode/index combination accepted')
manager = MemoryManager(clock, mode='basic')
operations = [
    lambda: manager.extract_entities('Ada walked home'),
    lambda: manager.extract_keywords('Ada walked home'),
    lambda: manager.retrieve_memories('home'),
    lambda: manager.get_query_embedding('home'),
    lambda: manager.flat_access.initialize_faiss_index(384),
    lambda: manager.flat_access.find_similar_memories('home'),
]
for operation in operations:
    try:
        operation()
    except MemoryFeatureUnavailableError as error:
        assert 'basic' in str(error)
    else:
        raise AssertionError('Disabled semantic operation silently succeeded')
assert not (blocked & sys.modules.keys())
''')

    def test_full_construction_is_lazy_and_missing_service_preserves_cause(self):
        self.run_python('''
import importlib.abc, sys
class MissingSpacy(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] == 'spacy':
            raise ModuleNotFoundError('Controlled missing spaCy', name='spacy')
sys.meta_path.insert(0, MissingSpacy())
from tiny_memories import MemoryManager, MemoryFeatureUnavailableError
from tiny_time_manager import GameCalendar, GameTimeManager
manager = MemoryManager(GameTimeManager(GameCalendar()))
assert manager.get_capabilities()['nlp'] == 'not_initialized'
try:
    manager.extract_entities('Ada walked home')
except MemoryFeatureUnavailableError as error:
    assert isinstance(error.__cause__, ImportError)
else:
    raise AssertionError('Missing NLP feature silently succeeded')
assert manager.get_capabilities()['nlp'] == 'unavailable'
''')

    def test_basic_state_restores_without_importing_ai(self):
        self.run_python(NO_AI_IMPORTS + '''
import pickle
from tiny_memories import Memory, MemoryManager
from tiny_time_manager import GameCalendar, GameTimeManager
first = MemoryManager(GameTimeManager(GameCalendar()), mode='basic')
other = MemoryManager(GameTimeManager(GameCalendar()), mode='basic')
first.add_memory(Memory('Stored event', importance_score=3))
restored = pickle.loads(pickle.dumps(first))
assert list(other.flat_access.get_recent_memories()) == []
assert restored.mode == 'basic'
assert restored.flat_access.manager is restored
assert restored.flat_access.get_recent_memories()[0].description == 'Stored event'
assert restored.get_capabilities()['semantic_search'] == 'disabled'
assert not (blocked & sys.modules.keys())
''')

    def test_complete_real_characters_keep_independent_basic_memories(self):
        self.run_python(NO_AI_IMPORTS + '''
from datetime import datetime
from tiny_graph_manager import GraphManager
from actions import ActionSystem
from tiny_characters import Character, PersonalityTraits
from tiny_items import ItemInventory
from tiny_locations import Location
from tiny_time_manager import GameCalendar, GameTimeManager
graph = GraphManager()
system = ActionSystem(graph_manager=graph)
clock = GameTimeManager(GameCalendar())
characters = [Character(name=name, age=30, job='unemployed',
    personality_traits=PersonalityTraits(50,50,50,50,50), inventory=ItemInventory(),
    graph_manager=graph, action_system=system, gametime_manager=clock,
    location=Location(name+' home',0,0,10,10,system), memory_mode='basic')
    for name in ('Ada','Bob')]
for char in characters:
    assert char._initialized
    assert char.graph_manager is graph
    assert char.goap_planner.graph_manager is graph
    assert all(a.graph_manager is graph for a in char.character_actions)
characters[0].create_memory('Ada registered an event', datetime(2026,10,20), 4)
assert characters[0].recall_recent_memories()[0].importance_score == 4
assert list(characters[1].recall_recent_memories()) == []
assert all(graph.G.has_node(c) for c in characters)
assert not (blocked & sys.modules.keys())
''')

