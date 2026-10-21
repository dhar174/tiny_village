"""Explicit checks for installed NLP/retrieval profiles; no model downloads."""

from tests.test_graph_initialization import FreshPythonTestCase


class TestInstalledNLP(FreshPythonTestCase):
    def test_real_spacy_present_basic_runtime_does_not_initialize_services(self):
        self.run_python('''
import spacy, nltk
from unittest.mock import patch
document = spacy.blank('en')('Ada remembers an event')
assert isinstance(document[0], spacy.tokens.Token)
with patch.object(spacy, 'load', side_effect=AssertionError('Unexpected spaCy model load')), \
     patch.object(nltk, 'download', side_effect=AssertionError('Unexpected NLTK download')):
    from demo_minimal_integration import demonstrate_minimal_integration
    controller = demonstrate_minimal_integration({'characters':{'count':2}})
    assert all(len(c.recall_recent_memories()) == 1 for c in controller.characters.values())
    import tiny_memories as memory
    assert memory.nlp is None and memory.model is None and memory.sentiment_analysis is None
''')

    def test_full_nlp_missing_model_has_explicit_failure_and_cause(self):
        self.run_python('''
import spacy
from unittest.mock import patch
from tiny_memories import MemoryManager, MemoryFeatureUnavailableError
from tiny_time_manager import GameCalendar, GameTimeManager
manager = MemoryManager(GameTimeManager(GameCalendar()))
with patch.object(spacy, 'load', side_effect=OSError('Controlled missing model assets')):
    try:
        manager.extract_entities('Ada remembers an event')
    except MemoryFeatureUnavailableError as error:
        assert isinstance(error.__cause__, OSError)
        assert 'model' in str(error)
    else:
        raise AssertionError('Missing model silently became ready')
assert manager.get_capabilities()['nlp'] == 'unavailable'
''')


class TestInstalledRetrieval(FreshPythonTestCase):
    def test_real_faiss_index_is_lazy_and_works_without_loading_embedding_models(self):
        self.run_python('''
import sys, numpy as np
from tiny_memories import MemoryManager
from tiny_time_manager import GameCalendar, GameTimeManager
manager = MemoryManager(GameTimeManager(GameCalendar()))
assert 'faiss' not in sys.modules
assert 'torch' not in sys.modules
assert manager.flat_access.faiss_index is None
manager.flat_access.initialize_faiss_index(3, 'ip')
index = manager.flat_access.faiss_index
index.add(np.array([[1,0,0]], dtype='float32'))
scores, identifiers = index.search(np.array([[1,0,0]], dtype='float32'), 1)
assert identifiers[0,0] == 0 and scores[0,0] == 1
assert manager.get_capabilities()['vector_index'] == 'ready'
assert manager.get_capabilities()['embeddings'] == 'not_initialized'
assert 'torch' not in sys.modules
manager._require_feature('vector_index')
assert manager.flat_access.faiss_index is index
''')
