import unittest
import tiny_memories
from tiny_memories import SpecificMemory, MemoryManager
import numpy as np
from tiny_memories import SentimentAnalysis, MemoryQuery
from tiny_time_manager import GameTimeManager, GameCalendar
from unittest.mock import MagicMock, patch


class TestSentimentAnalysis(unittest.TestCase):
    def setUp(self):
        self.sa = SentimentAnalysis()

    def test_get_sentiment_score(self):
        text = "I love programming"
        sentiment_score = self.sa.get_sentiment_score(text)
        self.assertIsInstance(sentiment_score, dict)
        self.assertIn('polarity', sentiment_score)
        self.assertIn('subjectivity', sentiment_score)

    def test_get_emotion_classification(self):
        text = "I am very happy"
        emotion = self.sa.get_emotion_classification(text)
        self.assertIsInstance(emotion, str)

class TestMemoryQuery(unittest.TestCase):
    def setUp(self):
        calendar = GameCalendar()
        self.gametime_manager = GameTimeManager()
        self.mq = MemoryQuery("query", "query_time", gametime_manager=self.gametime_manager)

    def test_add_complex_query(self):
        self.mq.add_complex_query("attribute", "query")
        self.assertEqual(self.mq.complex_query["attribute"], "query")

    def test_add_query_function(self):
        def query_function():
            return True
        self.mq.add_query_function(query_function)
        self.assertEqual(self.mq.query_function, query_function)

    def test_by_complex_function(self):
        node = type('Node', (object,), {'memory': type('Memory', (object,), {'description': "memory description"})})()
        self.mq.add_complex_query("attribute", "*attribute* is *memory_description* relevant?")
        self.mq.by_complex_function(node)
        self.assertIn("attribute", self.mq.complex_query)

    def test_by_tags_function(self):
        node = type('Node', (object,), {'memory': type('Memory', (object,), {'tags': ["tag1", "tag2"]})})()
        self.mq.query_tags = ["tag1"]
        self.assertTrue(self.mq.by_tags_function(node))

    def test_by_time_function(self):
        node = type('Node', (object,), {'memory': type('Memory', (object,), {'time': "time"})})()
        self.mq.query_time = "time"
        self.assertIsNone(self.mq.by_time_function(node, None))
        
class TestMemoryManager(unittest.TestCase):
    def setUp(self):
        # Mock dependencies
        self.memory_manager = MemoryManager()
        self.memory_manager.hierarchy = MagicMock()
        self.memory_manager.flat_access = MagicMock()
        self.memory_manager.recent_queries_embeddings = MagicMock()

    def test_init_memories(self):
        general_memories = ['memory1', 'memory2']
        self.memory_manager.init_memories(general_memories)
        self.assertEqual(self.memory_manager.general_memories, general_memories)
        # Verify if `add_general_memory` and `update_embeddings` are called for each memory
        self.assertTrue(self.memory_manager.hierarchy.add_general_memory.called)
        self.assertTrue(self.memory_manager.flat_access.add_memory.called)

    @patch('tiny_memories.faiss.IndexFlatL2')
    def test_index_recent_queries(self, mock_faiss):
        # Mocking embeddings and faiss index for simplicity
        embeddings = MagicMock()
        self.memory_manager.recent_queries_embeddings = [embeddings]
        self.memory_manager.index_recent_queries()
        self.assertIsNotNone(self.memory_manager.faiss_index_recent_queries)
        mock_faiss.assert_called()

    def test_add_general_memory(self):
        general_memory = 'general_memory'
        self.memory_manager.add_general_memory(general_memory)
        self.memory_manager.hierarchy.add_general_memory.assert_called_with(general_memory)
        self.memory_manager.update_embeddings.assert_called_with(general_memory)

    # Example of testing a method that relies on external functionality
    def test_extract_entities(self):
        with patch('tiny_memories.nlp') as mock_nlp:
            mock_nlp.return_value.ents = [MagicMock(text='entity1'), MagicMock(text='entity2')]
            result = MemoryManager.extract_entities('some text')
            self.assertEqual(result, ['entity1', 'entity2'])

    def test_update_embeddings_specific_memory(self):
        specific_memory = MagicMock()
        specific_memory.get_embedding.return_value = 'embedding'
        self.memory_manager.update_embeddings(specific_memory)
        # Check if the embedding is updated correctly
        self.assertIn(specific_memory, self.memory_manager.memory_embeddings)
        self.assertEqual(self.memory_manager.memory_embeddings[specific_memory], 'embedding')

    def test_add_memory(self):
        memory = MagicMock()
        self.memory_manager.add_memory(memory)
        self.memory_manager.flat_access.add_memory.assert_called_with(memory)
        # Assuming update_hierarchy has a meaningful implementation
        # self.memory_manager.update_hierarchy.assert_called_with(memory)

    # Mock external function extract_entities for testing
    @patch('tiny_memories.MemoryManager.extract_entities', return_value=['entity1', 'entity2'])
    def test_extract_keywords(self, mock_extract_entities):
        text = "some random text"
        # Mock methods called within extract_keywords
        with patch.object(self.memory_manager, 'extract_lda_keywords', return_value=set(['keyword1'])), \
             patch.object(self.memory_manager, 'extract_tfidf_keywords', return_value=set(['keyword2'])), \
             patch.object(self.memory_manager, 'extract_rake_keywords', return_value=set(['keyword3'])):
            keywords = self.memory_manager.extract_keywords(text)
            self.assertTrue('entity1' in keywords)
            self.assertTrue('keyword1' in keywords)
            self.assertTrue('keyword2' in keywords)
            self.assertTrue('keyword3' in keywords)

    @patch('tiny_memories.cosine_similarity', return_value=0.5)
    def test_is_relevant_general_memory(self, mock_cosine_similarity):
        general_memory = MagicMock()
        general_memory.tags = ['tag1']
        query = MagicMock()
        query.tags = ['tag1']
        query.analysis = {'keywords': ['keyword1'], 'embedding': 'embedding'}
        general_memory.description = 'description'
        self.memory_manager.get_memory_embedding = MagicMock(return_value='memory_embedding')
        self.memory_manager.extract_keywords = MagicMock(return_value=set(['keyword1']))
        result = self.memory_manager.is_relevant_general_memory(general_memory, query)
        self.assertTrue(result)

    @patch('tiny_memories.cosine_similarity', return_value=0.8)
    
    # This method demonstrates how to test a complex function that includes external dependencies and complex logic.
    # Given the complexity and external dependencies like `nlp`, `tokenizer`, and `model` in `analyze_query_context`,
    # you would mock these dependencies similarly to previous examples, focusing on the return values necessary to
    # test the logic within `analyze_query_context`. This approach requires familiarity with the data structures and
    # types these external functions and methods return.

    # Continue with additional tests for methods like `retrieve_from_hierarchy`, `traverse_specific_memories`,
    # `search_memories`, etc., applying similar mocking and patching strategies to isolate and test specific behaviors.
    def test_update_embeddings_with_specific_memory(self):
        specific_memory = MagicMock()
        specific_memory.get_embedding.return_value = 'embedding'
        self.memory_manager.update_embeddings(specific_memory)
        # Assert the embedding is updated
        specific_memory.get_embedding.assert_called_once()
        self.assertIn(specific_memory, self.memory_manager.memory_embeddings)

    def test_update_embeddings_with_general_memory(self):
        general_memory = MagicMock()
        specific_memory = MagicMock()
        specific_memory.get_embedding.return_value = 'embedding'
        general_memory.specific_memories = [specific_memory]
        self.memory_manager.update_embeddings(general_memory)
        # Assert the embeddings are updated for specific memories within a general memory
        specific_memory.get_embedding.assert_called_once()
        self.assertIn(specific_memory, self.memory_manager.memory_embeddings)

    def test_add_memory(self):
        memory = MagicMock()
        self.memory_manager.add_memory(memory)
        self.memory_manager.flat_access.add_memory.assert_called_with(memory)
        # This method would need more logic if `update_hierarchy` did more.

    @patch('tiny_memories.nlp')
    def test_extract_entities(self, mock_nlp):
        mock_nlp.return_value.ents = [MagicMock(text='entity1'), MagicMock(text='entity2')]
        result = MemoryManager.extract_entities('some text')
        self.assertEqual(result, ['entity1', 'entity2'])

    @patch('tiny_memories.RegexpTokenizer')
    @patch('tiny_memories.corpora.Dictionary')
    @patch('tiny_memories.models.LdaModel')
    def test_extract_lda_keywords(self, mock_lda, mock_dictionary, mock_tokenizer):
        mock_tokenizer.return_value.tokenize.return_value = ['doc1', 'doc2']
        mock_dictionary.return_value.doc2bow.return_value = 'bow'
        mock_lda.return_value.show_topic.return_value = [('keyword', 1)]
        result = MemoryManager.extract_lda_keywords(['doc'], num_topics=1, num_words=1)
        self.assertIn('keyword', result)

    @patch('tiny_memories.TfidfVectorizer')
    def test_extract_tfidf_keywords(self, mock_vectorizer):
        mock_vectorizer.return_value.fit_transform.return_value.toarray.return_value = np.array([[0, 1, 2]])
        mock_vectorizer.return_value.get_feature_names_out.return_value = np.array(['word1', 'word2', 'word3'])
        result = MemoryManager.extract_tfidf_keywords(['doc'])
        self.assertIn('word3', result)

    @patch('tiny_memories.Rake')
    def test_extract_rake_keywords(self, mock_rake):
        mock_rake.return_value.get_ranked_phrases.return_value = ['keyword1', 'keyword2']
        result = MemoryManager.extract_rake_keywords('doc')
        self.assertEqual(result, {'keyword1', 'keyword2'})

    @patch('tiny_memories.tokenizer')
    @patch('tiny_memories.model')
    def test_get_query_embedding(self, mock_model, mock_tokenizer):
        mock_model.return_value.last_hidden_state.mean.return_value.detach.return_value.numpy.return_value = np.array([1])
        mock_tokenizer.return_value = {'input_ids': 1}
        result = self.memory_manager.get_query_embedding('query')
        self.assertEqual(result.shape, (1,))

    # Skipping some methods due to their dependency on complex external resources or the need for extensive mocking.

    def test_cosine_similarity(self):
        vec1 = np.array([1, 0, 0])
        vec2 = np.array([0, 1, 0])
        result = MemoryManager.cosine_similarity(vec1, vec2)
        self.assertEqual(result, 0.0)  # Orthogonal vectors have a cosine similarity of 0

    def test_keyword_specificity(self):
        self.memory_manager.complex_keywords.add('complex')
        result = self.memory_manager.keyword_specificity('complex')
        self.assertEqual(result, 1)
        result = self.memory_manager.keyword_specificity('simple')
        self.assertEqual(result, 0.5)

    @patch('tiny_memories.np')
    def test_normalize_scores(self, mock_np):
        mock_np.array.return_value = np.array([1, 2, 3])
        mock_np.min.return_value = 1
        mock_np.max.return_value = 3 
        result = MemoryManager.normalize_scores([1, 2, 3])
        mock_np.assert_has_calls([call.array([1, 2, 3]), call.min(), call.max()])
        self.assertTrue(np.array_equal(result, np.array([0.0, 0.5, 1.0])))

    # Further tests would require mocking of methods not shown here (e.g., `search_by_tag`, `calculate_recency_score`, etc.),
    # as well as handling of additional dependencies and potentially complex data structures.

    @patch('tiny_memories.cosine_similarity')
    def test_retrieve_from_hierarchy(self, mock_cosine_similarity):
        mock_cosine_similarity.return_value = 0.5
        general_memory = MagicMock()
        self.memory_manager.hierarchy.general_memories = [general_memory]
        self.memory_manager.is_relevant_general_memory = MagicMock(return_value=True)
        self.memory_manager.traverse_specific_memories = MagicMock(return_value=['specific_memory'])
        
        query = MagicMock()
        results = self.memory_manager.retrieve_from_hierarchy(query)
        self.assertIn('specific_memory', results)
        self.memory_manager.is_relevant_general_memory.assert_called_with(general_memory, query)
        self.memory_manager.traverse_specific_memories.assert_called_with(general_memory, query)

    def test_traverse_specific_memories_with_key(self):
        general_memory = MagicMock()
        query = MagicMock()
        key = 'test_key'
        self.memory_manager.get_common_memories = MagicMock(return_value=['memory1', 'memory2'])

        results = self.memory_manager.traverse_specific_memories(general_memory, query, key=key)
        self.assertIn('memory1', results)
        self.memory_manager.get_common_memories.assert_called_with(key)

    @patch('tiny_memories.cosine_similarity')
    def test_is_relevant_general_memory_with_tags(self, mock_cosine_similarity):
        general_memory = MagicMock(tags={'science'})
        query = MagicMock(tags={'science'}, analysis={'keywords': ['research'], 'embedding': np.array([1])})
        self.memory_manager.get_memory_embedding = MagicMock(return_value=np.array([1]))
        mock_cosine_similarity.return_value = 0.9

        result = self.memory_manager.is_relevant_general_memory(general_memory, query)
        self.assertTrue(result)
        mock_cosine_similarity.assert_called()

    def test_retrieve_memories_bst(self):
        specific_memories_myself = [SpecificMemory("I am planning a trip to Europe", "myself", 8), SpecificMemory("I am learning to play the guitar", "myself", 6), SpecificMemory("I am studying for a Chemistry test", "myself", 3)]
        manager.add_general_memory(GeneralMemory("myself", "Memories about myself")).init_specific_memories(specific_memories_myself)
        general_memory.specific_memories_root = MagicMock()  # Assuming this is the root of a BST
        query = MagicMock()
        query.query_function = MagicMock(return_value=True)  # Mocking the query's criteria as always true for simplicity
        
        # Assuming retrieve_memories_bst is modified to accept and return specific memories for simplicity
        self.memory_manager.retrieve_memories_bst(general_memory, query)
        # Validate that the query function was used, implying traversal occurred
        query.query_function.assert_called()

    @patch('tiny_memories.MemoryManager.retrieve_memories')
    def test_search_memories_with_string_query(self, mock_retrieve_memories):
        query = "test query"
        mock_retrieve_memories.return_value = ['memory1', 'memory2']
        result = self.memory_manager.search_memories(query)
        self.assertIn('memory1', result)
        mock_retrieve_memories.assert_called_once()

    @patch('tiny_memories.MemoryManager.retrieve_memories')
    def test_search_memories_with_memory_query_object(self, mock_retrieve_memories):
        query = MagicMock()  # Simulating a MemoryQuery object
        mock_retrieve_memories.return_value = ['memory1']
        result = self.memory_manager.search_memories(query)
        self.assertIn('memory1', result)
        mock_retrieve_memories.assert_called_once_with(query, False, False)

    @patch('tiny_memories.MemoryManager.is_complex_query')
    @patch('tiny_memories.MemoryManager.retrieve_from_hierarchy')
    @patch('tiny_memories.MemoryManager.retrieve_from_flat_access')
    def test_retrieve_memories_for_complex_query(self, mock_retrieve_from_flat_access, mock_retrieve_from_hierarchy, mock_is_complex_query):
        query = MagicMock()
        mock_is_complex_query.return_value = True
        mock_retrieve_from_hierarchy.return_value = ['complex_memory']
        result = self.memory_manager.retrieve_memories(query)
        self.assertIn('complex_memory', result)
        mock_retrieve_from_hierarchy.assert_called_once_with(query)
        mock_retrieve_from_flat_access.assert_not_called()

class TestMemoryManagerIntegration(unittest.TestCase):
    def setUp(self):
        self.memory_manager = MemoryManager()
        # Mock external dependencies here if necessary, for example:
        # self.memory_manager.nlp = MagicMock()
        # self.memory_manager.faiss = MagicMock()

    def test_memory_integration_flow(self):
        # Step 1: Initialize MemoryManager with a set of general memories
        general_memory1 = GeneralMemory(description="General memory about Python programming.", tags={"programming", "python"})
        general_memory2 = GeneralMemory(description="General memory about machine learning.", tags={"machine learning", "data science"})
        self.memory_manager.init_memories([general_memory1, general_memory2])

        # Mock embeddings for the general memories (in a real scenario, this would be handled by an external library)
        self.memory_manager.memory_embeddings[general_memory1] = np.array([0.1, 0.2, 0.3])
        self.memory_manager.memory_embeddings[general_memory2] = np.array([0.4, 0.5, 0.6])

        # Step 2: Add a specific memory to the system
        specific_memory = SpecificMemory(description="Specific memory about Python list comprehensions.", tags={"programming", "python"})
        self.memory_manager.add_memory(specific_memory)

        # Mock embedding for the specific memory
        self.memory_manager.memory_embeddings[specific_memory] = np.array([0.15, 0.25, 0.35])

        # Step 3: Simulate a query that retrieves memories related to Python programming
        query = {"tags": {"python"}, "description": "Looking for information on Python programming."}
        # Mock the analysis of the query to match the tags and content
        self.memory_manager.analyze_query_context = MagicMock(return_value={"tags": {"python"}, "keywords": ["Python", "programming"]})

        # Retrieve memories based on the query
        retrieved_memories = self.memory_manager.retrieve_memories(query)

        # Verify that the retrieved memories are relevant to the query
        self.assertIn(general_memory1, retrieved_memories)
        self.assertIn(specific_memory, retrieved_memories)
        self.assertNotIn(general_memory2, retrieved_memories)

class TestMemory(unittest.TestCase):
    def setUp(self):
        self.memory = Memory("description", "creation_time", 0)

    def test_update_access_time(self):
        access_time = "new_access_time"
        self.memory.update_access_time(access_time)
        self.assertEqual(self.memory.access_time, access_time)

class TestBSTNode(unittest.TestCase):
    def setUp(self):
        self.node = BSTNode("key", "memory")

    def test_update_height(self):
        self.node.update_height()
        self.assertIsNotNone(self.node.height)

    def test_get_balance(self):
        balance = self.node.get_balance()
        self.assertIsInstance(balance, int)

class TestMemoryBST(unittest.TestCase):
    def setUp(self):
        self.bst = MemoryBST("key_attr")

    def test_get_height(self):
        height = self.bst.get_height(self.bst.root)
        self.assertIsInstance(height, int)

    def test_update_height(self):
        self.bst.update_height(self.bst.root)
        self.assertIsNotNone(self.bst.root.height)

    def test_get_balance(self):
        balance = self.bst.get_balance(self.bst.root)
        self.assertIsInstance(balance, int)

    def test_right_rotate(self):
        self.bst.right_rotate(self.bst.root)
        self.assertIsNotNone(self.bst.root)

    def test_left_rotate(self):
        self.bst.left_rotate(self.bst.root)
        self.assertIsNotNone(self.bst.root)

    def test_insert(self):
        memory = Memory("description", "creation_time", 0)
        self.bst.insert(self.bst.root, "key", memory)
        self.assertIsNotNone(self.bst.root)



from tiny_memories import GeneralMemory, SpecificMemory

class TestGeneralMemory(unittest.TestCase):
    def setUp(self):
        self.gm = GeneralMemory("attribute", "description")

    def test_init_specific_memories(self):
        specific_memories = [SpecificMemory("description", self.gm, 0)]
        self.gm.init_specific_memories(specific_memories)
        self.assertEqual(self.gm.specific_memories, specific_memories)

    def test_init_trees(self):
        self.gm.init_trees()
        self.assertIsNotNone(self.gm.timestamp_tree)
        self.assertIsNotNone(self.gm.importance_tree)

    def test_add_specific_memory(self):
        specific_memory = SpecificMemory("description", self.gm, 0)
        self.gm.add_specific_memory(specific_memory)
        self.assertIn(specific_memory, self.gm.specific_memories)

    def test_update_embeddings(self):
        specific_memory = SpecificMemory("description", self.gm, 0)
        self.gm.update_embeddings(specific_memory)
        self.assertIn(specific_memory, self.gm.memory_embeddings)

    def test_index_memories(self):
        self.gm.index_memories()
        self.assertIsNotNone(self.gm.faiss_index)

class TestSpecificMemory(unittest.TestCase):
    def setUp(self):
        self.gm = GeneralMemory("attribute", "description")
        self.sm = SpecificMemory("description", self.gm, 0)

    def test_update_importance_score(self):
        importance_score = 10
        self.sm.update_importance_score(importance_score)
        self.assertEqual(self.sm.importance_score, importance_score)

    def test_update_parent_memory(self):
        new_gm = GeneralMemory("new_attribute", "new_description")
        self.sm.update_parent_memory(new_gm)
        self.assertEqual(self.sm.parent_memory, new_gm)

    def test_update_related_memories(self):
        related_memories = [SpecificMemory("related_description", self.gm, 0)]
        self.sm.update_related_memories(related_memories)
        self.assertEqual(self.sm.related_memories, related_memories)

    def test_update_keywords(self):
        keywords = ["keyword1", "keyword2"]
        self.sm.update_keywords(keywords)
        self.assertEqual(self.sm.keywords, keywords)

    def test_add_related_memory(self):
        related_memory = SpecificMemory("related_description", self.gm, 0)
        self.sm.add_related_memory(related_memory)
        self.assertIn(related_memory, self.sm.related_memories)

    def test_get_embedding(self):
        embedding = self.sm.get_embedding()
        self.assertIsNotNone(embedding)

if __name__ == '__main__':
    unittest.main()


if __name__ == '__main__':
    unittest.main()
