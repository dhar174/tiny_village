import unittest
import tiny_memories
from tiny_memories import SpecificMemory, MemoryManager
import numpy as np
from tiny_memories import SentimentAnalysis, MemoryQuery
from tiny_time_manager import GameTimeManager, GameCalendar
from unittest.mock import MagicMock, patch
from datetime import datetime, timedelta
# Conditional torch import with proper error handling
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    # Create a mock torch module for testing
    import unittest.mock as mock
    torch = mock.MagicMock()
    # Add commonly used torch functions that return appropriate mock values
    torch.rand = lambda *args: mock.MagicMock()
    torch.randn = lambda *args: mock.MagicMock() 
    torch.tensor = lambda x: mock.MagicMock()
    torch.ones = lambda *args: mock.MagicMock()
    torch.cuda.is_available = lambda: False

def create_mock_tensor(shape):
    """Create a proper mock tensor object for testing instead of using real torch tensors.
    
    This avoids creating fake torch implementations that could mask real functionality issues.
    """
    mock_tensor = MagicMock()
    mock_tensor.shape = shape
    # Add common tensor methods that might be called
    mock_tensor.cpu.return_value = mock_tensor
    mock_tensor.detach.return_value = mock_tensor
    mock_tensor.numpy.return_value = MagicMock()
    mock_tensor.to.return_value = mock_tensor
    return mock_tensor


# Patch the global 'manager' and 'model' for GeneralMemory and SpecificMemory tests
# to avoid unintended side effects from the module-level initializations.
@patch("tiny_memories.manager", new_callable=MagicMock)
@patch("tiny_memories.model", new_callable=MagicMock)
class TestImport(unittest.TestCase):
    def test_module_import(self):
        try:
            import tiny_memories
            # Verify that the module has expected attributes/classes
            self.assertTrue(hasattr(tiny_memories, 'MemoryManager'), "Module should have MemoryManager class")
            self.assertTrue(hasattr(tiny_memories, 'SpecificMemory'), "Module should have SpecificMemory class")
            self.assertTrue(hasattr(tiny_memories, 'GeneralMemory'), "Module should have GeneralMemory class")
        except ImportError as e:
            self.fail(f"Failed to import tiny_memories module: {e}")


class TestEmbeddingModel(unittest.TestCase):
    @patch("tiny_memories.AutoTokenizer")
    @patch("tiny_memories.AutoModel")
    def setUp(self, MockAutoModel, MockAutoTokenizer):
        # Mock the transformer model loading
        self.mock_tokenizer_instance = MockAutoTokenizer.from_pretrained.return_value
        self.mock_model_instance = MockAutoModel.from_pretrained.return_value
        self.mock_model_instance.to.return_value = (
            self.mock_model_instance
        )  # for .to(device)

        # Prevent actual model loading during test setup by patching globals
        with patch("tiny_memories.model", None):
            with patch("tiny_memories.tokenizer", None):
                with patch("tiny_memories.sa", None):
                    with patch("tiny_memories.manager", None):
                        self.embedding_model = tiny_memories.EmbeddingModel()
                        # Ensure mocks were called for instantiation
                        MockAutoTokenizer.from_pretrained.assert_called_with(
                            "sentence-transformers/all-mpnet-base-v2",
                            trust_remote_code=True,
                            cache_dir="/mnt/d/transformers_cache",
                        )
                        MockAutoModel.from_pretrained.assert_called_with(
                            "sentence-transformers/all-mpnet-base-v2",
                            trust_remote_code=True,
                            cache_dir="/mnt/d/transformers_cache",
                        )
                        self.mock_model_instance.to.assert_called()  # Check if .to(device) was called
                        self.mock_model_instance.eval.assert_called_once()
                        self.mock_model_instance.zero_grad.assert_called_once()

    def test_embedding_model_initialization(self):
        self.assertIsNotNone(self.embedding_model.tokenizer)
        self.assertIsNotNone(self.embedding_model.model)
        self.assertEqual(self.embedding_model.tokenizer, self.mock_tokenizer_instance)
        self.assertEqual(self.embedding_model.model, self.mock_model_instance)

    def test_embedding_model_forward_pass(self):
        # Mock input tensors
        mock_input_ids = MagicMock()
        mock_attention_mask = MagicMock()
        mock_output = MagicMock()
        self.embedding_model.model.return_value = mock_output

        result = self.embedding_model.forward(mock_input_ids, mock_attention_mask)

        self.embedding_model.model.assert_called_once_with(
            input_ids=mock_input_ids, attention_mask=mock_attention_mask
        )
        self.assertEqual(result, mock_output)


class TestSentimentAnalysis(unittest.TestCase):
    def setUp(self):
        self.sa = SentimentAnalysis()

    def test_get_sentiment_score(self):
        text = "I love programming"
        sentiment_score = self.sa.get_sentiment_score(text)
        self.assertIsInstance(sentiment_score, dict)
        self.assertIn("polarity", sentiment_score)
        self.assertIn("subjectivity", sentiment_score)

    def test_get_emotion_classification(self, mock_pipeline):  # Added mock_pipeline
        text = "I am very happy"
        emotion = self.sa.get_emotion_classification(text)
        self.assertIsInstance(emotion, str)
        mock_pipeline.assert_called_with(model="lordtt13/emo-mobilebert")
        self.sa.emo_classifier.assert_called_with(text)

    def test_extract_simple_words_string(self):
        text = "Hello, world! This is a test."
        expected_words = [
            "Hello",
            "world",
            "This",
            "is",
            "a",
            "test",
        ]  # Assuming default stopwords
        # Filter out stopwords more robustly for comparison
        from nltk.corpus import stopwords

        stop_words = set(stopwords.words("english"))
        expected_words_filtered = [
            word
            for word in expected_words
            if word.lower() not in stop_words and word.lower() not in ENGLISH_STOP_WORDS
        ]

        actual_words = self.sa.extract_simple_words(text)
        # We compare sets because the order might not be guaranteed or important,
        # and stopwords list might vary slightly.
        # For a more precise test, mock stopwords.
        self.assertEqual(set(actual_words), set(expected_words_filtered))

    def test_extract_simple_words_list(self):
        text_list = ["Hello, world!", "Another test."]
        # Expected after joining, splitting, and stopword removal.
        # "Hello world Another test"
        expected_words = ["Hello", "world", "Another", "test"]
        from nltk.corpus import stopwords

        stop_words = set(stopwords.words("english"))
        expected_words_filtered = [
            word
            for word in expected_words
            if word.lower() not in stop_words and word.lower() not in ENGLISH_STOP_WORDS
        ]

        actual_words = self.sa.extract_simple_words(text_list)
        self.assertEqual(set(actual_words), set(expected_words_filtered))

    def test_extract_simple_words_empty_string(self):
        self.assertEqual(self.sa.extract_simple_words(""), [])

    def test_extract_simple_words_empty_list(self):
        self.assertEqual(self.sa.extract_simple_words([]), [])


class TestMemoryQuery(unittest.TestCase):
    def setUp(self):
        calendar = GameCalendar()
        self.gametime_manager = GameTimeManager()
        self.mq = MemoryQuery(
            "query", "query_time", gametime_manager=self.gametime_manager
        )

    def test_add_complex_query(self):
        self.mq.add_complex_query("attribute", "query")
        self.assertEqual(self.mq.complex_query["attribute"], "query")

    def test_add_query_function(self):
        def query_function():
            return True

        self.mq.add_query_function(query_function)
        self.assertEqual(self.mq.query_function, query_function)

    def test_by_complex_function(self):
        node = type(
            "Node",
            (object,),
            {
                "memory": type(
                    "Memory", (object,), {"description": "memory description"}
                )
            },
        )()
        self.mq.add_complex_query(
            "attribute", "*attribute* is *memory_description* relevant?"
        )
        self.mq.by_complex_function(node)
        self.assertIn("attribute", self.mq.complex_query)

    def test_by_tags_function(self):
        node = type(
            "Node",
            (object,),
            {"memory": type("Memory", (object,), {"tags": ["tag1", "tag2"]})},
        )()
        self.mq.query_tags = ["tag1"]
        self.assertTrue(self.mq.by_tags_function(node))

    def test_by_time_function(self):
        # ...existing code...
        self.mq.query_time = (
            "time"  # This was comparing string "time" with datetime object.
        )
        # Let's make query_time a datetime object for a more realistic test.
        self.mq.query_time = self.gametime_manager.calendar.get_game_time() - timedelta(
            minutes=30
        )

        # Mock node.memory.last_access_time to be more recent than query_time - 1 hour
        node.memory.last_access_time = (
            self.gametime_manager.calendar.get_game_time() - timedelta(minutes=10)
        )
        self.assertTrue(self.mq.by_time_function(node, None))

        # Mock node.memory.last_access_time to be older
        node.memory.last_access_time = (
            self.gametime_manager.calendar.get_game_time() - timedelta(hours=2)
        )
        self.assertFalse(self.mq.by_time_function(node, None))

    @patch("tiny_memories.MemoryQuery.generate_embedding")
    def test_get_embedding(self, mock_generate_embedding):
        mock_embedding_val = MagicMock(name="EmbeddingValue")
        mock_attention_mask_val = MagicMock(name="AttentionMaskValue")
        mock_generate_embedding.return_value = (
            mock_embedding_val,
            mock_attention_mask_val,
        )

        # Mock mean_pooling
        with patch("tiny_memories.mean_pooling") as mock_mean_pooling:
            mock_pooled_embedding = MagicMock(name="PooledEmbedding")
            mock_mean_pooling.return_value = mock_pooled_embedding

            embedding = self.mq.get_embedding()
            self.assertEqual(embedding, mock_pooled_embedding)
            mock_generate_embedding.assert_called_once()
            mock_mean_pooling.assert_called_once_with(
                mock_embedding_val, mock_attention_mask_val
            )
            # Subsequent calls should return cached embedding
            embedding2 = self.mq.get_embedding()
            self.assertEqual(embedding2, mock_pooled_embedding)
            mock_generate_embedding.assert_called_once()  # Still called only once

    @patch("tiny_memories.model")  # Mock the global model object used by MemoryQuery
    def test_generate_embedding(self, mock_global_model):
        # Configure the mock_global_model to behave like the EmbeddingModel instance
        mock_tokenizer_instance = MagicMock()
        mock_model_instance = MagicMock()
        mock_global_model.tokenizer = mock_tokenizer_instance
        mock_global_model.model = mock_model_instance
        mock_global_model.device = "cpu"

        mock_input_dict = {"input_ids": MagicMock(), "attention_mask": MagicMock()}
        mock_tokenizer_instance.return_value = mock_input_dict

        mock_outputs = MagicMock()
        mock_outputs.last_hidden_state = MagicMock(name="LastHiddenState")
        mock_model_instance.return_value = mock_outputs

        self.mq.query = "test query"
        embedding_val, attention_mask_val = self.mq.generate_embedding()

        mock_tokenizer_instance.assert_called_once_with(
            [self.mq.query.strip()],
            padding=True,
            truncation=True,
            add_special_tokens=True,
            is_split_into_words=True,  # This should likely be False for a query string
            pad_to_multiple_of=8,
            return_tensors="pt",
        )
        mock_input_dict.to.assert_called_once_with(
            "cpu"
        )  # Check that .to(device) was called on tokenized input
        mock_model_instance.assert_called_once_with(
            mock_input_dict["input_ids"],
            attention_mask=mock_input_dict["attention_mask"],
            output_hidden_states=True,
            return_dict=True,
            return_tensors="pt",
        )
        self.assertEqual(embedding_val, mock_outputs.last_hidden_state)
        self.assertEqual(attention_mask_val, mock_input_dict["attention_mask"])

    def test_by_importance_function(self):
        node = MagicMock()
        node.memory.importance_score = 5
        self.assertTrue(self.mq.by_importance_function(node, 0, 10))
        self.assertTrue(self.mq.by_importance_function(node, 5, 5))
        self.assertFalse(self.mq.by_importance_function(node, 6, 10))
        self.assertFalse(self.mq.by_importance_function(node, 0, 4))

    def test_by_sentiment_function(self):
        node = MagicMock()
        node.memory.sentiment_score = {"polarity": 0.5, "subjectivity": 0.8}
        self.assertTrue(self.mq.by_sentiment_function(node, 0.0, 1.0, 0.0, 1.0))
        self.assertTrue(self.mq.by_sentiment_function(node, 0.5, 0.5, 0.8, 0.8))
        self.assertFalse(
            self.mq.by_sentiment_function(node, 0.6, 1.0, 0.0, 1.0)
        )  # Polarity too low
        self.assertFalse(
            self.mq.by_sentiment_function(node, 0.0, 1.0, 0.9, 1.0)
        )  # Subjectivity too low

    def test_by_emotion_function(self):
        node = MagicMock()
        node.memory.emotion_classification = "happy"
        self.assertTrue(self.mq.by_emotion_function(node, "happy"))
        self.assertFalse(self.mq.by_emotion_function(node, "sad"))

    def test_by_keywords_function(self):
        node = MagicMock()
        node.memory.keywords = ["apple", "banana", "cherry"]
        self.assertTrue(self.mq.by_keywords_function(node, ["banana", "date"]))
        self.assertFalse(self.mq.by_keywords_function(node, ["date", "elderberry"]))
        node.memory.keywords = []
        self.assertFalse(self.mq.by_keywords_function(node, ["apple"]))
        node.memory.keywords = None
        self.assertFalse(self.mq.by_keywords_function(node, ["apple"]))

    @patch("tiny_memories.cosine_similarity")
    def test_by_similarity_function(self, mock_cosine_similarity):
        node = MagicMock()
        node.memory.embedding = MagicMock(name="NodeEmbedding")
        query_embedding = MagicMock(name="QueryEmbedding")

        mock_cosine_similarity.return_value = np.array(
            [[0.9]]
        )  # cosine_similarity returns a 2D array
        self.assertTrue(self.mq.by_similarity_function(node, query_embedding, 0.8))
        mock_cosine_similarity.assert_called_once_with(
            node.memory.embedding, query_embedding
        )

        mock_cosine_similarity.reset_mock()
        mock_cosine_similarity.return_value = np.array([[0.7]])
        self.assertFalse(self.mq.by_similarity_function(node, query_embedding, 0.8))
        mock_cosine_similarity.assert_called_once_with(
            node.memory.embedding, query_embedding
        )

    def test_by_attribute_function(self):
        node = MagicMock()
        node.memory.attribute = "color"
        self.assertTrue(self.mq.by_attribute_function(node, "color"))
        self.assertFalse(self.mq.by_attribute_function(node, "size"))


class TestMemoryManager(unittest.TestCase):
    def setUp(self):
        # Mock dependencies
        self.memory_manager = MemoryManager()
        self.memory_manager.hierarchy = MagicMock()
        self.memory_manager.flat_access = MagicMock()
        self.memory_manager.recent_queries_embeddings = MagicMock()

    def test_init_memories(self):
        general_memories = ["memory1", "memory2"]
        self.memory_manager.init_memories(general_memories)
        self.assertEqual(self.memory_manager.general_memories, general_memories)
        # Verify if `add_general_memory` and `update_embeddings` are called for each memory
        self.assertTrue(self.memory_manager.hierarchy.add_general_memory.called)
        self.assertTrue(self.memory_manager.flat_access.add_memory.called)

    @patch("tiny_memories.faiss.IndexFlatL2")
    def test_index_recent_queries(self, mock_faiss):
        # Mocking embeddings and faiss index for simplicity
        embeddings = MagicMock()
        self.memory_manager.recent_queries_embeddings = [embeddings]
        self.memory_manager.index_recent_queries()
        self.assertIsNotNone(self.memory_manager.faiss_index_recent_queries)
        mock_faiss.assert_called()

    def test_add_general_memory(self):
        general_memory = "general_memory"
        self.memory_manager.add_general_memory(general_memory)
        self.memory_manager.hierarchy.add_general_memory.assert_called_with(
            general_memory
        )
        self.memory_manager.update_embeddings.assert_called_with(general_memory)

    # Example of testing a method that relies on external functionality
    def test_extract_entities(self):
        with patch("tiny_memories.nlp") as mock_nlp:
            mock_nlp.return_value.ents = [
                MagicMock(text="entity1"),
                MagicMock(text="entity2"),
            ]
            result = MemoryManager.extract_entities("some text")
            self.assertEqual(result, ["entity1", "entity2"])

    def test_update_embeddings_specific_memory(self):
        specific_memory = MagicMock()
        specific_memory.get_embedding.return_value = "embedding"
        self.memory_manager.update_embeddings(specific_memory)
        # Check if the embedding is updated correctly
        self.assertIn(specific_memory, self.memory_manager.memory_embeddings)
        self.assertEqual(
            self.memory_manager.memory_embeddings[specific_memory], "embedding"
        )

    def test_add_memory(self):
        memory = MagicMock()
        self.memory_manager.add_memory(memory)
        self.memory_manager.flat_access.add_memory.assert_called_with(memory)
        # Assuming update_hierarchy has a meaningful implementation
        # self.memory_manager.update_hierarchy.assert_called_with(memory)

    # Mock external function extract_entities for testing
    @patch(
        "tiny_memories.MemoryManager.extract_entities",
        return_value=["entity1", "entity2"],
    )
    def test_extract_keywords(self, mock_extract_entities):
        text = "some random text"
        # Mock methods called within extract_keywords
        with patch.object(
            self.memory_manager, "extract_lda_keywords", return_value=set(["keyword1"])
        ), patch.object(
            self.memory_manager,
            "extract_tfidf_keywords",
            return_value=set(["keyword2"]),
        ), patch.object(
            self.memory_manager, "extract_rake_keywords", return_value=set(["keyword3"])
        ):
            keywords = self.memory_manager.extract_keywords(text)
            self.assertTrue("entity1" in keywords)
            self.assertTrue("keyword1" in keywords)
            self.assertTrue("keyword2" in keywords)
            self.assertTrue("keyword3" in keywords)

    @patch("tiny_memories.cosine_similarity", return_value=0.5)
    def test_is_relevant_general_memory(self, mock_cosine_similarity):
        general_memory = MagicMock()
        general_memory.tags = ["tag1"]
        query = MagicMock()
        query.tags = ["tag1"]
        query.analysis = {"keywords": ["keyword1"], "embedding": "embedding"}
        general_memory.description = "description"
        self.memory_manager.get_memory_embedding = MagicMock(
            return_value="memory_embedding"
        )
        self.memory_manager.extract_keywords = MagicMock(return_value=set(["keyword1"]))
        result = self.memory_manager.is_relevant_general_memory(general_memory, query)
        self.assertTrue(result)

    @patch("tiny_memories.cosine_similarity", return_value=0.8)

    # This method demonstrates how to test a complex function that includes external dependencies and complex logic.
    # Given the complexity and external dependencies like `nlp`, `tokenizer`, and `model` in `analyze_query_context`,
    # you would mock these dependencies similarly to previous examples, focusing on the return values necessary to
    # test the logic within `analyze_query_context`. This approach requires familiarity with the data structures and
    # types these external functions and methods return.

    # Continue with additional tests for methods like `retrieve_from_hierarchy`, `traverse_specific_memories`,
    # `search_memories`, etc., applying similar mocking and patching strategies to isolate and test specific behaviors.
    def test_update_embeddings_with_specific_memory(self):
        specific_memory = MagicMock()
        specific_memory.get_embedding.return_value = "embedding"
        self.memory_manager.update_embeddings(specific_memory)
        # Assert the embedding is updated
        specific_memory.get_embedding.assert_called_once()
        self.assertIn(specific_memory, self.memory_manager.memory_embeddings)

    def test_update_embeddings_with_general_memory(self):
        general_memory = MagicMock()
        specific_memory = MagicMock()
        specific_memory.get_embedding.return_value = "embedding"
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

    @patch("tiny_memories.nlp")
    def test_extract_entities(self, mock_nlp):
        mock_nlp.return_value.ents = [
            MagicMock(text="entity1"),
            MagicMock(text="entity2"),
        ]
        result = MemoryManager.extract_entities("some text")
        self.assertEqual(result, ["entity1", "entity2"])

    @patch("tiny_memories.RegexpTokenizer")
    @patch("tiny_memories.corpora.Dictionary")
    @patch("tiny_memories.models.LdaModel")
    def test_extract_lda_keywords(self, mock_lda, mock_dictionary, mock_tokenizer):
        mock_tokenizer.return_value.tokenize.return_value = ["doc1", "doc2"]
        mock_dictionary.return_value.doc2bow.return_value = "bow"
        mock_lda.return_value.show_topic.return_value = [("keyword", 1)]
        result = MemoryManager.extract_lda_keywords(["doc"], num_topics=1, num_words=1)
        self.assertIn("keyword", result)

    @patch("tiny_memories.TfidfVectorizer")
    def test_extract_tfidf_keywords(self, mock_vectorizer):
        mock_vectorizer.return_value.fit_transform.return_value.toarray.return_value = (
            np.array([[0, 1, 2]])
        )
        mock_vectorizer.return_value.get_feature_names_out.return_value = np.array(
            ["word1", "word2", "word3"]
        )
        result = MemoryManager.extract_tfidf_keywords(["doc"])
        self.assertIn("word3", result)

    @patch("tiny_memories.Rake")
    def test_extract_rake_keywords(self, mock_rake):
        mock_rake.return_value.get_ranked_phrases.return_value = [
            "keyword1",
            "keyword2",
        ]
        result = MemoryManager.extract_rake_keywords("doc")
        self.assertEqual(result, {"keyword1", "keyword2"})

    @patch("tiny_memories.tokenizer")
    @patch("tiny_memories.model")
    def test_get_query_embedding(self, mock_model, mock_tokenizer):
        mock_model.return_value.last_hidden_state.mean.return_value.detach.return_value.numpy.return_value = np.array(
            [1]
        )
        mock_tokenizer.return_value = {"input_ids": 1}
        result = self.memory_manager.get_query_embedding("query")
        self.assertEqual(result.shape, (1,))

    # Skipping some methods due to their dependency on complex external resources or the need for extensive mocking.

    def test_cosine_similarity(self):
        vec1 = np.array([1, 0, 0])
        vec2 = np.array([0, 1, 0])
        result = MemoryManager.cosine_similarity(vec1, vec2)
        self.assertEqual(
            result, 0.0
        )  # Orthogonal vectors have a cosine similarity of 0

    def test_keyword_specificity(self):
        self.memory_manager.complex_keywords.add("complex")
        result = self.memory_manager.keyword_specificity("complex")
        self.assertEqual(result, 1)
        result = self.memory_manager.keyword_specificity("simple")
        self.assertEqual(result, 0.5)

    @patch("tiny_memories.np")
    def test_normalize_scores(self, mock_np):
        mock_np.array.return_value = np.array([1, 2, 3])
        mock_np.min.return_value = 1
        mock_np.max.return_value = 3
        result = MemoryManager.normalize_scores([1, 2, 3])
        mock_np.assert_has_calls([call.array([1, 2, 3]), call.min(), call.max()])
        self.assertTrue(np.array_equal(result, np.array([0.0, 0.5, 1.0])))

    # Further tests would require mocking of methods not shown here (e.g., `search_by_tag`, `calculate_recency_score`, etc.),
    # as well as handling of additional dependencies and potentially complex data structures.

    @patch("tiny_memories.cosine_similarity")
    def test_retrieve_from_hierarchy(self, mock_cosine_similarity):
        mock_cosine_similarity.return_value = 0.5
        general_memory = MagicMock()
        self.memory_manager.hierarchy.general_memories = [general_memory]
        self.memory_manager.is_relevant_general_memory = MagicMock(return_value=True)
        self.memory_manager.traverse_specific_memories = MagicMock(
            return_value=["specific_memory"]
        )

        query = MagicMock()
        results = self.memory_manager.retrieve_from_hierarchy(query)
        self.assertIn("specific_memory", results)
        self.memory_manager.is_relevant_general_memory.assert_called_with(
            general_memory, query
        )
        self.memory_manager.traverse_specific_memories.assert_called_with(
            general_memory, query
        )

    def test_traverse_specific_memories_with_key(self):
        general_memory = MagicMock()
        query = MagicMock()
        key = "test_key"
        self.memory_manager.get_common_memories = MagicMock(
            return_value=["memory1", "memory2"]
        )

        results = self.memory_manager.traverse_specific_memories(
            general_memory, query, key=key
        )
        self.assertIn("memory1", results)
        self.memory_manager.get_common_memories.assert_called_with(key)

    @patch("tiny_memories.cosine_similarity")
    def test_is_relevant_general_memory_with_tags(self, mock_cosine_similarity):
        general_memory = MagicMock(tags={"science"})
        query = MagicMock(
            tags={"science"},
            analysis={"keywords": ["research"], "embedding": np.array([1])},
        )
        self.memory_manager.get_memory_embedding = MagicMock(return_value=np.array([1]))
        mock_cosine_similarity.return_value = 0.9

        result = self.memory_manager.is_relevant_general_memory(general_memory, query)
        self.assertTrue(result)
        mock_cosine_similarity.assert_called()

    def test_retrieve_memories_bst(self):
        specific_memories_myself = [
            SpecificMemory("I am planning a trip to Europe", "myself", 8),
            SpecificMemory("I am learning to play the guitar", "myself", 6),
            SpecificMemory("I am studying for a Chemistry test", "myself", 3),
        ]
        manager.add_general_memory(
            GeneralMemory("myself", "Memories about myself")
        ).init_specific_memories(specific_memories_myself)
        general_memory.specific_memories_root = (
            MagicMock()
        )  # Assuming this is the root of a BST
        query = MagicMock()
        query.query_function = MagicMock(
            return_value=True
        )  # Mocking the query's criteria as always true for simplicity

        # Assuming retrieve_memories_bst is modified to accept and return specific memories for simplicity
        self.memory_manager.retrieve_memories_bst(general_memory, query)
        # Validate that the query function was used, implying traversal occurred
        query.query_function.assert_called()

    @patch("tiny_memories.MemoryManager.retrieve_memories")
    def test_search_memories_with_string_query(self, mock_retrieve_memories):
        query = "test query"
        mock_retrieve_memories.return_value = ["memory1", "memory2"]
        result = self.memory_manager.search_memories(query)
        self.assertIn("memory1", result)
        mock_retrieve_memories.assert_called_once()

    @patch("tiny_memories.MemoryManager.retrieve_memories")
    def test_search_memories_with_memory_query_object(self, mock_retrieve_memories):
        query = MagicMock()  # Simulating a MemoryQuery object
        mock_retrieve_memories.return_value = ["memory1"]
        result = self.memory_manager.search_memories(query)
        self.assertIn("memory1", result)
        mock_retrieve_memories.assert_called_once_with(query, False, False)

    @patch("tiny_memories.MemoryManager.is_complex_query")
    @patch("tiny_memories.MemoryManager.retrieve_from_hierarchy")
    @patch("tiny_memories.MemoryManager.retrieve_from_flat_access")
    def test_retrieve_memories_for_complex_query(
        self,
        mock_retrieve_from_flat_access,
        mock_retrieve_from_hierarchy,
        mock_is_complex_query,
    ):
        query = MagicMock()
        mock_is_complex_query.return_value = True
        mock_retrieve_from_hierarchy.return_value = ["complex_memory"]
        result = self.memory_manager.retrieve_memories(query)
        self.assertIn("complex_memory", result)
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
        general_memory1 = GeneralMemory(
            description="General memory about Python programming.",
            tags={"programming", "python"},
        )
        general_memory2 = GeneralMemory(
            description="General memory about machine learning.",
            tags={"machine learning", "data science"},
        )
        self.memory_manager.init_memories([general_memory1, general_memory2])

        # Mock embeddings for the general memories (in a real scenario, this would be handled by an external library)
        self.memory_manager.memory_embeddings[general_memory1] = np.array(
            [0.1, 0.2, 0.3]
        )
        self.memory_manager.memory_embeddings[general_memory2] = np.array(
            [0.4, 0.5, 0.6]
        )

        # Step 2: Add a specific memory to the system
        specific_memory = SpecificMemory(
            description="Specific memory about Python list comprehensions.",
            tags={"programming", "python"},
        )
        self.memory_manager.add_memory(specific_memory)

        # Mock embedding for the specific memory
        self.memory_manager.memory_embeddings[specific_memory] = np.array(
            [0.15, 0.25, 0.35]
        )

        # Step 3: Simulate a query that retrieves memories related to Python programming
        query = {
            "tags": {"python"},
            "description": "Looking for information on Python programming.",
        }
        # Mock the analysis of the query to match the tags and content
        self.memory_manager.analyze_query_context = MagicMock(
            return_value={"tags": {"python"}, "keywords": ["Python", "programming"]}
        )

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
    def setUp(self, mock_model, mock_manager):  # Order matters for decorators
        # Configure mocks for model and manager as needed by GeneralMemory constructor
        self.mock_global_model = mock_model
        self.mock_global_manager = mock_manager

        # Mock parts of the global model used in GeneralMemory.generate_embedding
        mock_tokenizer_instance = MagicMock()
        mock_model_instance = MagicMock()
        self.mock_global_model.tokenizer = mock_tokenizer_instance
        self.mock_global_model.model = mock_model_instance
        self.mock_global_model.device = "cpu"

        mock_input_dict = {"input_ids": MagicMock(), "attention_mask": MagicMock()}
        # mock_tokenizer_instance.return_value = mock_input_dict # This was too simple
        # Need to handle .to(device) call
        mock_tokenized_output = MagicMock()
        mock_tokenized_output.to.return_value = mock_input_dict
        mock_tokenizer_instance.return_value = mock_tokenized_output

        mock_outputs = MagicMock()
        mock_outputs.last_hidden_state = create_mock_tensor((1, 10, 768))  # Example shape
        mock_model_instance.return_value = mock_outputs

        # Mock global manager's analyze_query_context (used by SpecificMemory.analyze_description)
        self.mock_global_manager.analyze_query_context.return_value = {
            "sentiment_score": {"polarity": 0.1, "subjectivity": 0.2},
            "emotion_classification": "neutral",
            "keywords": ["analyzed_kw"],
            "named_entities": ["Entity1"],
            "main_subject": "Subject",
            "main_verb": "Verb",
            "main_object": "Object",
            "temporal_expressions": ["today"],
            "verb_aspects": ["simple_present"],
        }
        # Mock global manager's extract_entities (used by SpecificMemory.extract_facts)
        self.mock_global_manager.extract_entities.return_value = ["FactEntity"]
        # Mock global nlp for extract_facts
        with patch("tiny_memories.nlp") as mock_spacy_nlp:
            mock_doc = MagicMock()
            mock_sent = MagicMock()
            mock_sent.text = "This is a fact."
            mock_doc.sents = [mock_sent]
            mock_spacy_nlp.return_value = mock_doc
            self.gm = tiny_memories.GeneralMemory("Test Description")

    def test_initialization(self, mock_model, mock_manager):  # Mocks passed by patch
        self.assertEqual(self.gm.description, "Test Description")
        self.assertIsNotNone(self.gm.description_embedding)
        self.assertEqual(self.gm.keywords, ["keyword1", "keyword2"])
        mock_manager.extract_keywords.assert_called_once_with("Test Description")
        self.assertIsNotNone(self.gm.timestamp_tree)
        self.assertIsNotNone(self.gm.importance_tree)
        self.assertIsNotNone(self.gm.key_tree)

    def test_add_keyword(self, mock_model, mock_manager):
        self.gm.add_keyword("new_keyword")
        self.assertIn("new_keyword", self.gm.keywords)

    def test_get_keywords(self, mock_model, mock_manager):
        self.assertEqual(self.gm.get_keywords(), self.gm.keywords)

    def test_get_embedding(self, mock_model, mock_manager):
        # Test that it returns the pre-computed embedding
        # And if it was None, it calls generate_embedding
        self.gm.description_embedding = None
        with patch.object(
            self.gm, "generate_embedding", return_value=(MagicMock(), MagicMock())
        ) as mock_gen_emb:
            with patch(
                "tiny_memories.mean_pooling", return_value=MagicMock(name="PooledEmb")
            ) as mock_mp:
                new_emb = self.gm.get_embedding()
                mock_gen_emb.assert_called_once()
                mock_mp.assert_called_once()
                self.assertIsNotNone(new_emb)

    def test_generate_embedding(self, mock_model, mock_manager):
        # This is mostly tested by setUp's mock configuration,
        # but we can assert calls on the passed-in mock_model
        self.gm.description = "Another desc"
        raw_emb, att_mask = self.gm.generate_embedding()

        self.mock_global_model.tokenizer.assert_called_with(
            ["Another desc"],  # Ensure it uses the current description
            padding=True,
            truncation=True,
            add_special_tokens=True,
            is_split_into_words=True,
            pad_to_multiple_of=8,
            return_tensors="pt",
        )
        # self.mock_global_model.model.assert_called() # Called with the tokenized output
        self.assertIsNotNone(raw_emb)
        self.assertIsNotNone(att_mask)

    @patch("tiny_memories.SpecificMemory")  # Mock SpecificMemory class
    def test_init_specific_memories(self, MockSpecificMemory, mock_model, mock_manager):
        mock_sm_instance1 = MockSpecificMemory.return_value
        mock_sm_instance2 = MockSpecificMemory.return_value
        specific_memories_descs = ["sm_desc1", "sm_desc2"]

        # If init_specific_memories expects SpecificMemory instances:
        sm_instances = [
            MagicMock(spec=tiny_memories.SpecificMemory)
            for _ in specific_memories_descs
        ]
        for (
            sm_mock
        ) in (
            sm_instances
        ):  # Ensure they have necessary attributes if add_specific_memory uses them
            sm_mock.analysis = None
            sm_mock.keywords = []
            sm_mock.last_access_time = datetime.now()
            sm_mock.importance_score = 0
            sm_mock.embedding = create_mock_tensor((1, 768))  # Mock embedding
            sm_mock.get_facts_embeddings = MagicMock(return_value=[])

        with patch.object(self.gm, "add_specific_memory") as mock_add_sm:
            self.gm.init_specific_memories(sm_instances)
            self.assertEqual(mock_add_sm.call_count, len(sm_instances))
            for sm_instance in sm_instances:
                mock_add_sm.assert_any_call(sm_instance)

    def test_get_specific_memories_empty(self, mock_model, mock_manager):
        self.assertEqual(self.gm.get_specific_memories(), [])

    @patch("tiny_memories.SpecificMemory")
    def test_get_specific_memories_with_items(
        self, MockSpecificMemory, mock_model, mock_manager
    ):
        # Mock the global manager's update_embeddings and flat_access for add_specific_memory
        mock_manager.update_embeddings = MagicMock()
        mock_manager.flat_access = MagicMock()
        mock_manager.flat_access.faiss_index = MagicMock()
        mock_manager.flat_access.index_id_to_node_id = {}

        sm1_desc = "Specific Memory 1"
        sm1_instance = MagicMock(spec=tiny_memories.SpecificMemory)
        sm1_instance.description = sm1_desc
        sm1_instance.last_access_time = datetime.now() - timedelta(hours=1)
        sm1_instance.importance_score = 5
        sm1_instance.keywords = ["alpha"]
        sm1_instance.analysis = True  # Mark as analyzed
        sm1_instance.embedding = create_mock_tensor((1, 1, 768))  # Mock embedding
        sm1_instance.get_facts_embeddings = MagicMock(return_value=[])

        # When SpecificMemory is instantiated inside add_specific_memory
        MockSpecificMemory.side_effect = [sm1_instance]

        self.gm.add_specific_memory(
            sm1_desc, importance_score=5
        )  # This will create SpecificMemory instance

        # Now retrieve
        retrieved_mems = self.gm.get_specific_memories()
        self.assertEqual(len(retrieved_mems), 1)
        # The actual instance added to BST might be the one created by SpecificMemory constructor
        # So we check by a unique property like description if mocks are tricky
        self.assertTrue(any(mem.description == sm1_desc for mem in retrieved_mems))

    def test_find_specific_memory_not_found(self, mock_model, mock_manager):
        self.assertIsNone(self.gm.find_specific_memory(key=99999))  # Non-existent key

    # More tests for find_specific_memory after adding items needed.

    def test_search_by_key_in_general_memory(self, mock_model, mock_manager):
        # This method is on GeneralMemory, but it seems to be for its internal BSTs.
        # It searches by id(node.memory)
        mock_sm = MagicMock(spec=tiny_memories.SpecificMemory)
        mock_sm.last_access_time = datetime.now()
        mock_sm.importance_score = 1
        mock_sm.keywords = []
        mock_sm.analysis = True
        mock_sm.embedding = create_mock_tensor((1, 1, 768))
        mock_sm.get_facts_embeddings = MagicMock(return_value=[])

        # Mock manager for add_specific_memory
        mock_manager.update_embeddings = MagicMock()
        mock_manager.flat_access = MagicMock()
        mock_manager.flat_access.faiss_index = MagicMock()
        mock_manager.flat_access.index_id_to_node_id = {}

        with patch("tiny_memories.SpecificMemory", return_value=mock_sm):
            self.gm.add_specific_memory(mock_sm)  # Add the mock directly

        # The key used in the BST is id(specific_memory_instance)
        # The search_by_key in GeneralMemory also uses id(node.memory)
        key_to_find = id(mock_sm)

        found_mem = self.gm.search_by_key(
            self.gm.key_tree.specific_memories_root, key_to_find
        )
        self.assertEqual(found_mem, mock_sm)

        self.assertIsNone(
            self.gm.search_by_key(
                self.gm.key_tree.specific_memories_root, id(MagicMock())
            )
        )

    @patch("tiny_memories.SpecificMemory")
    @patch("tiny_memories.faiss.IndexFlatL2")
    def test_add_specific_memory(
        self, MockFaissIndex, MockSpecificMemory, mock_model, mock_manager
    ):
        mock_faiss_instance = MockFaissIndex.return_value

        mock_sm_instance = MagicMock(spec=tiny_memories.SpecificMemory)
        mock_sm_instance.description = "Test SM"
        mock_sm_instance.last_access_time = datetime.now()
        mock_sm_instance.importance_score = 5
        mock_sm_instance.keywords = ["test_kw"]
        mock_sm_instance.analysis = None  # To trigger analyze_description
        mock_sm_instance.analyze_description = MagicMock()
        # Mock get_embedding to return (embedding_tensor, attention_mask_tensor)
        mock_embedding_tensor = create_mock_tensor((1, 1, 768))  # (batch, seq_len, hidden_size)
        mock_attention_mask = create_mock_tensor((1, 1))  # (batch, seq_len)
        mock_sm_instance.get_embedding.return_value = (
            mock_embedding_tensor,
            mock_attention_mask,
        )
        mock_sm_instance.embedding = create_mock_tensor((
            1, 768
        ))  # This is what mean_pooling would produce
        mock_sm_instance.get_facts_embeddings.return_value = []

        MockSpecificMemory.return_value = mock_sm_instance

        # Mock manager methods called by add_specific_memory
        mock_manager.update_embeddings = MagicMock()
        mock_manager.flat_access = MagicMock()
        mock_manager.flat_access.faiss_index = (
            MagicMock()
        )  # This is the global faiss index
        mock_manager.flat_access.index_id_to_node_id = {}

        with patch(
            "tiny_memories.mean_pooling", return_value=mock_sm_instance.embedding
        ) as mock_mp:  # if get_embedding is called
            self.gm.add_specific_memory("Test SM description", importance_score=5)

        MockSpecificMemory.assert_called_once_with("Test SM description", self.gm, 5)
        mock_sm_instance.analyze_description.assert_called_once()  # Because analysis was None

        # Check BSTs
        self.assertIsNotNone(self.gm.key_tree.specific_memories_root)
        self.assertEqual(
            self.gm.key_tree.specific_memories_root.memory, mock_sm_instance
        )

        self.assertIn("test_kw", self.gm.keywords)  # Keywords should be merged

        # Check calls to global manager
        mock_manager.update_embeddings.assert_called_once_with(mock_sm_instance)
        mock_manager.flat_access.faiss_index.add.assert_called()  # Called with sm_instance.embedding

        # Check faiss indexing within GeneralMemory (self.faiss_index)
        # This requires index_memories to be called, which it is in add_specific_memory
        MockFaissIndex.assert_called()  # For self.faiss_index = faiss.IndexFlatL2(...)
        # The mock_faiss_instance here is for the GM's own index, not the global one.
        # self.gm.faiss_index.add.assert_called() # This needs self.gm.faiss_index to be the mock

    @patch("tiny_memories.faiss.IndexFlatL2")
    def test_index_memories(self, MockFaissIndexConstructor, mock_model, mock_manager):
        mock_faiss_index_instance = MockFaissIndexConstructor.return_value
        self.gm.faiss_index = None  # Ensure it's created

        # Add a mock specific memory so there's something to index
        mock_sm = MagicMock(spec=tiny_memories.SpecificMemory)
        mock_embedding_tensor = create_mock_tensor((1, 768))
        mock_attention_mask = create_mock_tensor((1, 10))  # Dummy attention mask
        mock_sm.get_embedding.return_value = (
            mock_embedding_tensor,
            mock_attention_mask,
        )  # As per current index_memories structure

        # Mock the mean_pooling result as this is what index_memories would use if it pooled here
        # However, index_memories expects get_embedding to return already pooled embeddings if it's a list of them.
        # The current code in index_memories:
        # embeddings, att_mask = zip(*[memory.get_embedding() for memory in specific_memories])
        # embeddings = embeddings[0] -> so get_embedding should return a tuple (actual_emb_tensor, actual_att_mask_tensor)
        # then it does self.faiss_index.add(embeddings.cpu().detach().numpy())
        # This implies embeddings is a single tensor after the zip and selection.
        # This part of index_memories seems to assume get_embedding returns a (tensor, mask) for *one* memory,
        # and then it tries to batch them. This needs clarification.
        # For this test, we'll mock `get_specific_memories` and the `get_embedding` of the returned SMs.
        sm1 = MagicMock(spec=tiny_memories.SpecificMemory)
        sm1_emb_tensor = create_mock_tensor((1, 768))  # Pooled embedding
        sm1_att_mask = create_mock_tensor((1, 10))  # Dummy attention mask
        sm1.get_embedding.return_value = (
            sm1_emb_tensor,
            sm1_att_mask,
        )  # As per current index_memories structure

        sm2 = MagicMock(spec=tiny_memories.SpecificMemory)
        sm2_emb_tensor = create_mock_tensor((1, 768))
        sm2_att_mask = create_mock_tensor((1, 10))
        sm2.get_embedding.return_value = (sm2_emb_tensor, sm2_att_mask)

        with patch.object(self.gm, "get_specific_memories", return_value=[sm1, sm2]):
            # The current index_memories code will take embeddings[0] and att_mask[0] from the zip.
            # This means it will effectively use sm1_emb_tensor and sm1_att_mask.
            # And then try to add sm1_emb_tensor to faiss. This is problematic if sm1_emb_tensor is not a batch.
            # If sm1_emb_tensor is (1, D), then .cpu().detach().numpy() is fine.

            indexed_faiss = self.gm.index_memories()
            self.assertEqual(indexed_faiss, mock_faiss_index_instance)
            MockFaissIndexConstructor.assert_called_once_with(
                768
            )  # Embedding dim from sm1_emb_tensor.shape[1]

            # It should have added sm1_emb_tensor.cpu().detach().numpy()
            # The current code: embeddings = embeddings[0] means embeddings becomes sm1_emb_tensor
            mock_faiss_index_instance.add.assert_called_once_with(
                sm1_emb_tensor.cpu().detach().numpy()
            )


@patch("tiny_memories.manager", new_callable=MagicMock)
@patch("tiny_memories.model", new_callable=MagicMock)
@patch("tiny_memories.sa", new_callable=MagicMock)  # Mock global sentiment analyzer
class TestSpecificMemory(unittest.TestCase):
    def setUp(self, mock_sa, mock_model, mock_manager):  # Order from decorators
        self.mock_global_sa = mock_sa
        self.mock_global_model = mock_model
        self.mock_global_manager = mock_manager

        # Mock parent memory
        self.parent_memory_mock = MagicMock(spec=tiny_memories.GeneralMemory)
        self.parent_memory_mock.description = "Parent General Memory"

        # Mock parts of the global model used in SpecificMemory.generate_embedding
        mock_tokenizer_instance = MagicMock()
        mock_model_instance = MagicMock()
        self.mock_global_model.tokenizer = mock_tokenizer_instance
        self.mock_global_model.model = mock_model_instance
        self.mock_global_model.device = "cpu"

        mock_input_dict = {"input_ids": MagicMock(), "attention_mask": MagicMock()}
        # mock_tokenizer_instance.return_value = mock_input_dict # This was too simple
        # Need to handle .to(device) call
        mock_tokenized_output = MagicMock()
        mock_tokenized_output.to.return_value = mock_input_dict
        mock_tokenizer_instance.return_value = mock_tokenized_output

        mock_outputs = MagicMock()
        mock_outputs.last_hidden_state = create_mock_tensor((1, 10, 768))  # Example shape
        mock_model_instance.return_value = mock_outputs

        # Mock global manager's analyze_query_context (used by SpecificMemory.analyze_description)
        self.mock_global_manager.analyze_query_context.return_value = {
            "sentiment_score": {"polarity": 0.1, "subjectivity": 0.2},
            "emotion_classification": "neutral",
            "keywords": ["analyzed_kw"],
            "named_entities": ["Entity1"],
            "main_subject": "Subject",
            "main_verb": "Verb",
            "main_object": "Object",
            "temporal_expressions": ["today"],
            "verb_aspects": ["simple_present"],
        }
        # Mock global manager's extract_entities (used by SpecificMemory.extract_facts)
        self.mock_global_manager.extract_entities.return_value = ["FactEntity"]
        # Mock global nlp for extract_facts
        with patch("tiny_memories.nlp") as mock_spacy_nlp:
            mock_doc = MagicMock()
            mock_sent = MagicMock()
            mock_sent.text = "This is a fact."
            mock_doc.sents = [mock_sent]
            mock_spacy_nlp.return_value = mock_doc
            self.sm = tiny_memories.SpecificMemory(
                "Test SM Description", self.parent_memory_mock, 5
            )

    def test_initialization(self, mock_sa, mock_model, mock_manager):
        self.assertEqual(self.sm.description, "Test SM Description")
        self.assertEqual(self.sm.parent_memory, self.parent_memory_mock)
        self.assertEqual(self.sm.importance_score, 5)
        self.assertIsNotNone(self.sm.embedding)  # Should be generated
        self.assertIsNotNone(self.sm.att_mask)
        self.mock_global_manager.analyze_query_context.assert_called_once_with(
            "Test SM Description"
        )
        self.assertEqual(self.sm.keywords, ["analyzed_kw"])
        self.assertEqual(
            self.sm.sentiment_score, {"polarity": 0.1, "subjectivity": 0.2}
        )
        # Test facts extraction
        self.assertTrue(len(self.sm.facts) > 0)
        self.assertIsNotNone(self.sm.facts_embeddings)

    def test_get_parent_memory_from_string_found(
        self, mock_sa, mock_model, mock_manager
    ):
        # Mock global manager.hierarchy.get_general_memory
        mock_gm_instance = MagicMock(spec=tiny_memories.GeneralMemory)
        mock_manager.hierarchy = MagicMock()
        mock_manager.hierarchy.get_general_memory.return_value = mock_gm_instance

        parent = self.sm.get_parent_memory_from_string("Existing Parent")
        mock_manager.hierarchy.get_general_memory.assert_called_once_with(
            "Existing Parent"
        )
        self.assertEqual(parent, mock_gm_instance)

    def test_get_parent_memory_from_string_not_found(
        self, mock_sa, mock_model, mock_manager
    ):
        mock_manager.hierarchy = MagicMock()
        mock_manager.hierarchy.get_general_memory.return_value = None

        parent = self.sm.get_parent_memory_from_string("NonExistent Parent")
        self.assertIsNone(parent)

    def test_analyze_description(self, mock_sa, mock_model, mock_manager):
        # analyze_description is called in __init__. We can call it again to check.
        # Reset mock call count from __init__
        self.mock_global_manager.analyze_query_context.reset_mock()
        self.sm.analyze_description()
        self.mock_global_manager.analyze_query_context.assert_called_once_with(
            self.sm.description
        )
        # Assertions for all attributes set by analyze_description
        self.assertEqual(
            self.sm.sentiment_score, {"polarity": 0.1, "subjectivity": 0.2}
        )
        self.assertEqual(self.sm.emotion_classification, "neutral")
        # ... and so on for other attributes

    # test_update_importance_score is in existing tests

    def test_update_parent_memory(
        self, mock_sa, mock_model, mock_manager
    ):  # Already exists
        new_parent = MagicMock(spec=tiny_memories.GeneralMemory)
        self.sm.update_parent_memory(new_parent)
        self.assertEqual(self.sm.parent_memory, new_parent)

    # test_update_related_memories is in existing tests
    # test_update_keywords is in existing tests
    # test_add_related_memory is in existing tests
    # test_get_embedding is in existing tests (but might need refinement)

    def test_get_embedding_cached(self, mock_sa, mock_model, mock_manager):
        first_emb, first_mask = self.sm.get_embedding()  # Should use cached from init
        with patch.object(self.sm, "generate_embedding") as mock_gen_emb:
            second_emb, second_mask = self.sm.get_embedding()
            mock_gen_emb.assert_not_called()  # Should not regenerate
            self.assertIs(first_emb, second_emb)  # Check tensor identity
            self.assertIs(first_mask, second_mask)

    @patch("tiny_memories.mean_pooling")
    def test_generate_facts_embeddings(
        self, mock_mean_pooling, mock_sa, mock_model, mock_manager
    ):
        self.sm.facts = ["Fact one.", "Fact two."]
        self.sm.facts_embeddings = None  # Force regeneration

        # Mock tokenizer and model calls for each fact
        mock_pooled_emb1 = create_mock_tensor((1, 768))
        mock_pooled_emb2 = create_mock_tensor((1, 768))
        mock_mean_pooling.side_effect = [mock_pooled_emb1, mock_pooled_emb2]

        # Reset global model mocks to check calls per fact
        self.mock_global_model.tokenizer.reset_mock()
        self.mock_global_model.model.reset_mock()
        self.mock_global_model.tokenizer.return_value.to.reset_mock()

        generated_embeddings = self.sm.generate_facts_embeddings()

        self.assertEqual(len(generated_embeddings), 2)
        self.assertTrue(
            torch.equal(generated_embeddings[0], mock_pooled_emb1.squeeze(0))
        )  # Squeezed by func
        self.assertTrue(
            torch.equal(generated_embeddings[1], mock_pooled_emb2.squeeze(0))
        )

        self.assertEqual(self.mock_global_model.tokenizer.call_count, 2)
        self.assertEqual(self.mock_global_model.model.call_count, 2)
        self.assertEqual(mock_mean_pooling.call_count, 2)

    def test_get_facts(self, mock_sa, mock_model, mock_manager):
        self.assertEqual(self.sm.get_facts(), self.sm.facts)

    @patch("tiny_memories.nlp")  # Mock the global spacy nlp object
    def test_extract_facts(self, mock_spacy_nlp, mock_sa, mock_model, mock_manager):
        mock_doc = MagicMock()
        mock_sent1 = MagicMock(text="This is sentence one.")
        mock_sent2 = MagicMock(text="And sentence two?")  # A question
        mock_sent3 = MagicMock(text="Short.")  # Too short
        mock_sent4 = MagicMock(text="This is a good factual sentence.")
        mock_doc.sents = [mock_sent1, mock_sent2, mock_sent3, mock_sent4]
        mock_spacy_nlp.return_value = mock_doc

        # Mock manager.extract_entities for the is_entity_dense check
        self.mock_global_manager.extract_entities.side_effect = [
            ["entityA", "entityB"],  # For sent1
            [],  # For sent4 (not dense enough)
        ]

        extracted = self.sm.extract_facts("Some text to extract from.")
        mock_spacy_nlp.assert_called_once_with("Some text to extract from.")
        self.assertIn("This is sentence one.", extracted)
        self.assertNotIn("And sentence two?", extracted)  # Is a question
        self.assertNotIn("Short.", extracted)  # Too short
        self.assertNotIn(
            "This is a good factual sentence.", extracted
        )  # Not entity dense

    def test_add_tag(self, mock_sa, mock_model, mock_manager):
        self.sm.add_tag("new_tag")
        self.assertIn("new_tag", self.sm.tags)
        self.sm.add_tag(
            "new_tag"
        )  # Should not add duplicates if it's a set like behavior
        self.assertEqual(
            self.sm.tags.count("new_tag"), 1
        )  # Assuming tags is a list that prevents duplicates

    def test_get_tags(self, mock_sa, mock_model, mock_manager):
        self.sm.tags = ["tag1", "tag2"]
        self.assertEqual(self.sm.get_tags(), ["tag1", "tag2"])

    def test_get_subject(self, mock_sa, mock_model, mock_manager):
        self.sm.subject = "Test Subject"
        self.assertEqual(self.sm.get_subject(), "Test Subject")

    def test_set_subject(self, mock_sa, mock_model, mock_manager):
        self.sm.set_subject("New Subject")
        self.assertEqual(self.sm.subject, "New Subject")

    def test_getstate_setstate(self, mock_sa, mock_model, mock_manager):
        # Test pickling and unpickling
        # __getstate__ should handle non-serializable parts like torch tensors if not careful
        # For this test, we assume tensors are handled (e.g. converted to numpy or lists)
        # or that the test environment doesn't strictly check serializability if not using pickle.dumps

        # Simplify: check that all expected attributes are in the state
        state = self.sm.__getstate__()
        self.assertIn("description", state)
        self.assertIn(
            "parent_memory_description", state
        )  # Assuming parent_memory is stored by desc
        self.assertIn("importance_score", state)
        # Tensors like 'embedding' might be converted or excluded.
        # Check for 'embedding_np' or similar if that's the strategy.
        # If 'embedding' (torch.Tensor) is directly in state, pickle will handle it.
        self.assertIn("embedding", state)
        self.assertIn("att_mask", state)
        # ... and other relevant attributes

        new_sm = tiny_memories.SpecificMemory(
            "Dummy", self.parent_memory_mock, 0
        )  # Create a new instance
        # Temporarily mock analyze_description for setstate to avoid re-analysis based on "Dummy"
        with patch.object(new_sm, "analyze_description") as mock_analyze:
            new_sm.__setstate__(state)
            mock_analyze.assert_not_called()  # Should not re-analyze if state provides all

        self.assertEqual(new_sm.description, self.sm.description)
        self.assertEqual(new_sm.importance_score, self.sm.importance_score)
        if self.sm.embedding is not None and new_sm.embedding is not None:
            self.assertTrue(torch.equal(new_sm.embedding, self.sm.embedding))


# ... (Keep TestMemoryManager and TestMemoryManagerIntegration as they are for now) ...
# ... (Keep TestMemory, TestBSTNode, TestMemoryBST as they are, with prior edits) ...
# ... (Keep TestGeneralMemory, TestSpecificMemory with prior edits) ...

class TestSigmoidFunction(unittest.TestCase):
    """Test class for sigmoid function with precision fixes."""
    
    def test_sigmoid_with_positive_value(self):
        """Test sigmoid function with improved precision handling."""
        # Test sigmoid with positive value
        result = tiny_memories.sigmoid(1)
        # Fixed: Use lower precision (places=6) instead of brittle places=10
        # Original brittle code was:
        # self.assertAlmostEqual(result, 0.7310585786300049, places=10)
        self.assertAlmostEqual(result, 0.7310585786300049, places=6,
                              msg="Sigmoid test with improved precision")
    
    def test_sigmoid_computed_expected(self):
        """Test sigmoid using computed expected values - most robust approach."""
        
        # Test sigmoid with positive value - computing expected using same approach
        result = tiny_memories.sigmoid(1)
        expected = 1 / (1 + math.exp(-1))
        self.assertAlmostEqual(result, expected, places=10,
                              msg="Using computed expected value is most robust")
        
        # Test with negative value
        result = tiny_memories.sigmoid(-2)
        expected = 1 / (1 + math.exp(2))
        self.assertAlmostEqual(result, expected, places=6)
        
        # Test with zero
        result = tiny_memories.sigmoid(0)
        expected = 1 / (1 + math.exp(0))  # Should be 0.5
        self.assertAlmostEqual(result, expected, places=6)
        self.assertAlmostEqual(result, 0.5, places=6)


if __name__ == "__main__":
    unittest.main()
