import unittest
import numpy as np
from tiny_memories import EmbeddingModel, mean_pooling
from cosine_threshold_query import CosineThresholdQuery, IncrementalCosineThresholdQuery

class TestCosineThresholdQuery(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = EmbeddingModel()
        cls.model.model.eval()

    def generate_embedding(self, description):
        description = [description.strip()]
        input = self.model.tokenizer(description, padding=True,
        truncation=True,
        add_special_tokens=True,
        is_split_into_words=True,
        pad_to_multiple_of=8,
        return_tensors="pt").to(self.model.device)
        outputs = self.model.model(input["input_ids"],attention_mask=input["attention_mask"],
        output_hidden_states=True,
        return_dict=True,
        return_tensors="pt",)
        return mean_pooling(outputs.last_hidden_state, input["attention_mask"]).cpu().detach().numpy()

    def setUp(self):
        descriptions = ["This is a test description", "Another test description", "Yet another test description"]
        database_vectors = [self.generate_embedding(desc) for desc in descriptions]
        self.cosine_query = CosineThresholdQuery(database_vectors)

    def test_build_inverted_index_with_hulls(self):
        self.cosine_query.build_inverted_index_with_hulls()
        self.assertIsNotNone(self.cosine_query.index)
        self.assertIsNotNone(self.cosine_query.hulls)

    def test_cosine_threshold_query(self):
        query_vector = self.generate_embedding("This is a query description")
        threshold = 0.5
        result = self.cosine_query.cosine_threshold_query(query_vector, threshold)
        self.assertIsInstance(result, set)

    def test_calculate_max_similarity(self):
        query_vector = self.generate_embedding("This is a query description")
        b_positions = np.zeros(self.cosine_query.dimension, dtype=int)
        max_similarity = self.cosine_query.calculate_max_similarity(query_vector, b_positions)
        self.assertIsInstance(max_similarity, float)


class TestIncrementalCosineThresholdQuery(unittest.TestCase):
    def setUp(self):
        self.model = EmbeddingModel()
        self.model.model.eval()

    def generate_embedding(self, description):
        description = [description.strip()]
        input = self.model.tokenizer(description, padding=True,
        truncation=True,
        add_special_tokens=True,
        is_split_into_words=True,
        pad_to_multiple_of=8,
        return_tensors="pt").to(self.model.device)
        outputs = self.model.model(input["input_ids"],attention_mask=input["attention_mask"],
        output_hidden_states=True,
        return_dict=True,
        return_tensors="pt",)
        return mean_pooling(outputs.last_hidden_state, input["attention_mask"]).cpu().detach().numpy()

    def test_process_query_incrementally(self):
        # Generate some embeddings
        database_vectors = [self.generate_embedding("This is a test."), self.generate_embedding("Another test."), self.generate_embedding("Yet another test.")]
        query_vector = self.generate_embedding("This is a query.")

        # Create an instance of IncrementalCosineThresholdQuery
        query = IncrementalCosineThresholdQuery(database_vectors)

        # Test process_query_incrementally
        result = query.process_query_incrementally(query_vector, 0.5)
        self.assertIsInstance(result, list)
        self.assertTrue(all(isinstance(i, int) for i in result))

    def test_incremental_cosine_threshold_query(self):
        # Generate some embeddings
        database_vectors = [self.generate_embedding("This is a test."), self.generate_embedding("Another test."), self.generate_embedding("Yet another test.")]
        query_vector = self.generate_embedding("This is a query.")

        # Create an instance of IncrementalCosineThresholdQuery
        query = IncrementalCosineThresholdQuery(database_vectors)

        # Test update_current_values_and_partial_sums
        b_positions = np.zeros(query.dimension, dtype=int)
        query.update_current_values_and_partial_sums(b_positions)
        self.assertTrue(np.all(query.current_values >= 0))
        self.assertTrue(np.all(query.partial_sums >= 0))

        # Test calculate_max_similarity_incrementally
        max_similarity = query.calculate_max_similarity_incrementally()
        self.assertIsInstance(max_similarity, float)
        self.assertTrue(max_similarity >= 0)

        # Test stopping_condition_incremental
        stopping_condition = query.stopping_condition_incremental(0.5)
        self.assertIsInstance(stopping_condition, bool)

        # Test process_query_incrementally
        result = query.process_query_incrementally(query_vector, 0.5)
        self.assertIsInstance(result, list)
        self.assertTrue(all(isinstance(i, int) for i in result))

        # Test cosine_threshold_query
        result = query.cosine_threshold_query(query_vector, 0.5)
        self.assertIsInstance(result, list)
        self.assertTrue(all(isinstance(i, int) for i in result))

        # Test adjust_tau
        query.adjust_tau()
        self.assertTrue(query.tau >= 0)

if __name__ == '__main__':
    unittest.main()