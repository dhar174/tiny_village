from ast import Set
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
from typing import Dict, List, Tuple


class CosineThresholdQuery:
    def __init__(self, database_vectors: List[np.ndarray]):
        """
        Initialize the system with a database of vectors.
        Each vector is assumed to be normalized and of consistent dimensionality.
        """
        if not all(len(v) == len(database_vectors[0]) for v in database_vectors):
            raise ValueError("All vectors must have the same dimensionality.")
        
        self.database_vectors = [v / np.linalg.norm(v) for v in database_vectors]
        self.dimension = len(database_vectors[0])
        self.index = self.build_inverted_index_with_hulls()

        
    def build_inverted_index_with_hulls(self):
        """
        Builds an efficient inverted index from the database of vectors and computes the convex hull for each dimension.
        This index will be used to accelerate the query process by allowing fast lookup of vectors based on their components.
        """
        index = {}
        hulls = {}

        for dim in range(self.dimension):
            dim_values = [(vector_id, vector[dim]) for vector_id, vector in enumerate(self.database_vectors)]
            sorted_dim_values = sorted(dim_values, key=lambda x: x[1], reverse=True)
            
            if len(sorted_dim_values) > 2:
                points = np.array([[v[1], 0] for v in sorted_dim_values])  # Simplify to 1D points for hull computation
                hull = ConvexHull(points)
                hull_vertices = sorted(set([sorted_dim_values[i][0] for i in hull.vertices]), reverse=True)
                hulls[dim] = hull_vertices
            else:
                hulls[dim] = [v[0] for v in sorted_dim_values]
            
            index[dim] = sorted_dim_values

        self.index = index
        self.hulls = hulls




    def cosine_threshold_query(self, query_vector, threshold):
        """
        Executes a cosine threshold query using the optimized and refined methods.
        """
        query_vector = query_vector / np.linalg.norm(query_vector)
        max_dim = max([len(vector) for vector in self.database_vectors])  # Determine the maximum dimension across all vectors
        b_positions = np.zeros(max_dim, dtype=int)  # Initialize b_positions with the maximum dimension
        candidates = set()

        while True:
            next_dim = self.select_next_dimension_hull_based(query_vector, b_positions)
            if next_dim == -1:
                break  # No more dimensions to traverse

            b_positions[next_dim] += 1
            if next_dim in self.index and b_positions[next_dim] < len(self.index[next_dim]):
                vector_id, _ = self.index[next_dim][b_positions[next_dim]]
                candidates.add(vector_id)

            if self.stopping_condition(query_vector, b_positions, threshold):
                break

        return self.verify_candidates(query_vector, candidates, threshold)


    def calculate_max_similarity(self, query_vector: np.ndarray, b_positions: np.ndarray) -> float:
        """
        Efficiently finds τ and calculates the maximum similarity based on the current traversal state.
        """
        # Refining the approach to find τ by leveraging analytical insights or efficient search techniques
        
        # Extracting the current values from the index based on b_positions
        current_values = [self.index[dim][b_positions[dim]][1] if b_positions[dim] < len(self.index[dim]) else 0 for dim in range(self.dimension)]
        
        # Sorting query vector and current values by the ratio of query_vector[dim] to current_values[dim]
        ratios = [query_vector[dim] / val if val > 0 else float('inf') for dim, val in enumerate(current_values)]
        sorted_indices = sorted(range(self.dimension), key=lambda dim: ratios[dim])
        
        # Initial guess for tau can be refined based on sorted ratios
        tau_initial = [ratios[sorted_indices[0]]]  # Starting from the smallest ratio
        
        # Define an optimized objective function that leverages sorted ratios for a more efficient search
        def objective(tau):
            return sum(min(query_vector[dim] ** 2 * tau ** 2, 2 * query_vector[dim] * tau * current_values[dim]) for dim in sorted_indices) - 1
        
        # Solve the optimization problem using a more efficient method, possibly leveraging scipy.optimize's minimize method
        result = minimize(fun=objective, x0=tau_initial, method='SLSQP', bounds=[(0, None)])
        
        if not result.success:
            raise ValueError("Optimization to find tau failed.")
        
        tau_opt = result.x[0]
        max_similarity_estimate = sum(min(query_vector[dim] * tau_opt, current_values[dim]) * query_vector[dim] for dim in range(self.dimension))
        return max_similarity_estimate


    def stopping_condition(self, query_vector: np.ndarray, b_positions: np.ndarray, threshold: float) -> bool:
        """
        Evaluates the stopping condition for the given query, based on the optimization
        of the maximum possible similarity that any unseen vector can have.
        """
        max_similarity = self.calculate_max_similarity(query_vector, b_positions)
        return max_similarity < threshold
    

    def select_next_dimension_hull_based(self, query_vector: np.ndarray, b_positions: np.ndarray) -> int:
        """
        Selects the next dimension to traverse using the hull-based strategy, prioritizing based on data skewness.
        """
        max_impact = -np.inf
        next_dim = -1
        for dim, hull in self.hulls.items():
            if not hull:  # Skip if the hull is empty
                continue

            current_pos = b_positions[dim]
            if current_pos < len(self.index[dim]) and self.index[dim][current_pos][0] in hull:
                # Estimate the impact of this dimension based on the query vector and the value at the current position.
                impact = query_vector[dim] * self.index[dim][current_pos][1]
                if impact > max_impact:
                    max_impact = impact
                    next_dim = dim

        return next_dim


    def verify_candidates(self, query_vector: np.ndarray, candidates: set, threshold: float) -> list:
        """
        Optimizes the verification phase using a partial verification strategy. This method iteratively
        computes the cosine similarity and terminates early if the threshold cannot be met.
        """
        verified_candidates = []
        query_norm = np.linalg.norm(query_vector)

        for candidate_id in candidates:
            candidate_vector = self.database_vectors[candidate_id]
            candidate_norm = np.linalg.norm(candidate_vector)
            
            # Early termination based on norm product and threshold
            if candidate_norm * query_norm < threshold:
                continue
            
            dot_product = 0.0
            max_possible_similarity = 0.0
            
            # Iterate over each dimension
            for dim, query_val in enumerate(query_vector):
                candidate_val = candidate_vector[dim]
                
                # Update dot product and max possible similarity
                dot_product += query_val * candidate_val
                max_possible_similarity += abs(query_val) * abs(candidate_val)
                
                current_similarity = dot_product / (query_norm * candidate_norm)
                
                # Check if the current similarity already meets the threshold
                if current_similarity >= threshold:
                    verified_candidates.append(candidate_id)
                    break
                
                # Estimate the maximum possible similarity with remaining dimensions
                remaining_max_similarity = (dot_product + (max_possible_similarity - dot_product)) / (query_norm * candidate_norm)
                
                # If the max possible similarity cannot meet the threshold, terminate early
                if remaining_max_similarity < threshold:
                    break

        return verified_candidates

def are_vectors_linearly_dependent(vectors):
    matrix = np.stack(vectors)
    rank = np.linalg.matrix_rank(matrix)
    return (rank < len(vectors)).all()

class IncrementalCosineThresholdQuery(CosineThresholdQuery):
    def __init__(self, database_vectors: List[np.ndarray]):
        super().__init__(database_vectors)
        self.index = self.build_inverted_index_with_hulls()
        # Initialize additional attributes for incremental maintenance
        self.tau = 0.1  # Initial guess for tau
        self.partial_sums = np.zeros(self.dimension)  # To store partial contributions to the objective function
        self.current_values = np.zeros(self.dimension)  # To store current index values based on b_positions
        # self.query_vector = None  # To store the query vector for incremental processing
        # self.update_current_values_and_partial_sums(np.zeros(self.dimension, dtype=int))  # Initialize current values and partial sums
        self.adjust_tau()  # Adjust tau based on the initial state
        
    
    def update_current_values_and_partial_sums(self, b_positions: np.ndarray):
        """
        Updates current values and partial sums based on the current state of b_positions.
        This method is called after each update to b_positions to ensure the state is current.
        """
        for dim in range(self.dimension):
            if b_positions[dim] < len(self.index[dim]):
                new_value = self.index[dim][b_positions[dim]][1]
                self.current_values[dim] = new_value
                # Update the partial sum for this dimension
                self.partial_sums[dim] = min(self.query_vector[dim] * self.tau, new_value) ** 2
            else:
                # Handle cases where we've exhausted this dimension
                self.current_values[dim] = 0
                self.partial_sums[dim] = 0
    
    def calculate_max_similarity_incrementally(self) -> float:
        """
        Incrementally calculates the maximum similarity using updated partial sums.
        This method leverages the current state without recalculating from scratch.
        """
        # Objective function is the sum of partial_sums minus 1, adjusted for tau
        total_sum = sum(self.partial_sums) - 1
        
        # Update tau based on the new total sum if necessary
        if total_sum > 0:
            self.tau = np.sqrt(1 / total_sum)
        
        # Calculate and return the maximum similarity using the current tau
        max_similarity_estimate = sum(min(self.query_vector[dim] * self.tau, self.current_values[dim]) * self.query_vector[dim] for dim in range(self.dimension))
        return max_similarity_estimate

    def stopping_condition_incremental(self, threshold: float) -> bool:
        """
        Evaluates the stopping condition using an efficient incremental technique.
        """
        # Efficiently updated maximum similarity estimate for the stopping condition
        max_similarity = self.calculate_max_similarity_incrementally()
        return max_similarity < threshold
    
    def process_query_incrementally(self, query_vector: np.ndarray, threshold: float) -> List[int]:
        """
        Processes a cosine threshold query incrementally, maintaining and updating state for efficient computation.
        """
        self.query_vector = query_vector / np.linalg.norm(query_vector)  # Normalize the query vector
        b_positions = np.zeros(self.dimension, dtype=int)  # Initialize b_positions for traversal
        candidates = set()  # Initialize the set of candidate vectors

        # Update current values and partial sums initially
        self.update_current_values_and_partial_sums(b_positions)

        while True:
            # Select the next dimension to traverse using the hull-based strategy
            next_dim = self.select_next_dimension_hull_based(self.query_vector, b_positions)
            if next_dim == -1:  # No more dimensions to traverse
                break

            # Advance the position in the selected dimension
            b_positions[next_dim] += 1
            # Update current values and partial sums based on the new state of b_positions
            self.update_current_values_and_partial_sums(b_positions)

            # Dynamically adjust tau based on the updated partial sums and current values
            self.adjust_tau()

            # Add the vector to candidates if it hasn't been processed
            if b_positions[next_dim] < len(self.index[next_dim]):
                vector_id, _ = self.index[next_dim][b_positions[next_dim]]
                candidates.add(vector_id)

            # Evaluate the stopping condition incrementally
            if self.stopping_condition_incremental(threshold):
                break

        # Verify candidates against the threshold
        verified_candidates = self.verify_candidates(self.query_vector, candidates, threshold)
        return verified_candidates

    
    def cosine_threshold_query(self, query_vector: np.ndarray, threshold: float) -> List[int]:
        """
        Executes a cosine threshold query using the optimized and refined methods.
        """
        return self.process_query_incrementally(query_vector, threshold)
    
    def adjust_tau(self):
        """
        Dynamically adjusts tau to ensure that the sum of squared contributions equals 1.
        This iterative adjustment refines tau based on the current state of partial sums.
        """
        # Calculate the current sum of squared contributions
        current_sum = sum(min(self.query_vector[dim] * self.tau, self.current_values[dim])**2 for dim in range(self.dimension))
        
        # Define a tolerance for how close we need to get to 1
        tolerance = 1e-6
        learning_rate = 0.1  # Learning rate for adjustments
        
        # Iteratively adjust tau
        while abs(current_sum - 1) > tolerance:
            # Adjust tau based on the difference from the objective
            self.tau *= (1 + learning_rate * (1 - current_sum))
            
            # Recalculate the current sum with the adjusted tau
            current_sum = sum(min(self.query_vector[dim] * self.tau, self.current_values[dim])**2 for dim in range(self.dimension))
        
        # No need to return anything; tau is adjusted in place



# Test the cosine threshold query. Remember, the database vectors should be normalized, and they should represent text embeddings from a language model.
# Example usage:
# database_vectors = [generate_embedding("This is a test."), generate_embedding("Another test."), generate_embedding("Yet another test.")]
# query_vector = generate_embedding("This is a query.")
# query = CosineThresholdQuery(database_vectors)
# result = query.cosine_threshold_query(query_vector, 0.5)
# print(result)
            
# Example usage for incremental cosine threshold query:
# query = IncrementalCosineThresholdQuery(database_vectors)
# result = query.cosine_threshold_query(query_vector, 0.5)
# print(result)
            
# The result will be a list of indices of vectors in the database that satisfy the cosine similarity threshold with the query vector.
# The indices can then be used to retrieve the corresponding documents or entities from the database.

# Test: cosine_threshold_query
from tiny_memories import EmbeddingModel, mean_pooling
import torch
import unittest

class TestIncrementalCosineThresholdQuery(unittest.TestCase):
    def setUp(self):
        self.model = EmbeddingModel()
    
    def generate_embedding(self, text):
        input = self.model.tokenizer([text], padding=True, truncation=True, add_special_tokens=True, is_split_into_words=True, pad_to_multiple_of=8, return_tensors="pt").to(self.model.device)
        outputs = self.model.model(input["input_ids"], attention_mask=input["attention_mask"], output_hidden_states=True, return_dict=True, return_tensors="pt",)
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