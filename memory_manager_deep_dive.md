# Component Deep Dive: MemoryManager

The `MemoryManager` in `tiny_memories.py` handles how characters perceive, store, process, and retrieve information. It's a sophisticated system blending structured data with NLP for rich memory representation.

## Internal Structure & Key Components

*   **`Memory` (Base Class):** Basic attributes: `description`, `creation_time`, `last_access_time`.
*   **`GeneralMemory(Memory)`:**
    *   **Purpose:** Represents broader memory categories (e.g., "Social Interactions").
    *   **Attributes:** Embedding of its description, a FAISS index for its `SpecificMemory` instances, and `MemoryBST` instances (for timestamp, importance, key-based sorting of specific memories), aggregated keywords.
    *   **Functionality:** Manages a collection of `SpecificMemory` objects, including adding them and indexing them for local search.
*   **`SpecificMemory(Memory)`:**
    *   **Purpose:** Represents individual experiences (e.g., "Saw John at the park").
    *   **Attributes:** `parent_memory` (link to `GeneralMemory`), `embedding` (of its description), `keywords`, `tags`, `importance_score`, `sentiment_score`, `emotion_classification`, extracted `facts` and their `facts_embeddings`, `main_subject/verb/object`.
    *   **Functionality:** Undergoes NLP analysis via `analyze_description()` to populate its attributes.
*   **`MemoryBST`:**
    *   **Purpose:** Balanced AVL tree for organizing `SpecificMemory` objects by attributes like timestamp or importance, allowing efficient O(log n) lookups.
*   **`EmbeddingModel`:**
    *   **Purpose:** Generates sentence embeddings using models like "all-mpnet-base-v2".
*   **`SentimentAnalysis`:**
    *   **Purpose:** Performs sentiment (polarity/subjectivity via `TextBlob`) and emotion analysis (via Hugging Face models).
*   **`FlatMemoryAccess`:**
    *   **Purpose:** Provides global search across *all* `SpecificMemory` instances.
    *   **Attributes:** Global FAISS index for all specific memory embeddings (and their extracted facts), `index_id_to_node_id` mapping, collections for recent/common/important memories.
    *   **Functionality:** Initializes and manages the global FAISS index, `find_memories_by_query()` for global similarity search, persistence for embeddings and index.
*   **`MemoryQuery`:**
    *   **Purpose:** Encapsulates a search query for the memory system.
    *   **Attributes:** Query text, embedding, tags, complex filter criteria.
*   **`MemoryManager` (Main Class):**
    *   **Purpose:** Orchestrates the entire memory system.
    *   **Attributes:** Instances of `FlatMemoryAccess`, `EmbeddingModel`, `SentimentAnalysis`, NLP tools (spaCy, NLTK, scikit-learn for keyword extraction, topic modeling - though LDA seems less used).
    *   **Key Functionality:**
        *   `analyze_query_context()`: Core NLP pipeline (see below).
        *   `retrieve_memories()`: Main search entry point, uses `FlatMemoryAccess`.
        *   Keyword/Fact extraction methods.

## NLP Pipeline (within `MemoryManager.analyze_query_context`)

1.  **Input Text** (Memory description or query).
2.  **Preprocessing:** Basic cleaning.
3.  **spaCy Processing:** Tokenization, POS tagging, dependency parsing, NER.
4.  **Embedding Generation:** Using `EmbeddingModel`.
5.  **Sentiment & Emotion Analysis:** Using `SentimentAnalysis`.
6.  **Keyword Extraction:** RAKE, TF-IDF, and entities from spaCy.
7.  **Fact Extraction:** Using spaCy's dependency tree and potentially `tsm.main` to identify SVO triples and clauses.
8.  **Linguistic Feature Extraction:** Complexity scores, lexical diversity, voice detection, temporal expressions, verb aspects.
9.  **Output:** A rich dictionary of analyzed features for the input text.

## Memory Retrieval Process

1.  A query (string or `MemoryQuery`) is passed to `MemoryManager.retrieve_memories()`.
2.  The query text is processed via `analyze_query_context()` to get its embedding and features.
3.  `FlatMemoryAccess.find_memories_by_query()` searches the global FAISS index using the query embedding.
4.  FAISS returns `top_k` similar memory/fact embeddings.
5.  These are mapped back to `SpecificMemory` objects and returned with similarity scores.

## Strengths

*   **Rich Memory Representation:** Memories are deeply analyzed and structured.
*   **Efficient Semantic Search:** FAISS allows fast retrieval from many memories.
*   **Hybrid Retrieval:** Combines semantic search with structured filtering.
*   **Detailed NLP Analysis:** Provides deep understanding of textual content.

## Potential Challenges

*   **Complexity & NLP Accuracy:** The system is intricate, and its effectiveness hinges on the NLP pipeline's accuracy.
*   **Fact Extraction Robustness:** Reliably extracting correct facts is challenging.
*   **Computational Cost:** NLP, embeddings, and indexing can be resource-intensive.
*   **Memory Coherence:** No explicit mechanism for handling contradictory memories or beliefs.

The `MemoryManager` provides a powerful cognitive layer, enabling characters to learn and recall information in a human-like, context-aware manner.
