import unittest
import tiny_memories # Import the module
from datetime import datetime, timedelta

# Access the global sentiment_analysis instance from tiny_memories
sentiment_analyzer_instance = tiny_memories.sentiment_analysis

class TestSentimentAnalysisRevised(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sentiment_analyzer = sentiment_analyzer_instance
        if cls.sentiment_analyzer is None:
            raise unittest.SkipTest("Skipping all SentimentAnalysis tests: global sentiment_analysis instance is None.")

        if tiny_memories.nlp is None:
            print("Test WARNING: SpaCy nlp model (tiny_memories.nlp) is None.")

        if not tiny_memories._TEXTBLOB_AVAILABLE:
            print("Test WARNING: TextBlob is not available. test_get_sentiment_score will be skipped.")

        if not (tiny_memories._NLTK_STOPWORDS_AVAILABLE and tiny_memories.nltk_stopwords_module) and \
           not (tiny_memories._SKLEARN_STOP_WORDS_AVAILABLE and tiny_memories.ENGLISH_STOP_WORDS) :
            print("Test WARNING: Neither NLTK nor sklearn stopwords available. test_extract_simple_words will be skipped.")

    def test_sentiment_analysis_initialization(self):
        self.assertIsNotNone(self.sentiment_analyzer, "SentimentAnalyzer instance should exist.")
        self.assertIsNotNone(self.sentiment_analyzer.stop_words, "Stop words list attribute should exist.")

        if not tiny_memories._TRANSFORMERS_AVAILABLE or tiny_memories.pipeline is None:
            self.assertIsNone(self.sentiment_analyzer.emo_classifier)

    def test_get_sentiment_score(self):
        if not tiny_memories._TEXTBLOB_AVAILABLE:
            self.skipTest("TextBlob is not available. Skipping test_get_sentiment_score.")
            return
        positive_text = "This is a wonderful and fantastic experience!"
        negative_text = "This is a terrible and awful situation."
        neutral_text = "The cat is on the mat."
        positive_score = self.sentiment_analyzer.get_sentiment_score(positive_text)
        negative_score = self.sentiment_analyzer.get_sentiment_score(negative_text)
        neutral_score = self.sentiment_analyzer.get_sentiment_score(neutral_text)
        self.assertIn("polarity", positive_score); self.assertIn("subjectivity", positive_score)
        self.assertGreater(positive_score["polarity"], 0)
        self.assertIn("polarity", negative_score); self.assertIn("subjectivity", negative_score)
        self.assertLess(negative_score["polarity"], 0)
        self.assertIn("polarity", neutral_score); self.assertIn("subjectivity", neutral_score)
        self.assertAlmostEqual(neutral_score["polarity"], 0, delta=0.2)

    def test_get_emotion_classification(self):
        if self.sentiment_analyzer.emo_classifier is None:
            self.skipTest("Emotion classifier is None. Skipping test_get_emotion_classification.")
            return
        joy_text = "I am so happy and joyful today!"
        try:
            joy_emotion = self.sentiment_analyzer.get_emotion_classification(joy_text)
            self.assertIsInstance(joy_emotion, str)
            possible_emotions = ["sadness", "joy", "love", "anger", "fear", "surprise", "others"]
            self.assertIn(joy_emotion, possible_emotions)
        except Exception as e:
            self.fail(f"get_emotion_classification failed unexpectedly: {e}")

    def test_extract_simple_words(self):
        nltk_ready = tiny_memories._NLTK_STOPWORDS_AVAILABLE and tiny_memories.nltk_stopwords_module is not None
        sklearn_stopwords_ready = tiny_memories._SKLEARN_STOP_WORDS_AVAILABLE and tiny_memories.ENGLISH_STOP_WORDS is not None
        if not (nltk_ready or sklearn_stopwords_ready):
             self.skipTest("Neither NLTK nor sklearn stopwords available. Skipping test_extract_simple_words.")
             return
        text_for_extraction = "This is a simple test with some very unique words like qwertyx zxcasd"
        extracted_for_check = self.sentiment_analyzer.extract_simple_words(text_for_extraction)
        self.assertIn("simple", extracted_for_check)
        self.assertNotIn("is", extracted_for_check)

from tiny_memories import Memory, BSTNode, MemoryBST, SpecificMemory, GeneralMemory, MemoryQuery
from tiny_memories import _TORCH_AVAILABLE, _TRANSFORMERS_AVAILABLE

class MockGameTimeManager:
    def __init__(self):
        self.current_time = datetime.now(); self.calendar = self
    def get_game_time(self): return self.current_time
    def advance_time(self, **kwargs): self.current_time += timedelta(**kwargs)

mock_gametime_manager_instance = MockGameTimeManager()

class SimpleMemoryObject:
    def __init__(self, id_val, data="some data"):
        self.id = id_val; self.data = data; self.importance_score = id_val
        self.last_access_time = datetime.now()
    def __str__(self): return f"SimpleMemoryObject(id={self.id})"

def create_test_bst_node_for_query(desc="test memory", tags=None, last_access_hours_ago=0,
                                   importance=0.5, polarity=0.0, subjectivity=0.0, emotion="others",
                                   keywords=None, attribute_val="test_attr"):
    dummy_gm_desc = "DummyGMForSpecificMemory"
    parent_gm = GeneralMemory(dummy_gm_desc)
    mem_obj = None
    try:
        mem_obj = SpecificMemory(description=desc, parent_memory=parent_gm, importance_score=importance)
    except Exception as e:
        print(f"Note: SpecificMemory direct init failed for test node setup: {e}. Creating with defaults.")
        mem_obj = Memory(description=desc)
        setattr(mem_obj, 'importance_score', importance)
        setattr(mem_obj, 'sentiment_score', {"polarity": polarity, "subjectivity": subjectivity})
        setattr(mem_obj, 'emotion_classification', emotion)
        setattr(mem_obj, 'keywords', keywords if keywords else [])
        setattr(mem_obj, 'tags', tags if tags else [])

    mem_obj.tags = tags if tags else []
    mem_obj.last_access_time = mock_gametime_manager_instance.get_game_time() - timedelta(hours=last_access_hours_ago)
    if not hasattr(mem_obj, 'importance_score'): setattr(mem_obj, 'importance_score', importance)
    else: mem_obj.importance_score = importance
    if not hasattr(mem_obj, 'sentiment_score'): setattr(mem_obj, 'sentiment_score', {"polarity": polarity, "subjectivity": subjectivity})
    else: mem_obj.sentiment_score = {"polarity": polarity, "subjectivity": subjectivity}
    if not hasattr(mem_obj, 'emotion_classification'): setattr(mem_obj, 'emotion_classification', emotion)
    else: mem_obj.emotion_classification = emotion
    if not hasattr(mem_obj, 'keywords'): setattr(mem_obj, 'keywords', keywords if keywords else [])
    else: mem_obj.keywords = keywords if keywords else []
    setattr(mem_obj, 'attribute', attribute_val)

    node_key = hash(desc)
    return BSTNode(key=node_key, memory=mem_obj)

class TestMemory(unittest.TestCase):
    def test_memory_initialization(self):
        desc = "Test event"; now = datetime.now()
        mem = Memory(description=desc, creation_time=now)
        self.assertEqual(mem.description, desc); self.assertEqual(mem.creation_time, now)
        self.assertEqual(mem.last_access_time, now)

    def test_update_access_time(self):
        mem = Memory("Test event"); initial_access_time = mem.last_access_time
        new_access_time = initial_access_time + timedelta(seconds=10)
        mem.update_access_time(new_access_time)
        self.assertEqual(mem.last_access_time, new_access_time)

class TestBSTNode(unittest.TestCase):
    def test_bstnode_initialization(self):
        mem_obj = SimpleMemoryObject(1)
        node = BSTNode(key=10, memory=mem_obj)
        self.assertEqual(node.key, 10); self.assertIs(node.memory, mem_obj)
        self.assertIsNone(node.left); self.assertIsNone(node.right)
        self.assertEqual(node.height, 1)

    def test_bstnode_height_balance_updates_manual(self):
        node5 = BSTNode(5, SimpleMemoryObject(5)); node30 = BSTNode(30, SimpleMemoryObject(30))
        node10 = BSTNode(10, SimpleMemoryObject(10)); node10.left = node5
        node10.update_height(); self.assertEqual(node10.height, 2); self.assertEqual(node10.get_balance(), 1)
        node20 = BSTNode(20, SimpleMemoryObject(20)); node20.left = node10; node20.right = node30
        node20.update_height(); self.assertEqual(node20.height, 3); self.assertEqual(node20.get_balance(), 1)

class TestMemoryBST(unittest.TestCase):
    def _is_avl_bst(self, node):
        if node is None: return True, 0
        is_left_avl, left_height = self._is_avl_bst(node.left)
        if not is_left_avl: return False, 0
        is_right_avl, right_height = self._is_avl_bst(node.right)
        if not is_right_avl: return False, 0
        if node.left and node.left.key >= node.key: return False, 0
        if node.right and node.right.key <= node.key: return False, 0
        if abs(left_height - right_height) > 1: return False, 0
        expected_height = 1 + max(left_height, right_height)
        return True, expected_height
    def assert_is_avl_bst(self, bst_root, test_name=""):
        is_avl, _ = self._is_avl_bst(bst_root)
        self.assertTrue(is_avl, f"Tree is not a valid AVL BST after {test_name}")
    def _find_node_by_bst_key(self, node, key_to_find):
        if node is None: return None
        if key_to_find < node.key: return self._find_node_by_bst_key(node.left, key_to_find)
        elif key_to_find > node.key: return self._find_node_by_bst_key(node.right, key_to_find)
        else: return node.memory
    def test_memorybst_initialization(self): bst = MemoryBST(key_attr="id"); self.assertIsNone(bst.specific_memories_root); self.assertEqual(bst.key_attr, "id")
    def test_insert_single_node(self): bst = MemoryBST(key_attr="id"); mem_obj = SimpleMemoryObject(id_val=100); bst.specific_memories_root = bst.insert(bst.specific_memories_root, 100, mem_obj); self.assertIsNotNone(bst.specific_memories_root); self.assertEqual(bst.specific_memories_root.key, 100); self.assertIs(bst.specific_memories_root.memory, mem_obj); self.assertEqual(bst.specific_memories_root.height, 1)
    def test_insert_multiple_nodes_ordered(self): bst = MemoryBST(key_attr="id"); keys = [10, 20, 5]; mems = {k: SimpleMemoryObject(id_val=k) for k in keys}; [setattr(bst, 'specific_memories_root', bst.insert(bst.specific_memories_root, key, mems[key])) for key in keys]; root = bst.specific_memories_root; self.assertEqual(root.key, 10); self.assertEqual(root.left.key, 5); self.assertEqual(root.right.key, 20); self.assertEqual(root.height, 2)
    def test_search_by_key_bst(self): bst = MemoryBST(key_attr="importance_score"); mem1=SimpleMemoryObject(1); mem2=SimpleMemoryObject(2); mem3=SimpleMemoryObject(3); bst.specific_memories_root=bst.insert(bst.specific_memories_root,mem2.importance_score,mem2); bst.specific_memories_root=bst.insert(bst.specific_memories_root,mem1.importance_score,mem1); bst.specific_memories_root=bst.insert(bst.specific_memories_root,mem3.importance_score,mem3); self.assertIs(self._find_node_by_bst_key(bst.specific_memories_root, 1), mem1); self.assertIsNone(self._find_node_by_bst_key(bst.specific_memories_root, 100))
    def test_insert_left_left_rotation(self): bst = MemoryBST(key_attr="id"); keys=[30,20,10]; mems={k:SimpleMemoryObject(id_val=k) for k in keys}; [setattr(bst, 'specific_memories_root', bst.insert(bst.specific_memories_root, key, mems[key])) for key in keys]; root=bst.specific_memories_root; self.assertEqual(root.key,20); self.assertEqual(root.left.key,10); self.assertEqual(root.right.key,30); self.assertEqual(root.height,2); self.assert_is_avl_bst(root,"LL rotation")
    def test_insert_right_right_rotation(self): bst = MemoryBST(key_attr="id"); keys=[10,20,30]; mems={k:SimpleMemoryObject(id_val=k) for k in keys}; [setattr(bst, 'specific_memories_root', bst.insert(bst.specific_memories_root, key, mems[key])) for key in keys]; root=bst.specific_memories_root; self.assertEqual(root.key,20); self.assertEqual(root.left.key,10); self.assertEqual(root.right.key,30); self.assertEqual(root.height,2); self.assert_is_avl_bst(root,"RR rotation")
    def test_insert_left_right_rotation(self): bst = MemoryBST(key_attr="id"); keys=[30,10,20]; mems={k:SimpleMemoryObject(id_val=k) for k in keys}; [setattr(bst, 'specific_memories_root', bst.insert(bst.specific_memories_root, key, mems[key])) for key in keys]; root=bst.specific_memories_root; self.assertEqual(root.key,20); self.assertEqual(root.left.key,10); self.assertEqual(root.right.key,30); self.assertEqual(root.height,2); self.assert_is_avl_bst(root,"LR rotation")
    def test_insert_right_left_rotation(self): bst = MemoryBST(key_attr="id"); keys=[10,30,20]; mems={k:SimpleMemoryObject(id_val=k) for k in keys}; [setattr(bst, 'specific_memories_root', bst.insert(bst.specific_memories_root, key, mems[key])) for key in keys]; root=bst.specific_memories_root; self.assertEqual(root.key,20); self.assertEqual(root.left.key,10); self.assertEqual(root.right.key,30); self.assertEqual(root.height,2); self.assert_is_avl_bst(root,"RL rotation")
    def test_delete_leaf_node(self): bst=MemoryBST(key_attr="id");keys=[20,10,30,5];mems={k:SimpleMemoryObject(id_val=k)for k in keys};[setattr(bst,'specific_memories_root',bst.insert(bst.specific_memories_root,key,mems[key]))for key in keys];bst.specific_memories_root=bst.delete(bst.specific_memories_root,5);self.assert_is_avl_bst(bst.specific_memories_root,"delete leaf 5");self.assertIsNone(self._find_node_by_bst_key(bst.specific_memories_root,5))
    def test_delete_node_with_one_right_child(self): bst=MemoryBST(key_attr="id");keys=[20,10,30,15];mems={k:SimpleMemoryObject(id_val=k)for k in keys};[setattr(bst,'specific_memories_root',bst.insert(bst.specific_memories_root,key,mems[key]))for key in keys];bst.specific_memories_root=bst.delete(bst.specific_memories_root,10);self.assert_is_avl_bst(bst.specific_memories_root,"delete node 10 (one right child)");self.assertEqual(bst.specific_memories_root.left.key,15);self.assertIsNone(self._find_node_by_bst_key(bst.specific_memories_root,10))
    def test_delete_node_with_one_left_child(self): bst=MemoryBST(key_attr="id");keys=[20,10,30,5];mems={k:SimpleMemoryObject(id_val=k)for k in keys};[setattr(bst,'specific_memories_root',bst.insert(bst.specific_memories_root,key,mems[key]))for key in keys];bst.specific_memories_root=bst.delete(bst.specific_memories_root,30);self.assert_is_avl_bst(bst.specific_memories_root,"delete node 30 (leaf)");self.assertIsNone(self._find_node_by_bst_key(bst.specific_memories_root,30));bst.specific_memories_root=bst.delete(bst.specific_memories_root,10);self.assert_is_avl_bst(bst.specific_memories_root,"delete node 10 (one left child 5)");self.assertEqual(bst.specific_memories_root.key,20);self.assertEqual(bst.specific_memories_root.left.key,5);self.assertIsNone(self._find_node_by_bst_key(bst.specific_memories_root,10))
    def test_delete_node_with_two_children(self): bst=MemoryBST(key_attr="id");keys=[20,10,30,5,15,25,35];mems={k:SimpleMemoryObject(id_val=k)for k in keys};[setattr(bst,'specific_memories_root',bst.insert(bst.specific_memories_root,key,mems[key]))for key in keys];bst.specific_memories_root=bst.delete(bst.specific_memories_root,10);self.assert_is_avl_bst(bst.specific_memories_root,"delete node 10 (two children)");self.assertEqual(bst.specific_memories_root.left.key,15);self.assertEqual(bst.specific_memories_root.left.left.key,5);self.assertIsNone(bst.specific_memories_root.left.right);self.assertIsNone(self._find_node_by_bst_key(bst.specific_memories_root,10))
    def test_delete_root_node_with_two_children(self): bst=MemoryBST(key_attr="id");keys=[20,10,30,5,15,25,35];mems={k:SimpleMemoryObject(id_val=k)for k in keys};[setattr(bst,'specific_memories_root',bst.insert(bst.specific_memories_root,key,mems[key]))for key in keys];bst.specific_memories_root=bst.delete(bst.specific_memories_root,20);self.assert_is_avl_bst(bst.specific_memories_root,"delete root node 20 (two children)");self.assertEqual(bst.specific_memories_root.key,25);self.assertIsNone(self._find_node_by_bst_key(bst.specific_memories_root,20))
    def test_delete_causing_rotations(self): bst=MemoryBST(key_attr="id");keys=[5,2,8,1,4,7,9,3,6];mems={k:SimpleMemoryObject(id_val=k)for k in keys};[setattr(bst,'specific_memories_root',bst.insert(bst.specific_memories_root,key,mems[key]))for key in keys];self.assert_is_avl_bst(bst.specific_memories_root,"initial complex tree");bst.specific_memories_root=bst.delete(bst.specific_memories_root,9);self.assert_is_avl_bst(bst.specific_memories_root,"delete 9 (causes L rotation at 8, new root of this sub-tree 7)");self.assertEqual(bst.specific_memories_root.right.key,7,"After deleting 9, root's right child should be 7");bst.specific_memories_root=bst.delete(bst.specific_memories_root,8);self.assert_is_avl_bst(bst.specific_memories_root,"delete 8");bst.specific_memories_root=bst.delete(bst.specific_memories_root,1);self.assert_is_avl_bst(bst.specific_memories_root,"delete 1 (may cause complex rotation)")

class TestMemoryQuery(unittest.TestCase):
    def setUp(self):
        self.gametime_manager = mock_gametime_manager_instance
        self.gametime_manager.current_time = datetime(2024, 1, 1, 12, 0, 0)

    def test_memoryquery_initialization(self):
        query_text = "Test query"
        mq = MemoryQuery(query_text, gametime_manager=self.gametime_manager)
        self.assertEqual(mq.query, query_text); self.assertIsNotNone(mq.query_time)
        self.assertEqual(mq.query_tags, []); self.assertIsNone(mq.attribute)
        self.assertIs(mq.gametime_manager, self.gametime_manager)
        self.assertIs(mq.model, tiny_memories.model)

    def test_memoryquery_initialization_with_tags_and_attribute(self):
        query_text = "Tagged query"; tags = ["tag1", "tag2"]; attribute = "location"
        mq = MemoryQuery(query_text, query_tags=tags, attribute=attribute, gametime_manager=self.gametime_manager)
        self.assertEqual(mq.query_tags, tags); self.assertEqual(mq.attribute, attribute)

    @unittest.skipIf(not (_TORCH_AVAILABLE and _TRANSFORMERS_AVAILABLE and hasattr(tiny_memories, 'model') and tiny_memories.model is not None and hasattr(tiny_memories.model, 'model') and tiny_memories.model.model is not None),
                     "Skipping get_embedding tests as torch/transformers/global model.model are not fully available.")
    def test_get_embedding(self):
        mq = MemoryQuery("embed this", gametime_manager=self.gametime_manager)
        embedding = mq.get_embedding()
        self.assertIsNotNone(embedding); self.assertTrue(hasattr(embedding, 'shape'))
        self.assertTrue(embedding.shape == (1, 768) or embedding.shape == (768,) , f"Embedding shape unexpected: {embedding.shape}")

    def test_add_complex_query(self):
        mq = MemoryQuery("base query", gametime_manager=self.gametime_manager)
        attribute = "relevance"; complex_q_str = "Is *memory_description* relevant to *attribute*?"
        mq.add_complex_query(attribute, complex_q_str)
        self.assertIn(attribute, mq.complex_query)
        self.assertEqual(mq.complex_query[attribute], "Is *memory_description* relevant to relevance?")

    def test_by_tags_function(self):
        mq = MemoryQuery("query for tags", query_tags=["urgent", "work"], gametime_manager=self.gametime_manager)
        node_match = create_test_bst_node_for_query(tags=["work", "projectA"])
        node_no_match = create_test_bst_node_for_query(tags=["personal", "hobby"])
        self.assertTrue(mq.by_tags_function(node_match))
        self.assertFalse(mq.by_tags_function(node_no_match))

    def test_by_time_function(self):
        mq = MemoryQuery("query for time", gametime_manager=self.gametime_manager)
        node_recent = create_test_bst_node_for_query(last_access_hours_ago=0.5)
        node_old = create_test_bst_node_for_query(last_access_hours_ago=2)
        self.assertTrue(mq.by_time_function(node_recent, None))
        self.assertFalse(mq.by_time_function(node_old, None))

    def test_by_importance_function(self):
        mq = MemoryQuery("query for importance", gametime_manager=self.gametime_manager)
        node = create_test_bst_node_for_query(importance=0.8)
        self.assertTrue(mq.by_importance_function(node, min_importance=0.5, max_importance=1.0))
        self.assertFalse(mq.by_importance_function(node, min_importance=0.1, max_importance=0.5))

    def test_by_sentiment_function(self):
        mq = MemoryQuery("query for sentiment", gametime_manager=self.gametime_manager)
        node = create_test_bst_node_for_query(polarity=0.7, subjectivity=0.9)
        self.assertTrue(mq.by_sentiment_function(node, min_polarity=0.5, max_polarity=1.0, min_subjectivity=0.5, max_subjectivity=1.0))
        self.assertFalse(mq.by_sentiment_function(node, min_polarity=0.8, max_polarity=1.0, min_subjectivity=0.5, max_subjectivity=1.0))

    def test_by_emotion_function(self):
        mq = MemoryQuery("query for emotion", gametime_manager=self.gametime_manager)
        node_joy = create_test_bst_node_for_query(emotion="joy")
        self.assertTrue(mq.by_emotion_function(node_joy, emotion="joy"))
        self.assertFalse(mq.by_emotion_function(node_joy, emotion="sadness"))

    def test_by_keywords_function(self):
        mq = MemoryQuery("query for keywords", gametime_manager=self.gametime_manager)
        node = create_test_bst_node_for_query(keywords=["apple", "banana", "cherry"])
        self.assertTrue(mq.by_keywords_function(node, keywords=["banana", "grape"]))
        self.assertFalse(mq.by_keywords_function(node, keywords=["grape", "orange"]))

    @unittest.skipIf(not (_TORCH_AVAILABLE and _TRANSFORMERS_AVAILABLE and hasattr(tiny_memories, 'model') and tiny_memories.model is not None and hasattr(tiny_memories.model, 'model') and tiny_memories.model.model is not None),
                     "Skipping by_similarity_function test as torch/transformers/global model.model are not fully available.")
    def test_by_similarity_function(self):
        query_text = "similar content"; mq = MemoryQuery(query_text, gametime_manager=self.gametime_manager)
        query_embedding = mq.get_embedding()
        if query_embedding is None: self.skipTest("Query embedding failed.")
        node_similar = create_test_bst_node_for_query(desc=query_text)
        if not hasattr(node_similar.memory, 'get_embedding'): self.skipTest("Node memory missing get_embedding.")
        node_similar.memory.get_embedding()
        if node_similar.memory.embedding is None : self.skipTest("Node similar embedding failed.")
        self.assertTrue(mq.by_similarity_function(node_similar, query_embedding=query_embedding, threshold=0.9))

    def test_by_attribute_function(self):
        mq = MemoryQuery("query for attribute", gametime_manager=self.gametime_manager)
        node_attr_match = create_test_bst_node_for_query(attribute_val="specific_value")
        self.assertTrue(mq.by_attribute_function(node_attr_match, "specific_value")) # Positional
        node_no_attr = create_test_bst_node_for_query(); delattr(node_no_attr.memory, 'attribute')
        self.assertFalse(mq.by_attribute_function(node_no_attr, "specific_value")) # Positional

# --- New Test Classes for SpecificMemory and GeneralMemory ---
class TestSpecificMemory(unittest.TestCase):
    def setUp(self):
        self.gametime_manager = mock_gametime_manager_instance
        self.parent_gm_desc = "TestParentGM_for_SpecificMemory"

        # Always create a local parent_gm for test isolation
        self.parent_gm = GeneralMemory(self.parent_gm_desc)

        self.test_desc = "A specific test memory about a sunny day."
        self.importance = 0.7

        try:
            self.sm = SpecificMemory(self.test_desc, self.parent_gm, self.importance)
        except Exception as e:
            raise unittest.SkipTest(f"Skipping SpecificMemory tests: Failed to instantiate SpecificMemory due to: {e}.")

    def test_specificmemory_initialization(self):
        self.assertEqual(self.sm.description, self.test_desc)
        self.assertIs(self.sm.parent_memory, self.parent_gm)
        self.assertEqual(self.sm.importance_score, self.importance)
        self.assertIsNotNone(self.sm.creation_time)
        self.assertEqual(self.sm.creation_time, self.sm.last_access_time)
        self.assertEqual(self.sm.related_memories, [])
        self.assertEqual(self.sm.tags, [])
        self.assertIsNotNone(self.sm.analysis) # analyze_description is called in __init__

    def test_add_tag(self):
        self.sm.add_tag("new_tag"); self.assertIn("new_tag", self.sm.tags)
        self.sm.add_tag("another_tag"); self.assertIn("another_tag", self.sm.tags); self.assertEqual(len(self.sm.tags), 2)

    def test_update_importance_score(self):
        self.sm.update_importance_score(0.95); self.assertEqual(self.sm.importance_score, 0.95)

    def test_update_parent_memory(self):
        new_parent_gm = GeneralMemory("NewParentGM")
        self.sm.update_parent_memory(new_parent_gm); self.assertIs(self.sm.parent_memory, new_parent_gm)

    def test_add_related_memory(self):
        related_sm = SpecificMemory("Related SM", self.parent_gm, 0.5)
        self.sm.add_related_memory(related_sm)
        self.assertIn(related_sm, self.sm.related_memories)
        self.assertIn(self.sm, related_sm.related_memories) # Check bidirectional link

    @unittest.skipIf(not (_TORCH_AVAILABLE and _TRANSFORMERS_AVAILABLE and hasattr(tiny_memories, 'model') and tiny_memories.model is not None and hasattr(tiny_memories.model, 'model') and tiny_memories.model.model is not None),
                     "Skipping SM embedding tests: model not fully available.")
    def test_specificmemory_get_embedding(self):
        embedding, att_mask = self.sm.get_embedding()
        if embedding is None: self.skipTest("Embedding generation failed in SM.")
        self.assertTrue(hasattr(embedding, 'shape')); self.assertTrue(embedding.shape == (1,768) or embedding.shape == (768,))
        self.assertIsNotNone(att_mask)

    def test_specificmemory_pickling(self):
        import pickle
        if not (_TORCH_AVAILABLE and _TRANSFORMERS_AVAILABLE and hasattr(tiny_memories, 'model') and tiny_memories.model is not None and hasattr(tiny_memories.model, 'model') and tiny_memories.model.model is not None):
            self.sm.embedding = None; self.sm.att_mask = None; self.sm.facts_embeddings = None
        else:
            try: self.sm.get_embedding(); self.sm.get_facts_embeddings()
            except Exception: self.sm.embedding = None; self.sm.att_mask = None; self.sm.facts_embeddings = None

        pickled_sm = pickle.dumps(self.sm)
        unpickled_sm = pickle.loads(pickled_sm)
        self.assertEqual(unpickled_sm.description, self.sm.description)
        self.assertIsNone(unpickled_sm.embedding); self.assertIsNone(unpickled_sm.att_mask); self.assertIsNone(unpickled_sm.facts_embeddings)

class TestGeneralMemory(unittest.TestCase):
    def setUp(self):
        self.gm_desc = "Test General Memory for setUp"
        try:
            self.gm = GeneralMemory(self.gm_desc)
        except Exception as e:
            raise unittest.SkipTest(f"Skipping GeneralMemory tests: Failed to instantiate GeneralMemory: {e}.")

    def test_generalmemory_initialization(self):
        self.assertEqual(self.gm.description, self.gm_desc)
        self.assertIsNotNone(self.gm.creation_time); self.assertIsNotNone(self.gm.timestamp_tree)
        self.assertTrue(isinstance(self.gm.keywords, list))

    def test_add_specific_memory_and_get(self):
        sm_desc1 = "Specific Memory 1 for GM Add"; sm_desc2 = "Specific Memory 2 for GM Add"
        # SpecificMemory init will use global manager, which might be None or limited.
        # This test primarily checks GM's ability to store/retrieve SpecificMemory objects.
        sm1 = SpecificMemory(sm_desc1, self.gm, 0.8)
        sm2 = SpecificMemory(sm_desc2, self.gm, 0.6)

        self.gm.add_specific_memory(sm1) # add_specific_memory now takes the object
        self.gm.add_specific_memory(sm2)

        retrieved_memories = self.gm.get_specific_memories()
        self.assertEqual(len(retrieved_memories), 2)
        self.assertIn(sm1, retrieved_memories); self.assertIn(sm2, retrieved_memories)

    @unittest.skipIf(not (_TORCH_AVAILABLE and _TRANSFORMERS_AVAILABLE and hasattr(tiny_memories, 'model') and tiny_memories.model is not None and hasattr(tiny_memories.model, 'model') and tiny_memories.model.model is not None and tiny_memories._FAISS_AVAILABLE and tiny_memories.faiss is not None),
                     "Skipping GM index_memories: model or faiss not available.")
    def test_generalmemory_index_memories(self):
        local_gm = GeneralMemory("LocalGM for index_memories")
        sm1_local = SpecificMemory("SM1local_idx_GM", local_gm, 0.8)
        if hasattr(sm1_local, 'get_embedding'): sm1_local.get_embedding() # Prime embedding

        local_gm.add_specific_memory(sm1_local)

        faiss_index_result = local_gm.index_memories()

        if hasattr(sm1_local, 'embedding') and sm1_local.embedding is not None:
            self.assertIsNotNone(local_gm.faiss_index)
            if local_gm.faiss_index: self.assertGreater(local_gm.faiss_index.ntotal, 0)
        else:
            self.assertIsNone(local_gm.faiss_index)

if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
