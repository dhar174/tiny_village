import html
import json
import pickle
import random
import math
import re
import time
from collections import deque
from datetime import datetime, timedelta
import os
import sys
import heapq
import functools
import inspect

# --- Conditional Imports & Availability Flags ---
_PANDAS_AVAILABLE = False; pd = None
try:
    import pandas as pd; _PANDAS_AVAILABLE = True; print("Successfully imported pandas.")
except ImportError: print("WARNING: pandas library not found.")

_SCIPY_AVAILABLE = False; sp = None
try:
    import scipy as sp; _SCIPY_AVAILABLE = True; print("Successfully imported scipy.")
except ImportError: print("WARNING: scipy library not found.")

_SKLEARN_AVAILABLE = False
_SKLEARN_STOP_WORDS_AVAILABLE = False
_SKLEARN_KMEANS_AVAILABLE = False
_SKLEARN_LDA_AVAILABLE = False
_SKLEARN_CV_AVAILABLE = False
_SKLEARN_NMF_AVAILABLE = False

tree = None; cluster = None; decomposition = None; cosine_similarity = None
TfidfVectorizer = None; ENGLISH_STOP_WORDS = None; CountVectorizer = None
LatentDirichletAllocation = None; KMeans = None; NMF = None
try:
    from sklearn import tree, cluster, decomposition
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
    try:
        from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
        _SKLEARN_STOP_WORDS_AVAILABLE = True
    except ImportError: ENGLISH_STOP_WORDS = None; print("WARNING: sklearn ENGLISH_STOP_WORDS not available.")
    try:
        from sklearn.cluster import KMeans
        _SKLEARN_KMEANS_AVAILABLE = True
    except ImportError: KMeans = None; print("WARNING: sklearn KMeans not available.")
    try:
        from sklearn.decomposition import LatentDirichletAllocation
        _SKLEARN_LDA_AVAILABLE = True
    except ImportError: LatentDirichletAllocation = None; print("WARNING: sklearn LatentDirichletAllocation not available.")
    try:
        from sklearn.decomposition import NMF
        _SKLEARN_NMF_AVAILABLE = True
    except ImportError: NMF = None; print("WARNING: sklearn NMF not available.")

    _SKLEARN_AVAILABLE = True
    print("Successfully imported sklearn components (or tried to).")
except ImportError:
    print("WARNING: Core scikit-learn library not found. Sklearn-dependent features will be unavailable.")

_NETWORKX_AVAILABLE = False; nx = None; node_link_data = None
try:
    import networkx as nx
    from networkx import node_link_data # This is a function, not a class to instantiate
    _NETWORKX_AVAILABLE = True; print("Successfully imported networkx.")
except ImportError: print("WARNING: networkx library not found.")

_SYMPY_AVAILABLE = False; Q = None; comp = None; lex = None; per = None
try:
    from sympy import Q, comp, lex, per
    _SYMPY_AVAILABLE = True; print("Successfully imported sympy.")
except ImportError: print("WARNING: sympy library not found.")

_TORCH_AVAILABLE = False; torch = None
try:
    import torch; _TORCH_AVAILABLE = True; print("Successfully imported torch.")
except ImportError: print("WARNING: PyTorch (torch) not found.")

_TRANSFORMERS_AVAILABLE = False
AutoModel = None; AutoTokenizer = None; pipeline = None; AutoModelForSequenceClassification = None
try:
    from transformers import AutoModel, AutoTokenizer, pipeline, AutoModelForSequenceClassification
    _TRANSFORMERS_AVAILABLE = True; print("Successfully imported transformers.")
except ImportError: print("WARNING: Transformers library not found.")

_FAISS_AVAILABLE = False; faiss = None
try:
    import faiss; _FAISS_AVAILABLE = True; print("Successfully imported faiss.")
except ImportError: print("WARNING: FAISS library not found.")

_SPACY_AVAILABLE = False; _SPACY_TRF_AVAILABLE = False; _SPACY_SM_AVAILABLE = False
spacy = None; nlp = None
try:
    import spacy; _SPACY_AVAILABLE = True; print("Successfully imported spacy.")
    try:
        nlp = spacy.load("en_core_web_trf"); _SPACY_TRF_AVAILABLE = True
        print("Successfully loaded en_core_web_trf.")
    except OSError:
        print("WARNING: en_core_web_trf not found. Trying en_core_web_sm.")
        try:
            nlp = spacy.load("en_core_web_sm"); _SPACY_SM_AVAILABLE = True
            print("Successfully loaded en_core_web_sm.")
        except OSError: print("WARNING: No spaCy models (trf or sm) found, nlp set to None.")
except ImportError: print("WARNING: spaCy library not found, nlp set to None.")

_RAKE_NLTK_AVAILABLE = False; Rake = None
try:
    from rake_nltk import Rake; _RAKE_NLTK_AVAILABLE = True; print("Successfully imported rake_nltk.")
except ImportError: print("WARNING: rake_nltk library not found.")

_GENSIM_AVAILABLE = False; corpora = None; models = None
try:
    from gensim import corpora, models; _GENSIM_AVAILABLE = True; print("Successfully imported gensim.")
except ImportError: print("WARNING: gensim library not found.")

_NLTK_AVAILABLE = False; _NLTK_STOPWORDS_AVAILABLE = False; _NLTK_WORDNET_AVAILABLE = False
_NLTK_PUNKT_AVAILABLE = False; _NLTK_TAGGER_AVAILABLE = False
nltk = None; RegexpTokenizer = None; nltk_stopwords_module = None; wordnet = None
pos_tag = None; PorterStemmer = None; word_tokenize = None
try:
    import nltk; _NLTK_AVAILABLE = True
    try: from nltk.tokenize import RegexpTokenizer, word_tokenize; _NLTK_PUNKT_AVAILABLE = True
    except ImportError: RegexpTokenizer = None; word_tokenize = None; print("NLTK tokenizers not found.")
    try: from nltk.corpus import stopwords as nltk_stopwords_module; _NLTK_STOPWORDS_AVAILABLE = True
    except ImportError: nltk_stopwords_module = None; print("nltk.corpus.stopwords not found.")
    try: from nltk.corpus import wordnet; _NLTK_WORDNET_AVAILABLE = True
    except ImportError: wordnet = None; print("nltk.corpus.wordnet not found.")
    try: from nltk import pos_tag; _NLTK_TAGGER_AVAILABLE = True
    except ImportError: pos_tag = None; print("nltk.pos_tag not found.")
    try: from nltk.stem import PorterStemmer
    except ImportError: PorterStemmer = None; print("nltk.stem.PorterStemmer not found.")
    print("Successfully imported NLTK base (specific components might be missing).")
except ImportError: print("WARNING: NLTK library itself not found.")

_TEXTBLOB_AVAILABLE = False; TextBlob = None
try:
    from textblob import TextBlob; _TEXTBLOB_AVAILABLE = True; print("Successfully imported textblob.")
except ImportError: print("WARNING: textblob library not found.")

_MATPLOTLIB_AVAILABLE = False; matplotlib = None; plt = None
try:
    import matplotlib; matplotlib.use("TkAgg"); import matplotlib.pyplot as plt
    _MATPLOTLIB_AVAILABLE = True; print("Successfully imported matplotlib.")
except ImportError: print("WARNING: matplotlib library not found.")

_SEABORN_AVAILABLE = False; sns = None
try:
    import seaborn as sns; _SEABORN_AVAILABLE = True; print("Successfully imported seaborn.")
except ImportError: print("WARNING: seaborn library not found.")

# Local imports
import tiny_brain_io as tbi
import tiny_time_manager as ttm
import tiny_sr_mapping as tsm

os.environ["TRANSFORMERS_CACHE"] = "/mnt/d/transformers_cache"
remove_list = [r"\)", r"\(", r"–", r'"', r"”", r'"', r"\[.*\]", r".*\|.*", r"—"]

if _NLTK_AVAILABLE and nltk is not None:
    try:
        if _NLTK_STOPWORDS_AVAILABLE: nltk.download("stopwords", quiet=True)
        if _NLTK_WORDNET_AVAILABLE: nltk.download("wordnet", quiet=True)
        if _NLTK_PUNKT_AVAILABLE: nltk.download("punkt", quiet=True)
        if _NLTK_TAGGER_AVAILABLE: nltk.download("averaged_perceptron_tagger", quiet=True)
    except Exception as e: print(f"NLTK download error: {e}.")
else: print("NLTK or its components not available, skipping NLTK data downloads.")

class_interaction_graph = nx.DiGraph() if _NETWORKX_AVAILABLE and nx else None
call_flow_diagram = nx.DiGraph() if _NETWORKX_AVAILABLE and nx else None

debug = True
def debug_print(text):
    if debug: print(f"{text}")

urgent_words = ["now", "immediately", "urgently", "help", "hurry", "quickly", "fast", "soon", "asap", "as soon as possible", "quick", "urgent", "important", "critical", "essential", "vital", "crucial", "imperative", "necessary", "compulsory", "mandatory", "pressing", "acute", "severe", "desperate", "dire", "extreme", "serious", "grave", "momentous", "weighty", "significant", "paramount", "decisive", "conclusive", "decisive", "deciding", "determining", "key", "pivotal", "Oh no!", "Help!", "Yikes!", "very", "extremely", "absolutely", "within", "by", "before", "until", "speed", "rush", "must", "need"]
def sigmoid(x): return 1 / (1 + math.exp(-x))

# --- Class Definitions ---
class EmbeddingModel:
    def __init__(self):
        if not (_TORCH_AVAILABLE and _TRANSFORMERS_AVAILABLE and AutoTokenizer is not None and AutoModel is not None):
            self.tokenizer = None; self.model = None; self.device = None
            print("WARNING: EmbeddingModel init failed: PyTorch/Transformers missing.")
            return
        try:
            self.tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-mpnet-base-v2", trust_remote_code=True, cache_dir="/mnt/d/transformers_cache")
            self.model = AutoModel.from_pretrained("sentence-transformers/all-mpnet-base-v2", trust_remote_code=True, cache_dir="/mnt/d/transformers_cache")
            self.device = "cuda" if _TORCH_AVAILABLE and torch.cuda.is_available() else "cpu"
            if self.model and self.device: self.model.to(self.device); self.model.eval()
        except Exception as e:
            print(f"Error initializing EmbeddingModel: {e}")
            self.tokenizer = None; self.model = None; self.device = None

    def forward(self, input_ids, attention_mask):
        if self.model is None or not (_TORCH_AVAILABLE and torch is not None) : return None
        try:
            with torch.no_grad(): outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            return outputs
        except Exception as e: print(f"Error in EmbeddingModel.forward: {e}"); return None

def get_wordnet_pos(spacy_token):
    if not (_SPACY_AVAILABLE and spacy and _NLTK_WORDNET_AVAILABLE and wordnet): return ""
    if spacy_token.pos_ == "ADJ": return wordnet.ADJ
    elif spacy_token.pos_ == "VERB": return wordnet.VERB
    elif spacy_token.pos_ == "NOUN": return wordnet.NOUN
    elif spacy_token.pos_ == "ADV": return wordnet.ADV
    else: return ""

class SentimentAnalysis:
    def __init__(self):
        self.emo_classifier = None
        if _TRANSFORMERS_AVAILABLE and pipeline is not None:
            try: self.emo_classifier = pipeline(model="lordtt13/emo-mobilebert")
            except Exception as e: print(f"WARNING: Failed to load emo_classifier: {e}")

        self.stop_words = []
        try:
            sklearn_sw, nltk_sw = [], []
            if _SKLEARN_STOP_WORDS_AVAILABLE and ENGLISH_STOP_WORDS is not None: sklearn_sw = list(ENGLISH_STOP_WORDS)
            if _NLTK_STOPWORDS_AVAILABLE and nltk_stopwords_module is not None:
                try: nltk_sw = nltk_stopwords_module.words("english")
                except LookupError: print("NLTK stopwords corpus not found for SA.")
                except AttributeError: print("NLTK stopwords module not correctly loaded for SA.")
            self.stop_words = list(set(sklearn_sw).union(set(nltk_sw)))
            if not self.stop_words and (_SKLEARN_STOP_WORDS_AVAILABLE or _NLTK_STOPWORDS_AVAILABLE): # Print warning only if a source was expected
                print("WARNING: SentimentAnalysis loaded no stopwords, though one or more sources were available.")
        except Exception as e: print(f"Error loading stopwords in SentimentAnalysis: {e}")


    def get_sentiment_score(self, text):
        if not (_TEXTBLOB_AVAILABLE and TextBlob): return {"polarity": 0.0, "subjectivity": 0.0}
        try: tb = TextBlob(text); return {"polarity": tb.polarity, "subjectivity": tb.subjectivity}
        except Exception as e: print(f"Error in get_sentiment_score: {e}"); return {"polarity": 0.0, "subjectivity": 0.0}

    def get_emotion_classification(self, text):
        if self.emo_classifier is None: return "unknown"
        try: result = self.emo_classifier(text); return result[0]["label"]
        except Exception as e: print(f"Error in get_emotion_classification: {e}"); return "error"

    def extract_simple_words(self, text):
        current_stop_words = self.stop_words if self.stop_words is not None else []
        if type(text) is str:
            text_no_punct = re.sub(r"[^\w\s]", "", text)
            return [word for word in text_no_punct.split() if word.lower() not in current_stop_words and word]
        if type(text) is list:
            processed_list = []
            for item in text:
                item_clean = re.sub(r"[^\w\s]", "", str(item))
                if item_clean.lower() not in current_stop_words and item_clean: processed_list.append(item_clean)
            return processed_list
        return []

def mean_pooling(model_output_obj, attention_mask):
    if not (_TORCH_AVAILABLE and torch is not None): return None
    if model_output_obj is None or not hasattr(model_output_obj, 'last_hidden_state') or model_output_obj.last_hidden_state is None: return None
    try:
        token_embeddings = model_output_obj.last_hidden_state
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask
    except Exception as e: print(f"Error in mean_pooling: {e}"); return None

class MemoryQuery:
    def __init__(self, query, query_time=datetime.now(), query_tags=None, attribute=None, gametime_manager: ttm.GameTimeManager = None):
        self.query = query; self.query_time = query_time
        self.query_tags = query_tags if query_tags is not None else []
        self.attribute = attribute; self.gametime_manager = gametime_manager
        self.model = model
        self.query_embedding = None; self.complex_query = {}; self.retrieved_memories = []
        self.retrieval_time = None; self.retrieval_method = None; self.retrieval_parameters = None; self.analysis = None
        if self.gametime_manager is None: print("MemoryQuery WARNING: Game time manager is required but not provided.")

    def generate_embedding(self):
        if self.model is None or self.model.tokenizer is None or self.model.model is None or self.model.device is None:
            return None, None
        description = [self.query.strip()]
        try:
            input_data = self.model.tokenizer(description, padding=True, truncation=True, add_special_tokens=True, is_split_into_words=False, pad_to_multiple_of=8, return_tensors="pt").to(self.model.device)
            model_outputs = None
            if _TORCH_AVAILABLE and torch is not None:
                with torch.no_grad(): model_outputs = self.model.forward(input_data["input_ids"], input_data["attention_mask"])
            else: return None, None
            return model_outputs, input_data["attention_mask"]
        except Exception as e: print(f"Error in MemoryQuery.generate_embedding: {e}"); return None, None

    def get_embedding(self):
        if self.query_embedding is None:
            model_output_obj, attention_mask = self.generate_embedding()
            if model_output_obj and attention_mask:
                self.query_embedding = mean_pooling(model_output_obj, attention_mask)
            else: self.query_embedding = None
        return self.query_embedding

    def add_complex_query(self, attribute, query_str):
        self.attribute = attribute
        if "*attribute*" in query_str: query_str = query_str.replace("*attribute*", self.attribute)
        self.complex_query[attribute] = query_str

    def by_tags_function(self, node):
        return node.memory.tags and any(tag in node.memory.tags for tag in self.query_tags)
    def by_time_function(self, node, time_period):
        cutoff = time_period if time_period else (self.gametime_manager.get_game_time() - timedelta(hours=1))
        return node.memory.last_access_time > cutoff
    def by_importance_function(self, node, min_importance, max_importance):
        return min_importance <= node.memory.importance_score <= max_importance
    def by_sentiment_function(self, node, min_polarity, max_polarity, min_subjectivity, max_subjectivity):
        if hasattr(node.memory, 'sentiment_score') and node.memory.sentiment_score:
            return min_polarity <= node.memory.sentiment_score["polarity"] <= max_polarity and \
                   min_subjectivity <= node.memory.sentiment_score["subjectivity"] <= max_subjectivity
        return False
    def by_emotion_function(self, node, emotion):
        return hasattr(node.memory, 'emotion_classification') and node.memory.emotion_classification == emotion
    def by_keywords_function(self, node, keywords):
        return node.memory.keywords and any(kw in node.memory.keywords for kw in keywords)
    def by_similarity_function(self, node, query_embedding, threshold):
        if not (_SKLEARN_AVAILABLE and cosine_similarity and hasattr(node.memory, 'embedding') and node.memory.embedding is not None and query_embedding is not None): return False
        try:
            # Assuming embeddings are 2D numpy arrays or compatible for cosine_similarity
            mem_emb = node.memory.embedding.cpu().detach().numpy() if _TORCH_AVAILABLE and torch and isinstance(node.memory.embedding, torch.Tensor) else node.memory.embedding
            query_emb = query_embedding.cpu().detach().numpy() if _TORCH_AVAILABLE and torch and isinstance(query_embedding, torch.Tensor) else query_embedding
            if not hasattr(mem_emb, 'shape') or not hasattr(query_emb, 'shape'): return False # Not array-like
            if len(mem_emb.shape) == 1: mem_emb = mem_emb.reshape(1, -1)
            if len(query_emb.shape) == 1: query_emb = query_emb.reshape(1, -1)
            sim = cosine_similarity(mem_emb, query_emb)
            return sim[0][0] > threshold
        except Exception as e: print(f"Error in by_similarity_function: {e}"); return False
    def by_attribute_function(self, node, attribute_name_param):
        return hasattr(node.memory, 'attribute') and getattr(node.memory, 'attribute') == attribute_name_param

class Memory:
    def __init__(self, description, creation_time=datetime.now()):
        self.description = description; self.creation_time = creation_time; self.last_access_time = creation_time
    def update_access_time(self, access_time): self.last_access_time = access_time

class BSTNode:
    def __init__(self, key, memory): self.key = key; self.memory = memory; self.left = None; self.right = None; self.height = 1
    def update_height(self): left_h = self.left.height if self.left else 0; right_h = self.right.height if self.right else 0; self.height = 1 + max(left_h, right_h)
    def get_balance(self): left_h = self.left.height if self.left else 0; right_h = self.right.height if self.right else 0; return left_h - right_h
    def rotate_left(self): new_root = self.right; self.right = new_root.left; new_root.left = self; self.update_height(); new_root.update_height(); return new_root
    def rotate_right(self): new_root = self.left; self.left = new_root.right; new_root.right = self; self.update_height(); new_root.update_height(); return new_root

class MemoryBST:
    def __init__(self, key_attr): self.specific_memories_root = None; self.key_attr = key_attr
    def get_height(self, node): return node.height if node else 0
    def update_height(self, node): node.height = 1 + max(self.get_height(node.left), self.get_height(node.right))
    def get_balance(self, node): return self.get_height(node.left) - self.get_height(node.right) if node else 0
    def right_rotate(self, y): x = y.left; T2 = x.right; x.right = y; y.left = T2; self.update_height(y); self.update_height(x); return x
    def left_rotate(self, x): y = x.right; T2 = y.left; y.left = x; x.right = T2; self.update_height(x); self.update_height(y); return y
    def insert(self, node, key, memory):
        if not node: return BSTNode(key, memory)
        elif key < node.key: node.left = self.insert(node.left, key, memory)
        else: node.right = self.insert(node.right, key, memory)
        self.update_height(node); balance = self.get_balance(node)
        if balance > 1 and key < node.left.key: return self.right_rotate(node)
        if balance < -1 and key > node.right.key: return self.left_rotate(node)
        if balance > 1 and key > node.left.key: node.left = node.left.rotate_left(); return self.right_rotate(node)
        if balance < -1 and key < node.right.key: node.right = node.right.rotate_right(); return self.left_rotate(node)
        return node
    def delete(self, node, key):
        if not node: return node
        if key < node.key: node.left = self.delete(node.left, key)
        elif key > node.key: node.right = self.delete(node.right, key)
        else:
            if node.left is None: temp = node.right; node = None; return temp
            elif node.right is None: temp = node.left; node = None; return temp
            temp = self.minValueNode(node.right); node.key = temp.key; node.memory = temp.memory
            node.right = self.delete(node.right, temp.key)
        if node is None: return node
        node.height = 1 + max(self.get_height(node.left), self.get_height(node.right)); balance = self.get_balance(node)
        if balance > 1 and self.get_balance(node.left) >= 0: return self.right_rotate(node)
        if balance > 1 and self.get_balance(node.left) < 0: node.left = self.left_rotate(node.left); return self.right_rotate(node)
        if balance < -1 and self.get_balance(node.right) <= 0: return self.left_rotate(node)
        if balance < -1 and self.get_balance(node.right) > 0: node.right = self.right_rotate(node.right); return self.left_rotate(node)
        return node
    def minValueNode(self, node_param):
        current = node_param
        while current.left is not None: current = current.left
        return current
    def search_by_key(self, node, key):
        if node is None or key == node.key: return node.memory if node else None
        if key < node.key: return self.search_by_key(node.left, key)
        return self.search_by_key(node.right, key)

class GeneralMemory(Memory):
    def __init__(self, description, creation_time=datetime.now()):
        super().__init__(description, creation_time)
        self.description = description; self.description_embedding = None
        self.faiss_index = None; self.timestamp_tree = None; self.importance_tree = None; self.key_tree = None
        self.init_trees(); self.keywords = []
        if manager and hasattr(manager, 'extract_keywords'): self.keywords = manager.extract_keywords(self.description)
        self.analysis = None; self.sentiment_score = None; self.emotion_classification = None
        self.entities = []; self.main_subject = None; self.main_verb = None; self.main_object = None
        self.temporal_expressions = None; self.verb_aspects = None
        self.get_embedding()

    def generate_embedding(self):
        if model is None or not hasattr(model, 'tokenizer') or not hasattr(model, 'model') or not hasattr(model, 'device') or \
           model.tokenizer is None or model.model is None or model.device is None: return None, None
        description = [self.description.strip()]
        try:
            input_data = model.tokenizer(description, padding=True, truncation=True, add_special_tokens=True, is_split_into_words=False, pad_to_multiple_of=8, return_tensors="pt").to(model.device)
            model_outputs = None
            if _TORCH_AVAILABLE and torch is not None:
                with torch.no_grad(): model_outputs = model.forward(input_data["input_ids"], input_data["attention_mask"])
            else: return None, None
            return model_outputs, input_data["attention_mask"]
        except Exception as e: print(f"Error in GeneralMemory.generate_embedding: {e}"); return None, None

    def get_embedding(self):
        if self.description_embedding is None:
            model_output_obj, attention_mask = self.generate_embedding()
            if model_output_obj and attention_mask:
                self.description_embedding = mean_pooling(model_output_obj, attention_mask)
            else: self.description_embedding = None
        return self.description_embedding
    def init_trees(self): self.timestamp_tree = MemoryBST("last_access_time"); self.importance_tree = MemoryBST("importance_score"); self.key_tree = MemoryBST("keys")
    def get_specific_memories(self, key=None, attribute=None, tree=None):
        tree_to_traverse = tree if tree is not None else self.key_tree
        if tree_to_traverse is None: return []
        specific_memories = []; self._inorder_traversal(tree_to_traverse.specific_memories_root, lambda node_item: specific_memories.append(node_item.memory)); return specific_memories
    def _inorder_traversal(self, node, visit):
        if node is not None: self._inorder_traversal(node.left, visit); visit(node); self._inorder_traversal(node.right, visit)
    def add_specific_memory(self, specific_memory_obj): # Now expects an object
        key = id(specific_memory_obj)
        if self.timestamp_tree is None: self.init_trees()
        if manager and specific_memory_obj.analysis is None : specific_memory_obj.analyze_description()
        self.timestamp_tree.specific_memories_root = self.timestamp_tree.insert(self.timestamp_tree.specific_memories_root, specific_memory_obj.last_access_time, specific_memory_obj)
        self.importance_tree.specific_memories_root = self.importance_tree.insert(self.importance_tree.specific_memories_root, specific_memory_obj.importance_score, specific_memory_obj)
        self.key_tree.specific_memories_root = self.key_tree.insert(self.key_tree.specific_memories_root, key, specific_memory_obj)
        if manager and hasattr(manager, 'update_embeddings'): manager.update_embeddings(specific_memory_obj)
    def index_memories(self):
        if not (_FAISS_AVAILABLE and faiss and _TORCH_AVAILABLE and torch) : return None
        # ... (FAISS logic) ...

class SpecificMemory(Memory):
    def __init__(self, description, parent_memory, importance_score, subject=None, normalized_embeddings=False):
        super().__init__(description)
        self.normalized_embeddings = normalized_embeddings; self.subject = subject
        self.parent_memory = parent_memory
        self.related_memories = []; self.embedding = None; self.att_mask = None; self.keywords = []
        self.tags = []; self.importance_score = importance_score
        self.sentiment_score = None; self.emotion_classification = None; self.entities = []
        self.analysis = None; self.main_subject = None; self.main_verb = None; self.main_object = None
        self.temporal_expressions = None; self.verb_aspects = None; self.facts = []
        self.facts_embeddings = None; self.map_fact_embeddings = {}
        if manager: self.analyze_description()

    def analyze_description(self):
        if manager and hasattr(manager, 'analyze_query_context'):
            self.analysis = manager.analyze_query_context(self.description)
            if self.analysis:
                self.sentiment_score = self.analysis.get("sentiment_score"); self.emotion_classification = self.analysis.get("emotion_classification")
                self.keywords = self.analysis.get("keywords", []); self.main_subject = self.analysis.get("main_subject")
                self.main_verb = self.analysis.get("main_verb"); self.main_object = self.analysis.get("main_object")
                self.verb_aspects = self.analysis.get("verb_aspects"); self.facts = self.analysis.get("facts", [])
                self.facts_embeddings = self.get_facts_embeddings()
        else: print("WARNING: SpecificMemory.analyze_description: manager or analyze_query_context not available.")

    def generate_embedding(self, string=None, normalize=None):
        if model is None or not hasattr(model, 'tokenizer') or not hasattr(model, 'model') or not hasattr(model, 'device') or \
           model.tokenizer is None or model.model is None or model.device is None: return None, None
        if string is None: string = self.description
        description = [string.strip()]
        try:
            input_data = model.tokenizer(description, padding=True, truncation=True, add_special_tokens=True, is_split_into_words=False, pad_to_multiple_of=8, return_tensors="pt").to(model.device)
            model_outputs = None
            if _TORCH_AVAILABLE and torch is not None:
                with torch.no_grad(): model_outputs = model.forward(input_data["input_ids"], input_data["attention_mask"])
            else: return None, None
            if model_outputs and hasattr(model_outputs, 'last_hidden_state') and model_outputs.last_hidden_state is not None and normalize and _TORCH_AVAILABLE and torch is not None:
                model_outputs.last_hidden_state = torch.nn.functional.normalize(model_outputs.last_hidden_state, p=2, dim=-1)
            return model_outputs, input_data["attention_mask"]
        except Exception as e: print(f"Error in SpecificMemory.generate_embedding: {e}"); return None, None

    def get_embedding(self, normalize=None):
        if normalize is not None and self.normalized_embeddings != normalize:
            self.normalized_embeddings = normalize; self.embedding = None
        if self.embedding is None:
            model_output_obj, attention_mask = self.generate_embedding(normalize=self.normalized_embeddings)
            if model_output_obj and attention_mask:
                self.embedding = mean_pooling(model_output_obj, attention_mask)
                self.att_mask = attention_mask
            else: self.embedding = None; self.att_mask = None
        return self.embedding, self.att_mask

    def get_facts_embeddings(self):
        if not (_TORCH_AVAILABLE and torch is not None and _TRANSFORMERS_AVAILABLE): return []
        if self.facts_embeddings is None and self.facts:
            facts_embeddings_list = []
            for fact in self.facts:
                if not fact: continue
                fact_embedding_output, fact_attention_mask = self.generate_embedding(fact)
                if fact_embedding_output and fact_attention_mask:
                    fact_embedding = mean_pooling(fact_embedding_output, fact_attention_mask)
                    if fact_embedding is not None: facts_embeddings_list.append(fact_embedding)
            self.facts_embeddings = facts_embeddings_list
        return self.facts_embeddings if self.facts_embeddings is not None else []

    def add_tag(self, tag): self.tags.append(tag)
    def update_importance_score(self, score): self.importance_score = score
    def update_parent_memory(self, parent_mem): self.parent_memory = parent_mem
    def add_related_memory(self, rel_mem):
        if rel_mem not in self.related_memories: self.related_memories.append(rel_mem)
        if hasattr(rel_mem, 'add_related_memory') and self not in rel_mem.related_memories: rel_mem.add_related_memory(self)


class FlatMemoryAccess:
    def __init__(self, memory_embeddings={}, json_file=None, index_load_filename=None):
        self.index_load_filename = index_load_filename; self.index_is_normalized = False
        self.recent_memories = deque(maxlen=50); self.common_memories = {}; self.repetitive_memories = {}
        self.urgent_query_memories = {}; self.most_importance_memories = []
        self.index_id_to_node_id = {}; self.faiss_index = None; self.euclidean_threshold = 0.35
        self.index_build_count = 0; self.memory_embeddings = memory_embeddings; self.specific_memories = {}
        if _FAISS_AVAILABLE and faiss is not None:
             self.initialize_faiss_index(768, "ip", index_load_filename=index_load_filename)
    def initialize_faiss_index(self, dimension, metric="l2", normalize=False, index_load_filename=None):
        if not (_FAISS_AVAILABLE and faiss): self.faiss_index = None; return
        # ... (FAISS init logic) ...
    def find_memories_by_query(self, query, top_k=10, threshold=0.5, similarity_metric="euclidian", num_clusters=10, use_hnsw=False):
        if not (_FAISS_AVAILABLE and faiss and _TORCH_AVAILABLE and torch and self.faiss_index): return {}
        query_embedding = query.get_embedding()
        if query_embedding is None: return {}
        try:
            query_vec = query_embedding.cpu().detach().numpy()
            # ... (FAISS search logic) ...
            return {}
        except Exception as e: print(f"Error in FAISS search: {e}"); return {}
    # ... (Other FlatMemoryAccess methods) ...

class MemoryManager:
    def __init__(self, gametime_manager, index_load_filename=None, global_sentiment_analysis=None):
        self.flat_access = FlatMemoryAccess(index_load_filename=index_load_filename)
        self.index_load_filename = index_load_filename; self.memory_embeddings = {}
        self.complex_keywords = set(); self.similarity_threshold_specific = 0.4
        self.similarity_threshold_general = 0.25; self.recent_queries = deque(maxlen=50)
        self.faiss_index_recent_queries_flatl2 = None; self.map_tags = {}; self.general_memories = {}
        self.gametime_manager = gametime_manager

        self.lda_tokenizer = RegexpTokenizer(r"\w+") if _NLTK_AVAILABLE and RegexpTokenizer else None
        if not self.lda_tokenizer: print("WARNING: NLTK RegexpTokenizer not available for MemoryManager.")

        if _SKLEARN_AVAILABLE and CountVectorizer and LatentDirichletAllocation and self.lda_tokenizer:
            self.lda_vectorizer = CountVectorizer(stop_words="english", min_df=1)
            self.lda = LatentDirichletAllocation(n_components=3, max_iter=5, learning_method="online", learning_offset=50.0, random_state=0)
        else:
            self.lda_vectorizer = None; self.lda = None
            print("WARNING: sklearn/NLTK components for LDA not available in MemoryManager.")

        current_stop_words_for_tfidf = []
        sa_to_use = global_sentiment_analysis if global_sentiment_analysis else sentiment_analysis
        if sa_to_use and hasattr(sa_to_use, 'stop_words') and sa_to_use.stop_words:
            current_stop_words_for_tfidf = sa_to_use.stop_words
        elif _SKLEARN_STOP_WORDS_AVAILABLE and ENGLISH_STOP_WORDS is not None: # Check is not None
            current_stop_words_for_tfidf = list(ENGLISH_STOP_WORDS)

        if _SKLEARN_AVAILABLE and TfidfVectorizer:
            self.tfidf_vectorizer = TfidfVectorizer(stop_words=current_stop_words_for_tfidf if current_stop_words_for_tfidf else None)
        else:
            self.tfidf_vectorizer = None; print("WARNING: sklearn TfidfVectorizer not available.")

        self.rake = Rake() if _RAKE_NLTK_AVAILABLE and Rake else None
        if not self.rake: print("WARNING: Rake (rake_nltk) not available.")
        assert self.gametime_manager is not None, "Game time manager is required"

    def get_query_embedding(self, query_text):
        if model is None or not hasattr(model, 'tokenizer') or not hasattr(model, 'model') or not hasattr(model, 'device') or \
           model.tokenizer is None or model.model is None or model.device is None: return None
        try:
            input_data = model.tokenizer(query_text, return_tensors="pt", padding=True, truncation=True).to(model.device)
            if _TORCH_AVAILABLE and torch is not None:
                with torch.no_grad(): model_output_obj = model.forward(input_data["input_ids"], input_data["attention_mask"])
                if model_output_obj is None: return None
                return mean_pooling(model_output_obj, input_data["attention_mask"])
            else: return None
        except Exception as e: print(f"Error in MM.get_query_embedding: {e}"); return None

    def analyze_query_context(self, query_text):
        if not isinstance(query_text, str): # Ensure query_text is a string
            print(f"WARNING: analyze_query_context expected string, got {type(query_text)}. Returning empty analysis.")
            return {"keywords": [], "facts": [], "text": str(query_text)} # Basic fallback

        docs = None; words = query_text.split(); num_words = len(words)
        avg_word_length = sum(len(w) for w in words) / num_words if num_words > 0 else 0
        features = {"tokens": words, "lemmas": [], "pos_tags": []}
        main_subject, main_verb, main_object = "Unknown", "Unknown", "Unknown"
        keywords = query_text.split() if query_text else []; templates = []
        complexity = len(words) // 5 if num_words > 0 else 0; is_common_query = False; is_urgent = False
        lexical_density = 1.0 if num_words > 0 else 0.0; type_token_ratio = 1.0 if num_words > 0 else 0.0
        verb_aspects = {}; named_entities = {}

        if _SPACY_AVAILABLE and nlp is not None:
            try:
                docs = nlp(query_text)
                if docs and len(docs) > 0 :
                    features = {"tokens": [t.text for t in docs], "lemmas": [t.lemma_ for t in docs], "pos_tags": [t.pos_ for t in docs]}
                    if _NLTK_AVAILABLE and tsm and hasattr(tsm, 'main'):
                        fa_templates = tsm.main(docs, nlp)
                        templates = fa_templates if fa_templates else []
            except Exception as e: print(f"Error in MM.analyze_query_context (spaCy): {e}"); docs = None

        query_embedding_val = self.get_query_embedding(query_text)
        for word in words:
            if word.lower() in urgent_words: is_urgent = True; break
        if _TORCH_AVAILABLE and _FAISS_AVAILABLE and query_embedding_val is not None and \
           hasattr(self, 'faiss_index_recent_queries_flatl2') and self.faiss_index_recent_queries_flatl2 and \
           self.faiss_index_recent_queries_flatl2.ntotal > 0:
            pass
        sa_instance = sentiment_analysis
        sentiment_score_val = sa_instance.get_sentiment_score(query_text) if sa_instance else {"polarity":0.0, "subjectivity":0.0}
        emotion_classification_val = sa_instance.get_emotion_classification(query_text) if sa_instance else "unknown"

        return {
            "embedding": query_embedding_val, "sentiment_score": sentiment_score_val,
            "emotion_classification": emotion_classification_val, "ambiguity_score": complexity / 10,
            "keywords": keywords, "text": query_text, "main_subject": main_subject, "main_verb": main_verb,
            "main_object": main_object, "verb_aspects": verb_aspects, "complexity": complexity,
            "is_common_query": is_common_query, "is_urgent": is_urgent, "lexical_density": lexical_density,
            "type_token_ratio": type_token_ratio, "avg_word_length": avg_word_length,
            "lexical_diversity": type_token_ratio, "facts": templates, "named_entities": named_entities,
        }

    def extract_keywords(self, text):
        if not isinstance(text, str): # Ensure text is a string for keyword extraction
            print(f"WARNING: extract_keywords expected string, got {type(text)}. Returning empty list.")
            return []
        keywords_set = set()
        if self.rake:
            try:
                self.rake.extract_keywords_from_text(text)
                keywords_set.update(self.rake.get_ranked_phrases()[:5])
            except Exception as e: print(f"Error with Rake keyword extraction: {e}")
        if self.tfidf_vectorizer:
            try: keywords_set.update(self.extract_tfidf_keywords([text]))
            except Exception as e: print(f"Error with TFIDF keyword extraction: {e}")
        if _SPACY_AVAILABLE and nlp:
            try:
                doc = nlp(text)
                keywords_set.update(ent.text for ent in doc.ents)
            except Exception as e: print(f"Error with spaCy entity extraction for keywords: {e}")
        return list(keywords_set)


    def extract_tfidf_keywords(self, docs, top_n=3):
        if not (_SKLEARN_AVAILABLE and self.tfidf_vectorizer): return []
        if not isinstance(docs, list): docs = [docs]
        if not docs or not all(isinstance(d, str) for d in docs): # Ensure all docs are strings
            print("WARNING: extract_tfidf_keywords received non-string data in docs list.")
            return []
        try:
            X = self.tfidf_vectorizer.fit_transform(docs)
            if X.shape[0] == 0 or X.shape[1] == 0: return []
            feature_array = self.tfidf_vectorizer.get_feature_names_out()
            actual_top_n = min(top_n, X.shape[1]);
            if actual_top_n == 0: return []
            tfidf_keywords = []
            for row_indices_array in X.toarray().argsort(axis=1)[:, -actual_top_n:]:
                 for i_val in row_indices_array:
                    if i_val < len(feature_array): tfidf_keywords.append(feature_array[i_val])
            return tfidf_keywords
        except Exception as e: print(f"Error in extract_tfidf_keywords: {e}"); return []
    # ... (Other MemoryManager methods) ...

# --- Global Instance Initializations ---
model = None
sentiment_analysis = None
manager = None

try:
    if _TORCH_AVAILABLE and _TRANSFORMERS_AVAILABLE and AutoTokenizer is not None and AutoModel is not None:
        model = EmbeddingModel()
    else:
        print("Global 'model' (EmbeddingModel) could not be initialized due to missing PyTorch/Transformers.")
except Exception as e:
    print(f"CRITICAL ERROR during global model initialization: {e}")
    model = None

try:
    sentiment_analysis = SentimentAnalysis()
    print(f"Global 'sentiment_analysis' instance initialized. emo_classifier available: {hasattr(sentiment_analysis, 'emo_classifier') and sentiment_analysis.emo_classifier is not None}")
except Exception as e:
    print(f"CRITICAL ERROR during global sentiment_analysis initialization: {e}")
    sentiment_analysis = None

try:
    _tiny_calendar_global = ttm.GameCalendar()
    _tiny_time_manager_global = ttm.GameTimeManager(_tiny_calendar_global)
    manager = MemoryManager(_tiny_time_manager_global, "ip_no_norm.bin", global_sentiment_analysis=sentiment_analysis)
    print(f"Global 'manager' (MemoryManager) instance initialized.")
except Exception as e:
    print(f"CRITICAL ERROR during global manager initialization: {e}")
    manager = None

if __name__ == "__main__":
    tiny_brain_io_instance = None
    if tbi and hasattr(tbi, 'TinyBrainIO'):
        try: tiny_brain_io_instance = tbi.TinyBrainIO("alexredna/TinyLlama-1.1B-Chat-v1.0-reasoning-v2")
        except Exception as e: print(f"Error initializing TinyBrainIO in __main__: {e}")
    else: print("Skipping TinyBrainIO initialization in __main__.")

    print("Main block reached. Module partially runnable depending on available libraries.")
    if manager:
        try:
            test_query = "What is the weather like?"
            results = manager.search_memories(test_query)
        except Exception as e: print(f"Error during __main__ simplified test query: {e}")

        all_fine = _SPACY_AVAILABLE and nlp and model and sentiment_analysis and hasattr(sentiment_analysis, 'emo_classifier') and sentiment_analysis.emo_classifier
        if all_fine: print("All main components (Spacy, Model, Sentiment with EmoClassifier) seem to be available.")
        else: print("One or more main components NOT available. __main__ functionality limited.")
    else: print("MemoryManager not initialized. Cannot run __main__ test queries.")
    # ... (Original __main__ example run commented out)
