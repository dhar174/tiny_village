from ast import Raise
from calendar import c
import json
from locale import normalize
import math
import re
import time
import token
from typing import final
from llama_cpp import deque
from networkx import node_link_data
import numpy as np
from datetime import datetime, timedelta
import pandas as pd
from regex import F, P
import scipy as sp
from sklearn import tree
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
from sympy import Q, lex, per
import torch
from transformers import AutoModel, AutoTokenizer, pipeline
from rake_nltk import Rake
import faiss
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim import corpora, models
import nltk
from nltk.tokenize import RegexpTokenizer
from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from nltk.corpus import stopwords
import heapq
import tiny_brain_io as tbi
import tiny_time_manager as ttm
import os
import sys

from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

os.environ["TRANSFORMERS_CACHE"] = "/mnt/d/transformers_cache"
remove_list = ["\)", "\(", "–", '"', "”", '"', "\[.*\]", ".*\|.*", "—"]
lda = LatentDirichletAllocation(n_components=3)
nlp = spacy.load("en_core_web_sm")
import nltk
import os
from spacy.matcher import Matcher
from spacy.tokens import Span

if not os.path.exists(nltk.data.find("corpora/stopwords")):
    nltk.download("stopwords")

import matplotlib

matplotlib.use("TkAgg")  # or 'Qt5Agg', or 'inline' for Jupyter notebooks
import matplotlib.pyplot as plt
import seaborn as sns

from transformers import AutoModelForSequenceClassification, AutoTokenizer
import functools

import inspect
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize
from nltk import pos_tag

class_interaction_graph = nx.DiGraph()
call_flow_diagram = nx.DiGraph()
from nltk.stem import PorterStemmer
import gensim

# Write a line of code that will identify how many classes are in this module:
num_classes = 0
for name, obj in inspect.getmembers(sys.modules[__name__]):
    if inspect.isclass(obj):
        num_classes += 1

# import scann


# #Make a heat map that shows the number of times each class calls each other class
# heat_map = {}
# for name, obj in inspect.getmembers(sys.modules[__name__]):
#     if inspect.isclass(obj):
#         heat_map[obj.__name__] = {}
#         for name2, obj2 in inspect.getmembers(sys.modules[__name__]):
#             if inspect.isclass(obj2):
#                 heat_map[obj.__name__][obj2.__name__] = 0
# print(heat_map)

# def track_calls(cls):
#     class Wrapper:
#         def __init__(self, *args, **kwargs):
#             self.wrapped = cls(*args, **kwargs)

#         def __getattr__(self, name):
#             attr = getattr(self.wrapped, name)
#             if callable(attr):

#                 @functools.wraps(attr)
#                 def wrapper(*args, **kwargs):
#                     start_time = time.time()
#                     # result = func(*args, **kwargs)
#                     end_time = time.time()

#                     # Update the Class Interaction Graph
#                     if args and isinstance(args[0], EmbeddingModel):
#                         class_interaction_graph.add_edge(args[0].__class__.__name__, attr.__name__)
#                     if args and isinstance(args[0], SentimentAnalysis):
#                         class_interaction_graph.add_edge(args[0].__class__.__name__, attr.__name__)
#                     if args and isinstance(args[0], MemoryQuery):
#                         class_interaction_graph.add_edge(args[0].__class__.__name__, attr.__name__)
#                     if args and isinstance(args[0], Memory):
#                         class_interaction_graph.add_edge(args[0].__class__.__name__, attr.__name__)
#                     if args and isinstance(args[0], BSTNode):
#                         class_interaction_graph.add_edge(args[0].__class__.__name__, attr.__name__)
#                     if args and isinstance(args[0], MemoryBST):
#                         class_interaction_graph.add_edge(args[0].__class__.__name__, attr.__name__)
#                     if args and isinstance(args[0], GeneralMemory):
#                         class_interaction_graph.add_edge(args[0].__class__.__name__, attr.__name__)
#                     if args and isinstance(args[0], SpecificMemory):
#                         class_interaction_graph.add_edge(args[0].__class__.__name__, attr.__name__)
#                     if args and isinstance(args[0], MemoryHierarchy):
#                         class_interaction_graph.add_edge(args[0].__class__.__name__, attr.__name__)
#                     if args and isinstance(args[0], FlatMemoryAccess):
#                         class_interaction_graph.add_edge(args[0].__class__.__name__, attr.__name__)
#                     if args and isinstance(args[0], MemoryManager):
#                         class_interaction_graph.add_edge(args[0].__class__.__name__, attr.__name__)
#                     class_name = None
#                     # Update the Call Flow Diagram
#                     caller = inspect.stack()[1].function
#                     if caller != '<module>':
#                         func_name = f"{attr.__self__.__class__.__name__}.{name}" if hasattr(attr, '__self__') else name

#                         if not call_flow_diagram.has_edge(caller, func_name):
#                             call_flow_diagram.add_edge(caller, func_name, num_calls=0, total_time=0)
#                         call_flow_diagram[caller][func_name]['num_calls'] += 1
#                         call_flow_diagram[caller][func_name]['total_time'] += end_time - start_time
#                         call_flow_diagram[caller][func_name]['average_time'] = call_flow_diagram[caller][func_name]['total_time'] / call_flow_diagram[caller][func_name]['num_calls']
#                         call_flow_diagram[caller][func_name]['weight'] = call_flow_diagram[caller][func_name]['num_calls'] * call_flow_diagram[caller][func_name]['average_time']
#                         call_flow_diagram[caller][func_name]['color'] = 'red' if call_flow_diagram[caller][func_name]['average_time'] > 0.1 else 'black'
#                         call_flow_diagram[caller][func_name]['penwidth'] = call_flow_diagram[caller][func_name]['num_calls'] * 0.1

#                     # Update heat_map using the class names and function names
#                         class_name = args[0].__class__.__name__ if args and hasattr(args[0], '__class__') else None
#                         func_name = f"{attr.__self__.__class__.__name__}.{name}" if hasattr(attr, '__self__') else name
#                         if class_name not in heat_map:
#                             heat_map[class_name] = {}
#                         if func_name not in heat_map[class_name]:
#                             heat_map[class_name][func_name] = 0
#                         heat_map[class_name][func_name] += 1
#                         # ...
#                     return attr(*args, **kwargs)


#                     # return result
#                 return wrapper
#             return attr
#     return Wrapper


urgent_words = [
    "now",
    "immediately",
    "urgently",
    "help",
    "hurry",
    "quickly",
    "fast",
    "soon",
    "asap",
    "as soon as possible",
    "quick",
    "urgent",
    "important",
    "critical",
    "essential",
    "vital",
    "crucial",
    "imperative",
    "necessary",
    "compulsory",
    "mandatory",
    "pressing",
    "acute",
    "severe",
    "desperate",
    "dire",
    "extreme",
    "serious",
    "grave",
    "momentous",
    "weighty",
    "significant",
    "paramount",
    "decisive",
    "conclusive",
    "decisive",
    "deciding",
    "determining",
    "key",
    "pivotal",
    "critical",
    "crucial",
    "vital",
    "essential",
    "indispensable",
    "necessary",
    "compulsory",
    "mandatory",
    "obligatory",
    "required",
    "requisite",
    "needful",
    "requisite",
    "needful",
    "urgent",
    "imperative",
    "pressing",
    "acute",
    "serious",
    "grave",
    "momentous",
    "weighty",
    "significant",
    "paramount",
    "decisive",
    "conclusive",
    "decisive",
    "deciding",
    "determining",
    "key",
    "pivotal",
    "critical",
    "crucial",
    "vital",
    "essential",
    "indispensable",
    "necessary",
    "compulsory",
    "mandatory",
    "obligatory",
    "required",
    "requisite",
    "needful",
    "requisite",
    "needful",
    "urgent",
    "imperative",
    "pressing",
    "acute",
    "serious",
    "grave",
    "momentous",
    "weighty",
    "significant",
    "paramount",
    "decisive",
    "conclusive",
    "decisive",
    "deciding",
    "determining",
    "key",
    "pivotal",
    "critical",
    "crucial",
    "vital",
    "essential",
    "indispensable",
    "necessary",
    "compulsory",
    "mandatory",
    "obligatory",
    "required",
    "requisite",
    "needful",
    "requisite",
    "needful",
    "urgent",
    "imperative",
    "pressing",
    "acute",
    "serious",
    "grave",
    "momentous",
    "weighty",
    "significant",
    "paramount",
    "decisive",
    "conclusive",
    "decisive",
    "deciding",
    "determining",
    "key",
    "pivotal",
    "Oh no!",
    "Help!",
    "Yikes!",
    "now",
    "immediately",
    "instantly",
    "right away",
    "very",
    "extremely",
    "absolutely",
    "urgently",
    "within",
    "by",
    "before",
    "until",
    "speed",
    "rush",
    "must",
    "need",
]


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


# Example usage:
# @track_calls
class EmbeddingModel:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(
            "sentence-transformers/all-mpnet-base-v2",
            trust_remote_code=True,
            cache_dir="/mnt/d/transformers_cache",
        )
        self.model = AutoModel.from_pretrained(
            "sentence-transformers/all-mpnet-base-v2",
            trust_remote_code=True,
            cache_dir="/mnt/d/transformers_cache",
        )
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        self.model.eval()
        self.model.zero_grad()

    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        print(f"Shape of outputs: {outputs.last_hidden_state.shape}")
        return outputs


def get_wordnet_pos(spacy_token):
    if spacy_token.pos_ == "ADJ":
        return wordnet.ADJ
    elif spacy_token.pos_ == "VERB":
        return wordnet.VERB
    elif spacy_token.pos_ == "NOUN":
        return wordnet.NOUN
    elif spacy_token.pos_ == "ADV":
        return wordnet.ADV
    else:
        return ""


# @track_calls
class SentimentAnalysis:
    def __init__(self):
        # self.classifier = pipeline('sentiment-analysis')
        self.emo_classifier = pipeline(model="lordtt13/emo-mobilebert")
        self.stop_words = list(ENGLISH_STOP_WORDS.union(stopwords.words("english")))

    def get_sentiment_score(self, text):
        # result = self.classifier(text)
        # # Converting result to a numerical score (e.g., positive to 1, negative to -1)
        # sentiment_score = 1 if result[0]['label'] == 'POSITIVE' else -1
        analysisPol = TextBlob(text).polarity
        analysisSub = TextBlob(text).subjectivity
        sentiment_score = {"polarity": analysisPol, "subjectivity": analysisSub}
        print(f"Sentiment score: {sentiment_score}")
        return sentiment_score

    def get_emotion_classification(self, text):
        result = self.emo_classifier(text)
        print(f"Emotion classification: {result[0]['label']}")
        return result[0]["label"]

    def extract_simple_words(self, text):
        if type(text) is str:
            text = re.sub(r"[^\w\s]", "", text)
            return [word for word in text.split() if word not in self.stop_words]
        if type(text) is list:
            for i in range(len(text)):
                text[i] = re.sub(r"[^\w\s]", "", text[i])
            return [word for word in text if word not in self.stop_words]


def mean_pooling(model_output, attention_mask):
    # print(type(model_output))
    # print(type(attention_mask))
    token_embeddings = (
        model_output  # Assuming model_output is already just the embeddings
    )
    input_mask_expanded = (
        attention_mask.unsqueeze(-1).expand_as(token_embeddings).float()
    )
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask


# @track_calls
class MemoryQuery:
    def __init__(
        self,
        query,
        query_time=datetime.now(),
        query_tags=[],
        attribute=None,
        gametime_manager: ttm.GameTimeManager = None,
    ):
        self.complex_query = {}
        self.retrieved_memories = {}
        self.attribute = None
        if attribute is not None:
            self.attribute = attribute
        self.query = query
        self.query_function = None
        self.query_time = query_time
        self.retrieved_memories = []
        self.retrieval_time = None
        self.retrieval_method = None
        self.retrieval_parameters = None
        self.analysis = None
        self.gametime_manager = (
            gametime_manager
            if gametime_manager is not None
            else Raise("Game time manager is required")
        )
        self.query_embedding = None
        self.model = model

        self.query_tags = []
        if query_tags is not None:
            self.query_tags = query_tags

    def get_embedding(self):
        if self.query_embedding is None:
            query_embedding_and_attention_mask = self.generate_embedding()
            self.query_embedding = mean_pooling(
                query_embedding_and_attention_mask[0],
                query_embedding_and_attention_mask[1],
            )
        return self.query_embedding

    def generate_embedding(self):
        description = [self.query.strip()]
        input = model.tokenizer(
            description,
            padding=True,
            truncation=True,
            add_special_tokens=True,
            is_split_into_words=True,
            pad_to_multiple_of=8,
            return_tensors="pt",
        ).to(model.device)
        outputs = model.model(
            input["input_ids"],
            attention_mask=input["attention_mask"],
            output_hidden_states=True,
            return_dict=True,
            return_tensors="pt",
        )
        return [outputs.last_hidden_state, input["attention_mask"]]

    def add_complex_query(self, attribute, query):
        self.attribute = attribute
        if "*attribute*" in query:
            query = query.replace("*attribute*", self.attribute)

        # Example: "Answer only with yes or no: is *memory_description* relevant to *attribute*?"
        self.complex_query[attribute] = query

    def add_query_function(self, query_function):
        self.query_function = query_function

    def by_complex_function(self, node):
        if "*memory_description*" in node.memory.description:
            self.complex_query[self.attribute] = node.memory.description.replace(
                "*memory_description*", node.memory.description
            )

        response = tiny_brain_io.input_to_model(self.attribute_query[self.attribute])[0]
        response = response.replace(self.complex_query[self.attribute], "")
        if (
            "yes" in response
            or "affirmative" in response
            or "positive" in response
            or "correct" in response
            or "true" in response
            or "indeed" in response
            or "is relevant" in response
            or "is related" in response
            or "is directly related" in response
        ):
            return True
        elif (
            "no" in response
            or "negative" in response
            or "incorrect" in response
            or "false" in response
            or "not relevant" in response
            or "not related" in response
        ):
            return False

    def by_tags_function(self, node):
        if node.memory.tags is not None and len(node.memory.tags) > 0:
            if any(tag in node.memory.tags for tag in self.query_tags):
                return True
        return False

    def by_time_function(self, node, time_period):
        if time_period is None:
            time_period = self.gametime_manager.calendar.get_game_time() - timedelta(
                hours=1
            )
        if node.memory.last_access_time > time_period:
            return True
        return False

    def by_importance_function(self, node, min_importance, max_importance):
        if min_importance <= node.memory.importance_score <= max_importance:
            return True
        return False

    def by_sentiment_function(
        self, node, min_polarity, max_polarity, min_subjectivity, max_subjectivity
    ):
        if (
            min_polarity <= node.memory.sentiment_score["polarity"] <= max_polarity
            and min_subjectivity
            <= node.memory.sentiment_score["subjectivity"]
            <= max_subjectivity
        ):
            return True
        return False

    def by_emotion_function(self, node, emotion):
        if node.memory.emotion_classification == emotion:
            return True
        return False

    def by_keywords_function(self, node, keywords):
        if node.memory.keywords is not None and len(node.memory.keywords) > 0:
            if any(keyword in node.memory.keywords for keyword in keywords):
                return True
        return False

    def by_similarity_function(self, node, query_embedding, threshold):
        similarity = cosine_similarity(node.memory.embedding, query_embedding)
        if similarity > threshold:
            return True
        return False

    def by_attribute_function(self, node, attribute):
        if node.memory.attribute == attribute:
            return True
        return False


# @track_calls
class Memory:
    def __init__(self, description, creation_time=datetime.now()):
        self.description = description
        self.creation_time = creation_time
        self.last_access_time = creation_time

    def update_access_time(self, access_time):
        self.last_access_time = access_time


# @track_calls
class BSTNode:
    def __init__(self, key, memory):
        self.key = key
        self.memory = memory
        self.left = None
        self.right = None
        self.height = 1  # Initial height set to 1 for leaf nodes

    def update_height(self):
        # Update the node's height based on the heights of its children
        left_height = self.left.height if self.left else 0
        right_height = self.right.height if self.right else 0
        self.height = 1 + max(left_height, right_height)

    def get_balance(self):
        # Calculate and return the balance factor of the node
        left_height = self.left.height if self.left else 0
        right_height = self.right.height if self.right else 0
        return left_height - right_height

    def rotate_left(self):
        # Perform a left rotation on the node
        new_root = self.right
        self.right = new_root.left
        new_root.left = self
        self.update_height()
        new_root.update_height()
        return new_root

    def rotate_right(self):
        # Perform a right rotation on the node
        new_root = self.left
        self.left = new_root.right
        new_root.right = self
        self.update_height()
        new_root.update_height()
        return new_root

    def rotate_left_right(self):
        # Perform a left-right rotation on the node
        self.left = self.left.rotate_left()
        return self.rotate_right()

    def rotate_right_left(self):
        # Perform a right-left rotation on the node
        self.right = self.right.rotate_right()
        return self.rotate_left()


# @track_calls
class MemoryBST:
    def __init__(self, key_attr):
        self.specific_memories_root = None  # BST root
        self.key_attr = key_attr  # Determines which attribute to organize by

    def get_height(self, node):
        if not node:
            return 0
        return node.height

    def update_height(self, node):
        node.height = 1 + max(self.get_height(node.left), self.get_height(node.right))

    def get_balance(self, node):
        if not node:
            return 0
        return self.get_height(node.left) - self.get_height(node.right)

    def right_rotate(self, y):
        x = y.left
        T2 = x.right
        x.right = y
        y.left = T2
        self.update_height(y)
        self.update_height(x)
        return x

    def left_rotate(self, x):
        y = x.right
        T2 = y.left
        y.left = x
        x.right = T2
        self.update_height(x)
        self.update_height(y)
        return y

    # def add_specific_memory(self, specific_memory,importance_score=0):
    #     if isinstance(specific_memory, SpecificMemory):
    #         if specific_memory not in self.specific_memories:
    #             self.specific_memories.append(specific_memory)
    #             specific_memory.update_parent_memory(self)
    #             self.update_embeddings(specific_memory)
    #             self.index_memories()
    #     elif isinstance(specific_memory, str):
    #         new_specific_memory = SpecificMemory(specific_memory, self, importance_score)
    #         self.specific_memories.append(new_specific_memory)
    #         self.update_embeddings(new_specific_memory)
    #         self.index_memories()

    def insert(self, node, key, memory):
        # Step 1: Perform the normal BST insertion
        if not node:
            return BSTNode(key, memory)
        elif key < node.key:
            node.left = self.insert(node.left, key, memory)
        else:
            node.right = self.insert(node.right, key, memory)

        # Step 2: Update the height of the ancestor node
        self.update_height(node)

        # Step 3: Get the balance factor to check whether this node became unbalanced
        balance = self.get_balance(node)

        # Step 4: If the node is unbalanced, then try out the 4 cases of rotations

        # Case 1 - Left Left
        if balance > 1 and key < node.left.key:
            return self.right_rotate(node)

        # Case 2 - Right Right
        if balance < -1 and key > node.right.key:
            return self.left_rotate(node)

        # Case 3 - Left Right
        if balance > 1 and key > node.left.key:
            node.left = self.left_rotate(node.left)
            return self.right_rotate(node)

        # Case 4 - Right Left
        if balance < -1 and key < node.right.key:
            node.right = self.right_rotate(node.right)
            return self.left_rotate(node)

        return node

    def _insert_into_bst(self, node, memory):
        if node is None:
            return BSTNode(memory)
        if memory.importance_score < node.memory.importance_score:
            node.left = self._insert_into_bst(node.left, memory)
        else:
            node.right = self._insert_into_bst(node.right, memory)
        return node  # Return the (possibly updated) node pointer

    def delete(self, node, key):
        # Step 1: Perform standard BST delete
        if not node:
            return node

        if key < node.key:
            node.left = self.delete(node.left, key)
        elif key > node.key:
            node.right = self.delete(node.right, key)
        else:
            # Node with only one child or no child
            if node.left is None:
                temp = node.right
                node = None
                return temp
            elif node.right is None:
                temp = node.left
                node = None
                return temp

            # Node with two children: Get the inorder successor (smallest in the right subtree)
            temp = self.minValueNode(node.right)

            # Copy the inorder successor's content to this node
            node.key = temp.key
            node.memory = temp.memory

            # Delete the inorder successor
            node.right = self.delete(node.right, temp.key)

        # If the tree had only one node then return
        if node is None:
            return node

        # Step 2: Update the height of the current node
        node.height = 1 + max(self.get_height(node.left), self.get_height(node.right))

        # Step 3: Get the balance factor
        balance = self.get_balance(node)

        # Step 4: Balance the tree
        # Left Left Case
        if balance > 1 and self.get_balance(node.left) >= 0:
            return self.right_rotate(node)

        # Left Right Case
        if balance > 1 and self.get_balance(node.left) < 0:
            node.left = self.left_rotate(node.left)
            return self.right_rotate(node)

        # Right Right Case
        if balance < -1 and self.get_balance(node.right) <= 0:
            return self.left_rotate(node)

        # Right Left Case
        if balance < -1 and self.get_balance(node.right) > 0:
            node.right = self.right_rotate(node.right)
            return self.left_rotate(node)

        return node

    def minValueNode(node):
        current = node
        while current.left is not None:
            current = current.left
        return current

    def add_tag(self, tag):
        self.tags.append(tag)

    def search_by_key(self, node, key):
        """
        Search for a memory in the BST based on the given key.

        :param node: The current node in the BST (start with the root).
        :param key: The key value to search for.
        :return: The memory object if found, or None if not found.
        """
        # Base case: The node is None or the key matches the node's key
        if node is None or key == getattr(node.memory, self.key_attr):
            return node.memory if node else None

        # If the key is smaller than the node's key, search in the left subtree
        if key < getattr(node.memory, self.key_attr):
            return self.search(node.left, key)
        # If the key is greater than the node's key, search in the right subtree
        return self.search_by_key(node.right, key)


# @track_calls
class GeneralMemory(Memory):
    def __init__(self, description, creation_time=datetime.now()):
        super().__init__(description, creation_time)
        self.description = description

        self.description_embedding = None
        self.description_embedding = self.generate_embedding()
        # print(self.description_embedding[0].shape)
        # print(self.description_embedding[1].shape)

        self.description_embedding = mean_pooling(
            self.description_embedding[0], self.description_embedding[1]
        )
        # self.specific_memories = []
        self.faiss_index = None
        self.timestamp_tree = None
        self.importance_tree = None
        self.key_tree = None
        self.init_trees()
        self.keywords = []
        self.keywords = manager.extract_keywords(self.description)
        self.analysis = None
        self.sentiment_score = None
        self.emotion_classification = None
        self.entities = []
        self.main_subject = None
        self.main_verb = None
        self.main_object = None
        self.temporal_expressions = None
        self.verb_aspects = None
        self.analyze_description()

    def analyze_description(self):
        self.analysis = manager.analyze_query_context(self.description)
        self.sentiment_score = self.analysis["sentiment_score"]
        self.emotion_classification = self.analysis["emotion_classification"]
        self.keywords = self.analysis["keywords"]
        self.entities = self.analysis["named_entities"]
        self.main_subject = self.analysis["main_subject"]
        self.main_verb = self.analysis["main_verb"]
        self.main_object = self.analysis["main_object"]
        self.temporal_expressions = self.analysis["temporal_expressions"]
        self.verb_aspects = self.analysis["verb_aspects"]

    def add_keyword(self, keyword):
        self.keywords.append(keyword)

    def get_keywords(self):
        return self.keywords

    def get_embedding(self):
        if self.description_embedding is None:
            self.description_embedding = self.generate_embedding()
        return self.description_embedding

    def generate_embedding(self):
        description = [self.description.strip()]
        input = model.tokenizer(
            description,
            padding=True,
            truncation=True,
            add_special_tokens=True,
            is_split_into_words=True,
            pad_to_multiple_of=8,
            return_tensors="pt",
        ).to(model.device)
        outputs = model.model(
            input["input_ids"],
            attention_mask=input["attention_mask"],
            output_hidden_states=True,
            return_dict=True,
            return_tensors="pt",
        )
        return [outputs.last_hidden_state, input["attention_mask"]]

    def init_specific_memories(self, specific_memories):
        # self.specific_memories = specific_memories
        for specific_memory in specific_memories:
            self.add_specific_memory(specific_memory)

    def init_trees(self):
        self.timestamp_tree = MemoryBST("last_access_time")
        self.importance_tree = MemoryBST("importance_score")
        self.key_tree = MemoryBST("keys")

    def get_specific_memories(self, key=None, attribute=None, tree=None):
        if attribute is not None:
            return self.traverse_specific_memories(self, None, attribute)
        if tree is None:
            tree = self.key_tree
        specific_memories = []
        self._inorder_traversal(
            tree.specific_memories_root,
            lambda node: specific_memories.append(node.memory),
        )
        # print(f"Specific memories: {[memory.description for memory in specific_memories]}")

        return specific_memories

    def find_specific_memory(self, key=None, tree=None, node=None):
        if tree is None:
            tree = self.key_tree
        if key is None and node is not None:
            key = id(node.memory)

        return self.search_by_key(tree.specific_memories_root, key)

    def search_by_key(self, node, key):
        """
        Search for a memory in the BST based on the given key.

        :param node: The current node in the BST (start with the root).
        :param key: The key value to search for.
        :return: The memory object if found, or None if not found.
        """

        # Base case: The node is None or the key matches the node's key
        if node is None or key == id(node.memory):
            return node.memory if node else None

        # If the key is smaller than the node's key, search in the left subtree
        if key < id(node.memory):
            return self.search_by_key(node.left, key)
        # If the key is greater than the node's key, search in the right subtree
        return self.search_by_key(node.right, key)

    def _inorder_traversal(self, node, visit):
        if node is not None:
            self._inorder_traversal(node.left, visit)
            visit(node)
            self._inorder_traversal(node.right, visit)

    def add_specific_memory(self, specific_memory, importance_score=0):
        # Key will be the instance id of the specific memory
        key = id(specific_memory)
        # Modified to insert into BST
        if not isinstance(specific_memory, SpecificMemory):
            specific_memory = SpecificMemory(specific_memory, self, importance_score)
        if (
            self.timestamp_tree is None
            or self.importance_tree is None
            or self.key_tree is None
        ):
            self.init_trees()
        if self.find_specific_memory(key, self.key_tree) is None:
            self.timestamp_tree.specific_memories_root = self.timestamp_tree.insert(
                self.timestamp_tree.specific_memories_root,
                specific_memory.last_access_time,
                specific_memory,
            )
            self.importance_tree.specific_memories_root = self.importance_tree.insert(
                self.importance_tree.specific_memories_root,
                specific_memory.importance_score,
                specific_memory,
            )
            self.key_tree.specific_memories_root = self.key_tree.insert(
                self.key_tree.specific_memories_root, key, specific_memory
            )

            self.keywords.extend(
                list(set(specific_memory.keywords).difference(set(self.keywords)))
            )

            # manager.update_embeddings(specific_memory)
            self.index_memories()
            manager.hierarchy.update_general_memory(self)
            manager.flat_access.faiss_index.add(
                specific_memory.embedding.cpu().detach().numpy()
            )
            manager.flat_access.index_id_to_node_id[
                manager.flat_access.faiss_index.ntotal - 1
            ] = specific_memory

    def index_memories(self):
        specific_memories = self.get_specific_memories()
        if not specific_memories:
            return None

        embeddings, att_mask = zip(
            *[memory.get_embedding() for memory in specific_memories]
        )
        # unpack embeddings and att_mask from tuple
        embeddings = embeddings[0]
        att_mask = att_mask[0]

        # embeddings = mean_pooling(embeddings, attention_mask=att_mask)
        # embeddings = torch.stack(embeddings)
        # attention_mask = torch.stack(att_mask)

        # print(embeddings)
        self.faiss_index = faiss.IndexFlatL2(embeddings.shape[1])
        # for embedding in embeddings:
        # print(embeddings[0].shape)
        self.faiss_index.add(embeddings.cpu().detach().numpy())
        return self.faiss_index


# @track_calls
class SpecificMemory(Memory):
    def __init__(self, description, parent_memory, importance_score, subject=None):
        super().__init__(description)
        self.subject = None
        if subject is not None:
            self.subject = subject
        self.description = description
        self.parent_memory = parent_memory
        if isinstance(parent_memory, str):
            self.parent_memory = self.get_parent_memory_from_string(parent_memory)
        self.related_memories = []
        self.embedding = None
        self.keywords = []
        self.tags = []
        self.importance_score = importance_score
        self.sentiment_score = None
        self.emotion_classification = None
        self.att_mask = None
        self.entities = []
        self.analysis = None
        self.main_subject = None
        self.main_verb = None
        self.main_object = None
        self.temporal_expressions = None
        self.verb_aspects = None

        # "features": features,
        #     "embedding": query_embedding,
        #     'sentiment_score': sentiment_score,
        #     'emotion_classification': emotion_classification,
        #     'ambiguity_score': ambiguity_score,
        #     'keywords': keywords,
        #     'text': query,
        #     'main_subject': main_subject,
        #     'main_verb': main_verb,
        #     'main_object': main_object,
        #     'named_entities': named_entities,
        #     'temporal_expressions': spans,
        #     'verb_aspects': verb_aspects,
        #     'complexity': complexity,
        #     'is_common_query': is_common_query,
        #     'is_urgent': is_urgent,
        #     'lexical_density': lexical_density,
        #     'type_token_ratio': type_token_ratio,
        #     'avg_word_length': avg_word_length,
        #     'lexical_diversity': lexical_diversity,
        #     'passive_voice': passive_voice,
        #     'active_voice': active_voice,
        #     'modals': modals,
        #     'determiners': determiners,
        #     'semantic_roles': semantic_roles,
        #     'dependencies': dependency_tree,
        #     'relationships': relationships,
        #     'proper_nouns': proper_nouns,
        #     'common_nouns': common_nouns,
        #     'verbs': verbs,
        #     'adpositions': adpositions,
        #     'adverbs': adverbs,
        #     'auxiliaries': auxiliaries,
        #     'conjunctions': conjunctions,
        #     'advanced_vocabulary': advanced_vocabulary,
        #     'simple_vocabulary': simple_vocabulary,
        #     'numbers': numbers,
        #     'symbols': symbols,
        #     'punctuations': punctuations,
        #     'particles': particles,
        #     'interjections': interjections,
        #     'prepositions': prepositions,
        #     'conjunctions': conjunctions,
        #     'pronouns': pronouns,
        #     'adjectives': adjectives,
        #     'adverbs': adverbs,
        #     'word_frequency': word_frequencies

        self.analyze_description()

    def analyze_description(self):
        self.analysis = manager.analyze_query_context(self.description)
        self.sentiment_score = self.analysis["sentiment_score"]
        self.emotion_classification = self.analysis["emotion_classification"]
        self.keywords = self.analysis["keywords"]
        self.entities = self.analysis["named_entities"]
        self.main_subject = self.analysis["main_subject"]
        self.main_verb = self.analysis["main_verb"]
        self.main_object = self.analysis["main_object"]
        self.temporal_expressions = self.analysis["temporal_expressions"]
        self.verb_aspects = self.analysis["verb_aspects"]

    def get_parent_memory_from_string(self, parent_memory):
        try:
            if parent_memory in [
                memory.description for memory in manager.hierarchy.general_memories
            ]:
                return [
                    memory
                    for memory in manager.hierarchy.general_memories
                    if memory.description == parent_memory
                ][0]
            else:
                return parent_memory
        except:
            return parent_memory

    def add_entity(self, entity):
        self.entities.append(entity)

    def add_tag(self, tag):
        self.tags.append(tag)

    def update_importance_score(self, importance_score):
        self.importance_score = importance_score

    def update_parent_memory(self, parent_memory):
        self.parent_memory = parent_memory

    def update_related_memories(self, related_memories):
        self.related_memories = related_memories

    def update_keywords(self, keywords):
        self.keywords = keywords

    def add_related_memory(self, related_memory):
        if related_memory not in self.related_memories:
            self.related_memories.append(related_memory)
            related_memory.add_related_memory(self)  # Ensuring bidirectional link

    def get_embedding(self):
        if self.embedding is None:
            self.embedding_and_mask = self.generate_embedding()
            self.embedding = mean_pooling(
                self.embedding_and_mask[0], self.embedding_and_mask[1]
            )
        return self.embedding, self.att_mask

    def generate_embedding(self):
        description = [self.description.strip()]
        input = model.tokenizer(
            description,
            padding=True,
            truncation=True,
            add_special_tokens=True,
            is_split_into_words=True,
            pad_to_multiple_of=8,
            return_tensors="pt",
        ).to(model.device)
        outputs = model.model(
            input["input_ids"],
            attention_mask=input["attention_mask"],
            output_hidden_states=True,
            return_dict=True,
            return_tensors="pt",
        )
        return [outputs.last_hidden_state, input["attention_mask"]]


# @track_calls
class MemoryHierarchy:
    def __init__(self):
        self.general_memories = []
        self.memory_graph = nx.MultiGraph()
        self.searcher = None

    # Key: SpecificMemory, Value: List of connected SpecificMemories
    # self.memory_connections = {}

    def add_general_memory(self, general_memory):
        # Check that the general memory is not already in the hierarchy
        if general_memory.description not in [
            memory.description for memory in self.general_memories
        ]:
            self.general_memories.append(general_memory)
            self.memory_graph.add_node(general_memory)
            general_memory.init_specific_memories(
                general_memory.get_specific_memories()
            )
            for specific_memory in general_memory.get_specific_memories():
                self.add_memory_node(specific_memory, general_memory)
            self.update_general_memory(general_memory)
            return self.general_memories[-1]
        else:
            # find the general memory and update it
            print("Memory already exists, updating it")
            for memory in self.general_memories:
                if memory.description == general_memory.description:
                    for specific_memory in general_memory.get_specific_memories():
                        if specific_memory.description not in [
                            memory.description
                            for memory in memory.get_specific_memories()
                        ]:
                            memory.add_specific_memory(specific_memory)
                        if not self.memory_graph.has_node(specific_memory):
                            self.add_memory_node(specific_memory, memory)
                            memory.index_memories()
                        elif self.memory_graph.has_node(specific_memory):
                            for edge in self.memory_graph.edges(
                                specific_memory, keys=True
                            ):
                                if (
                                    self.memory_graph.get_edge_data(*edge)["weight"]
                                    != 1
                                ):
                                    self.memory_graph.remove_edge(*edge)

                    return memory

    def update_general_memory(self, general_memory: GeneralMemory):
        for memory in self.general_memories:
            if memory.description == general_memory.description:
                for specific_memory in general_memory.get_specific_memories():
                    if specific_memory.description not in [
                        memory.description for memory in memory.get_specific_memories()
                    ]:
                        memory.add_specific_memory(specific_memory)
                    if not self.memory_graph.has_node(specific_memory):
                        self.add_memory_node(specific_memory, memory)
                    else:
                        # Get the specific memory node from the graph, remove it's current edges and add new ones
                        self.memory_graph.remove_node(specific_memory)
                        self.add_memory_node(specific_memory, memory)

    def add_memory_node(self, specific_memory, parent_memory: GeneralMemory):
        self.memory_graph.add_node(specific_memory)
        # parent_memory.add_specific_memory(specific_memory)
        self.add_memory_connection(
            parent_memory, specific_memory, relationship="related"
        )
        self.determine_related_memories(specific_memory)

    def add_memory_connection(self, memory1, memory2, **kwargs):
        if not self.memory_graph.has_edge(memory1, memory2):
            self.memory_graph.add_edge(memory1, memory2, **kwargs)
        else:
            # remove the edge and add a new one
            self.memory_graph.remove_edge(memory1, memory2)
            self.memory_graph.add_edge(memory1, memory2, **kwargs)

    def traverse_memories(self, memory, function):
        if function is None:
            return [
                related_memory for related_memory in self.memory_graph.neighbors(memory)
            ]
        return [
            related_memory
            for related_memory in self.memory_graph.neighbors(memory)
            if function(related_memory)
        ]

    # def initialize_scann_searcher(self, num_neighbors=10, distance_measure="dot_product", num_leaves=100, num_leaves_to_search=10, training_sample_size=0, dimensions_per_block=2, reordering_size=100):
    #     """
    #     Initializes a ScaNN searcher based on the vectors associated with the nodes in the graph.
    #     Assumes each node in the graph has a 'weight' attribute which is a numpy array.
    #     """
    #     vectors = np.array([data['weight'] for _, data in self.graph.nodes(data=True)])

    #     searcher = scann.ScannBuilder(vectors, num_neighbors, distance_measure).tree(
    #         num_leaves=num_leaves, num_leaves_to_search=num_leaves_to_search, training_sample_size=len(vectors) if training_sample_size is 0 else training_sample_size
    #     ).score_ah(dimensions_per_block=dimensions_per_block).reorder(reordering_size).build()

    #     return searcher

    # def find_similar_nodes(self, node_id, num_neighbors=5):
    #     """
    #     Finds nodes similar to the specified node using the ScaNN searcher.

    #     Parameters:
    #     - node_id: The ID of the node for which to find similar nodes.
    #     - num_neighbors: Number of similar nodes to find.

    #     Returns:
    #     A list of tuples (neighbor_node_id, similarity_score), sorted by similarity score in descending order.
    #     """
    #     if node_id not in self.graph:
    #         print(f"Node {node_id} not found in the graph.")
    #         return []

    #     query_vector = self.graph.nodes[node_id]['weight']
    #     neighbors, similarity_scores = self.searcher.search(query_vector, final_num_neighbors=num_neighbors)

    #     # Map neighbors' indices back to node IDs
    #     node_ids = list(self.graph.nodes)
    #     similar_nodes = [(node_ids[neighbor], score) for neighbor, score in zip(neighbors, similarity_scores)]

    #     return similar_nodes

    def traverse_memories_by_edge_attribute_dfs(
        self,
        memory,
        attribute,
        value,
        max_value=None,
        min_value=None,
        max_depth=3,
        current_depth=0,
        visited_nodes=[],
        visited_edges=[],
        mem_type=None,
    ):
        """
        Traverse the memory graph using depth-first search (DFS) based on the given edge attribute and value.

        Args:
            memory: The starting memory node for the traversal.
            attribute: The edge attribute to filter by.
            value: The value of the edge attribute to match.
            max_value: The maximum value for the edge attribute (optional).
            min_value: The minimum value for the edge attribute (optional).
            max_depth: The maximum depth to traverse (optional).
            current_depth: The current depth level in the traversal (internal use).
            visited_nodes: A list of visited memory nodes (internal use).
            visited_edges: A list of visited edges (internal use).
            mem_type: The type of memory to filter by (optional).

        Returns:
            A list of memory nodes that match the given edge attribute and value.

        Example Usage:
            traverse_memories_by_edge_attribute_dfs(memory, "relationship", "related", max_depth=3)
            or
            traverse_memories_by_edge_attribute_dfs(specific_memory, "relationship", "related", max_depth=3, mem_type=SpecificMemory)

        Use Case for finding similiar nodes in context of SpecificMemory instances as nodes:
            This function can be used to find related nodes in the memory graph based on specific edge attributes and values.
            For example, to find all related memories to a specific memory, or to find memories with a specific relationship type.
            Depth-first search (DFS) means that the function will traverse the graph by following a path as far as possible before backtracking.
            It uses a stack data structure to remember to get the next vertex to start a search when a dead end occurs in any iteration.

        Advantages:
            DFS may be useful in finding connected components or paths in a graph.
            It uses less memory than BFS.
        Disadvantages:
            DFS isnt the best solution if youre trying to find the shortest path between two nodes.
            It can get trapped in a cycle in the graph if its not acyclic.

        """
        memories_values = {}
        visited_nodes = [memory]
        for current_depth, (parent, current_memory, edge_data) in enumerate(
            nx.dfs_edges(self.memory_graph, source=memory, depth_limit=max_depth), 1
        ):
            if current_memory not in visited_nodes:
                if mem_type is not None and not isinstance(current_memory, mem_type):
                    continue
                visited_nodes.append(current_memory)
                edge_data = self.memory_graph.get_edge_data(parent, current_memory)
                if edge_data and edge_data.get(attribute) == value:
                    memories_values[current_memory] = edge_data[attribute]
        return memories_values

    def traverse_memories_by_edge_attribute_bfs(
        self,
        memory,
        attribute,
        value,
        max_value=None,
        min_value=None,
        max_depth=1,
        current_depth=0,
        visited_nodes=[],
        visited_edges=[],
        mem_type=None,
    ):
        """
        Traverse the memory graph using breadth-first search (BFS) based on the given edge attribute and value.

        Args:
            memory: The starting memory node for the traversal.
            attribute: The edge attribute to filter by.
            value: The value of the edge attribute to match.
            max_value: The maximum value for the edge attribute (optional).
            min_value: The minimum value for the edge attribute (optional).
            max_depth: The maximum depth to traverse (optional).
            current_depth: The current depth level in the traversal (internal use).
            visited_nodes: A list of visited memory nodes (internal use).
            visited_edges: A list of visited edges (internal use).
            mem_type: The type of memory to filter by (optional).

        Returns:
            A list of memory nodes that match the given edge attribute and value.

        Example Usage:
            traverse_memories_by_edge_attribute_bfs(memory, "relationship", "related", max_depth=3)
            or
            traverse_memories_by_edge_attribute_bfs(specific_memory, "relationship", "related", max_depth=3, mem_type=SpecificMemory)

        Use Case for finding similiar nodes in context of SpecificMemory instances as nodes:
            This function can be used to find related nodes in the memory graph based on specific edge attributes and values.
            For example, to find all related memories to a specific memory, or to find memories with a specific relationship type.
            BFS is a graph traversal method that explores all the vertices of a graph in breadth-first order, i.e., it explores all the vertices at the present depth before going to the next level of depth. It uses a queue data structure.
        Advantages:
            BFS can find the shortest path between two nodes in an unweighted graph.
            It can be used to test bipartiteness of a graph.
        Disadvantages:
            BFS uses more memory than DFS.
            It can be slower than DFS for some types of problems, such as finding connected components.


        """

        memories_values = {}
        visited_nodes = [memory]
        for current_depth, (parent, current_memory, edge_data) in enumerate(
            nx.bfs_edges(self.memory_graph, source=memory, depth_limit=max_depth), 1
        ):
            if current_memory not in visited_nodes:
                if mem_type is not None and not isinstance(current_memory, mem_type):
                    continue
                visited_nodes.append(current_memory)
                if edge_data and edge_data.get(attribute) == value:
                    memories_values[current_memory] = edge_data[attribute]
        return memories_values

    def query_relevant_nodes(
        self,
        query_node,
        graph=None,
        simrank_threshold=0.1,
        top_k_pagerank=100,
        top_k_voterank=50,
    ):
        """
        Queries the relevant nodes in the memory graph based on the given query node.

        Group of Functions (Incorporating SimRank, PageRank, and Custom VoteRank):
        Combination of Algorithms:
          This approach combines three distinct methodologies (SimRank for similarity, PageRank for global importance, and a custom VoteRank for local influence) to analyze nodes from different perspectives.
        Sequential Processing:
          The methodology involves sequential processing where nodes are first filtered by similarity (SimRank), then by global importance (PageRank), and finally by local influence (using a VoteRank-like approach).
          This sequential refinement helps in narrowing down the most relevant and influential nodes related to a specific query or node.
        Comprehensive Ranking:
          The final ranking is achieved by merging and normalizing scores from all three algorithms, offering a balanced consideration of similarity, global importance, and local influence.
        Use Case:
          This approach is designed for complex scenarios where multiple factors (similarity to a query node, global importance, and local influence) are crucial in determining the relevance and ranking of nodes.
          It's particularly useful when the objective is to identify nodes that are not just globally important but also closely related to the query node and influential within their local context.

        Args:
            query_node: The node to query for relevant nodes.
            graph: The memory graph to search in. If None, the default memory graph is used.
            simrank_threshold: The similarity threshold for SimRank. Nodes with a similarity score above this threshold will be considered similar to the query node.
            top_k_pagerank: The number of top globally important nodes to select using PageRank.
            top_k_voterank: The number of most influential nodes to select using VoteRank.

        Returns:
            A dictionary containing the final ranked nodes, where the keys are the node names and the values are their corresponding ranking scores.
        """
        if graph is None:
            graph = self.memory_graph

        # Step 1: Use SimRank to find nodes similar to the query node
        simrank_scores = nx.simrank_similarity(graph, source=query_node)
        similar_nodes = {
            node: score
            for node, score in simrank_scores.items()
            if score >= simrank_threshold
        }

        # Step 2: Use PageRank to find globally important nodes, filtered by similarity
        pagerank_scores = nx.pagerank(graph)
        important_similar_nodes = {
            node: pagerank_scores[node]
            for node in similar_nodes
            if node in pagerank_scores
        }
        # Sort by PageRank score and select the top_k
        top_important_nodes = dict(
            sorted(
                important_similar_nodes.items(), key=lambda item: item[1], reverse=True
            )[:top_k_pagerank]
        )

        # Step 3: Use VoteRank to identify the most influential nodes, further filtered by the top PageRank nodes
        # Note: VoteRank implementation might need adjustment or a custom approach if not directly available in NetworkX
        # voterank_scores = self.custom_voterank(top_important_nodes.keys(), top_k=top_k_voterank)

        voterank_scores = nx.voterank(
            self.memory_graph.subgraph(top_important_nodes.keys()), top_k=top_k_voterank
        )

        # Combine results, potentially weighting them according to specific needs
        final_ranked_nodes = self.merge_and_rank_results(
            similar_nodes, top_important_nodes, voterank_scores
        )

        return final_ranked_nodes

    def custom_voterank(self, nodes, graph=None, top_k=50):
        """
        Computes the VoteRank scores for the given nodes in the memory graph.

        Args:
            nodes: The nodes to compute VoteRank scores for.
            graph: The memory graph to use. If None, the default memory graph is used.
            top_k: The number of top influential nodes to select.

        Returns:
            A dictionary containing the VoteRank scores for the nodes.
        """
        if graph is None:
            graph = self.memory_graph
        # Use degree centrality within NetworkX as a proxy for influence.
        centrality_scores = nx.degree_centrality(graph)
        # Filter scores for the subset of nodes and sort them
        filtered_scores = {node: centrality_scores[node] for node in nodes}
        sorted_scores = dict(
            sorted(filtered_scores.items(), key=lambda item: item[1], reverse=True)[
                :top_k
            ]
        )
        return sorted_scores

    def merge_and_rank_results(self, simrank_scores, pagerank_scores, voterank_scores):
        # Normalize scores (convert each set of scores to a 0-1 scale)
        simrank_norm = self.normalize_scores(simrank_scores)
        pagerank_norm = self.normalize_scores(pagerank_scores)
        voterank_norm = self.normalize_scores(voterank_scores)

        # Combine scores by averaging
        combined_scores = {}
        for node in set(simrank_norm) | set(pagerank_norm) | set(voterank_norm):
            scores = [
                simrank_norm.get(node, 0),
                pagerank_norm.get(node, 0),
                voterank_norm.get(node, 0),
            ]
            combined_scores[node] = sum(scores) / len(scores)

        # Sort by combined score
        final_ranked_nodes = dict(
            sorted(combined_scores.items(), key=lambda item: item[1], reverse=True)
        )
        return final_ranked_nodes

    def normalize_scores(self, scores):
        if not scores:
            return {}
        max_score = max(scores.values())
        min_score = min(scores.values())
        range_score = max_score - min_score
        return {
            node: (score - min_score) / range_score if range_score else 0
            for node, score in scores.items()
        }

    def traverse_memories_by_relevance(
        self,
        start_memory,
        attribute,
        target_value=None,
        max_value=None,
        min_value=None,
        max_depth=3,
        current_depth=0,
        visited_nodes=None,
        mem_type=None,
    ):
        """
        Traverses the memories in the memory graph based on relevance to the start memory and returns a dictionary of memories and their corresponding attribute values.

        traverse_memories_by_relevance Function:
        Single Algorithm Focus:
          This function primarily focuses on traversing the memory graph based on the relevance of nodes to a specific start memory, using a specified attribute.
        Flexible Filtering:
            It allows for flexible filtering based on attribute values, including target_value, max_value, and min_value, to narrow down the relevant nodes.
        Use Case:
            The primary use case for this function is to identify and retrieve memories that are relevant to a specific start memory based on a given attribute, with the ability to filter the results based on specific attribute values.
        Greedy Best-First Search:
            Use Case:
                This algorithm chooses which path to explore next based solely on a heuristic that estimates the closeness to the goal, without considering the path cost so far.
                It's less efficient than A* but can be faster in finding a relevant result if the heuristic is well-designed.
            Application:
                Suitable for scenarios where you need to quickly suggest a single or a few highly relevant memories based on specific attributes, rather than finding the most optimal path.

        Args:
            start_memory: The memory to start the traversal from.
            attribute: The attribute to filter memories based on.
            target_value: The target value to filter memories based on the attribute.
            max_value: The maximum value to filter memories based on the attribute.
            min_value: The minimum value to filter memories based on the attribute.
            max_depth: The maximum depth for the traversal.
            current_depth: The current depth of the traversal (used for recursion).
            visited_nodes: A set of visited nodes to avoid revisiting the same nodes (used for recursion).
            mem_type: The type of memories to consider. Only memories of this type will be considered.

        Returns:
            A dictionary of memories and their corresponding attribute values.


        """
        if visited_nodes is None:
            visited_nodes = set()

        # Priority queue: (negative attribute value, current depth, memory node)
        # Negative attribute value used because heapq is a min-heap, but we want to prioritize high values
        pq = [(-float("inf"), 0, start_memory)]
        memories_values = {}

        while pq:
            attr_value, depth, memory = heapq.heappop(pq)
            attr_value = -attr_value  # Convert back to positive

            if memory in visited_nodes or depth > max_depth:
                continue

            visited_nodes.add(memory)
            for related_memory in self.memory_graph.neighbors(memory):
                if related_memory not in visited_nodes:
                    if mem_type is not None and not isinstance(
                        related_memory, mem_type
                    ):
                        continue

                    edge_data = self.memory_graph.get_edge_data(memory, related_memory)

                    # Ensure the edge has the specified attribute
                    if attribute in edge_data:
                        edge_value = edge_data[attribute]
                        # Apply filters based on max_value, min_value, and optionally target_value
                        if (
                            (max_value is None or edge_value <= max_value)
                            and (min_value is None or edge_value >= min_value)
                            and (target_value is None or edge_value == target_value)
                        ):

                            # Add to the priority queue with priority based on edge attribute value
                            heapq.heappush(pq, (-edge_value, depth + 1, related_memory))

                            # Store or update the memory's value if it meets the criteria
                            memories_values[related_memory] = edge_value

        return memories_values

    def traverse_memories_with_pagerank(
        self,
        start_memory=None,
        attribute=None,
        target_value=None,
        max_value=None,
        min_value=None,
        personalization=None,
        mem_type=None,
    ):
        """
        Traverses the memories in the memory graph using the PageRank algorithm and returns a dictionary of memories and their corresponding PageRank scores.

        traverse_memories_with_pagerank Function:
        Single Algorithm Focus:
          This function primarily utilizes the PageRank algorithm to rank nodes in the graph based on their global importance, as determined by the structure of the graph itself.
        Specific Filtering:
          It allows for filtering nodes based on a given attribute and value ranges (target_value, max_value, min_value). The filtering is applied after the PageRank scores are calculated.
        Personalization Option:
          The function includes an option for personalization, which can adjust the PageRank scores based on predefined preferences or relevances for specific nodes.
        Use Case:
          The primary use case for this function is to identify and retrieve nodes that are globally significant within the entire graph, potentially adjusted by personalized relevances or
          specific attribute filters. It's best suited for applications where the overall network structure significantly influences node relevance.

        Use Case for finding similiar nodes in context of SpecificMemory instances as nodes:
        Traversal with PageRank:
            This function is designed to traverse the memory graph using the PageRank algorithm, providing a global importance score for each memory node.
        Flexible Filtering:
            The function allows for flexible filtering based on attributes and value ranges, enabling the selection of nodes based on specific criteria.
        Personalization:
            The function includes an option for personalization, allowing for the prioritization of nodes based on predefined preferences or relevances.
        Use Case:
            This function is particularly useful in scenarios where the overall importance of nodes in the graph is a key consideration, and the ability to filter nodes based on specific attributes or personalization is desired.

        Args:
            start_memory (optional): The specific memory to start the traversal from. If provided, only memories connected to the start memory will be considered. Defaults to None.
            attribute (optional): The attribute to filter memories based on. Only memories with this attribute will be considered. Defaults to None.
            target_value (optional): The target value to filter memories based on the attribute. Only memories with the attribute equal to the target value will be considered. Defaults to None.
            max_value (optional): The maximum value to filter memories based on the attribute. Only memories with the attribute less than or equal to the max value will be considered. Defaults to None.
            min_value (optional): The minimum value to filter memories based on the attribute. Only memories with the attribute greater than or equal to the min value will be considered. Defaults to None.
            personalization (optional): A dictionary of personalization values for the PageRank algorithm. If provided, it will prioritize nodes based on the personalization dict values. Defaults to None.
            mem_type (optional): The type of memories to consider. Only memories of this type will be considered. Defaults to None.

        Returns:
            dict: A dictionary of memories and their corresponding PageRank scores.

        """
        # Initialize PageRank scores for the entire memory graph
        # If personalization is provided, it will prioritize nodes based on the personalization dict values
        pagerank_scores = nx.pagerank(
            self.memory_graph, personalization=personalization
        )

        memories_values = {}

        for memory, score in pagerank_scores.items():
            if mem_type is not None and not isinstance(memory, mem_type):
                continue

            # If a specific start memory is defined, skip unrelated memories
            if start_memory is not None and memory != start_memory:
                continue

            edge_data = self.memory_graph.nodes[memory].get(attribute, None)
            if edge_data is not None:
                # Apply filters based on target_value, max_value, and min_value
                if (
                    (target_value is None or edge_data == target_value)
                    and (max_value is None or edge_data <= max_value)
                    and (min_value is None or edge_data >= min_value)
                ):
                    memories_values[memory] = score

        return memories_values

    def find_nodes_by_edge_weight(self, node_id, graph=None, min_weight=10):
        """
        Find and rank nodes connected to the specified node based on edge weights.

        Parameters:
        - graph: A NetworkX graph object
        - node_id: Identifier for the node of interest
        - min_weight: Minimum edge weight to consider a node as similar

        Returns:
        A dictionary where the keys are the neighbor nodes and the values are the weights, sorted by weight in descending order.

        Use Case for finding similiar nodes in context of SpecificMemory instances as nodes:
        Customized Similarity Calculation:
          This function is designed to calculate the similarity between SpecificMemory instances in a graph, based on the edge weights between nodes.
        """
        if graph is None:
            graph = self.memory_graph
        if node_id not in graph:
            print(f"Node {node_id} not found in the graph.")
            return {}

        # Use adjacency iterator for efficient traversal
        similar_nodes = {
            neighbor: attrs["weight"]
            for neighbor, attrs in graph.adj[node_id].items()
            if attrs.get("weight", 0) >= min_weight
        }

        # Sort the dictionary by the weights in descending order
        similar_nodes = dict(
            sorted(similar_nodes.items(), key=lambda item: item[1], reverse=True)
        )

        return similar_nodes

    def find_nodes_by_similarity(
        self, node_id, graph=None, similarity_metric="similarity", min_similarity=0.5
    ):
        """
        Find and rank nodes similar to the specified node based on a given similarity metric.

        Parameters:
        - graph: NetworkX graph object
        - node_id: Identifier for the node of interest
        - similarity_metric: String representing the edge attribute to use for similarity
        - min_similarity: Minimum similarity score to consider a node as similar

        Returns:
        A list of tuples (neighbor_node_id, similarity_score), sorted by similarity score in descending order.


        Use Case for finding similiar nodes in context of SpecificMemory instances as nodes:
        Customized Similarity Calculation:
          This function is designed to calculate the similarity between SpecificMemory instances in a graph, based on a given edge attribute representing the similarity metric.
        Flexible Similarity Metric:
            The function allows for flexibility in the choice of similarity metric, enabling the comparison of nodes based on different attributes or measures of similarity.
        Minimum Similarity Threshold:
            The function includes a minimum similarity threshold, ensuring that only nodes with a similarity score above the threshold are considered as similar.
        Use Case:
            This function is particularly useful in scenarios where the relationships between nodes in a graph are influenced by various similarity metrics,
            and a flexible and customizable similarity measure is desired.

        """
        similar_nodes = []
        if graph is None:
            graph = self.memory_graph
        # Check if the node exists in the graph
        if node_id not in graph:
            print(f"Node {node_id} not found in the graph.")
            return similar_nodes

        # Iterate over all neighbors of the given node and their edge attributes
        for neighbor, edges in graph[node_id].items():
            for edge_key, edge_data in edges.items():
                # Retrieve the similarity score from the edge attributes
                similarity_score = edge_data.get(similarity_metric, 0)

                # If the score meets or exceeds the minimum similarity, add it to the list
                if similarity_score >= min_similarity:
                    similar_nodes.append((neighbor, similarity_score))

        # Sort the similar nodes by their similarity score in descending order
        similar_nodes.sort(key=lambda x: x[1], reverse=True)

        return similar_nodes

    def custom_simrank_with_weights(self, G=None, importance_decay=0.8, iterations=10):
        """
        Compute the SimRank similarity scores for nodes in the graph, incorporating edge weights into the similarity calculation.

        Parameters:
        - G: NetworkX graph object
        - importance_decay: Weight decay factor for incorporating edge weights
        - iterations: Number of iterations for the similarity computation

        Returns:
        A dictionary of similarity scores for node pairs, where the keys are tuples of node identifiers and the values are the similarity scores.

        Use Case for Custom SimRank with Weights in context of SpecificMemory instances as nodes:
        Customized Similarity Calculation:
          This function is designed to calculate the similarity between SpecificMemory instances in a graph, incorporating edge weights into the similarity computation.
        Edge Weight Incorporation:
            The function utilizes edge weights to influence the similarity calculation, allowing for a more nuanced and context-aware similarity measure.
        Iterative Computation:
            The similarity scores are computed iteratively over multiple iterations, allowing for a more refined and stable similarity measure.
        Use Case:
            This function is particularly useful in scenarios where the relationships between nodes in a graph are influenced by edge weights, and a more nuanced similarity measure is desired.

        """
        # Initialize similarity scores: set initial similarity to 1 for same nodes, 0 otherwise
        if G is None:
            G = self.memory_graph
        sim_scores = {
            (u, v): 1.0 if u == v else 0.0 for u in G.nodes() for v in G.nodes()
        }

        # Temporary storage for updating similarity scores during iterations
        temp_scores = sim_scores.copy()

        # Iterate to compute updated similarity scores
        for _ in range(iterations):
            for u in G.nodes():
                for v in G.nodes():
                    if u != v:
                        # Get predecessors (or neighbors in the case of undirected graphs)
                        u_neighbors = set(G.neighbors(u))
                        v_neighbors = set(G.neighbors(v))

                        # Calculate weighted sum of similarities of neighbors
                        sum_sim = 0.0
                        for n_u in u_neighbors:
                            for n_v in v_neighbors:
                                edge_weight_u = G[u][n_u].get(
                                    "weight", 1
                                )  # Default weight is 1 if not specified
                                edge_weight_v = G[v][n_v].get(
                                    "weight", 1
                                )  # Default weight is 1 if not specified
                                weight_avg = (edge_weight_u + edge_weight_v) / 2.0

                                # Incorporate the edge weights into the similarity calculation
                                sum_sim += weight_avg * sim_scores[(n_u, n_v)]

                        # Update the similarity score with normalization by the count of neighbors
                        denom = len(u_neighbors) * len(v_neighbors)
                        temp_scores[(u, v)] = (
                            (importance_decay * sum_sim / denom) if denom != 0 else 0
                        )

            # Update the similarity scores for the next iteration
            sim_scores = temp_scores.copy()

        return sim_scores

    def get_related_memories(self, memory: SpecificMemory):
        for general_memory in self.general_memories:
            for specific_memory in general_memory.get_specific_memories():
                if specific_memory == memory or not isinstance(
                    specific_memory, SpecificMemory
                ):
                    continue
                keywords_in_common = list(
                    set(memory.keywords).intersection(set(specific_memory.keywords))
                )
                if keywords_in_common:
                    self.add_memory_connection(
                        memory,
                        specific_memory,
                        keywords_in_common=len(keywords_in_common),
                    )
                # tags_in_common = list(set(memory.tags).intersection(set(specific_memory.tags))) if memory.tags is not None and specific_memory.tags is not None else []
                # if tags_in_common:
                #     self.add_memory_connection(memory, specific_memory, tags_in_common = tags_in_common)
                time_relations = 0
                time_diff = (
                    (specific_memory.last_access_time - memory.last_access_time)
                    if specific_memory.last_access_time - memory.last_access_time
                    > timedelta(hours=1)
                    else (
                        (specific_memory.creation_time - memory.creation_time)
                        if specific_memory.creation_time - memory.creation_time
                        > timedelta(hours=1)
                        else (
                            (memory.last_access_time - specific_memory.last_access_time)
                            if memory.last_access_time
                            - specific_memory.last_access_time
                            > timedelta(hours=1)
                            else (
                                (memory.creation_time - specific_memory.creation_time)
                                if memory.creation_time - specific_memory.creation_time
                                > timedelta(hours=1)
                                else 0
                            )
                        )
                    )
                )
                if (
                    specific_memory.last_access_time - memory.last_access_time
                    > timedelta(hours=1)
                    or specific_memory.creation_time - memory.creation_time
                    > timedelta(hours=1)
                ):
                    self.add_memory_connection(
                        memory, specific_memory, time_diff=time_diff
                    )
                elif (
                    memory.last_access_time - specific_memory.last_access_time
                    > timedelta(hours=1)
                    or memory.creation_time - specific_memory.creation_time
                    > timedelta(hours=1)
                ):
                    self.add_memory_connection(
                        memory,
                        specific_memory,
                        time_diff=memory.last_access_time
                        - specific_memory.last_access_time,
                    )
                sentiment_relations = [
                    memory.sentiment_score,
                    specific_memory.sentiment_score,
                ]
                subjectivity_diff = (
                    specific_memory.sentiment_score["subjectivity"]
                    - memory.sentiment_score["subjectivity"]
                )
                polarity_diff = (
                    specific_memory.sentiment_score["polarity"]
                    - memory.sentiment_score["polarity"]
                )
                if (
                    specific_memory.sentiment_score["polarity"]
                    - memory.sentiment_score["polarity"]
                    > 0.1
                    or specific_memory.sentiment_score["subjectivity"]
                    - memory.sentiment_score["subjectivity"]
                    > 0.1
                ):
                    self.add_memory_connection(
                        memory,
                        specific_memory,
                        polarity_diff=polarity_diff,
                        subjectivity_diff=subjectivity_diff,
                    )
                elif (
                    memory.sentiment_score["polarity"]
                    - specific_memory.sentiment_score["polarity"]
                    > 0.1
                    or memory.sentiment_score["subjectivity"]
                    - specific_memory.sentiment_score["subjectivity"]
                    > 0.1
                ):
                    self.add_memory_connection(
                        memory,
                        specific_memory,
                        polarity_diff=memory.sentiment_score["polarity"]
                        - specific_memory.sentiment_score["polarity"],
                        subjectivity_diff=memory.sentiment_score["subjectivity"]
                        - specific_memory.sentiment_score["subjectivity"],
                    )
                if (
                    specific_memory.emotion_classification
                    == memory.emotion_classification
                ):
                    self.add_memory_connection(
                        memory, specific_memory, emotion=memory.emotion_classification
                    )
                entity_relations = list(
                    set(memory.entities)
                    .union(
                        set(memory.analysis["main_subject"])
                        .union(set(memory.analysis["proper_nouns"]))
                        .union(set(memory.analysis["pronouns"]))
                    )
                    .intersection(
                        set(specific_memory.entities).union(
                            set(specific_memory.analysis["main_subject"])
                        )
                    )
                )
                len_entity_relations = len(entity_relations)
                if entity_relations:
                    self.add_memory_connection(
                        memory, specific_memory, entities=len_entity_relations
                    )
                importance_relations = [
                    (memory.importance_score / 10),
                    (specific_memory.importance_score / 10),
                ]
                importance_diff = abs(
                    specific_memory.importance_score - memory.importance_score
                )
                if specific_memory.importance_score == memory.importance_score:
                    self.add_memory_connection(
                        memory, specific_memory, importance_diff=importance_diff
                    )
                # for entity in list(set(memory.entities).union(set(memory.analysis['keywords']).union(set(memory.analysis['advanced_vocabulary'])).union(set(memory.analysis['proper_nouns'])).union(set(memory.analysis['pronouns'])))):
                #     if entity in specific_memory.description:
                #         self.add_memory_connection(memory, specific_memory, entities = entity)
                cos_sim = cosine_similarity(
                    memory.get_embedding()[0].cpu().detach().numpy(),
                    specific_memory.get_embedding()[0].cpu().detach().numpy(),
                )
                if cos_sim > 0:
                    self.add_memory_connection(
                        memory, specific_memory, similarity=cos_sim
                    )

                final_relations = [
                    keywords_in_common,
                    time_relations,
                    sentiment_relations,
                    entity_relations,
                    importance_relations,
                    cos_sim,
                ]
                score = 0
                score = (
                    len(keywords_in_common)
                    / len(
                        list(set(memory.keywords).union(set(specific_memory.keywords)))
                    )
                ) * 2
                # Assuming time_relations is in hours
                time_relations_timedelta = timedelta(hours=time_relations)
                score += 1 - (
                    time_relations_timedelta.total_seconds()
                    / timedelta(hours=1).total_seconds()
                )
                score += 1 - (
                    abs(
                        sentiment_relations[0]["polarity"]
                        - sentiment_relations[1]["polarity"]
                    )
                    / 1
                )
                score += 1 - (
                    abs(
                        sentiment_relations[0]["subjectivity"]
                        - sentiment_relations[1]["subjectivity"]
                    )
                    / 1
                )
                score += len(entity_relations)
                score += 1 - (
                    abs(importance_relations[0] - importance_relations[1]) / 1
                )
                score += cos_sim * 4
                score = score / 12
                score = score * 100
                self.add_memory_connection(memory, specific_memory, weight=score)

        return self.memory_graph.neighbors(memory)

    def get_related_memories_by_type(self, memory, memory_type):
        return [
            related_memory
            for related_memory in self.memory_graph.neighbors(memory)
            if isinstance(related_memory, memory_type)
        ]

    def get_related_memories_by_entity(self, memory, entity):
        return [
            related_memory
            for related_memory in self.memory_graph.neighbors(memory)
            if related_memory == entity
        ]

    def get_related_memories_by_attribute(self, memory, attribute):
        return [
            related_memory
            for related_memory in self.memory_graph.neighbors(memory)
            if related_memory.attribute == attribute
        ]

    def get_related_memories_by_attribute_value(self, memory, attribute, value):
        return [
            related_memory
            for related_memory in self.memory_graph.neighbors(memory)
            if getattr(related_memory, attribute) == value
        ]

    def get_related_memories_by_function(self, memory, function):
        return [
            related_memory
            for related_memory in self.memory_graph.neighbors(memory)
            if function(related_memory)
        ]

    def get_related_memories_by_keywords(self, memory, keywords):
        return [
            related_memory
            for related_memory in self.memory_graph.neighbors(memory)
            if any(keyword in related_memory.keywords for keyword in keywords)
        ]

    def get_related_memories_by_similarity(self, memory, query_embedding, threshold):
        return [
            related_memory
            for related_memory in self.memory_graph.neighbors(memory)
            if cosine_similarity(related_memory.embedding, query_embedding) > threshold
        ]

    def get_related_memories_by_sentiment(
        self, memory, min_polarity, max_polarity, min_subjectivity, max_subjectivity
    ):
        return [
            related_memory
            for related_memory in self.memory_graph.neighbors(memory)
            if min_polarity
            <= related_memory.sentiment_score["polarity"]
            <= max_polarity
            and min_subjectivity
            <= related_memory.sentiment_score["subjectivity"]
            <= max_subjectivity
        ]

    def get_related_memories_by_emotion(self, memory, emotion):
        return [
            related_memory
            for related_memory in self.memory_graph.neighbors(memory)
            if related_memory.emotion_classification == emotion
        ]

    def get_related_memories_by_time(self, memory, time_period):
        return [
            related_memory
            for related_memory in self.memory_graph.neighbors(memory)
            if related_memory.last_access_time > time_period
        ]

    def get_related_memories_by_importance(
        self, memory, min_importance, max_importance
    ):
        return [
            related_memory
            for related_memory in self.memory_graph.neighbors(memory)
            if min_importance <= related_memory.importance_score <= max_importance
        ]

    def get_related_memories_by_tags(self, memory, tags):
        return [
            related_memory
            for related_memory in self.memory_graph.neighbors(memory)
            if any(tag in related_memory.tags for tag in tags)
        ]

    def get_related_memories_by_complex_query(self, memory, attribute, query):
        return [
            related_memory
            for related_memory in self.memory_graph.neighbors(memory)
            if "*attribute*" in query
            and query.replace("*attribute*", attribute) in related_memory.description
        ]

    def determine_related_memories(self, memory, query=None):
        if isinstance(query, list):
            return self.get_related_memories_by_keywords(memory, query)
        if isinstance(query, str):
            return self.get_related_memories_by_tags(memory, query)
        if isinstance(query, dict):
            attribute = query["attribute"]
            query = query["query"]
            return self.get_related_memories_by_complex_query(memory, attribute, query)
        if isinstance(query, tuple):
            if query[0] == "time":
                return self.get_related_memories_by_time(memory, query[1])
            if query[0] == "importance":
                return self.get_related_memories_by_importance(
                    memory, query[1], query[2]
                )
            if query[0] == "sentiment":
                return self.get_related_memories_by_sentiment(
                    memory, query[1], query[2], query[3], query[4]
                )
            if query[0] == "emotion":
                return self.get_related_memories_by_emotion(memory, query[1])
            if query[0] == "keywords":
                return self.get_related_memories_by_keywords(memory, query[1])
            if query[0] == "similarity":
                return self.get_related_memories_by_similarity(
                    memory, query[1], query[2]
                )
            if query[0] == "tags":
                return self.get_related_memories_by_tags(memory, query[1])
            if query[0] == "function":
                return self.get_related_memories_by_function(memory, query[1])
            if query[0] == "attribute":
                return self.get_related_memories_by_attribute(memory, query[1])
            if query[0] == "attribute_value":
                return self.get_related_memories_by_attribute_value(
                    memory, query[1], query[2]
                )
            if query[0] == "type":
                return self.get_related_memories_by_type(memory, query[1])
            return self.get_related_memories(memory)
        if callable(query):
            return self.get_related_memories_by_function(memory, query)
        if isinstance(query, MemoryQuery):
            keyword_relations = self.get_related_memories_by_keywords(
                memory, query.analysis["keywords"]
            )
            keyword_relations.extend(query.analysis["advanced_vocabulary"])
            keyword_relations = list(set(keyword_relations))

            tag_relations = self.get_related_memories_by_tags(memory, query.query_tags)
            for relation in tag_relations:
                self.add_memory_connection(memory, relation, tags=query.query_tags)
            for relation in keyword_relations:
                self.add_memory_connection(
                    memory, relation, keywords=query.analysis["keywords"]
                )
            time_relations = self.get_related_memories_by_time(memory, query.query_time)
            for relation in time_relations:
                self.add_memory_connection(
                    memory, relation, last_access_time=query.query_time
                )

            sentiment_relations = self.get_related_memories_by_sentiment(
                memory,
                query.analysis["polarity"][0],
                query.analysis["polarity"][1],
                query.analysis["subjectivity"][0],
                query.analysis["subjectivity"][1],
            )
            for relation in sentiment_relations:
                self.add_memory_connection(
                    memory, relation, sentiment=query.analysis["sentiment"]
                )
            emotion_relations = self.get_related_memories_by_emotion(
                memory, query.analysis["emotion"]
            )
            for relation in emotion_relations:
                self.add_memory_connection(
                    memory, relation, emotion=query.analysis["emotion"]
                )

            entity_relations = self.get_related_memories_by_entity(
                memory, query.analysis["named_entities"]
            )
            for relation in entity_relations:
                self.add_memory_connection(
                    memory, relation, entity=query.analysis["named_entities"]
                )

            return list(
                set(
                    keyword_relations
                    + time_relations
                    + sentiment_relations
                    + emotion_relations
                )
            )
        if query is None:
            return self.get_related_memories(memory)


# @track_calls
class FlatMemoryAccess:
    def __init__(self, hierarchy: MemoryHierarchy):
        self.memory_graph = None
        self.recent_memories = deque(maxlen=50)
        self.common_memories = {}
        self.repetitive_memories = {}
        self.urgent_query_memories = {}
        self.most_importance_memories = {}
        self.index_id_to_node_id = {}
        self.faiss_index = None
        self.euclidean_threshold = 0.35

        if hierarchy is not None:
            self.hierarchy = hierarchy
        else:
            Raise(ValueError("Hierarchy cannot be None"))
        self.initialize_faiss_index(768)

        # self.json_file = json_file
        # self.cache = retrieve_cache(self.json_file)

    def initialize_faiss_index(self, dimension, metric="l2"):
        """
        Initializes a FAISS index for the embeddings and populates it with vectors
        from the SpecificMemory instances in the graph. Assumes embeddings are numpy arrays.

        Parameters:
        - dimension: The dimensionality of the embeddings.
        - metric: The distance metric to use (e.g., 'l2', 'ip', 'hnsw', 'pq', 'opq', 'nsw').

        Metric Explanations and Use Cases:
        - 'l2': Euclidean distance. Suitable for general-purpose similarity search.
            Use it when the dimensionality of the vectors is not too high.
        - 'ip': Inner product (dot product). Suitable for high-dimensional sparse vectors.
            Use it when the vectors are high-dimensional and sparse.
        - 'hnsw': Hierarchical Navigable Small World. Suitable for efficient approximate nearest neighbor search.
            Use it when you need to perform approximate nearest neighbor search, especially in high-dimensional spaces.
        - 'pq': Product Quantization. Suitable for high-dimensional vectors with reduced memory usage.
            Use it when you need to reduce memory usage for high-dimensional vectors.
        - 'opq': Optimized Product Quantization. Suitable for high-dimensional vectors with optimized memory usage.
            Use it when you need to optimize memory usage for high-dimensional vectors.
        - 'nsw': Navigable Small World. Suitable for efficient approximate nearest neighbor search.
            Use it when you need to perform approximate nearest neighbor search.
        - 'lsh': Locality-Sensitive Hashing. Suitable for binary vectors and approximate nearest neighbor search.
            Use it when you need to perform approximate nearest neighbor search for binary vectors. This is useful for text search. Also known as 'cosine' for cosine similarity.
        - 'ivf': Inverted File with exact post-verification. Suitable for high-dimensional vectors with exact search.
            Use it when you need to perform exact search for high-dimensional vectors. For instance, when the vectors are embeddings of text or images.
        - 'sq8': Scalar Quantizer with 8 bits per component. Suitable for high-dimensional vectors with reduced memory usage.
            Use it when you need to reduce memory usage for high-dimensional vectors. This is useful for large-scale image search.
        - 'ivfsq': Inverted File with scalar quantizer (8 bits per component). Suitable for high-dimensional vectors with reduced memory usage.
            Use this when you need to reduce memory usage for high-dimensional vectors. This is useful for large-scale image search.
        - 'ivfpq': Inverted File with product quantizer. Suitable for high-dimensional vectors with exact search.
            This is useful for high-dimensional vectors with exact search. For instance, when the vectors are embeddings of text or images.
        - 'ivfpqr': Inverted File with product quantizer and re-ranking. Suitable for high-dimensional vectors with exact search and re-ranking.
            Use this when you need to perform exact search and re-ranking for high-dimensional vectors. For instance, when the vectors are embeddings of text or images.


        """
        if dimension <= 0 or dimension is None:
            dimension = 768  # Default dimension for BERT embeddings

        N = 1000000

        # Param of PQ
        M = 16  # The number of sub-vector. Typically this is 8, 16, 32, etc.
        nbits = 8  # bits per sub-vector. This is typically 8, so that each sub-vec is encoded by 1 byte
        # Param of IVF
        nlist = 1000  # The number of cells (space partition). Typical value is sqrt(N)
        # Param of HNSW
        hnsw_m = 32  # The number of neighbors for HNSW. This is typically 32

        if metric == "l2":
            # Exact Search for L2
            self.faiss_index = faiss.IndexFlatL2(dimension)
        elif metric == "ip":
            # Exact Search for Inner Product
            self.faiss_index = faiss.IndexFlatIP(dimension)
        elif metric == "hnsw" or metric == "nsw":
            # Hierarchical Navigable Small World graph exploration
            self.faiss_index = faiss.IndexHNSWFlat(dimension, 32)
        elif metric == "pq":
            # Product quantizer (PQ) in flat mode
            self.faiss_index = faiss.IndexPQ(dimension, 8, 8, 0)
        elif metric == "opq":
            # Optimized Product Quantizer (OPQ) in flat mode
            self.faiss_index = faiss.IndexFlatL2(dimension)
            self.faiss_index = faiss.IndexPreTransform(
                self.faiss_index, faiss.OPQMatrix(dimension, 64)
            )

        elif metric == "lsh":
            # Locality-Sensitive Hashing (binary flat index)
            nbits = 16
            self.faiss_index = faiss.IndexLSH(dimension, nbits)

        elif metric == "ivf":
            # Inverted file with exact post-verification
            self.faiss_index = faiss.IndexIVFFlat(
                self.quantizer, dimension, 100, faiss.METRIC_L2
            )

        elif metric == "sq8":
            # Scalar quantizer with 8 bits per component
            self.faiss_index = faiss.IndexScalarQuantizer(
                dimension, faiss.ScalarQuantizer.QT_8bit
            )
        elif metric == "ivfx,sq8" or metric == "ivfsq":
            # Inverted file with scalar quantizer (8 bits per component)
            self.faiss_index = faiss.IndexIVFScalarQuantizer(
                self.quantizer, dimension, 100, faiss.ScalarQuantizer.QT_8bit
            )
        elif metric == "ivfpq":
            # Inverted file with product quantizer
            self.faiss_index = faiss.IndexIVFPQ(self.quantizer, dimension, 100, 8, 8)
        elif metric == "ivfpqr":
            # Inverted file with product quantizer and re-ranking
            self.faiss_index = faiss.IndexIVFPQR(self.quantizer, dimension, 100, 8, 8)

        print(
            f"FAISS index initialized with dimension {self.faiss_index.d} and metric {metric}"
        )
        embeddings = []
        index_key = 0

        for specific_memory in self.get_graph().nodes:
            if (
                isinstance(specific_memory, SpecificMemory)
                and specific_memory.embedding is not None
            ):
                embeddings.append(specific_memory.get_embedding()[0])
                self.index_id_to_node_id[index_key] = specific_memory
                index_key += 1

        if embeddings:
            print(f"Adding {len(embeddings)} embeddings to the FAISS index")
            for em in embeddings:
                print(f"Embedding shape: {em.shape}")

            # embeddings = np.array([e.cpu().detach() for e in embeddings])
            embeddings = np.concatenate(embeddings)
            # embeddings = torch.stack([e.squeeze(0) for e in embeddings]).cpu().detach().numpy()
            print(f"Embedding shape: {embeddings.shape}")
            print(f"Embedding type: {type(embeddings)}")
            assert (
                embeddings.shape[1] == self.faiss_index.d
            ), f"Data dimensionality ({embeddings.shape[1]}) does not match index dimensionality ({self.faiss_index.d})"

            if metric == "l2" or metric == "ip" or metric == "cosine":
                self.faiss_index.add(embeddings)

            elif metric == "hnsw" or metric == "nsw":
                self.faiss_index.add_with_ids(
                    embeddings,
                    np.array(list(self.index_id_to_node_id.keys()))
                    .cpu()
                    .detach()
                    .numpy(),
                )
                try:  # efConstruction is the number of neighbors that are accessed during the construction of the graph. Higher efConstruction leads to higher accuracy but slower construction.
                    self.faiss_index.efConstruction = 32
                    self.faiss_index.efSearch = 64
                except:
                    pass
            elif metric == "ann" or metric == "approximate":
                # HNSW + IVFPQ
                # Define the index
                quantizer = faiss.IndexHNSWFlat(
                    dimension, hnsw_m
                )  # we use HNSW to assign the vectors to the cells
                self.faiss_index = faiss.IndexIVFPQ(
                    quantizer, dimension, nlist, M, nbits
                )

                # train the index
                self.faiss_index.train(embeddings.cpu().detach().numpy())
                self.faiss_index.add(embeddings.cpu().detach().numpy())
            elif metric == "pq":
                self.faiss_index.train(embeddings.cpu().detach().numpy())
                self.faiss_index.add(embeddings.cpu().detach().numpy())
            elif metric == "opq":
                self.faiss_index.train(embeddings.cpu().detach().numpy())
                self.faiss_index.add(embeddings.cpu().detach().numpy())
            elif metric == "lsh":
                self.faiss_index.add(embeddings.cpu().detach().numpy())
                if (
                    self.faiss_index.buckets is not None
                    and len(self.faiss_index.buckets) > 0
                ):
                    buckets = self.faiss_index.buckets
                else:
                    buckets = faiss.vector_to_array(self.faiss_index.codes)
                # use buckets for LSH forest techniques

            elif metric == "ivf":
                self.faiss_index.train(np.array(embeddings).cpu().detach().numpy())
                self.faiss_index.add(np.array(embeddings).cpu().detach().numpy())
            elif metric == "sq8":
                self.faiss_index.train(np.array(embeddings).cpu().detach().numpy())
                self.faiss_index.add(np.array(embeddings).cpu().detach().numpy())
            else:
                self.faiss_index.add(embeddings.cpu().detach().numpy())

            print(f"FAISS index populated with {self.faiss_index.ntotal} embeddings")

            # self.faiss_index.add(np.array(embeddings))
        else:
            print("No embeddings found to populate the FAISS index")
            time.sleep(2)

    def retrieve_cache(json_file):
        try:
            with open(json_file, "r") as file:
                cache = json.load(file)
        except FileNotFoundError:
            cache = {
                "questions": [],
                "embeddings": [],
                "answers": [],
                "response_text": [],
            }

        return cache

    def store_cache(json_file, cache):
        with open(json_file, "w") as file:
            json.dump(cache, file)

    def find_memories_by_query(
        self,
        query,
        top_k=10,
        threshold=0.5,
        similarity_metric="euclidian",
        num_clusters=10,
        use_hnsw=False,
    ):
        """
        Finds and returns top_k most similar memories to the given specific_memory_id,
        based on the cosine similarity of their embeddings.

        Parameters:
        - query: The query to search for.
        - top_k: Number of most similar memories to return.

        Returns:
        A list of tuples (memory_id, similarity_score).
        """

        start_time = time.time()
        query_embedding = None
        if isinstance(query, MemoryQuery):
            if query.query_embedding is None:
                query_embedding = query.get_embedding()
            else:
                query_embedding = query.query_embedding
        else:
            query = MemoryQuery(query, manager.gametime_manager)
            if query.query_embedding is None:
                query_embedding = query.get_embedding()
            else:
                query_embedding = query.query_embedding
        if query_embedding is None:
            print(f"Error retrieving embedding for query {query}")
            return {}
        if self.faiss_index is None:
            self.initialize_faiss_index(query_embedding.shape[1])

        # self.faiss_index.nprobe = num_clusters
        query_vec = query_embedding.cpu().detach().numpy()
        distances, indices = self.faiss_index.search(x=query_vec, k=top_k)

        if similarity_metric == "cosine":
            # Convert L2 distances to cosine similarity scores
            similarities = 1 - distances
        else:
            similarities = distances

        print(
            f"Similarities: {similarities},\n Indices: {indices},\n Distances: {distances}\n"
        )
        print(self.get_graph().nodes)
        print(self.index_id_to_node_id)

        # Convert indices to memory IDs
        for idx in indices[0]:
            if idx != -1:
                print(f"Memory ID: {self.index_id_to_node_id[idx]}")
                print(
                    f"Corresponding graph node: {manager.hierarchy.memory_graph.nodes[self.index_id_to_node_id[idx]]}"
                )

                print(
                    f"Corresponding graph node: {self.get_graph().nodes[self.index_id_to_node_id[idx]]}"
                )
        similar_memories = {
            self.index_id_to_node_id[idx]: float(similarity)
            for idx, similarity in zip(indices[0], similarities[0])
            if idx != -1 and similarity > threshold
        }
        assert isinstance(
            similar_memories, dict
        ), f"Similar memories is not a dictionary: {similar_memories}"

        end_time = time.time()
        print(
            f"Time taken to find similar memories: {end_time - start_time:.4f} seconds"
        )
        return similar_memories

    def find_similar_memories(
        self,
        specific_memory_id,
        top_k=10,
        threshold=0.5,
        similarity_metric="cosine",
        use_faiss=True,
        use_pagerank=False,
        num_clusters=10,
        use_hnsw=False,
    ):
        """
        Finds and returns top_k most similar memories to the given specific_memory_id,
        based on the cosine similarity of their embeddings.

        Parameters:
        - specific_memory_id: The ID of the SpecificMemory instance to search for.
        - top_k: Number of most similar memories to return.

        Returns:
        A list of tuples (memory_id, similarity_score).
        """

        if use_faiss:
            return self.find_similar_memories_faiss(
                specific_memory_id, top_k, threshold, similarity_metric, num_clusters
            )

        if use_pagerank:
            return self.find_similar_memories_pagerank(
                specific_memory_id, top_k, threshold, similarity_metric
            )

        start_time = time.time()

        query_memory = self.get_graph().nodes[specific_memory_id]
        if query_memory.embedding is None:
            try:
                query_memory.get_embedding()
            except Exception as e:
                print(
                    f"Error retrieving embedding for memory {specific_memory_id}: {e}"
                )
                return []

        similarities = {}
        for memory in self.get_graph().nodes:
            if memory == specific_memory_id:
                continue
            if self.get_graph().nodes[memory].embedding is None:
                try:
                    self.get_graph().nodes[memory].get_embedding()
                except Exception as e:
                    print(f"Error retrieving embedding for memory {memory}: {e}")
                    continue
            if similarity_metric == "cosine":
                similarities[memory] = 1 - cosine_similarity(
                    query_memory.embedding, self.get_graph().nodes[memory].embedding
                )
            else:
                similarities[memory] = self.euclidean_distance(
                    query_memory.embedding, self.get_graph().nodes[memory].embedding
                )

    def euclidean_distance(self, x, y):
        # Return basic euclidean distance of two vectors
        return np.linalg.norm(x - y)

    def find_similar_memories_pagerank(
        self, specific_memory_id, top_k=10, threshold=0.1, similarity_metric="cosine"
    ):
        start_time = time.time()

        query_memory = self.get_graph().nodes[specific_memory_id]
        if query_memory.embedding is None:
            try:
                query_memory.get_embedding()
            except Exception as e:
                print(
                    f"Error retrieving embedding for memory {specific_memory_id}: {e}"
                )
                return []

        pagerank_scores = nx.pagerank(self.memory_graph)

        similarities = {}
        for memory in self.get_graph().nodes:
            if memory == specific_memory_id:
                continue
            if self.get_graph().nodes[memory].embedding is None:
                try:
                    self.get_graph().nodes[memory].get_embedding()
                except Exception as e:
                    print(f"Error retrieving embedding for memory {memory}: {e}")
                    continue
            if similarity_metric == "cosine":
                similarities[memory] = 1 - cosine_similarity(
                    query_memory.embedding, self.get_graph().nodes[memory].embedding
                )
            else:
                similarities[memory] = self.euclidean_distance(
                    query_memory.embedding, self.get_graph().nodes[memory].embedding
                )

        end_time = time.time()
        print(
            f"Time taken to find similar memories: {end_time - start_time:.4f} seconds"
        )
        return similarities

    def find_similar_memories_faiss(
        self,
        specific_memory_id,
        top_k=10,
        threshold=0.1,
        similarity_metric="",
        num_clusters=10,
    ):
        start_time = time.time()

        query_memory = self.get_graph().nodes[specific_memory_id]
        if query_memory.embedding is None:
            try:
                query_memory.get_embedding()
            except Exception as e:
                print(
                    f"Error retrieving embedding for memory {specific_memory_id}: {e}"
                )
                return []

        if self.faiss_index is None or self.faiss_index.ntotal == 0:
            print(
                f"(Re)Initializing FAISS index with {query_memory.embedding.shape[0]} dimensions"
            )
            self.initialize_faiss_index(query_memory.embedding.shape[0])

        # self.faiss_index.nprobe = num_clusters
        query_vec = np.array([query_memory.embedding])
        distances, indices = self.faiss_index.search(query_vec, top_k)

        if similarity_metric == "cosine":
            # Convert L2 distances to cosine similarity scores
            similarities = 1 - distances
        else:
            similarities = distances

        # Convert indices to memory IDs
        similar_memories = [
            (self.index_id_to_node_id[idx], similarity)
            for idx, similarity in zip(indices[0], similarities[0])
            if similarity > threshold
        ]

        end_time = time.time()
        print(
            f"Time taken to find similar memories: {end_time - start_time:.4f} seconds"
        )
        return similar_memories

    def get_graph(self):
        self.memory_graph = self.hierarchy.memory_graph
        return self.hierarchy.memory_graph

    def add_memory(self, memory: SpecificMemory):
        self.recent_memories.append(memory)
        # Adding to priority queue with importance score as the priority
        if len(self.most_importance_memories) < 10:
            heapq.heappush(
                self.most_importance_memories, (memory.importance_score, memory)
            )
        else:
            # If the new memory is more important than the least important in the heap
            if memory.importance_score > self.most_importance_memories[0][0]:
                heapq.heappop(self.most_importance_memories)
                heapq.heappush(
                    self.most_importance_memories, (memory.importance_score, memory)
                )

    def add_common_memory(self, key, memory):
        self.common_memories[key] = memory

    def add_repetitive_memory(self, key, memory):
        self.repetitive_memories[key] = memory

    def add_urgent_query_memory(self, memory):
        self.urgent_query_memories[memory] = memory

    def get_recent_memories(self):
        return self.recent_memories

    def get_common_memories(self, key):
        return self.common_memories[key]

    def get_all_common_memories(self):
        return list(self.common_memories.values())

    def get_repetitive_memories(self, key):
        return self.repetitive_memories[key]

    def get_urgent_query_memories(self):
        return self.urgent_query_memories

    def get_most_importance_memories(self):
        return self.most_importance_memories

    def get_list_of_memories(self):
        self.recent_memories.extend(self.common_memories)
        self.recent_memories.extend(self.repetitive_memories)
        self.recent_memories.extend(self.urgent_query_memories)
        self.recent_memories.extend(self.most_importance_memories)
        return self.recent_memories


sentiment_analysis = SentimentAnalysis()


# @track_calls
class MemoryManager:
    def __init__(self, gametime_manager):
        self.hierarchy = MemoryHierarchy()
        self.flat_access = FlatMemoryAccess(self.hierarchy)
        self.memory_embeddings = {}
        self.complex_keywords = set()
        self.similarity_threshold_specific = 0.4
        self.similarity_threshold_general = 0.25
        self.recent_queries = deque(maxlen=50)
        self.faiss_index_recent_queries_flatl2 = None
        self.map_tags = {}
        # self.general_memories = []
        self.gametime_manager = gametime_manager
        assert self.gametime_manager is not None, "Game time manager is required"

    def init_memories(self, general_memories):
        # self.general_memories = general_memories
        for general_memory in general_memories:
            self.hierarchy.add_general_memory(general_memory)
            general_memory.index_memories()
            self.flat_access.add_memory(general_memory)

    def index_recent_queries_flatl2(self):
        print(f"\n Indexing recent queries \n")
        if len(self.recent_queries) < 1:
            return None
        embeddings = [query.query_embedding for query in self.recent_queries]
        print(f"Length of embeddings: {len(embeddings)}")

        if len(embeddings) > 0 and embeddings[0] is not None:
            embeddings = torch.cat(embeddings, dim=0)
            self.faiss_index_recent_queries_flatl2 = faiss.IndexFlatL2(
                embeddings.shape[1]
            )
            self.faiss_index_recent_queries_flatl2.add(
                embeddings.cpu().detach().numpy()
            )
            return self.faiss_index_recent_queries_flatl2
        return None

    def index_recent_queries_hnsw(self):
        print(f"\n Indexing recent queries \n")
        if len(self.recent_queries) < 1:
            return None
        embeddings = [query.query_embedding for query in self.recent_queries]
        assert len(embeddings) > 0, "No recent queries to index"
        embeddings = torch.cat(embeddings, dim=0)

        dimension = embeddings.shape[1]
        self.faiss_index_recent_queries = faiss.IndexHNSWFlat(
            dimension, 32
        )  # 32 is an example for the second parameter, you can adjust it according to your needs
        self.faiss_index_recent_queries.hnsw.efConstruction = (
            40  # example value, adjust it according to your needs
        )
        self.faiss_index_recent_queries.hnsw.efSearch = (
            40  # example value, adjust it according to your needs
        )
        self.faiss_index_recent_queries.add(embeddings.cpu().detach().numpy())
        return self.faiss_index_recent_queries

    def add_general_memory(self, general_memory):
        self.hierarchy.add_general_memory(general_memory)
        self.update_embeddings(general_memory)
        return general_memory

    def update_embeddings(self, memory=None):
        if isinstance(memory, SpecificMemory):
            self.memory_embeddings.update({memory: memory.get_embedding()})
        elif isinstance(memory, GeneralMemory):
            for specific_memory in memory.get_specific_memories():
                if specific_memory != None:
                    self.memory_embeddings[specific_memory] = (
                        specific_memory.get_embedding()
                    )

    def add_memory(self, memory):
        self.flat_access.add_memory(memory)
        self.update_hierarchy(memory)

    def update_hierarchy(self, memory):
        """
        Update the hierarchical memory structure with new memory.

        Args:
            memory: Memory object to integrate into hierarchy
        """
        try:
            # Extract memory properties for hierarchical organization
            timestamp = getattr(memory, "timestamp", None)
            importance = getattr(memory, "importance", 0.5)
            memory_type = getattr(memory, "memory_type", "general")
            description = getattr(memory, "description", "")

            # Organize by time periods
            if timestamp:
                # Group by time periods (daily, weekly, monthly)
                time_key = self._get_time_period_key(timestamp)

                if not hasattr(self, "time_hierarchy"):
                    self.time_hierarchy = {}

                if time_key not in self.time_hierarchy:
                    self.time_hierarchy[time_key] = []

                self.time_hierarchy[time_key].append(memory)

            # Organize by importance levels
            if not hasattr(self, "importance_hierarchy"):
                self.importance_hierarchy = {"high": [], "medium": [], "low": []}

            if importance >= 0.8:
                self.importance_hierarchy["high"].append(memory)
            elif importance >= 0.5:
                self.importance_hierarchy["medium"].append(memory)
            else:
                self.importance_hierarchy["low"].append(memory)

            # Organize by memory type/category
            if not hasattr(self, "type_hierarchy"):
                self.type_hierarchy = {}

            if memory_type not in self.type_hierarchy:
                self.type_hierarchy[memory_type] = []

            self.type_hierarchy[memory_type].append(memory)

            # Create semantic clusters if description available
            if description:
                self._update_semantic_clusters(memory, description)

            # Maintain hierarchy size limits (prevent unbounded growth)
            self._prune_hierarchy()

        except Exception as e:
            print(f"Error updating memory hierarchy: {e}")

    def _get_time_period_key(self, timestamp):
        """Generate time period key for hierarchical organization."""
        try:
            import datetime

            if isinstance(timestamp, (int, float)):
                dt = datetime.datetime.fromtimestamp(timestamp)
            else:
                dt = timestamp

            # Create hierarchical time keys
            year = dt.year
            month = dt.month
            day = dt.day

            return f"{year}-{month:02d}-{day:02d}"
        except:
            return "unknown"

    def _update_semantic_clusters(self, memory, description):
        """Update semantic clustering of memories."""
        try:
            # Extract keywords and entities
            keywords = self.extract_keywords(description)
            entities = self.extract_entities(description)

            if not hasattr(self, "semantic_clusters"):
                self.semantic_clusters = {}

            # Group by keywords
            for keyword in keywords[:5]:  # Top 5 keywords
                if keyword not in self.semantic_clusters:
                    self.semantic_clusters[keyword] = []
                self.semantic_clusters[keyword].append(memory)

            # Group by entities
            for entity in entities:
                entity_key = f"entity_{entity.lower()}"
                if entity_key not in self.semantic_clusters:
                    self.semantic_clusters[entity_key] = []
                self.semantic_clusters[entity_key].append(memory)

        except Exception as e:
            print(f"Error updating semantic clusters: {e}")

    def _prune_hierarchy(self):
        """Prune hierarchy to maintain reasonable size limits."""
        try:
            max_memories_per_cluster = 100

            # Prune time hierarchy
            if hasattr(self, "time_hierarchy"):
                for time_key in list(self.time_hierarchy.keys()):
                    memories = self.time_hierarchy[time_key]
                    if len(memories) > max_memories_per_cluster:
                        # Keep most important memories
                        memories.sort(
                            key=lambda m: getattr(m, "importance", 0), reverse=True
                        )
                        self.time_hierarchy[time_key] = memories[
                            :max_memories_per_cluster
                        ]

            # Prune semantic clusters
            if hasattr(self, "semantic_clusters"):
                for cluster_key in list(self.semantic_clusters.keys()):
                    memories = self.semantic_clusters[cluster_key]
                    if len(memories) > max_memories_per_cluster:
                        # Keep most recent and important
                        memories.sort(
                            key=lambda m: (
                                getattr(m, "importance", 0),
                                getattr(m, "timestamp", 0),
                            ),
                            reverse=True,
                        )
                        self.semantic_clusters[cluster_key] = memories[
                            :max_memories_per_cluster
                        ]

        except Exception as e:
            print(f"Error pruning hierarchy: {e}")

    def extract_entities(self, text):
        if isinstance(text, list):
            text = " ".join(text)
        doc = nlp(text)
        return [ent.text for ent in doc.ents]

    # Function to perform LDA topic modeling

    def extract_lda_keywords(self, docs, num_topics=3, num_words=3):
        tokenizer = RegexpTokenizer(r"\w+")
        docs = [
            re.sub(r"|".join(map(re.escape, remove_list)), "", docs) for docs in docs
        ]

        doc_tokens = [tokenizer.tokenize(doc.lower()) for doc in docs]
        docs = [" ".join(tokens) for tokens in doc_tokens]

        vectorizer = CountVectorizer()
        X = vectorizer.fit_transform(docs)

        lda = LatentDirichletAllocation(n_components=num_topics, random_state=0)
        lda.fit(X)

        lda_keywords = []
        for topic_idx, topic in enumerate(lda.components_):
            top_words = [
                vectorizer.get_feature_names_out()[i]
                for i in topic.argsort()[: -num_words - 1 : -1]
            ]
            lda_keywords.extend(top_words)

        return lda_keywords

    # Function to extract keywords using TF-IDF
    def extract_tfidf_keywords(self, docs, top_n=3):
        if not isinstance(docs, list):
            docs = [docs]

        vectorizer = TfidfVectorizer(stop_words=sentiment_analysis.stop_words)
        X = vectorizer.fit_transform(docs)
        feature_array = vectorizer.get_feature_names_out()
        tfidf_sorting = [
            X.toarray().argsort(axis=1)[:, -top_n:] for _ in range(X.shape[0])
        ]
        tfidf_keywords = [feature_array[i] for row in tfidf_sorting for i in row[0]]
        return tfidf_keywords

    # Function to extract keywords using RAKE
    def extract_rake_keywords(self, docs, top_n=2):
        print(type(docs))
        r = Rake()
        if isinstance(docs, list):
            docs = " ".join(docs)
        r.extract_keywords_from_text(docs)
        rake_keywords = set(r.get_ranked_phrases()[:top_n])
        return rake_keywords

    # def generate_embedding(self):
    #     self.description = [self.description.strip()]
    #     input = model.tokenizer(self.description, padding=True,
    #     truncation=True,
    #     add_special_tokens=True,
    #     is_split_into_words=True,
    #     pad_to_multiple_of=8,
    #     return_tensors="pt").to(model.device)
    #     outputs = model.model(input["input_ids"],attention_mask=input["attention_mask"],
    #     output_hidden_states=True,
    #     return_dict=True,
    #     return_tensors="pt",)
    #     return [outputs.last_hidden_state, input["attention_mask"]]

    def get_query_embedding(self, query):
        input = model.tokenizer(
            query, return_tensors="pt", padding=True, truncation=True
        )
        input = input.to(model.device)
        outputs = model.forward(input["input_ids"], input["attention_mask"])
        print(f"\n Shape of query embedding: {outputs.last_hidden_state.shape}")
        query_embedding = mean_pooling(
            outputs.last_hidden_state, input["attention_mask"]
        )
        return query_embedding

    def retrieve_from_hierarchy(self, query):
        if isinstance(query, MemoryQuery):
            print(f"\n Retrieving memories from hierarchy for query: {query.query}")
        else:
            query = MemoryQuery(query, gametime_manager=self.gametime_manager)
            print(f"\n Retrieving memories from hierarchy for query: {query}")
        relevant_memories = {}

        relevancy_scores = {}
        for general_memory in self.hierarchy.general_memories:

            is_rel, score = self.is_relevant_general_memory(general_memory, query)
            relevancy_scores[general_memory] = score
            if is_rel:
                relevant_memories.update(
                    self.traverse_specific_memories(general_memory, query)
                )
        print(
            f"\n Relevant memories, hierarchy: {[memory.description for memory in relevant_memories]} \n"
        )
        if len(relevant_memories) < 1:
            assert len(relevancy_scores) > 0, "No relevancy scores found"
            # If no relevant memories are found, return the highest scoring general memory
            highest_score = max(relevancy_scores.values())
            # Sort the relevancy scores by value
            possibly_relevant = {
                k: v
                for k, v in sorted(
                    relevancy_scores.items(), key=lambda item: item[1], reverse=True
                )
            }

            # Turn the relevancy scores back into a dict, now sorted by value
            relevancy_scores = {
                k: float(relevancy_scores[k]) for k in possibly_relevant
            }
            ids = list(relevancy_scores.keys())
            print(f"\n ids: {ids}")
            print(f"\n Relevancy scores: {relevancy_scores}")

            if len(possibly_relevant) > 1:
                for i in range(1, len(possibly_relevant)):
                    if (
                        possibly_relevant[ids[i]] in relevancy_scores
                        and relevancy_scores[possibly_relevant[ids[i]]] < highest_score
                    ):
                        possibly_relevant = possibly_relevant[:i]
                        break
            print(f"\n Length of possibly relevant: {len(possibly_relevant)}")
            num_gm_nodes = [
                node
                for node in self.hierarchy.memory_graph.nodes
                if isinstance(node, GeneralMemory)
            ]
            assert len(self.hierarchy.general_memories) == len(
                num_gm_nodes
            ), f"Number of general memories does not match number of nodes in memory graph: {len(self.hierarchy.general_memories)} != {len(num_gm_nodes)}"
            print(f" \n Top scoring general memory: {ids[0].description}")
            print(
                f"\n Second highest scoring general memory: {ids[1].description} with score: {relevancy_scores[ids[1]]}"
            )
            print(
                f"\n Third highest scoring general memory: {ids[2].description} with score: {relevancy_scores[ids[2]]}"
            )
            print(
                f"\n Fourth highest scoring general memory: {ids[3].description} with score: {relevancy_scores[ids[3]]}"
            )
            print(
                f"\n Fifth highest scoring general memory: {ids[4].description} with score: {relevancy_scores[ids[4]]}"
            )
            for i in range(5):
                relevant_memories.update(
                    self.traverse_specific_memories(
                        self.hierarchy.general_memories[possibly_relevant[ids[i]]],
                        query,
                    )
                )

        print(
            f"\n Relevant memories, hierarchy: {[memory.description for memory in relevant_memories]}"
        )
        return relevant_memories

    def retrieve_memories_bst(self, general_memory, query):
        print(
            f"\n Retrieving memories from BST for general memory: {general_memory.description} and query: {query.query}"
        )
        exit(0)
        # This function is designed to traverse a BST within a GeneralMemory instance
        # to find SpecificMemory instances that match the given query criteria.
        matching_memories = []
        query_criteria = query.query_function

        # Example BST traversal method based on criteria
        def bst_traversal(node):
            if node is None:
                return
            # Assuming query_criteria is a function that evaluates if a node matches the criteria
            if query_criteria(node.memory):
                matching_memories.append(node.memory)
            bst_traversal(node.left)
            bst_traversal(node.right)

        bst_traversal(general_memory.specific_memories_root)
        print(f"\n Matching memories BST: {matching_memories}")
        return matching_memories

    def traverse_specific_memories(
        self, general_memory, query, key=None, attribute=None
    ):
        print(
            f"\n Traversing specific memories for general memory: {general_memory.description} and query: {query.query}"
        )
        if query.analysis is None:
            query.analysis = self.analyze_query_context(query.query)
        if query.query_embedding is None:
            query.query_embedding = query.analysis["embedding"]
        if attribute is not None:
            query.attribute = attribute
            q = f"Answer only with yes or no: is *memory_description* relevant to *attribute*?"
            query.add_complex_query(attribute, q)
            return self.retrieve_memories_bst(general_memory, query)
        elif key is not None:
            return general_memory.find_specific_memory(key)
        elif query.attribute is not None:
            query.attribute = attribute
            q = f"Answer only with yes or no: is *memory_description* relevant to *attribute*?"
            query.add_complex_query(attribute, q)
            return self.retrieve_memories_bst(general_memory, query)
        dists, ids = self.search_by_index(general_memory)
        possible_memories = []
        for sm_id, dist in zip(ids, dists):
            possible_memories.append(
                general_memory.get_specific_memories()[int(sm_id[0])]
            )
            possible_memories.extend(self.flat_access.get_list_of_memories())
            print(
                "FAISS score for memory: ",
                general_memory.get_specific_memories()[int(sm_id[0])].description,
                " is ",
                dist,
            )
        print(f"length of possible memories: {len(possible_memories)}")
        print(
            f"possible memories: {[memory.description for memory in possible_memories]}"
        )

        specific_memories = {}  # {memory:total_score}
        for specific_memory in possible_memories:
            print(
                f"\n Specific Memory embedding shape: {specific_memory.embedding.shape}"
            )
            recency_score = self.calculate_recency_score(
                specific_memory, datetime.now()
            )
            relevance_score = cosine_similarity(
                specific_memory.embedding.cpu().detach().numpy(),
                query.query_embedding.cpu().detach().numpy(),
            )
            recency_score = self.normalize_scores(recency_score)
            relevance_score = self.normalize_scores(relevance_score)
            importance_score = self.normalize_scores(
                specific_memory.importance_score / 10
            )
            # Apply sigmoid function to relevance_score
            relevance_weight = sigmoid(relevance_score)

            # Calculate weights for other scores
            other_weight = (1 - relevance_weight) / 2

            # Calculate final score
            score = relevance_weight * relevance_score + other_weight * (
                recency_score + importance_score
            )
            specific_memories[specific_memory] = score
            print(
                f"\n Specific memory: {specific_memory.description} has a score of: {score} \n"
            )
            specific_memories.update(self.traverse_memory_graph(specific_memory, query))
        specific_memories = sorted(
            specific_memories.items(), key=lambda x: x[1], reverse=True
        )
        specific_memories = {k: v for k, v in specific_memories}
        print(f"\n types of specific memories: {type(specific_memories)}")
        if len(specific_memories) < 1:
            return {}
        print(
            f"\n type of first element in specific memories: {type(list(specific_memories.values())[0])}"
        )
        print(
            f"\n Specific memories in possible_memories: {[memory.description for memory in specific_memories]}"
        )
        return specific_memories

    def traverse_memory_graph(self, specific_memory, query):
        print(
            f"\n Traversing memory graph for specific memory: {specific_memory.description} and query: {query.query}"
        )
        related_memories = {}
        if self.hierarchy.memory_graph.has_node(specific_memory):
            print(
                f"\n Found specific memory: {specific_memory.description} in memory graph"
            )
            related_memories.update(
                self.hierarchy.find_nodes_by_edge_weight(specific_memory, min_weight=1)
            )
        print(
            f"\n Related memories in traverse_memory_graph: {[memory.description for memory in related_memories]}"
        )
        print(
            f"\n Length of related memories in traverse_memory_graph: {len(related_memories)}"
        )
        print(
            f"\n Types of related memories in traverse_memory_graph: {type(related_memories)}"
        )
        if len(related_memories) < 1:
            return {}
        print(
            f"\n Types of first element in related memories in traverse_memory_graph: {type(list(related_memories.values())[0])}"
        )

        return related_memories

    def is_relevant_general_memory(self, general_memory, query):
        # try to identify with tags first
        print(
            f"\n Checking relevance of general memory: {general_memory.description} for query: {query.query}"
        )
        rel_scores = [0]
        # print(f"Shape of query embedding: {query.analysis['embedding'].shape}")
        # print(f"Shape of general memory embedding: {general_memory.description_embedding.shape}")
        # if query.query_tags is not None and len(query.query_tags) > 0:
        #     if any(tag in general_memory.tags for tag in query.query_tags):
        #         score = len([tag for tag in query.query_tags if tag in general_memory.tags])/len(query.query_tags)

        #         print(f"\n General memory: {general_memory.description} is relevant to query: {query.query} because of tag overlap with score: {score} \n")
        #         # score is a percentage of the number of tags that match

        #         return True, score

        if query.analysis is None:
            query.analysis = self.analyze_query_context(query.query)

        if general_memory.description_embedding is None:
            general_memory.description_embedding = general_memory.get_embedding()
        if query.query_embedding is None:
            query.query_embedding = query.analysis["embedding"]

        similarity = cosine_similarity(
            general_memory.description_embedding.cpu().detach(),
            query.query_embedding.cpu().detach(),
        )

        if similarity > self.similarity_threshold_general:
            rel_scores.append(similarity)
            print(
                f"\n General memory: {general_memory.description} is relevant to query: {query.query} because of text similarity  {similarity} \n"
            )
            return True, similarity

        # Check for keyword overlap
        if general_memory.keywords is not None and len(general_memory.keywords) > 0:
            if any(
                keyword in general_memory.keywords
                for keyword in query.analysis["keywords"]
            ):
                score = (
                    len(
                        [
                            keyword
                            for keyword in query.analysis["keywords"]
                            if keyword in general_memory.keywords
                        ]
                    )
                    * 2
                ) / len(
                    list(
                        set(query.analysis["keywords"]).union(
                            set(general_memory.keywords)
                        )
                    )
                )
                rel_scores.append(score)
                final_score = (score + similarity) / 2
                print(
                    f"\n keywords in common: {[keyword for keyword in query.analysis['keywords'] if keyword in general_memory.keywords]}"
                )
                print(f"\n keywords in general memory: {general_memory.keywords}")
                print(f"\n keywords in query: {query.analysis['keywords']}")
                print(
                    f"\n General memory: {general_memory.description} is relevant to query: {query.query} because of keyword overlap with score: {score} with keywords: {list(set(general_memory.keywords).intersection(set(query.analysis['keywords'])))} \n"
                )
                return True, final_score
        else:
            try:
                print(
                    f"\n Extracting keywords from general memory: {general_memory.description}"
                )
                memory_keywords = self.extract_keywords(general_memory.description)
                query_keywords = query.analysis["keywords"]
            except Exception as e:
                memory_keywords = []
                query_keywords = []
            if len(memory_keywords) > 0 and len(query_keywords) > 0:
                if any(keyword in query_keywords for keyword in memory_keywords):
                    score = (
                        len(
                            [
                                keyword
                                for keyword in memory_keywords
                                if keyword in query_keywords
                            ]
                        )
                        * 2
                    ) / len(list(set(query_keywords).union(set(memory_keywords))))
                    final_score = (score + similarity) / 2
                    print(
                        f"\n keywords in common: {[keyword for keyword in memory_keywords if keyword in query_keywords]}"
                    )
                    print(f"\n keywords in general memory: {general_memory.keywords}")
                    print(f"\n keywords in query: {query.analysis['keywords']}")
                    print(
                        f"\n General memory: {general_memory.description} is relevant to query: {query.query} because of keyword overlap with score: {score} with keywords: {list(set(memory_keywords).intersection(set(query_keywords)))} \n"
                    )
                    return True, final_score
                # If no keywords are available, use a simple text similarity measure

                print(f"type of similarity: {type(similarity)}")
                # similarity = self.normalize_scores(similarity)
                # print(f"type of similarity: {type(similarity)}")
                print(
                    f"\n Similarity between general memory: {general_memory.description} and query: {query.query} is {similarity} \n"
                )

        if (
            general_memory.analysis["named_entities"] is not None
            and len(general_memory.analysis["named_entities"]) > 0
        ):
            if any(
                entity in general_memory.analysis["named_entities"]
                for entity in query.analysis["named_entities"]
            ):
                score = len(
                    [
                        entity
                        for entity in query.analysis["named_entities"]
                        if entity in general_memory.analysis["named_entities"]
                    ]
                ) / len(
                    list(
                        set(query.analysis["named_entities"]).union(
                            set(general_memory.analysis["named_entities"])
                        )
                    )
                )
                rel_scores.append(score)
                final_score = (score + similarity) / 2
                print(
                    f"\n entities in common: {[entity for entity in query.analysis['named_entities'] if entity in general_memory.analysis['named_entities']]}"
                )
                print(
                    f"\n General memory: {general_memory.description} is relevant to query: {query.query} because of entity overlap with score: {score} with entities: {list(set(general_memory.analysis['named_entities']).intersection(set(query.analysis['named_entities'])))} \n"
                )
                return True, final_score
        if query.analysis["main_subject"] is not None and (
            query.analysis["main_subject"] in general_memory.analysis["named_entities"]
            or query.analysis["main_subject"] in general_memory.analysis["keywords"]
            or query.analysis["main_subject"] in general_memory.description
        ):
            print(
                f"\n General memory: {general_memory.description} is relevant to query: {query.query} because of main subject overlap with main subject: {query.analysis['main_subject']} \n"
            )
            score = 0.5 + self.normalize_scores(rel_scores)
            return True, score
        final_score = max(rel_scores)
        print(
            f"\n Final score for general memory: {general_memory.description} is {final_score} \n"
        )
        if (
            self.normalize_scores(rel_scores) > self.similarity_threshold_specific
            or final_score > 0.5
        ):
            print(
                f"\n Wait up, General memory: {general_memory.description} is relevant to query: {query.query} because of normalized score: {self.normalize_scores(rel_scores)} \n"
            )
            return True, self.normalize_scores(rel_scores)
        print(
            f"\n General memory: {general_memory.description} is not relevant to query: {query.query} \n"
        )
        return False, final_score

    def retrieve_from_flat_access(self, query_features):
        print(f"\n Retrieving from flat access: {query_features.query}")
        # rel_mems = [memory for memory in self.flat_access.get_list_of_memories() if self.is_relevant_flat_memory(memory, query_features)]

        rel_mems = self.flat_access.find_memories_by_query(
            query_features, threshold=0.1
        )

        print(f"\n Relevant memories and scores, flat access: {rel_mems}")
        return rel_mems

    def is_relevant_specific_memory(self, specific_memory, query):

        print(
            f"\n Checking relevance of specific memory: {specific_memory.description} for query: {query.query}"
        )
        rel_scores = [0]
        if specific_memory.analysis is None:
            specific_memory.analysis = self.analyze_query_context(
                specific_memory.description
            )
        if specific_memory.embedding is None:
            specific_memory.embedding = specific_memory.get_embedding()
        if query.query_embedding is None:
            query.query_embedding = self.get_query_embedding(query.analysis["text"])[0]

        similarity = cosine_similarity(specific_memory.embedding, query.query_embedding)
        if (
            similarity > self.similarity_threshold_specific
        ):  # Define a suitable threshold, like 0.7
            rel_scores.append(similarity)

            return True, similarity
        if (
            specific_memory.analysis["named_entities"] is not None
            and len(specific_memory.analysis["named_entities"]) > 0
        ):
            if any(
                entity in specific_memory.analysis["named_entities"]
                for entity in query.analysis["named_entities"]
            ):
                score = len(
                    [
                        entity
                        for entity in query.analysis["named_entities"]
                        if entity in specific_memory.analysis["named_entities"]
                    ]
                ) / len(
                    list(
                        set(query.analysis["named_entities"]).union(
                            set(specific_memory.analysis["named_entities"])
                        )
                    )
                )
                rel_scores.append(score)
                final_score = (score + similarity) / 2
                print(
                    f"\n entities in common: {[entity for entity in query.analysis['named_entities'] if entity in specific_memory.analysis['named_entities']]}"
                )
                print(
                    f"\n Specific memory: {specific_memory.description} is relevant to query: {query.query} because of entity overlap with score: {score} with entities: {list(set(specific_memory.analysis['named_entities']).intersection(set(query.analysis['named_entities'])))} \n"
                )
                return True, final_score

        if (
            specific_memory.sentiment_score["polarity"] > 0.5
            or specific_memory.sentiment_score["polarity"] < -0.5
            or specific_memory.importance_score > 7
            or specific_memory.sentiment_score["subjectivity"] >= 0.7
        ) and (similarity > 0.3):
            senti_score = (
                specific_memory.sentiment_score["polarity"]
                + specific_memory.sentiment_score["subjectivity"]
                + (specific_memory.importance_score / 10) / 3
            )
            rel_and_senti = rel_scores
            rel_and_senti.append(senti_score)
            final_score = (self.normalize_scores(rel_and_senti) + similarity) / 2
            rel_scores.append(final_score)
            return True, final_score
        if specific_memory.keywords is not None and len(specific_memory.keywords) > 0:
            matches = len(
                [
                    keyword in specific_memory.keywords
                    for keyword in query.analysis["keywords"]
                ]
            )
            if (
                matches > 0
                and similarity > 0.3
                or (similarity > 0.2 and matches > 1)
                or (similarity > 0.1 and matches > 2)
                or (similarity > 0.05 and matches > 3)
                or (
                    similarity > 0.2
                    and query.analysis["emotion_classification"]
                    == specific_memory.emotion_classification
                )
            ):
                score = (matches * 2) / len(
                    list(
                        set(query.analysis["keywords"]).union(
                            set(specific_memory.keywords)
                        )
                    )
                )
                rel_scores.append(score)
                final_score = (score + similarity) / 2
                return True, final_score
        if specific_memory.importance_score + (similarity * 10) > 7:
            score = specific_memory.importance_score / 10
            rel_and_importance = rel_scores
            rel_and_importance.append(score)
            final_score = (self.normalize_scores(rel_and_importance) + similarity) / 2
            return True, final_score
        if specific_memory.last_access_time - query.query_time < timedelta(hours=24):
            score = 1 - (
                specific_memory.last_access_time - query.query_time
            ).total_seconds() / (24 * 60 * 60)
            rel_scores.append(score)
            # combine that score with the similarity score to get the final score
            final_score = (score + similarity) / 2
            return True, final_score
        if (
            self.normalize_scores(rel_scores) > self.similarity_threshold_specific
            or max(rel_scores) > 0.5
        ):
            return True, self.normalize_scores(rel_scores)
        return False, max(rel_scores)

    def is_relevant_related_memory(self, related_memory, query_features):
        print(
            f"\n Checking relevance of related memory: {related_memory.description} for query: {query_features.query}"
        )
        rel_scores = [0]
        if related_memory.embedding is None:
            related_memory.embedding = related_memory.get_embedding()
        if query_features.analysis is None:
            query_features.analysis = self.analyze_query_context(query_features.query)
        if query_features.query_embedding is None:
            query_features.query_embedding = query_features.analysis["embedding"]
        if related_memory.analysis is None:
            related_memory.analysis = self.analyze_query_context(
                related_memory.description
            )

        similarity = cosine_similarity(
            related_memory.embedding.cpu().detach(),
            query_features.query_embedding.cpu().detach(),
        )
        if (
            similarity > self.similarity_threshold_specific
        ):  # Define a suitable threshold, like 0.7
            rel_scores.append(similarity)
            return True, similarity
        if (
            related_memory.analysis["named_entities"] is not None
            and len(related_memory.analysis["named_entities"]) > 0
        ):
            if any(
                entity in related_memory.analysis["named_entities"]
                for entity in query_features.analysis["named_entities"]
            ):
                score = len(
                    [
                        entity
                        for entity in query_features.analysis["named_entities"]
                        if entity in related_memory.analysis["named_entities"]
                    ]
                ) / len(
                    list(
                        set(query_features.analysis["named_entities"]).union(
                            set(related_memory.analysis["named_entities"])
                        )
                    )
                )
                rel_scores.append(score)
                final_score = (score + similarity) / 2
                print(
                    f"\n entities in common: {[entity for entity in query_features.analysis['named_entities'] if entity in related_memory.analysis['named_entities']]}"
                )
                print(
                    f"\n Related memory: {related_memory.description} is relevant to query: {query_features.query} because of entity overlap with score: {score} with entities: {list(set(related_memory.analysis['named_entities']).intersection(set(query_features.analysis['named_entities'])))} \n"
                )
                return True, final_score

        # similarity = self.normalize_scores(similarity)

        elif related_memory.importance_score > 7 and similarity > 0.3:
            score = related_memory.importance_score / 10
            rel_and_importance = rel_scores
            rel_and_importance.append(score)
            final_score = (self.normalize_scores(rel_and_importance) + similarity) / 2
            return True, final_score
        elif (
            related_memory.sentiment_score["polarity"] > 0.5
            or related_memory.sentiment_score["polarity"] < -0.5
            or related_memory.importance_score > 5
            or related_memory.sentiment_score["subjectivity"] >= 0.7
        ) and (similarity > 0.3):
            senti_score = (
                related_memory.sentiment_score["polarity"]
                + related_memory.sentiment_score["subjectivity"]
                + (related_memory.importance_score / 10) / 3
            )
            rel_and_senti = rel_scores
            rel_and_senti.append(senti_score)
            final_score = (self.normalize_scores(rel_and_senti) + similarity) / 2
            rel_scores.append(final_score)
            return True, final_score
        elif related_memory.keywords is not None and len(related_memory.keywords) > 0:
            matches = len(
                [
                    keyword in related_memory.keywords
                    for keyword in query_features.analysis["keywords"]
                ]
            )
            if (
                matches > 0
                and similarity > 0.3
                or (similarity > 0.2 and matches > 1)
                or (similarity > 0.1 and matches > 2)
                or (similarity > 0.05 and matches > 3)
                or (
                    similarity > 0.2
                    and query_features.analysis["emotion_classification"]
                    == related_memory.emotion_classification
                )
            ):
                score = (matches * 2) / len(
                    list(
                        set(query_features.analysis["keywords"]).union(
                            set(related_memory.keywords)
                        )
                    )
                )
                rel_scores.append(score)
                final_score = (score + similarity) / 2
                return True, final_score
        if related_memory.last_access_time - query_features.query_time < timedelta(
            hours=24
        ):
            score = 1 - (
                related_memory.last_access_time - query_features.query_time
            ).total_seconds() / (24 * 60 * 60)
            rel_scores.append(score)
            # combine that score with the similarity score to get the final score
            final_score = (score + similarity) / 2
            return True, final_score

        if (
            self.normalize_scores(rel_scores) > self.similarity_threshold_specific
            or max(rel_scores) > 0.5
        ):
            return True, self.normalize_scores(rel_scores)

        return False, max(rel_scores)

    def get_sentiment_score(self, text):
        return sentiment_analysis.get_sentiment_score(text)

    def get_emotion_classification(self, text):
        return sentiment_analysis.get_emotion_classification(text)

    def analyze_query_context(self, query):
        print(f"\n Analyzing query: {query}")
        # Linguistic Analysis
        docs = nlp(query)
        print(docs)
        features = {
            "tokens": [token.text for token in docs],
            "lemmas": [token.lemma_ for token in docs],
            "pos_tags": [token.pos_ for token in docs],
            # Add more features as needed
        }

        is_urgent = False
        # if "urgent" in features["tokens"]:
        #     is_urgent = True
        is_common_query = False

        # Embedding Generation
        # inputs = tiny_brain_io.tokenizer(query, return_tensors='pt').to(tiny_brain_io.device)
        # tiny_brain_io.model.to(tiny_brain_io.device)

        # outputs = tiny_brain_io.model(**inputs, labels=inputs["input_ids"], return_dict=True, output_hidden_states=True)
        outputs = self.get_query_embedding(query)
        # response = tiny_brain_io.input_to_model(query)
        # print(f"Outputs: {outputs}")
        # inputs_decoded = tiny_brain_io.tokenizer.decode(outputs[1][0], skip_special_tokens=True)
        # print(f"Input ids decoded: {inputs_decoded}")
        # outputs_decodedb = tiny_brain_io.tokenizer.decode(outputs[0].logits.argmax(2)[0], skip_special_tokens=True)

        # print(f"Outputs decodedb: {outputs_decodedb}")

        # perplexity = torch.exp(outputs[0])
        # print(f"Perplexity: {perplexity}")
        words = query.split()
        num_words = len(words)
        avg_word_length = sum(len(word) for word in words) / num_words

        # Calculate the complexity of the sentence using the Flesch-Kincaid Grade Level
        # https://en.wikipedia.org/wiki/Flesch%E2%80%93Kincaid_readability_tests
        # Range: 0-100
        # 90-100: 5th grade
        # 80-90: 6th grade
        # 70-80: 7th grade
        # 60-70: 8th & 9th grade
        # 50-60: 10th to 12th grade
        # 30-50: College
        # 0-30: College graduate

        flesch_kincaid_grade_level = (
            0.39 * (num_words / 1) + 11.8 * (avg_word_length / 1) - 15.59
        )
        print(f"Flesch-Kincaid Grade Level: {flesch_kincaid_grade_level}")

        # Calculate the complexity of the sentence using the Gunning Fog Index
        # https://en.wikipedia.org/wiki/Gunning_fog_index
        # Range: 0-20
        # 6: 6th grade
        # 8: 8th grade
        # 12: 12th grade
        # 16: College
        # 20: College graduate
        gunning_fog_index = 0.4 * (
            (num_words / 1)
            + 100 * (sum([1 for word in words if len(word) > 3]) / num_words)
        )
        print(f"Gunning Fog Index: {gunning_fog_index}")

        # Calculate the complexity of the sentence using the Coleman-Liau Index
        # https://en.wikipedia.org/wiki/Coleman%E2%80%93Liau_index
        # Range of coleman_liau_index: 1-12
        # 1-6: 6th grade
        # 7-8: 7th & 8th grade
        # 9-12: 9th to 12th grade
        coleman_liau_index = 5.89 * (num_words / 1) - 29.6 * (1 / 1) - 15.8
        print(f"Coleman-Liau Index: {coleman_liau_index}")

        # Calculate the complexity of the sentence using the Automated Readability Index
        # https://en.wikipedia.org/wiki/Automated_readability_index
        # Range: 1-14
        # 1-3: 5-8 year old
        # 4-5: 9-10 year old
        # 6-7: 11-12 year old
        # 8-9: 13-15 year old
        # 10-12: 16-17 year old
        # 13-14: College
        automated_readability_index = (
            4.71 * (sum([1 for word in words if word.isalpha()]) / num_words)
            + 0.5 * (num_words / 1)
            - 21.43
        )
        print(f"Automated Readability Index: {automated_readability_index}")

        # Calculate the complexity of the sentence using the Simple Measure of Gobbledygook (SMOG) Index
        # https://en.wikipedia.org/wiki/SMOG
        # Range for smog_index: 1-20
        # 1-3: 6th grade
        # 4-6: 7th & 8th grade
        # 7-9: 9th & 10th grade
        # 10-12: 11th & 12th grade
        # 13-20: College

        smog_index = 1.043 * (
            30 * (sum([1 for word in words if word.isalpha()]) / num_words) ** 0.5
            + 3.1291
        )
        print(f"SMOG Index: {smog_index}")

        # Calculate the complexity of the sentence using the Dale-Chall Readability Score
        # https://en.wikipedia.org/wiki/Dale%E2%80%93Chall_readability_formula
        # Range: 0-10
        # 4.9: 4th grade
        # 5.9: 5th grade
        # 6.9: 6th grade
        # 7.9: 7th grade
        # 8.9: 8th grade
        # 9.9: 9th grade
        # 10.9: 10th grade

        dale_chall_readability_score = 0.1579 * (
            100
            * (
                sum(
                    [
                        1
                        for word in words
                        if word not in sentiment_analysis.extract_simple_words(words)
                    ]
                )
                / num_words
            )
        ) + 0.0496 * (num_words / 1)
        print(f"Dale-Chall Readability Score: {dale_chall_readability_score}")

        # Calculate the complexity of the sentence using the Linsear Write Formula
        # https://en.wikipedia.org/wiki/Linsear_Write
        # Range of linsear_write_formula: 0-20
        # 0-5: 5th grade
        # 6-10: 6th to 8th grade
        # 11-15: 9th to 12th grade
        # 16-20: College

        linsear_write_formula = (
            sum(
                [
                    1
                    for word in words
                    if word in sentiment_analysis.extract_simple_words(words)
                ]
            )
            + sum(
                [
                    3
                    for word in words
                    if word not in sentiment_analysis.extract_simple_words(words)
                ]
            )
        ) / num_words
        print(f"Linsear Write Formula: {linsear_write_formula}")

        # Calculate the complexity of the sentence using the Spache Readability Formula
        # https://en.wikipedia.org/wiki/Spache_Readability_Formula
        # Range of spache_readability_formula: 0-100
        # 0-30: 4th grade
        # 30-50: 5th to 6th grade
        # 50-60: 7th to 8th grade
        # 60-70: 9th to 12th grade
        # 70-80: College
        # 80-100: College graduate
        spache_readability_formula = (
            0.121
            * (
                sum(
                    [
                        1
                        for word in words
                        if word in sentiment_analysis.extract_simple_words(words)
                    ]
                )
                / num_words
            )
            + 0.082 * (num_words / 1)
            + 0.659
        )
        print(f"Spache Readability Formula: {spache_readability_formula}")

        # Determine the depth of the parse tree for the sentence
        parse_tree_depth = max([len(list(sent.subtree)) for sent in docs.sents])

        # Determine the number of entities in the sentence
        num_entities = len([ent for ent in docs.ents])

        # Determine the number of noun phrases in the sentence
        num_noun_phrases = len([chunk for chunk in docs.noun_chunks])

        # Determine the number of sentences in the input
        num_sentences = len([sent for sent in docs.sents])

        # Determine the number of subordinate clauses in the input
        num_subordinate_clauses = len([token for token in docs if token.dep_ == "mark"])

        # Determine the number of T-units in the input
        num_t_units = sum(
            1
            for sent in docs.sents
            for token in sent
            if token.tag_ in ["VBD", "VBZ", "VBP", "VBG", "VBN"]
        )
        # Determine the number of constituents in the input
        constituents = [
            token
            for token in docs
            if token.dep_
            in [
                "nsubj",
                "nsubjpass",
                "dobj",
                "iobj",
                "pobj",
                "attr",
                "ccomp",
                "xcomp",
                "acomp",
                "advcl",
                "advmod",
                "neg",
                "npadvmod",
                "prep",
                "pcomp",
                "agent",
                "csubj",
                "csubjpass",
                "expl",
                "poss",
                "possessive",
                "prt",
                "case",
                "amod",
                "appos",
                "nummod",
                "compound",
                "nmod",
                "nmod:npmod",
                "nmod:tmod",
                "nmod:poss",
                "acl",
                "acl:relcl",
                "det",
                "predet",
                "preconj",
                "infmod",
                "partmod",
                "advcl",
                "purpcl",
                "tmod",
                "npmod",
                "rcmod",
                "quantmod",
                "parataxis",
                "list",
                "dislocated",
                "ref",
                "orphan",
                "conj",
                "cc",
                "root",
                "dep",
                "mark",
                "punct",
                "conj",
                "cc",
                "root",
                "dep",
                "mark",
                "punct",
            ]
        ]
        num_constituents = len(constituents)
        constituent_dependencies = [constituent.dep_ for constituent in constituents]

        # Determine the number of crossing dependencies in the input
        num_crossing_dependencies = 0
        crossing_dependencies = []
        for i in range(len(constituents)):
            for j in range(i + 1, len(constituents)):
                if set(constituents[i].lefts).intersection(
                    set(constituents[j].rights)
                ) or set(constituents[i].rights).intersection(
                    set(constituents[j].lefts)
                ):
                    crossing_dependencies.append((constituents[i], constituents[j]))
                    num_crossing_dependencies += 1

        # Determine the number of function words in the input
        num_function_words = len([token for token in docs if token.is_stop])

        # Determine the number of content words in the input
        num_content_words = len([token for token in docs if not token.is_stop])

        # Determine the MLU (mean length of utterance) in the input
        mlu = num_words / num_sentences

        # Use parse_tree_depth, num_entities, num_noun_phrases, num_sentences, num_subordinate_clauses, num_t_units, num_constituents, num_crossing_dependencies, num_function_words, num_content_words, and mlu to determine the complexity of the input
        complexity = 0
        if parse_tree_depth > 5:
            complexity += 1
        if num_entities > 3:
            complexity += num_entities // 3
        if num_noun_phrases > 5:
            complexity += num_noun_phrases // 5
        if num_sentences > 1:
            complexity += num_sentences // 2
        if num_subordinate_clauses > 2:
            complexity += num_subordinate_clauses // 2
        if num_t_units > 2:
            complexity += num_t_units // 2
        if num_constituents > 5:
            complexity += num_constituents // 5
        if num_crossing_dependencies > 5:
            complexity += num_crossing_dependencies // 5
        if num_function_words > 20:
            complexity += num_function_words // 20
        if num_content_words > 5:
            complexity += num_content_words // 5
        if mlu > 5:
            complexity += mlu // 5

        # Calculate the lexical density of the input
        lexical_density = num_content_words / num_words
        print(f"\n Lexical Density: {lexical_density}")

        # Calculate the type-token ratio of the input
        type_token_ratio = len(set([token.text for token in docs])) / num_words

        # Find the most and least frequent words in the input
        word_frequencies = {
            token.text: token.prob for token in docs if token.prob != -1
        }
        most_frequent_word = max(word_frequencies, key=word_frequencies.get)
        least_frequent_word = min(word_frequencies, key=word_frequencies.get)

        # Calculate the average word length of the input
        avg_word_length = sum([len(token.text) for token in docs]) / num_words

        # Calculate the lexical diversity of the input
        lexical_diversity = len(set([token.text for token in docs])) / num_words
        print(f"\n Lexical Diversity: {lexical_diversity}")

        # Calculate the use of passive voice in the input
        passive_voice = len([token for token in docs if token.dep_ == "nsubjpass"])
        # Use passive_voice to determine the subject and object of the passive voice
        passive_voice_details = {}
        print(f"\n Passive Voice: {passive_voice}")
        for token in docs:
            if token.dep_ == "nsubjpass":
                subject = token.text
                object = token.head.text
                print(f"Passive Voice: Subject: {subject}, Object: {object} \n")
                passive_voice_details[subject] = object

        # Calculate the use of active voice in the input
        active_voice = len([token for token in docs if token.dep_ == "nsubj"])
        # Use active_voice to determine the subject and object of the active voice
        active_voice_details = {}
        print(f"\n Active Voice: {active_voice}")
        for token in docs:
            if token.dep_ == "nsubj":
                subject = token.text
                object = token.head.text
                print(f"Active Voice: Subject: {subject}, Object: {object} \n")
                active_voice_details[subject] = object

        # Calculate the use of modals in the input
        modals = [token for token in docs if token.tag_ == "MD"]
        modal_len = len(modals)
        modal_frequency = sum([token.prob for token in modals if token.prob != -1])

        # Calculate the use of adverbs in the input
        adverbs = [token for token in docs if token.pos_ == "ADV"]
        adverb_len = len(adverbs)
        adverb_frequency = sum([token.prob for token in adverbs if token.prob != -1])

        # Calculate the use of adjectives in the input
        adjectives = [token for token in docs if token.pos_ == "ADJ"]
        adjective_len = len(adjectives)
        adjective_frequency = sum(
            [token.prob for token in adjectives if token.prob != -1]
        )

        # Calculate the use of pronouns in the input
        pronouns = [token for token in docs if token.pos_ == "PRON"]
        pronoun_len = len(pronouns)
        pronoun_frequency = sum([token.prob for token in pronouns if token.prob != -1])

        # Calculate the use of conjunctions in the input
        conjunctions = [token for token in docs if token.pos_ == "CCONJ"]
        conjunction_len = len(conjunctions)
        conjunction_frequency = sum(
            [token.prob for token in conjunctions if token.prob != -1]
        )

        # Calculate the use of determiners in the input
        # determiners = [token for token in docs if token.pos_ == "DET"]

        determiners = {}
        print(f"\n Determiners: {determiners}")
        for token in docs:
            if token.pos_ == "DET":
                determiner = token.text
                associated_noun = token.head.text
                definiteness = (
                    "Definite" if token.text.lower() in ["the"] else "Indefinite"
                )
                quantity = (
                    "Singular" if token.text.lower() in ["a", "an", "the"] else "Plural"
                )
                possession = (
                    "Yes"
                    if token.text.lower()
                    in ["my", "your", "his", "her", "its", "our", "their"]
                    else "No"
                )
                print(
                    f"\n Determiner: {determiner}, Associated Noun: {associated_noun}, Definiteness: {definiteness}, Quantity: {quantity}, Possession: {possession}"
                )
                determiners[determiner] = {
                    "associated_noun": associated_noun,
                    "definiteness": definiteness,
                    "quantity": quantity,
                    "possession": possession,
                }

        for determiner, details in determiners.items():
            print(f"\n Determiner: {determiner}, Details: {details}")
            if determiner in self.map_tags:
                self.map_tags[determiner].append(details)

        for token in docs:

            if token.pos_ == "ADV" and token.text.lower() in [
                "now",
                "immediately",
                "urgently",
            ]:
                is_urgent = True
            if token.pos_ == "VERB" and token.tag_ == "VBZ":  # Imperative verbs
                is_urgent = True
            if token.pos_ in ["INTJ", "ADV"] or token.dep_ in ["prt"]:
                if token.text.lower() in urgent_words:
                    is_urgent = True
            if token.pos_ == "INTJ":  # Interjections are strong indicators
                is_urgent = True

        # Calculate the use of prepositions in the input
        # Example of how to structure prepositions
        prepositions = [
            (token, token.dep_, token.head) for token in docs if token.pos_ == "ADP"
        ]
        print(f"\n Prepositions: {prepositions}")
        preposition_len = len(prepositions)

        # Calculate the use of interjections in the input
        interjections = [token for token in docs if token.pos_ == "INTJ"]
        interjection_len = len(interjections)
        interjection_frequency = sum(
            [token.prob for token in interjections if token.prob != -1]
        )

        # Calculate the use of particles in the input
        particles = [token for token in docs if token.pos_ == "PART"]
        particle_len = len(particles)
        particle_frequency = sum(
            [token.prob for token in particles if token.prob != -1]
        )

        # Calculate the use of punctuations in the input
        punctuations = [token for token in docs if token.pos_ == "PUNCT"]
        punctuation_len = len(punctuations)
        punctuation_frequency = sum(
            [token.prob for token in punctuations if token.prob != -1]
        )

        # Calculate the use of symbols in the input
        symbols = [token for token in docs if token.pos_ == "SYM"]
        symbol_len = len(symbols)
        symbol_frequency = sum([token.prob for token in symbols if token.prob != -1])

        # Calculate the use of numbers in the input
        numbers = [token for token in docs if token.pos_ == "NUM"]
        number_len = len(numbers)
        number_frequency = sum([token.prob for token in numbers if token.prob != -1])

        # Calculate the use of foreign words in the input
        foreign_words = [token for token in docs if token.pos_ == "X"]
        foreign_word_len = len(foreign_words)
        foreign_word_frequency = sum(
            [token.prob for token in foreign_words if token.prob != -1]
        )

        # Calculate the use of proper nouns in the input
        proper_nouns = [token for token in docs if token.pos_ == "PROPN"]
        proper_noun_len = len(proper_nouns)
        proper_noun_frequency = sum(
            [token.prob for token in proper_nouns if token.prob != -1]
        )

        # Calculate the use of common nouns in the input
        common_nouns = [token for token in docs if token.pos_ == "NOUN"]
        common_noun_len = len(common_nouns)
        common_noun_frequency = sum(
            [token.prob for token in common_nouns if token.prob != -1]
        )

        # Calculate the use of verbs in the input
        verbs = [token for token in docs if token.pos_ == "VERB"]
        verb_len = len(verbs)
        verb_frequency = sum([token.prob for token in verbs if token.prob != -1])

        # Calculate the use of adpositions in the input
        adpositions = [token for token in docs if token.pos_ == "ADP"]
        adposition_len = len(adpositions)
        adposition_frequency = sum(
            [token.prob for token in adpositions if token.prob != -1]
        )

        # Calculate the use of adverbs in the input
        adverbs = [token for token in docs if token.pos_ == "ADV"]
        adverb_len = len(adverbs)
        adverb_frequency = sum([token.prob for token in adverbs if token.prob != -1])

        # Calculate the use of auxiliaries in the input
        auxiliaries = [token for token in docs if token.pos_ == "AUX"]
        auxiliary_len = len(auxiliaries)
        auxiliary_frequency = sum(
            [token.prob for token in auxiliaries if token.prob != -1]
        )

        # Calculate the use of conjunctions in the input
        conjunctions = [token for token in docs if token.pos_ == "CCONJ"]
        conjunction_len = len(conjunctions)
        conjunction_frequency = sum(
            [token.prob for token in conjunctions if token.prob != -1]
        )

        # Calculate the use of advanced vocabulary in the input
        advanced_vocabulary = [token for token in docs if token.prob > -15]
        advanced_vocabulary_len = len(advanced_vocabulary)
        advanced_vocabulary_frequency = sum(
            [token.prob for token in advanced_vocabulary if token.prob != -1]
        )

        # Calculate the use of simple vocabulary in the input
        simple_vocabulary = [token for token in docs if token.prob < -15]
        simple_vocabulary_len = len(simple_vocabulary)
        simple_vocabulary_frequency = sum(
            [token.prob for token in simple_vocabulary if token.prob != -1]
        )
        # Use modal_len and modal_frequency to determine the modality of the input
        modal_details = {}

        # analyze the relationships and modality
        # 1. Modality Analysis
        for modal in modals:
            modal_types = {
                "can": "Ability",
                "could": "Ability",
                "may": "Possibility",
                "might": "Possibility",
                "shall": "Obligation",
                "should": "Obligation",
                "will": "Prediction",
                "would": "Prediction",
                "must": "Necessity",
            }
            modality_type = modal_types[modal.text]
            modal_details[modal] = {
                "type": modality_type,
                "subject": modal.nbor(-1).text if modal.dep_ == "aux" else "None",
                "verb": modal.head.text,
                "object": [w.text for w in modal.head.rights if w.dep_ == "dobj"],
            }
            if modality_type == "Necessity":
                is_urgent = True

        # 2. Subject-Verb Relationship, subject_verb_relationships = {subject: [verb1, verb2, ...]}
        subject_verb_relationships = {}
        for key, value in modal_details.items():
            if value["subject"] not in subject_verb_relationships:
                subject_verb_relationships[value["subject"]] = []
            subject_verb_relationships[value["subject"]].append(value["verb"])
            print(
                f"Subject-Verb Relationship: {value['subject']}, Verb: {value['verb']}"
            )

        # 3. Verb-Object Relationship
        verb_object_relationships = {}
        for key, value in modal_details.items():
            if value["verb"] not in verb_object_relationships:
                verb_object_relationships[value["verb"]] = []
            verb_object_relationships[value["verb"]].append(value["object"])
            print(
                f"Verb-Object Relationship: {value['verb']}, Object: {value['object']}"
            )

        # 4. Subject-Object Relationship
        subject_object_relationships = {}
        for key, value in modal_details.items():
            if value["subject"] not in subject_object_relationships:
                subject_object_relationships[value["subject"]] = []
            subject_object_relationships[value["subject"]].append(value["object"])
            print(
                f"Subject-Object Relationship: {value['subject']}, Object: {value['object']}"
            )

        # 4. Semantic Role Labeling
        semantic_roles = {}

        for token in docs:
            if token.dep_ != "ROOT":
                if token.head.text not in semantic_roles:
                    semantic_roles[token.head.text] = []
                semantic_roles[token.head.text].append({token.text: token.dep_})
                print(f"Semantic Role: {token.text}, Role: {token.dep_}")
        print(f"\n Semantic Roles: {semantic_roles}")
        # 5. Named Entity Recognition
        named_entities = {}

        for ent in docs.ents:
            if ent.label_ not in named_entities:
                named_entities[ent.label_] = []
            named_entities[ent.label_].append(ent.text)
            print(f"Named Entity: {ent.text}, Label: {ent.label_}")
        print(f"\n Named Entities: {named_entities}")

        # # 6. Coreference Resolution
        # coreferences = {}
        # for cluster in docs._.coref_clusters:
        #     coreferences[cluster.main.text] = [mention.text for mention in cluster.mentions]
        #     print(f"Coreference: {cluster.main.text}, Mentions: {[mention.text for mention in cluster.mentions]}")

        # 7 Dependency Parsing
        dependency_tree = {}
        for token in docs:
            if token.dep_ != "ROOT":
                if token.head.text not in dependency_tree:
                    dependency_tree[token.head.text] = []
                dependency_tree[token.head.text].append({token.text: token.dep_})
                print(f"Dependency: {token.text}, Relation: {token.dep_}")

        # # 8. Constituency Parsing
        # constituency_tree = {}
        # for token in docs:
        #     if token._.constituents:
        #         if token.text not in constituency_tree:
        #             constituency_tree[token.text] = []
        #         constituency_tree[token.text].append(token._.constituents)
        #         print(f"Constituency: {token.text}, Constituents: {token._.constituents}")

        # 9. Semantic Parsing
        # semantic_tree = {}
        # for token in docs:
        #     if token._.semantics:
        #         if token.text not in semantic_tree:
        #             semantic_tree[token.text] = []
        #         semantic_tree[token.text].append(token._.semantics)
        #         print(f"Semantic: {token.text}, Semantics: {token._.semantics}")

        # # 10. Anaphora Resolution
        # anaphoras = {}
        # for token in docs:
        #     if token._.is_anaphora:
        #         if token.text not in anaphoras:
        #             anaphoras[token.text] = []
        #         anaphoras[token.text].append(token._.is_anaphora)
        #         print(f"Anaphora: {token.text}, Is Anaphora: {token._.is_anaphora}")

        # # 11. Ellipsis Resolution
        # ellipses = {}
        # for token in docs:
        #     if token._.is_ellipsis:
        #         if token.text not in ellipses:
        #             ellipses[token.text] = []
        #         ellipses[token.text].append(token._.is_ellipsis)
        #         print(f"Ellipsis: {token.text}, Is Ellipsis: {token._.is_ellipsis}")

        # 12. Temporal and Aspectual Analysis
        # temporal_aspectual_analysis = {}
        # for token in docs:
        #     if token.pos_ == "VERB":
        #         print(f"Verb Token: {token.text}, Tense: {token._.tense}, Aspect: {token._.aspect}, Mood: {token._.mood}, Polarity: {token._.polarity}, Voice: {token._.voice}")
        #         temporal_aspectual_analysis[token.text] = {'tense': token._.tense, 'aspect': token._.aspect, 'mood': token._.mood, 'polarity': token._.polarity, 'voice': token._.voice}
        # Define patterns for temporal expressions
        date_pattern = [{"SHAPE": "dddd"}]  # Matches a four-digit year, e.g., 2019
        future_pattern = [
            {"LEMMA": "will"},
            {"POS": "VERB"},
        ]  # Matches future tense, e.g., will finish

        # Initialize the Matcher and add the patterns
        matcher = Matcher(nlp.vocab)
        matcher.add("DATE_PATTERN", [date_pattern])
        matcher.add("FUTURE_PATTERN", [future_pattern])

        # Find matches in the doc
        matches = matcher(docs)

        # Create a function to categorize verb aspects
        verb_aspects = {}

        # Iterate over matches and print results
        spans = {}
        for match_id, start, end in matches:
            span = Span(docs, start, end)
            print(f"Temporal expression found: {span.text}")
            spans[span.text] = span
        print(f"\n Temporal Expressions: {spans}")

        # Use span to determine the temporal expression of the input

        # Iterate over tokens to identify verb aspects

        for token in verbs:
            if token.tag_ == "VBG":  # Gerund or present participle
                aspect = "progressive"
            elif token.tag_ == "VBN":  # Past participle
                aspect = "perfect"
            elif token.tag_ == "VBD":  # Simple past
                aspect = "simple past"
            else:
                aspect = "simple present"
            verb_aspects[token.text] = aspect

        for verb, aspect in verb_aspects.items():
            print(f"Verb: {verb}, Aspect: {aspect} \n")
        #     #Use verb and aspect to determine the aspect of the input

        for interjection in interjections:
            if interjection.lower in [
                "yay",
                "wow",
                "hooray",
                "bravo",
                "huzzah",
                "kudos",
                "hurray",
                "eureka",
                "bingo",
                "hallelujah",
                "phew",
                "yippee",
                "cheers",
                "hurrah",
                "whoop",
                "yeehaw",
                "yowza",
                "woohoo",
                "yes",
                "awesome",
                "fantastic",
                "amazing",
                "cool",
                "excellent",
                "great",
                "splendid",
                "wonderful",
                "right on",
                "good job",
                "well done",
            ]:
                intent = "Positive"
            if interjection.lower in [
                "oh",
                "oops",
                "uh-oh",
                "whoops",
                "ouch",
                "darn",
                "drat",
                "rats",
                "shoot",
                "blast",
                "buggar",
                "bother",
                "gosh",
                "golly",
                "gee",
                "dang",
                "dangit",
                "ugh",
                "oh no",
                "ouch",
                "yikes",
                "oops",
                "darn",
                "oh dear",
                "no way",
                "argh",
                "drat",
                "jeez",
                "phooey",
                "rats",
                "uh-oh",
                "yuck",
                "eww",
                "huh",
                "bah",
                "humbug",
                "bother",
                "fiddlesticks",
                "good grief",
                "oh my",
                "oh boy",
                "oh brother",
            ]:
                intent = "Negative"
                is_urgent = True

        # Entity and Relationship Extraction (More Detailed)
        entities = []
        relationships = {}
        for prep, relation, head in prepositions:
            print(
                f"\n Preposition: {prep.text}, Relation: {relation}, Head: {head.text}"
            )
            if relation == "pobj":  # Object of preposition
                if head.dep_ == "nsubj":  # Subject of the verb
                    entities.append(head.text)  # Subject of the verb
                    relationships[head.text] = (
                        prep.text
                    )  # head.text is the possessor, prep.text is the relationship
                elif head.dep_ == "dobj":  # Direct object of the verb
                    entities.append(head.text)  # Direct object of the verb
                    relationships[head.text] = (
                        prep.text
                    )  # head.text is the object, prep.text is the relationship

            elif relation == "prep":  # Covers cases beyond just objects
                # Example: '...belonging to Sarah' -> Sarah (possessor), belonging to (relationship)
                entities.append(head.text)

        print(f"\n Entities: {entities}, Relationships: {relationships} \n")

        main_subject = (
            next(iter(named_entities.values()), " ")[0] if named_entities else " "
        )
        main_object = None
        main_verb = None
        print(f"\n Named Entities: {named_entities}")
        print(f"\n Main Subject: {main_subject}")

        previous_token = None
        for token in docs:
            if token.dep_ == "ROOT":
                main_verb = token.text
            if "nsubj" in token.dep_ and (
                previous_token and previous_token.dep_ == "compound"
            ):
                print(f"token text: {token.text}")
                main_subject = f"{previous_token.text} {token.text}"
            elif token.dep_ == "nsubj":  # Subject of the verb
                main_subject = token.text
            if token.dep_ == "dobj":
                main_object = token.text
            previous_token = token if token.dep_ == "compound" else None
        print(f"\n Main Subject: {main_subject}")
        print(
            f"\n Main Subject: {main_subject.strip()}, Main Verb: {main_verb}, Main Object: {main_object} \n"
        )

        # Check if main_subject is in named_entities
        if main_subject:
            main_subject_lower = main_subject.lower()
            for entity in named_entities:
                entity_lower = entity.lower()
                if (
                    main_subject_lower in entity_lower
                    or entity_lower in main_subject_lower
                ):
                    main_subject = entity
        print(f"\n Main Subject: {main_subject}")

        # 5. Verb-Adjective Relationship

        # 6. Verb-Preposition Relationship

        # 7. Verb-Particle Relationship

        # 8. Verb-Adposition Relationship

        # Use the lexical_density, type_token_ratio, word_frequency, word_length, lexical_diversity, passive_voice, active_voice, modals, adverbs, adjectives, pronouns, conjunctions, determiners, prepositions, interjections, particles, punctuations, symbols, numbers, foreign_words, proper_nouns, common_nouns, verbs, adpositions, adverbs, auxiliaries, conjunctions, advanced_vocabulary, and simple_vocabulary to determine the complexity of the input
        complexity += lexical_density * 10
        # complexity += type_token_ratio * 10
        complexity += int(avg_word_length)

        complexity += lexical_diversity * 10

        # if adverbs > 0.5:
        #     complexity += 1
        # if adjectives > 0.5:
        #     complexity += 1
        # if pronouns > 0.5:
        #     complexity += 1
        # if conjunctions > 0.5:
        #     complexity += 1
        # if determiners > 0.5:
        #     complexity += 1
        # if prepositions > 0.5:
        #     complexity += 1
        # if interjections > 0.5:
        #     complexity += 1
        # if particles > 0.5:
        #     complexity += 1
        # if punctuations > 0.5:
        #     complexity += 1
        # if symbols > 0.5:
        #     complexity += 1
        # if numbers > 0.5:
        #     complexity += 1
        # if foreign_words > 0.5:
        #     complexity += 1
        # if proper_nouns > 0.5:
        #     complexity += 1

        # print(f"response: {response}")
        # perplexity = torch.exp(response[0][1])
        keywords = self.extract_keywords(query)
        print(f"\n Keywords: {keywords}\n ")
        # hs_decoded = tiny_brain_io.tokenizer.decode(outputs[0][-1][0].argmax(1), skip_special_tokens=True)
        # print(f"Hidden states decoded: {hs_decoded}")
        # print(f"\n Shape of hidden states: {outputs[-1].shape} \n ")
        # query_embedding = mean_pooling(outputs[0].cpu().detach(), outputs[1].cpu().detach())
        query_embedding = outputs
        print(f"Shape of query embedding: {query_embedding.shape}")
        # kmeans = KMeans(n_clusters=2)

        # kmeans.fit(outputs[0][-1].cpu().detach().numpy())
        # print(f"Kmeans labels: {kmeans.labels_}")
        # print(f"Kmeans cluster centers: {kmeans.cluster_centers_}")
        # # Step 3: Create a transition model
        # G = nx.DiGraph()
        # prev_cluster = None

        # for word in words:
        #     cluster = kmeans.predict([query_embedding])[0]
        #     if prev_cluster is not None:
        #         if G.has_edge(prev_cluster, cluster):
        #             G[prev_cluster][cluster]["weight"] += 1
        #         else:
        #             G.add_edge(prev_cluster, cluster, weight=1)
        #     prev_cluster = cluster

        # # Step 4: Measure complexity (e.g., by looking at the number of transitions)
        # complexity = sum([data["weight"] for _, _, data in G.edges(data=True)])
        # print(f"Complexity: {complexity}")

        # Check cosine similiarity of query with recent queries
        if (
            self.faiss_index_recent_queries_flatl2 is None
            and len(self.recent_queries) > 0
        ):
            self.index_recent_queries_flatl2()
            if self.faiss_index_recent_queries_flatl2 is not None:

                dists, indices = self.faiss_index_recent_queries_flatl2.search(
                    query_embedding.cpu().detach().numpy(), k=1
                )

                for dist, index in zip(dists, indices):
                    print(f"Distance: {dist}, Index: {index}")

                if dists[0][0] < 0.1:
                    print(f"Query is similar to recent queries \n")
                    is_common_query = True

        # reading_level =

        ambiguity_score = complexity / 10
        sentiment_score = self.get_sentiment_score(query)
        emotion_classification = self.get_emotion_classification(query)
        return {
            "features": features,
            "embedding": query_embedding,
            "sentiment_score": sentiment_score,
            "emotion_classification": emotion_classification,
            "ambiguity_score": ambiguity_score,
            "keywords": keywords,
            "text": query,
            "main_subject": main_subject,
            "main_verb": main_verb,
            "main_object": main_object,
            "named_entities": named_entities,
            "temporal_expressions": spans,
            "verb_aspects": verb_aspects,
            "complexity": complexity,
            "is_common_query": is_common_query,
            "is_urgent": is_urgent,
            "lexical_density": lexical_density,
            "type_token_ratio": type_token_ratio,
            "avg_word_length": avg_word_length,
            "lexical_diversity": lexical_diversity,
            "passive_voice": passive_voice,
            "active_voice": active_voice,
            "modals": modals,
            "determiners": determiners,
            "semantic_roles": semantic_roles,
            "dependencies": dependency_tree,
            "relationships": relationships,
            "proper_nouns": proper_nouns,
            "common_nouns": common_nouns,
            "verbs": verbs,
            "adpositions": adpositions,
            "adverbs": adverbs,
            "auxiliaries": auxiliaries,
            "conjunctions": conjunctions,
            "advanced_vocabulary": advanced_vocabulary,
            "simple_vocabulary": simple_vocabulary,
            "numbers": numbers,
            "symbols": symbols,
            "punctuations": punctuations,
            "particles": particles,
            "interjections": interjections,
            "prepositions": prepositions,
            "conjunctions": conjunctions,
            "pronouns": pronouns,
            "adjectives": adjectives,
            "adverbs": adverbs,
            "word_frequency": word_frequencies,
        }

    def is_relevant_flat_memory(self, flat_memory, query_features):
        print(
            f"\n Checking relevance of flat memory: {flat_memory.description} to query: {query_features.text} \n"
        )
        # Simpler logic compared to hierarchical/graph-based layers
        # Example: Check for keyword or category match, or use a basic text similarity measure

        # Extract keywords or categories from the memory and the query
        if flat_memory.keywords is None or len(flat_memory.keywords) == 0:
            print(
                f"\n Extracting keywords for flat memory: {flat_memory.description} \n"
            )
            if (
                flat_memory.analysis["keywords"] is None
                or len(flat_memory.analysis["keywords"]) == 0
            ):
                flat_memory.keywords = self.extract_keywords(flat_memory.description)
            else:
                flat_memory.keywords = flat_memory.analysis["keywords"]
            flat_memory.keywords.extend(flat_memory.named_entities)
            flat_memory.keywords.extend(flat_memory.analysis["keywords"])
            flat_memory.keywords = list(set(flat_memory.keywords))
        query_keywords = query_features.analysis["keywords"]
        query_keywords.extend(query_features.analysis["named_entities"])

        stemmer = PorterStemmer()
        query_keywords = [stemmer.stem(word) for word in query.keywords]
        flat_memory_keywords = [stemmer.stem(word) for word in flat_memory.keywords]
        # Check for overlaps or similarity
        return (
            True
            if any(keyword in query_keywords for keyword in flat_memory_keywords)
            else False
        )

    # or use a simple text similarity measure

    def get_common_memories(self, key):
        return self.flat_access.get_common_memories(key)

    def retrieve_memories_based_on_importance(
        self, general_memory, min_importance, max_importance
    ):
        # Example method to demonstrate retrieving memories within a range of importance scores
        def visit_func(specific_memory):
            if min_importance <= specific_memory.importance_score <= max_importance:
                matching_memories.append(specific_memory)

        matching_memories = []
        general_memory.in_order_traverse(
            general_memory.specific_memories_root, visit_func
        )
        return matching_memories

    def recall_sorted_specific_memories(self, general_memory):
        # This method demonstrates how to call get_all_specific_memories
        sorted_specific_memories = general_memory.get_all_specific_memories()
        # Process or return the sorted specific memories as needed
        return sorted_specific_memories

    def retrieve_memories(self, query, urgent=False, common=False, key=None):
        if query.analysis is None:
            print(f"\n Analyzing query: {query.query}")
            query.analysis = self.analyze_query_context(query.query)
            if query.query_embedding is None:
                query.query_embedding = query.analysis["embedding"]
            for k, value in query.analysis.items():
                if (
                    k != "embedding"
                    and value is not None
                    and value != {}
                    and value != []
                    and value != ""
                ):
                    print(f"Key: {k}, Value: {value}")
        self.recent_queries.append(query)
        if urgent or common:
            print(f"\n Retrieving urgent memories: {urgent}, common memories: {common}")
            return (
                self.flat_access.get_urgent_query_memories()
                if urgent
                else (
                    self.flat_access.get_common_memories(key)
                    if common and key is not None
                    else self.retrieve_from_flat_access(query)
                )
            )
        if not self.is_complex_query(query):
            return self.retrieve_from_flat_access(query)
        else:
            return self.retrieve_from_hierarchy(query)

    def is_complex_query(self, query_features):
        # Analyze query features to determine complexity
        length_threshold = 20  # Threshold for query length
        specificity_threshold = 0.7  # Threshold for keyword specificity
        polarity_threshold = 0.5  # Threshold for polarized content
        subjectivity_threshold = 0.5  # Threshold for subjective content
        ambiguity_threshold = 10  # Threshold for ambiguity
        if query_features.analysis is None:
            query_features.analysis = self.analyze_query_context(query_features.text)

        query_embedding = query_features.analysis["embedding"]
        if (
            self.faiss_index_recent_queries_flatl2 is None
            and len(self.recent_queries) > 0
        ):
            self.index_recent_queries_flatl2()
        if self.faiss_index_recent_queries_flatl2 is not None:
            dists, indices = self.faiss_index_recent_queries_flatl2.search(
                query_embedding.cpu().detach().numpy(), k=1
            )
            if dists[0][0] < 0.1:
                print(f"Query is similar to recent queries")
                return False

        is_long = len(query_features.analysis["text"].split()) > length_threshold
        if is_long:
            print(f"Query is long")
            return True
        has_specific_keywords = any(
            self.keyword_specificity(keyword) > specificity_threshold
            for keyword in query_features.analysis["keywords"]
        )
        if has_specific_keywords:
            print(f"Query has specific keywords")
            return True
        is_ambiguous = query_features.analysis["ambiguity_score"] > ambiguity_threshold
        if is_ambiguous:
            print(f"Query is ambiguous {query_features.analysis['ambiguity_score']}")
            return True

        is_polarized = (
            query_features.analysis["sentiment_score"]["polarity"] > polarity_threshold
        )
        if is_polarized:
            print(f"Query is polarized")
            return True

        is_subjective = (
            query_features.analysis["sentiment_score"]["subjectivity"]
            > subjectivity_threshold
        )

        if is_subjective:
            print(f"Query is subjective")
            return True

        is_emotional = query_features.analysis["emotion_classification"] != "others"
        if is_emotional:
            print(f"Query is emotional")
            return True
        print(f"Query is not complex")
        return False

    # def cosine_similarity(vec1, vec2):
    #     return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

    def keyword_specificity(self, keyword):
        if keyword in self.complex_keywords:
            return 1
        else:
            return 0.5

    def extract_keywords(self, text):
        # Extract keywords using different methods
        stemmer = PorterStemmer
        rake_keywords = self.extract_rake_keywords(text)
        print(f"Rake keywords: {rake_keywords}")
        print(f"Type of text: {type(text)}")
        print(f"Text: {text}")
        # if isinstance(text, list):
        #     lda_keywords = self.extract_lda_keywords(text)
        # else:
        #     lda_keywords = self.extract_lda_keywords([text])
        # #remove duplicates from lda_keywords, keep the order
        # lda_keywords = list(dict.fromkeys(lda_keywords))

        # print(f"LDA keywords: {lda_keywords}")
        tfidf_keywords = self.extract_tfidf_keywords(text)
        print(f"TF-IDF keywords: {tfidf_keywords}")
        print(f"Type of tfidf_keywords: {type(tfidf_keywords)}")
        entity_keywords = self.extract_entities(text)
        (
            print(f"Entity keywords: {entity_keywords}")
            if entity_keywords
            else print(f"No entity keywords found")
        )
        print(f"Type of entity_keywords: {type(entity_keywords)}")

        # Combine keywords from different methods
        keywords = list(rake_keywords.union(tfidf_keywords).union(entity_keywords))
        # synonyms = []
        # for word in keywords:
        #     token = nlp(word)[0]
        #     wn_pos = get_wordnet_pos(token)
        #     if wn_pos:
        #         for syn in wordnet.synsets(word, pos=wn_pos):
        #             if cosine_similarity
        #             for lemma in syn.lemmas():
        #                 synonyms.append(lemma.name())
        #         keywords = keywords.union(synonyms)
        return keywords

    def search_memories(self, query, urgent=False, common=False, tag=None):

        if isinstance(query, MemoryQuery):
            query.query_time = datetime.now()
            memories = self.retrieve_memories(query, urgent, common)

        elif isinstance(query, str):
            new_query = MemoryQuery(
                query,
                query_time=datetime.now(),
                query_tags=tag,
                gametime_manager=self.gametime_manager,
            )
            memories = self.retrieve_memories(new_query, urgent, common)

        else:
            raise TypeError("Query must be a string or a MemoryQuery object")
        return memories

    def normalize_scores(self, scores):
        if isinstance(scores, (list, tuple, set, dict, np.ndarray)):
            scores = np.array(scores)
            # First, remove zero scores
            scores = scores[scores != 0]
            if len(scores) == 0:
                return 0
        elif isinstance(scores, (float, int)):
            if scores == 0:
                return 0
            else:
                return scores

        min_score = np.min(scores)
        max_score = np.max(scores)

        if max_score == min_score or max_score == 0:
            return scores
        else:
            return (scores - min_score) / (max_score - min_score)

    def search_by_tag(self, tag):
        res = []
        for mem in self.hierarchy.general_memories:
            if tag in mem.tags:
                res.append(mem)
        return res

    def search_by_index(self, general_memory, k=10):
        return general_memory.faiss_index.search(
            general_memory.description_embedding.cpu().detach().numpy(), k
        )

    def calculate_recency_score(self, memory, current_time, decay_factor=0.995):
        time_since_last_access = (
            current_time - memory.last_access_time
        ).total_seconds() / 3600  # Convert to hours
        return decay_factor**time_since_last_access


# def cosine_similarity(vec1, vec2):
#     # Normalize vectors
#     norm_vec1 = vec1 / np.linalg.norm(vec1)
#     norm_vec2 = vec2 / np.linalg.norm(vec2)
#     return np.dot(norm_vec1, norm_vec2)


# def retrieve_memories(self,memories, query_embedding, current_time):
#     recency_scores = [calculate_recency_score(memory, current_time) for memory in memories]
#     relevance_scores = [cosine_similarity(self.get_embedding(memory.description), query_embedding) for memory in memories]
#     importance_scores = [memory.importance_score for memory in memories]

#     recency_scores = normalize_scores(recency_scores)
#     relevance_scores = normalize_scores(relevance_scores)
#     importance_scores = normalize_scores(importance_scores)

#     final_scores = [recency + relevance + importance for recency, relevance, importance in zip(recency_scores, relevance_scores, importance_scores)]
#     sorted_memories = sorted(zip(final_scores, memories), key=lambda x: x[0], reverse=True)

#     return [memory for score, memory in sorted_memories if score > 0]  # Filter out zero-scored memories

# Example usage
# current_time = datetime.now()
# memories = [
#     Memory("Isabella Rodriguez is setting out the pastries", current_time - timedelta(hours=1), 5),
#     Memory("Maria Lopez is studying for a Chemistry test while drinking coffee", current_time - timedelta(hours=2), 3),
#     # More memories...
# ]

# query_embedding = get_embedding("Discussing what to study for a chemistry test")
# retrieved_memories = retrieve_memories(memories, query_embedding, current_time)

# # Display the retrieved memories
# for memory in retrieved_memories:
#     print(memory.description)
if __name__ == "__main__":
    tiny_calendar = ttm.GameCalendar()
    tiny_time_manager = ttm.GameTimeManager(tiny_calendar)
    manager = MemoryManager(tiny_time_manager)
    tiny_brain_io = tbi.TinyBrainIO("alexredna/TinyLlama-1.1B-Chat-v1.0-reasoning-v2")

    model = EmbeddingModel()

    # NOTE: The descriptions should be "Memories about [type of nounn] and [specific topic]" OR "Memories about [type of noun] and [adjectives about the noun]"
    manager.add_general_memory(
        GeneralMemory("Memories about people and food")
    ).add_specific_memory("Isabella Rodriguez is setting out the pastries.", 5)
    manager.add_general_memory(
        GeneralMemory("Memories about people and their actions")
    ).add_specific_memory(
        "Maria Lopez is studying for a Chemistry test while drinking coffee.", 3
    )

    manager.add_general_memory(
        GeneralMemory("Memories about people and events")
    ).add_specific_memory("John Doe is planning a surprise party.", 7)
    manager.add_general_memory(
        GeneralMemory("Memories about people and their actions")
    ).add_specific_memory("Jane Smith is learning to play the guitar.", 6)
    manager.add_general_memory(
        GeneralMemory("Memories about places and travel")
    ).add_specific_memory("The Eiffel Tower is a popular tourist attraction.", 8)
    manager.add_general_memory(
        GeneralMemory("Memories about people and their preferences")
    ).add_specific_memory("George told me that he loves surprise birthday parties.", 9)

    manager.add_general_memory(
        GeneralMemory("Memories about people and their jobs")
    ).add_specific_memory("Bob is a travel agent.", 10)
    manager.add_general_memory(
        GeneralMemory("Memories about people and food")
    ).add_specific_memory("Alice told me that coq au vin is a popular French dish.", 7)
    manager.add_general_memory(
        GeneralMemory("Memories about people and book products")
    ).add_specific_memory("Eve told me that Digital Fortress is a popular book.", 6)
    manager.add_general_memory(
        GeneralMemory("Memories about people and electronic products")
    ).add_specific_memory(
        "Mallory told me that Smartwatches are selling like hotcakes.", 5
    )
    manager.add_general_memory(
        GeneralMemory("Memories about people and the night life")
    ).add_specific_memory(
        "Jack told me that the Rabbit Hole is a popular nightclub.", 4
    )
    manager.add_general_memory(
        GeneralMemory("Memories about people and the night life")
    ).add_specific_memory("Jimmy had said that the bar the Bunker has poor service.", 3)
    manager.add_general_memory(
        GeneralMemory("Memories about people abd fashion")
    ).add_specific_memory("Jill told me that bowties are in fashion.", 3)
    manager.add_general_memory(
        GeneralMemory("Memories about people and their opinions on technology")
    ).add_specific_memory("Kevin told me that the electric car is the future.", 2)
    manager.add_general_memory(
        GeneralMemory("Memories about people and healthcare emergencies")
    ).add_specific_memory("Linda told me that ebola is spreading.", 1)
    manager.add_general_memory(
        GeneralMemory("Memories about people and grain prices")
    ).add_specific_memory("Michael said rice is seeing a price hike.", 0)
    manager.add_general_memory(
        GeneralMemory("Memories about people and crop sales")
    ).add_specific_memory(
        "Nancy said potatoes are cheap these days but see the most demand.", 0
    )
    manager.add_general_memory(
        GeneralMemory("Memories about people and crop sales")
    ).add_specific_memory(
        "Oliver told me that ginger root crop has the highest profit margin.", 0
    )
    manager.add_general_memory(
        GeneralMemory("Memories about people and their opinions on electronic products")
    ).add_specific_memory(
        "Charlie told me that the new iPhone is not worth the price.", 4
    )
    manager.add_general_memory(
        GeneralMemory("Memories about places of natural beauty")
    ).add_specific_memory("The Grand Canyon is a natural wonder.", 9)
    manager.add_general_memory(
        GeneralMemory("Memories about places that are historical sites")
    ).add_specific_memory("The Great Wall of China is a historic site.", 10)
    manager.add_general_memory(
        GeneralMemory("Memories about places of symbolic significance")
    ).add_specific_memory("The Statue of Liberty is a symbol of freedom.", 7)
    manager.add_general_memory(
        GeneralMemory("Memories about events and places where they occurred")
    ).add_specific_memory("The 2016 Olympics were held in Rio de Janeiro.", 9)
    manager.add_general_memory(
        GeneralMemory("Memories about events and places where they occurred")
    ).add_specific_memory("The 2018 World Cup was held in Russia.", 8)
    manager.add_general_memory(
        GeneralMemory("Memories about events and places where they occurred")
    ).add_specific_memory("The 2020 Tokyo Olympics were postponed.", 7)
    manager.add_general_memory(
        GeneralMemory("Memories about events and places where they occurred")
    ).add_specific_memory("The 2022 Winter Olympics will be held in Beijing.", 6)
    manager.add_general_memory(
        GeneralMemory("Memories about events and places where they occurred")
    ).add_specific_memory("The 2024 Summer Olympics will be held in Paris.", 5)
    manager.add_general_memory(
        GeneralMemory("Memories about events and places where they occurred")
    ).add_specific_memory("The 2026 World Cup will be held in the United States.", 4)
    manager.add_general_memory(
        GeneralMemory("Memories about events and places where they occurred")
    ).add_specific_memory("The 2028 Summer Olympics will be held in Los Angeles.", 3)
    manager.add_general_memory(
        GeneralMemory("Memories about events and places where they occurred")
    ).add_specific_memory("The 2030 Winter Olympics will be held in Vancouver.", 2)
    manager.add_general_memory(
        GeneralMemory("Memories about events and places where they occurred")
    ).add_specific_memory("The 2032 Summer Olympics will be held in Sydney.", 1)

    specific_memories_myself = [
        SpecificMemory(
            "I am planning a trip to Europe in June.", "Memories about myself", 8
        ),
        SpecificMemory("I am learning to play the guitar.", "Memories about myself", 6),
        SpecificMemory(
            "I am studying for a Chemistry test", "Memories about myself", 3
        ),
    ]
    manager.add_general_memory(
        GeneralMemory("Memories about myself")
    ).init_specific_memories(specific_memories_myself)

    print(
        f"\n \n \n Keywords for each memory: {[(mem.description, mem.keywords) for mem in manager.hierarchy.general_memories]} \n \n \n"
    )

    time.sleep(2)

    print(
        f"\n \n \n List of specific memories for each general memory: {[(mem.description, [specific_memory.description for specific_memory in mem.get_specific_memories()]) for mem in manager.hierarchy.general_memories]} \n \n \n"
    )

    time.sleep(2)

    print(
        f"\n \n \n General Memories: {[mem.description for mem in manager.hierarchy.general_memories]} \n \n \n"
    )

    # Display several views of the initial networkx graph in the hierarchy using networkx and matplotlib
    # Create a new graph that contains only the structure of the original graph
    visualization_graph = nx.Graph()

    print(f"Memory graph: {manager.hierarchy.memory_graph.nodes()}")
    node_mapping = {
        node.description: node for node in manager.hierarchy.memory_graph.nodes()
    }
    print(f"Node mapping: {node_mapping}")

    edge_colors = []
    edge_list = []
    edge_widths = []

    node_colors = []
    node_sizes = []
    descriptions = []
    # Add nodes with integer labels
    for node in manager.hierarchy.memory_graph.nodes():
        print(f"Node: {node}")
        print(f"Node description: {node.description}")
        if node.description in descriptions:
            print(
                f"\n Node description: {node.description} already in descriptions! \n "
            )
        descriptions.append(node.description)
        visualization_graph.add_node(node.description)
        if isinstance(node, GeneralMemory):
            node_colors.append("green")
            node_sizes.append(50)
        elif isinstance(node, SpecificMemory):
            node_colors.append("yellow")
            node_sizes.append(node.importance_score * 10)
        else:
            print(f"\n atypical Node type: {type(node)} \n")
            node_colors.append("black")  # default color
            node_sizes.append(50)
        visualization_graph.remove_edges_from(list(visualization_graph.edges()))
        # Add only the edges from the original graph to the new graph

    # Map old nodes to new nodes
    for node1, node2, keys, data in manager.hierarchy.memory_graph.edges(
        data="weight", keys=True, default=0
    ):
        if node1.description == node2.description:
            continue
        if (
            node1.description not in descriptions
            or node2.description not in descriptions
        ):
            print(
                f"\n Node1: {node1.description}, Node2: {node2.description} not in descriptions! \n "
            )
        if (
            visualization_graph.has_node(node1.description)
            and visualization_graph.has_node(node2.description)
            and not visualization_graph.has_edge(node1.description, node2.description)
            and not visualization_graph.has_edge(node2.description, node1.description)
            and node1.description != node2.description
            and data > 0
        ):
            visualization_graph.add_edge(
                node1.description, node2.description, weight=data
            )
            edge_list.append((node1.description, node2.description))
            print(
                f"Edge added: {node1.description} to {node2.description} with weight: {data}"
            )
        elif not visualization_graph.has_node(node1.description):
            print(f"\n Node1: {node1.description} not in visualization graph! \n ")
        elif not visualization_graph.has_node(node2.description):
            print(f"\n Node2: {node2.description} not in visualization graph! \n ")
        elif visualization_graph.has_edge(node1.description, node2.description):
            print(
                f"\n Edge between {node1.description} and {node2.description} already exists! \n "
            )
        elif visualization_graph.has_edge(node2.description, node1.description):
            print(
                f"\n Edge between {node2.description} and {node1.description} already exists! \n "
            )
        elif node1.description == node2.description:
            print(
                f"\n Node1: {node1.description} is the same as Node2: {node2.description}! \n "
            )
        elif data <= 0:
            print(
                f"\n Edge weight is 0 for {node1.description} and {node2.description}! \n "
            )

    # # Add edges with colors based on their attributes
    # for edge in visualization_graph.edges():
    #     edge_list.append(edge)
    #     #find highest attribute value (relative to the edge) and assign a color based on the attribute value that is highest

    #     edge_data = manager.hierarchy.memory_graph.get_edge_data(*edge)
    #     weight = edge_data.get('weight', 0) if edge_data else 0
    #     if weight >= 90:
    #         edge_colors.append('red')
    #     elif weight >= 80:
    #         edge_colors.append('blue')
    #     elif weight >= 70:
    #         edge_colors.append('green')
    #     elif weight >= 60:
    #         edge_colors.append('yellow')
    #     elif weight >= 50:
    #         edge_colors.append('orange')
    #     elif weight >= 40:
    #         edge_colors.append('purple')
    #     elif weight >= 30:
    #         edge_colors.append('black')
    #     elif weight >= 20:
    #         edge_colors.append('pink')
    #     elif weight >= 10:
    #         edge_colors.append('brown')
    #     elif weight >= 2:
    #         edge_colors.append('gray')
    #     else:
    #         edge_colors.append('white')
    #     edge_widths.append(edge_data.get('similarity', 0) * 2 if edge_data else 1)
    # Draw the new graph
    print(f"Number of nodes: {visualization_graph.number_of_nodes()}")
    print(f"Number of node sizes: {len(node_sizes)}")

    print(f"Number of edges: {visualization_graph.number_of_edges()}")

    nx.draw_networkx(
        visualization_graph,
        with_labels=True,
        node_color=node_colors,
        node_size=node_sizes,
        font_size=8,
        font_color="blue",
        font_weight="bold",
        pos=nx.random_layout(visualization_graph),
    )
    # plt.show()

    # Create subgraphs for each general memory and its specific memories
    # node_count = 0
    # subgraph = nx.Graph()
    # node_colors = []
    # for general_memory in manager.hierarchy.general_memories:
    #     if node_count == 0:
    #         subgraph.clear()
    #         node_colors = []
    #     subgraph.add_node(general_memory.description, size=50)
    #     node_colors.append('green')
    #     for specific_memory in general_memory.get_specific_memories():
    #         subgraph.add_node(specific_memory.description,  size=20)
    #         node_colors.append('yellow')
    #         node_count += 1
    #         subgraph.add_edge(general_memory.description, specific_memory.description, color='black')
    #         for specific_memory2 in general_memory.get_specific_memories():
    #             if specific_memory != specific_memory2:
    #                 subgraph.add_edge(specific_memory.description, specific_memory2.description, color='green')
    #     if node_count > 10:
    #         node_count = 0
    #         #Try drawing several layouts of the subgraphs with nx.random_layout, nx.spring_layout, nx.spectral_layout, nx.shell_layout, nx.circular_layout, nx.kamada_kawai_layout, nx.planar_layout, nx.fruchterman_reingold_layout, nx.spiral_layout
    #         for lname, layout in {"random": nx.random_layout, "spring": nx.spring_layout, "spectral": nx.spectral_layout, "shell": nx.shell_layout, "circular": nx.circular_layout, "kamada_kawai": nx.kamada_kawai_layout, "planar": nx.planar_layout, "fruchterman_reingold": nx.fruchterman_reingold_layout, "spiral": nx.spiral_layout, "bipartite": nx.bipartite_layout, "multipartite": nx.multipartite_layout}.items():
    #             try:
    #                 #Title the graph with the general memory description and layout (as a string)
    #                 plt.title(f"Graph for general memory: {general_memory.description} with layout: {lname}")
    #                 nx.draw_networkx(subgraph, with_labels=True, font_weight='bold', pos=layout(subgraph), node_color=node_colors)
    #                 plt.show()

    #             except Exception as e:
    #                 print(f"Error with layout: {layout}, {e} for graph for general memory {general_memory.description}")

    # Create 2 subgraphs that contains only the specific memories that are relevant to a query, one just cosine and the other the score passed from the manager
    #
    # query = MemoryQuery(query,query_time=datetime.now(), gametime_manager=tiny_time_manager)
    # query_embedding = query.get_embedding()
    # relevant_specific_memories = manager.search_memories(query)
    # print(f"\n \n Relevant specific memories for query '{query}': {relevant_specific_memories} \n \n ")
    # query = "Where was the World Cup held in 2018?"

    # relevant_specific_memories = manager.search_memories(query)
    # print(f"\n \n Relevant specific memories for query '{query}': {relevant_specific_memories} \n \n ")
    # subgraph1 = nx.Graph()
    # subgraph2 = nx.Graph()
    # node_colors = []
    # subgraph1.add_node(query, size=50, label='query')
    # subgraph2.add_node(query,  size=50, label='query')
    # node_colors.append('blue')

    # for memory, score in relevant_specific_memories.items():
    #     similiarity_with_query = cosine_similarity(memory.embedding.cpu().detach().numpy(), query_embedding.cpu().detach().numpy())
    #     subgraph1.add_node(memory.description,  size=similiarity_with_query * 10)
    #     subgraph2.add_node(memory.description,  size=int(score))
    #     subgraph1.add_edge(query, memory.description, weight=similiarity_with_query)
    #     subgraph2.add_edge(query, memory.description, weight=score)
    #     node_colors.append('yellow')

    # #Try drawing several layouts of the subgraphs with nx.random_layout, nx.spring_layout, nx.spectral_layout, nx.shell_layout, nx.circular_layout, nx.kamada_kawai_layout, nx.planar_layout, nx.fruchterman_reingold_layout, nx.spiral_layout
    # if len(subgraph1.nodes()) > 1:
    #     for lname, layout in {"random": nx.random_layout, "spring": nx.spring_layout, "spectral": nx.spectral_layout, "shell": nx.shell_layout, "circular": nx.circular_layout, "kamada_kawai": nx.kamada_kawai_layout, "planar": nx.planar_layout, "fruchterman_reingold": nx.fruchterman_reingold_layout, "spiral": nx.spiral_layout, "bipartite": nx.bipartite_layout, "multipartite": nx.multipartite_layout}.items():

    #         try:
    #             nx.draw_networkx(subgraph1, with_labels=True, font_weight='bold', pos=layout(subgraph1), node_color=node_colors)
    #             plt.title(f"Graph for subgraph1 cosine similarity with layout: {lname}")
    #             plt.show()
    #             nx.draw_networkx(subgraph2, with_labels=True, font_weight='bold', pos=layout(subgraph2), node_color=node_colors)
    #             plt.title(f"Graph for subgraph2 score from manager with layout: {lname}")
    #             plt.show()
    #         except Exception as e:
    #             print(f"Error with layout: {layout}, {e} for subgraph1 cosine query similarity and subgraph2 score from manager")

    # Create a subgraph that splits the specific memories into clusters based on their similarity
    subgraph1 = nx.Graph()
    subgraph2 = nx.Graph()
    for general_memory in manager.hierarchy.general_memories:
        for specific_memory in general_memory.get_specific_memories():
            subgraph1.add_node(specific_memory.description)
            subgraph2.add_node(specific_memory.description)
            node_colors.append("yellow")
    for general_memory in manager.hierarchy.general_memories:
        for specific_memory in general_memory.get_specific_memories():
            for specific_memory2 in general_memory.get_specific_memories():
                if specific_memory != specific_memory2 and not subgraph1.has_edge(
                    specific_memory.description, specific_memory2.description
                ):
                    subgraph1.add_edge(
                        specific_memory.description,
                        specific_memory2.description,
                        weight=cosine_similarity(
                            specific_memory.embedding.cpu().detach().numpy(),
                            specific_memory2.embedding.cpu().detach().numpy(),
                        ),
                    )
                    subgraph2.add_edge(
                        specific_memory.description,
                        specific_memory2.description,
                        weight=manager.hierarchy.memory_graph.get_edge_data(
                            specific_memory, specific_memory2
                        ).get("weight", 0),
                    )

    if len(subgraph1.nodes()) > 1:
        for lname, layout in {
            "random": nx.random_layout,
            "spring": nx.spring_layout,
            "spectral": nx.spectral_layout,
            "shell": nx.shell_layout,
            "circular": nx.circular_layout,
            "kamada_kawai": nx.kamada_kawai_layout,
            "planar": nx.planar_layout,
            "fruchterman_reingold": nx.fruchterman_reingold_layout,
            "spiral": nx.spiral_layout,
            "bipartite": nx.bipartite_layout,
            "multipartite": nx.multipartite_layout,
        }.items():

            try:
                nx.draw_networkx(
                    subgraph1,
                    with_labels=True,
                    font_weight="bold",
                    pos=layout(subgraph1),
                    node_color=node_colors,
                )
                plt.title(f"Graph for subgraph1 cosine similarity with layout: {lname}")
                plt.show()
                nx.draw_networkx(
                    subgraph2,
                    with_labels=True,
                    font_weight="bold",
                    pos=layout(subgraph2),
                    node_color=node_colors,
                )
                plt.title(
                    f"Graph for subgraph2 weight from manager graph with layout: {lname}"
                )
                plt.show()
            except Exception as e:
                print(
                    f"Error with layout: {layout}, {e} for subgraph1 cosine similarity and subgraph2 weight from manager graph"
                )

    memories = {}

    query = "Where was the World Cup held in 2018?"
    memories[query] = manager.search_memories(query)
    print(
        f"\n \n Memories for query '{query}': {[mem.description for mem in memories[query]]}\n \n "
    )
    # Pause until the user presses a key
    time.sleep(2)

    query = "When will the 2022 Winter Olympics be held?"
    memories[query] = manager.search_memories(query)
    print(
        f"\n \n Memories for query '{query}': {[mem.description for mem in memories[query]]}\n \n "
    )
    # Pause until the user presses a key
    time.sleep(2)

    query = "I think someone is planning a surprise party"
    memories[query] = manager.search_memories(query)
    print(
        f"\n \n Memories for query '{query}': {[mem.description for mem in memories[query]]}\n \n "
    )
    # Pause until the user presses a key
    time.sleep(2)

    query = "I need to think of a popular tourist attraction"
    memories[query] = manager.search_memories(query)
    print(
        f"\n \n Memories for query '{query}': {[mem.description for mem in memories[query]]}\n \n "
    )
    time.sleep(2)

    query = "I am planning a trip to Europe"
    memories[query] = manager.search_memories(query)
    print(
        f"\n \n Memories for query '{query}': {[mem.description for mem in memories[query]]}\n \n "
    )
    time.sleep(2)

    query = "Who is learning to play the guitar?"
    memories[query] = manager.search_memories(query)
    print(
        f"\n \n Memories for query '{query}': {[mem.description for mem in memories[query]]}\n \n \n \n"
    )
    time.sleep(2)

    query = "I think someone is studying for a Chemistry test"
    memories[query] = manager.search_memories(query)
    print(
        f"\n \n Memories for query '{query}': {[mem.description for mem in memories[query]]}\n \n \n \n"
    )
    time.sleep(2)

    query = "What product should I sell in my electronics store?"
    memories[query] = manager.search_memories(query)
    print(
        f"\n \n Memories for query '{query}': {[mem.description for mem in memories[query]]}\n \n \n \n"
    )
    time.sleep(2)

    query = "What should I eat at the French restaurant?"
    memories[query] = manager.search_memories(query)
    print(
        f"\n \n Memories for query '{query}': {[mem.description for mem in memories[query]]}\n \n \n \n"
    )
    time.sleep(2)

    query = "What book should I read?"
    memories[query] = manager.search_memories(query)
    print(
        f"\n \n Memories for query '{query}': {[mem.description for mem in memories[query]]}\n \n \n \n"
    )
    time.sleep(2)

    query = "Where should I go for a night out to have drinks  and meet someone?"
    memories[query] = manager.search_memories(query)
    print(
        f"\n \n Memories for query '{query}': {[mem.description for mem in memories[query]]}\n \n \n \n"
    )
    time.sleep(2)

    query = "What bar should I avoid?"
    memories[query] = manager.search_memories(query)
    print(
        f"\n \n Memories for query '{query}': {[mem.description for mem in memories[query]]}\n \n \n \n"
    )
    time.sleep(2)

    query = "What fashion accessory should I wear to the party?"
    memories[query] = manager.search_memories(query)
    print(
        f"\n \n Memories for query '{query}': {[mem.description for mem in memories[query]]}\n \n \n \n"
    )
    time.sleep(2)

    query = "What is the future of transportation?"
    memories[query] = manager.search_memories(query)
    print(
        f"\n \n Memories for query '{query}': {[mem.description for mem in memories[query]]}\n \n \n \n"
    )
    time.sleep(2)

    query = "What is the current state of the ebola outbreak?"
    memories[query] = manager.search_memories(query)
    print(
        f"\n \n Memories for query '{query}': {[mem.description for mem in memories[query]]}\n \n \n \n"
    )
    time.sleep(2)
    time.sleep(2)

    query = "I am a farmer, what crop should I grow to make the most profit?"
    memories[query] = manager.search_memories(query)
    print(
        f"\n \n Memories for query '{query}': {[mem.description for mem in memories[query]]}\n \n \n \n"
    )
    time.sleep(2)

    query = "Is the new iPhone worth buying?"
    memories[query] = manager.search_memories(query)
    print(
        f"\n \n Memories for query '{query}': {[mem.description for mem in memories[query]]}\n \n \n \n"
    )
    time.sleep(2)

    query = "As a farmer, what crop would be make the most money?"
    memories[query] = manager.search_memories(query)
    print(
        f"\n \n Memories for query '{query}': {[mem.description for mem in memories[query]]}\n \n \n \n"
    )
    time.sleep(2)

    query = "What farm vegetable is selling the most these days?"
    memories[query] = manager.search_memories(query)
    print(
        f"\n \n Memories for query '{query}': {[mem.description for mem in memories[query]]}\n \n \n \n"
    )
    time.sleep(2)

    for query, memory in memories.items():
        print(f"\n \n \n \n Memory upper: {memory} \n for query: {query} \n ")
        index = 0
        for mem in memory:
            print(f"Memory lower: {mem} \n")
            print(f"Memory description: {mem.description} \n")
            print(f"Memory parent memory: {mem.parent_memory} \n")
            if not isinstance(mem.parent_memory, GeneralMemory) and isinstance(
                mem.parent_memory, str
            ):
                print(
                    f"Memory parent memory description: {[m.description for m in manager.hierarchy.general_memories if m.description == mem.parent_memory]} \n"
                )
                print(
                    f"Memory parent general memory keywords: {[m.keywords for m in manager.hierarchy.general_memories if m.description == mem.parent_memory]} \n"
                )
            elif isinstance(mem.parent_memory, GeneralMemory):
                print(
                    f"Memory parent memory description: {mem.parent_memory.description} \n"
                )
                print(
                    f"Memory parent general memory keywords: {mem.parent_memory.keywords} \n"
                )
            print(f"Memory related memories: {mem.related_memories} \n")
            print(f"Memory keywords: {mem.keywords} \n")
            print(f"Memory tags: {mem.tags} \n")
            print(f"Memory importance score: {mem.importance_score} \n")
            print(f"Memory sentiment score: {mem.sentiment_score} \n")
            print(f"Memory emotion classification: {mem.emotion_classification} \n")
            print(f"Memory last access time: {mem.last_access_time} \n")
            print(f"Memory recency index: {index} \n")

    # Save as a graphml file
    # nx.write_graphml(visualization_graph, "visualization_graph.graphml")

    # Save memories to a file
    try:
        with open("memories.txt", "w") as file:
            file.write(json.dumps(memories))
    except Exception as e:
        print(f"Error saving memories to file: {e}")

    # Save the manager to a file
    try:
        with open("manager.txt", "w") as file:
            file.write(json.dumps(manager))
    except Exception as e:
        print(f"Error saving manager to file: {e}")

    # Test queries through both hierarchy and flataccess
    hierarchy_results_weight = {}
    hierarchy_results_similarity = {}
    flat_access_results = {}
    time_hier_weight = 0
    time_hier_sim = 0
    time_flat = 0
    query = "What is a popular French dish?"
    # Time each query
    start = time.time()
    flat_access_results = manager.flat_access.find_memories_by_query(
        MemoryQuery(
            query, query_time=datetime.now(), gametime_manager=tiny_time_manager
        )
    )
    end = time.time()
    time_flat = end - start
    start = time.time()
    hierarchy_results_weight = manager.retrieve_from_hierarchy(query)
    end = time.time()
    time_hier_weight = end - start
    start = time.time()
    # hierarchy_results_similarity = manager.hierarchy.find_nodes_by_similarity(query, 0.5)
    end = time.time()
    time_hier_sim = end - start
    print(
        f"\n \n \n Hierarchy weight results for query '{query}': {hierarchy_results_weight} \n \n \n"
    )
    # print(f"\n \n \n Hierarchy similiarity results for query '{query}': {hierarchy_results_similarity} \n \n \n")
    print(
        f"\n \n \n Flat access results for query '{query}': {flat_access_results} \n \n \n"
    )
    time.sleep(2)
    # Write same to file
    with open("results.txt", "w") as file:
        file.write(f"Query: {query} \n \n")
        file.write(f"Hierarchy search time for edge weight: {time_hier_weight} \n")

        file.write("Hierarchy weight results: \n \n")
        for memory, weight in hierarchy_results_weight.items():
            file.write(f"{memory.description}, weight: {weight} \n")

        file.write(f"Flat access search time: {time_flat} \n")

        file.write("Flat access results: \n \n")
        for memory, weight in flat_access_results.items():
            file.write(f"{memory.description}, weight: {weight} \n")
        # file.write(json.dumps({"hierarchy_weight": hierarchy_results_weight, "hierarchy_similarity": hierarchy_results_similarity, "flat_access": flat_access_results}, default=lambda o: o.__dict__, sort_keys=True, indent=4))

    query = "Where was the World Cup held in 2018?"
    time_hier_weight = 0
    time_hier_sim = 0
    time_flat = 0

    # Time each query
    start = time.time()
    flat_access_results = manager.flat_access.find_memories_by_query(
        MemoryQuery(
            query, query_time=datetime.now(), gametime_manager=tiny_time_manager
        )
    )
    end = time.time()
    time_flat = end - start
    start = time.time()
    hierarchy_results_weight = manager.retrieve_from_hierarchy(query)
    end = time.time()
    time_hier_weight = end - start
    start = time.time()
    # hierarchy_results_similarity = manager.hierarchy.find_nodes_by_similarity(query, 0.5)
    end = time.time()
    time_hier_sim = end - start
    print(
        f"\n \n \n Hierarchy weight results for query '{query}': {hierarchy_results_weight} \n \n \n"
    )
    # print(f"\n \n \n Hierarchy similiarity results for query '{query}': {hierarchy_results_similarity} \n \n \n")
    print(
        f"\n \n \n Flat access results for query '{query}': {flat_access_results} \n \n \n"
    )
    time.sleep(2)
    # Write same to file
    with open("results.txt", "w") as file:
        file.write(f"Query: {query} \n \n")
        file.write(f"Hierarchy search time for edge weight: {time_hier_weight} \n")

        file.write("Hierarchy weight results: \n \n")
        for memory, weight in hierarchy_results_weight.items():
            file.write(f"{memory.description}, weight: {weight} \n")

        file.write(f"Flat access search time: {time_flat} \n")

        file.write("Flat access results: \n \n")
        for memory, weight in flat_access_results.items():
            file.write(f"{memory.description}, weight: {weight} \n")
        # file.write(json.dumps({"hierarchy_weight": hierarchy_results_weight, "hierarchy_similarity": hierarchy_results_similarity, "flat_access": flat_access_results}, default=lambda o: o.__dict__, sort_keys=True, indent=4))

    query = "I think someone is planning a surprise party"
    time_hier_weight = 0
    time_hier_sim = 0
    time_flat = 0

    # Time each query
    start = time.time()
    flat_access_results = manager.flat_access.find_memories_by_query(
        MemoryQuery(
            query, query_time=datetime.now(), gametime_manager=tiny_time_manager
        )
    )
    end = time.time()
    time_flat = end - start
    start = time.time()
    hierarchy_results_weight = manager.retrieve_from_hierarchy(query)
    end = time.time()
    time_hier_weight = end - start
    start = time.time()
    # hierarchy_results_similarity = manager.hierarchy.find_nodes_by_similarity(query, 0.5)
    end = time.time()
    time_hier_sim = end - start
    print(
        f"\n \n \n Hierarchy weight results for query '{query}': {hierarchy_results_weight} \n \n \n"
    )
    # print(f"\n \n \n Hierarchy similiarity results for query '{query}': {hierarchy_results_similarity} \n \n \n")
    print(
        f"\n \n \n Flat access results for query '{query}': {flat_access_results} \n \n \n"
    )
    time.sleep(2)
    # Write same to file
    with open("results.txt", "a") as file:
        file.write(f"Query: {query} \n \n")
        file.write(f"Hierarchy search time for edge weight: {time_hier_weight} \n")

        file.write("Hierarchy weight results: \n \n")
        for memory, weight in hierarchy_results_weight.items():
            file.write(f"{memory.description}, weight: {weight} \n")

        file.write(f"Flat access search time: {time_flat} \n")

        file.write("Flat access results: \n \n")
        for memory, weight in flat_access_results.items():
            file.write(f"{memory.description}, weight: {weight} \n")
        # file.write(json.dumps({"hierarchy_weight": hierarchy_results_weight, "hierarchy_similarity": hierarchy_results_similarity, "flat_access": flat_access_results}, default=lambda o: o.__dict__, sort_keys=True, indent=4))

    query = "I need to think of a popular tourist attraction"
    time_hier_weight = 0
    time_hier_sim = 0
    time_flat = 0

    # Time each query
    start = time.time()
    flat_access_results = manager.flat_access.find_memories_by_query(
        MemoryQuery(
            query, query_time=datetime.now(), gametime_manager=tiny_time_manager
        )
    )
    end = time.time()
    time_flat = end - start
    start = time.time()
    hierarchy_results_weight = manager.retrieve_from_hierarchy(query)
    end = time.time()
    time_hier_weight = end - start
    start = time.time()
    # hierarchy_results_similarity = manager.hierarchy.find_nodes_by_similarity(query, 0.5)
    end = time.time()
    time_hier_sim = end - start
    print(
        f"\n \n \n Hierarchy weight results for query '{query}': {hierarchy_results_weight} \n \n \n"
    )
    # print(f"\n \n \n Hierarchy similiarity results for query '{query}': {hierarchy_results_similarity} \n \n \n")
    print(
        f"\n \n \n Flat access results for query '{query}': {flat_access_results} \n \n \n"
    )
    time.sleep(2)
    # Write same to file
    with open("results.txt", "a") as file:
        file.write(f"Query: {query} \n \n")
        file.write(f"Hierarchy search time for edge weight: {time_hier_weight} \n")

        file.write("Hierarchy weight results: \n \n")
        for memory, weight in hierarchy_results_weight.items():
            file.write(f"{memory.description}, weight: {weight} \n")

        file.write(f"Flat access search time: {time_flat} \n")

        file.write("Flat access results: \n \n")
        for memory, weight in flat_access_results.items():
            file.write(f"{memory.description}, weight: {weight} \n")
        # file.write(json.dumps({"hierarchy_weight": hierarchy_results_weight, "hierarchy_similarity": hierarchy_results_similarity, "flat_access": flat_access_results}, default=lambda o: o.__dict__, sort_keys=True, indent=4))

    query = "I am planning a trip to Europe"
    time_hier_weight = 0
    time_hier_sim = 0
    time_flat = 0

    # Time each query
    start = time.time()
    flat_access_results = manager.flat_access.find_memories_by_query(
        MemoryQuery(
            query, query_time=datetime.now(), gametime_manager=tiny_time_manager
        )
    )
    end = time.time()
    time_flat = end - start
    start = time.time()
    hierarchy_results_weight = manager.retrieve_from_hierarchy(query)
    end = time.time()
    time_hier_weight = end - start
    start = time.time()
    # hierarchy_results_similarity = manager.hierarchy.find_nodes_by_similarity(query, 0.5)
    end = time.time()
    time_hier_sim = end - start
    print(
        f"\n \n \n Hierarchy weight results for query '{query}': {hierarchy_results_weight} \n \n \n"
    )
    # print(f"\n \n \n Hierarchy similiarity results for query '{query}': {hierarchy_results_similarity} \n \n \n")
    print(
        f"\n \n \n Flat access results for query '{query}': {flat_access_results} \n \n \n"
    )
    time.sleep(2)
    # Write same to file
    with open("results.txt", "a") as file:
        file.write(f"Query: {query} \n \n")
        file.write(f"Hierarchy search time for edge weight: {time_hier_weight} \n")

        file.write("Hierarchy weight results: \n \n")
        for memory, weight in hierarchy_results_weight.items():
            file.write(f"{memory.description}, weight: {weight} \n")

        file.write(f"Flat access search time: {time_flat} \n")

        file.write("Flat access results: \n \n")
        for memory, weight in flat_access_results.items():
            file.write(f"{memory.description}, weight: {weight} \n")
    query = "Who is learning to play the guitar?"
    time_hier_weight = 0
    time_hier_sim = 0
    time_flat = 0

    # Time each query
    start = time.time()
    flat_access_results = manager.flat_access.find_memories_by_query(
        MemoryQuery(
            query, query_time=datetime.now(), gametime_manager=tiny_time_manager
        )
    )
    end = time.time()
    time_flat = end - start
    start = time.time()
    hierarchy_results_weight = manager.retrieve_from_hierarchy(query)
    end = time.time()
    time_hier_weight = end - start
    start = time.time()
    # hierarchy_results_similarity = manager.hierarchy.find_nodes_by_similarity(query, 0.5)
    end = time.time()
    time_hier_sim = end - start
    print(
        f"\n \n \n Hierarchy weight results for query '{query}': {hierarchy_results_weight} \n \n \n"
    )
    # print(f"\n \n \n Hierarchy similiarity results for query '{query}': {hierarchy_results_similarity} \n \n \n")
    print(
        f"\n \n \n Flat access results for query '{query}': {flat_access_results} \n \n \n"
    )
    time.sleep(2)
    # Write same to file
    with open("results.txt", "a") as file:
        file.write(f"Query: {query} \n \n")
        file.write(f"Hierarchy search time for edge weight: {time_hier_weight} \n")

        file.write("Hierarchy weight results: \n \n")
        for memory, weight in hierarchy_results_weight.items():
            file.write(f"{memory.description}, weight: {weight} \n")

        file.write(f"Flat access search time: {time_flat} \n")

        file.write("Flat access results: \n \n")
        for memory, weight in flat_access_results.items():
            file.write(f"{memory.description}, weight: {weight} \n")
        # file.write(json.dumps({"hierarchy_weight": hierarchy_results_weight, "hierarchy_similarity": hierarchy_results_similarity, "flat_access": flat_access_results}, default=lambda o: o.__dict__, sort_keys=True, indent=4))

    query = "I think someone is studying for a Chemistry test"
    time_hier_weight = 0
    time_hier_sim = 0
    time_flat = 0

    # Time each query
    start = time.time()
    flat_access_results = manager.flat_access.find_memories_by_query(
        MemoryQuery(
            query, query_time=datetime.now(), gametime_manager=tiny_time_manager
        )
    )
    end = time.time()
    time_flat = end - start
    start = time.time()
    hierarchy_results_weight = manager.retrieve_from_hierarchy(query)
    end = time.time()
    time_hier_weight = end - start
    start = time.time()
    # hierarchy_results_similarity = manager.hierarchy.find_nodes_by_similarity(query, 0.5)
    end = time.time()
    time_hier_sim = end - start
    print(
        f"\n \n \n Hierarchy weight results for query '{query}': {hierarchy_results_weight} \n \n \n"
    )
    # print(f"\n \n \n Hierarchy similiarity results for query '{query}': {hierarchy_results_similarity} \n \n \n")
    print(
        f"\n \n \n Flat access results for query '{query}': {flat_access_results} \n \n \n"
    )
    time.sleep(2)
    # Write same to file
    with open("results.txt", "a") as file:
        file.write(f"Query: {query} \n \n")
        file.write(f"Hierarchy search time for edge weight: {time_hier_weight} \n")

        file.write("Hierarchy weight results: \n \n")
        for memory, weight in hierarchy_results_weight.items():
            file.write(f"{memory.description}, weight: {weight} \n")

        file.write(f"Flat access search time: {time_flat} \n")

        file.write("Flat access results: \n \n")
        for memory, weight in flat_access_results.items():
            file.write(f"{memory.description}, weight: {weight} \n")
        # file.write(json.dumps({"hierarchy_weight": hierarchy_results_weight, "hierarchy_similarity": hierarchy_results_similarity, "flat_access": flat_access_results}, default=lambda o: o.__dict__, sort_keys=True, indent=4))

    query = "What product should I sell in my electronics store?"
    time_hier_weight = 0
    time_hier_sim = 0
    time_flat = 0

    # Time each query
    start = time.time()
    flat_access_results = manager.flat_access.find_memories_by_query(
        MemoryQuery(
            query, query_time=datetime.now(), gametime_manager=tiny_time_manager
        )
    )
    end = time.time()
    time_flat = end - start
    start = time.time()
    hierarchy_results_weight = manager.retrieve_from_hierarchy(query)
    end = time.time()
    time_hier_weight = end - start
    start = time.time()
    # hierarchy_results_similarity = manager.hierarchy.find_nodes_by_similarity(query, 0.5)
    end = time.time()
    time_hier_sim = end - start
    print(
        f"\n \n \n Hierarchy weight results for query '{query}': {hierarchy_results_weight} \n \n \n"
    )
    # print(f"\n \n \n Hierarchy similiarity results for query '{query}': {hierarchy_results_similarity} \n \n \n")
    print(
        f"\n \n \n Flat access results for query '{query}': {flat_access_results} \n \n \n"
    )
    time.sleep(2)
    # Write same to file
    with open("results.txt", "a") as file:
        file.write(f"Query: {query} \n \n")
        file.write(f"Hierarchy search time for edge weight: {time_hier_weight} \n")

        file.write("Hierarchy weight results: \n \n")
        for memory, weight in hierarchy_results_weight.items():
            file.write(f"{memory.description}, weight: {weight} \n")

        file.write(f"Flat access search time: {time_flat} \n")

        file.write("Flat access results: \n \n")
        for memory, weight in flat_access_results.items():
            file.write(f"{memory.description}, weight: {weight} \n")
        # file.write(json.dumps({"hierarchy_weight": hierarchy_results_weight, "hierarchy_similarity": hierarchy_results_similarity, "flat_access": flat_access_results}, default=lambda o: o.__dict__, sort_keys=True, indent=4))

    query = "What should I eat at the French restaurant?"
    time_hier_weight = 0
    time_hier_sim = 0
    time_flat = 0

    # Time each query
    start = time.time()
    flat_access_results = manager.flat_access.find_memories_by_query(
        MemoryQuery(
            query, query_time=datetime.now(), gametime_manager=tiny_time_manager
        )
    )
    end = time.time()
    time_flat = end - start
    start = time.time()
    hierarchy_results_weight = manager.retrieve_from_hierarchy(query)
    end = time.time()
    time_hier_weight = end - start
    start = time.time()
    # hierarchy_results_similarity = manager.hierarchy.find_nodes_by_similarity(query, 0.5)
    end = time.time()
    time_hier_sim = end - start
    print(
        f"\n \n \n Hierarchy weight results for query '{query}': {hierarchy_results_weight} \n \n \n"
    )
    # print(f"\n \n \n Hierarchy similiarity results for query '{query}': {hierarchy_results_similarity} \n \n \n")
    print(
        f"\n \n \n Flat access results for query '{query}': {flat_access_results} \n \n \n"
    )
    time.sleep(2)
    # Write same to file
    with open("results.txt", "a") as file:
        file.write(f"Query: {query} \n \n")
        file.write(f"Hierarchy search time for edge weight: {time_hier_weight} \n")

        file.write("Hierarchy weight results: \n \n")
        for memory, weight in hierarchy_results_weight.items():
            file.write(f"{memory.description}, weight: {weight} \n")

        file.write(f"Flat access search time: {time_flat} \n")

        file.write("Flat access results: \n \n")
        for memory, weight in flat_access_results.items():
            file.write(f"{memory.description}, weight: {weight} \n")
        # file.write(json.dumps({"hierarchy_weight": hierarchy_results_weight, "hierarchy_similarity": hierarchy_results_similarity, "flat_access": flat_access_results}, default=lambda o: o.__dict__, sort_keys=True, indent=4))

    query = "What book should I read?"
    time_hier_weight = 0
    time_hier_sim = 0
    time_flat = 0

    # Time each query
    start = time.time()
    flat_access_results = manager.flat_access.find_memories_by_query(
        MemoryQuery(
            query, query_time=datetime.now(), gametime_manager=tiny_time_manager
        )
    )
    end = time.time()
    time_flat = end - start
    start = time.time()
    hierarchy_results_weight = manager.retrieve_from_hierarchy(query)
    end = time.time()
    time_hier_weight = end - start
    start = time.time()
    # hierarchy_results_similarity = manager.hierarchy.find_nodes_by_similarity(query, 0.5)
    end = time.time()
    time_hier_sim = end - start
    print(
        f"\n \n \n Hierarchy weight results for query '{query}': {hierarchy_results_weight} \n \n \n"
    )
    # print(f"\n \n \n Hierarchy similiarity results for query '{query}': {hierarchy_results_similarity} \n \n \n")
    print(
        f"\n \n \n Flat access results for query '{query}': {flat_access_results} \n \n \n"
    )
    time.sleep(2)
    # Write same to file
    with open("results.txt", "a") as file:
        file.write(f"Query: {query} \n \n")
        file.write(f"Hierarchy search time for edge weight: {time_hier_weight} \n")

        file.write("Hierarchy weight results: \n \n")
        for memory, weight in hierarchy_results_weight.items():
            file.write(f"{memory.description}, weight: {weight} \n")

        file.write(f"Flat access search time: {time_flat} \n")

        file.write("Flat access results: \n \n")
        for memory, weight in flat_access_results.items():
            file.write(f"{memory.description}, weight: {weight} \n")
        # file.write(json.dumps({"hierarchy_weight": hierarchy_results_weight, "hierarchy_similarity": hierarchy_results_similarity, "flat_access": flat_access_results}, default=lambda o: o.__dict__, sort_keys=True, indent=4))

    query = "Where should I go for a night out to have drinks  and meet someone?"
    time_hier_weight = 0
    time_hier_sim = 0
    time_flat = 0

    # Time each query
    start = time.time()
    flat_access_results = manager.flat_access.find_memories_by_query(
        MemoryQuery(
            query, query_time=datetime.now(), gametime_manager=tiny_time_manager
        )
    )
    end = time.time()
    time_flat = end - start
    start = time.time()
    hierarchy_results_weight = manager.retrieve_from_hierarchy(query)
    end = time.time()
    time_hier_weight = end - start
    start = time.time()
    # hierarchy_results_similarity = manager.hierarchy.find_nodes_by_similarity(query, 0.5)
    end = time.time()
    time_hier_sim = end - start
    print(
        f"\n \n \n Hierarchy weight results for query '{query}': {hierarchy_results_weight} \n \n \n"
    )
    # print(f"\n \n \n Hierarchy similiarity results for query '{query}': {hierarchy_results_similarity} \n \n \n")
    print(
        f"\n \n \n Flat access results for query '{query}': {flat_access_results} \n \n \n"
    )
    time.sleep(2)
    # Write same to file
    with open("results.txt", "a") as file:
        file.write(f"Query: {query} \n \n")
        file.write(f"Hierarchy search time for edge weight: {time_hier_weight} \n")

        file.write("Hierarchy weight results: \n \n")
        for memory, weight in hierarchy_results_weight.items():
            file.write(f"{memory.description}, weight: {weight} \n")

        file.write(f"Flat access search time: {time_flat} \n")

        file.write("Flat access results: \n \n")
        for memory, weight in flat_access_results.items():
            file.write(f"{memory.description}, weight: {weight} \n")
        # file.write(json.dumps({"hierarchy_weight": hierarchy_results_weight, "hierarchy_similarity": hierarchy_results_similarity, "flat_access": flat_access_results}, default=lambda o: o.__dict__, sort_keys=True, indent=4))

    query = "What bar should I avoid?"
    time_hier_weight = 0
    time_hier_sim = 0
    time_flat = 0

    # Time each query
    start = time.time()
    flat_access_results = manager.flat_access.find_memories_by_query(
        MemoryQuery(
            query, query_time=datetime.now(), gametime_manager=tiny_time_manager
        )
    )
    end = time.time()
    time_flat = end - start
    start = time.time()
    hierarchy_results_weight = manager.retrieve_from_hierarchy(query)
    end = time.time()
    time_hier_weight = end - start
    start = time.time()
    # hierarchy_results_similarity = manager.hierarchy.find_nodes_by_similarity(query, 0.5)
    end = time.time()
    time_hier_sim = end - start
    print(
        f"\n \n \n Hierarchy weight results for query '{query}': {hierarchy_results_weight} \n \n \n"
    )
    # print(f"\n \n \n Hierarchy similiarity results for query '{query}': {hierarchy_results_similarity} \n \n \n")
    print(
        f"\n \n \n Flat access results for query '{query}': {flat_access_results} \n \n \n"
    )
    time.sleep(2)
    # Write same to file
    with open("results.txt", "a") as file:
        file.write(f"Query: {query} \n \n")
        file.write(f"Hierarchy search time for edge weight: {time_hier_weight} \n")

        file.write("Hierarchy weight results: \n \n")
        for memory, weight in hierarchy_results_weight.items():
            file.write(f"{memory.description}, weight: {weight} \n")

        file.write(f"Flat access search time: {time_flat} \n")

        file.write("Flat access results: \n \n")
        for memory, weight in flat_access_results.items():
            file.write(f"{memory.description}, weight: {weight} \n")
        # file.write(json.dumps({"hierarchy_weight": hierarchy_results_weight, "hierarchy_similarity": hierarchy_results_similarity, "flat_access": flat_access_results}, default=lambda o: o.__dict__, sort_keys=True, indent=4))

    query = "What fashion accessory should I wear to the party?"

    time_hier_weight = 0
    time_hier_sim = 0
    time_flat = 0

    # Time each query
    start = time.time()
    flat_access_results = manager.flat_access.find_memories_by_query(
        MemoryQuery(
            query, query_time=datetime.now(), gametime_manager=tiny_time_manager
        )
    )
    end = time.time()
    time_flat = end - start
    start = time.time()
    hierarchy_results_weight = manager.retrieve_from_hierarchy(query)
    end = time.time()
    time_hier_weight = end - start
    start = time.time()
    # hierarchy_results_similarity = manager.hierarchy.find_nodes_by_similarity(query, 0.5)
    end = time.time()
    time_hier_sim = end - start
    print(
        f"\n \n \n Hierarchy weight results for query '{query}': {hierarchy_results_weight} \n \n \n"
    )
    # print(f"\n \n \n Hierarchy similiarity results for query '{query}': {hierarchy_results_similarity} \n \n \n")
    print(
        f"\n \n \n Flat access results for query '{query}': {flat_access_results} \n \n \n"
    )
    time.sleep(2)
    # Write same to file
    with open("results.txt", "a") as file:
        file.write(f"Query: {query} \n \n")
        file.write(f"Hierarchy search time for edge weight: {time_hier_weight} \n")

        file.write("Hierarchy weight results: \n \n")
        for memory, weight in hierarchy_results_weight.items():
            file.write(f"{memory.description}, weight: {weight} \n")

        file.write(f"Flat access search time: {time_flat} \n")

        file.write("Flat access results: \n \n")
        for memory, weight in flat_access_results.items():
            file.write(f"{memory.description}, weight: {weight} \n")
        # file.write(json.dumps({"hierarchy_weight": hierarchy_results_weight, "hierarchy_similarity": hierarchy_results_similarity, "flat_access": flat_access_results}, default=lambda o: o.__dict__, sort_keys=True, indent=4))

    query = "What is the future of transportation?"
    time_hier_weight = 0
    time_hier_sim = 0
    time_flat = 0

    # Time each query
    start = time.time()
    flat_access_results = manager.flat_access.find_memories_by_query(
        MemoryQuery(
            query, query_time=datetime.now(), gametime_manager=tiny_time_manager
        )
    )
    end = time.time()
    time_flat = end - start
    start = time.time()
    hierarchy_results_weight = manager.retrieve_from_hierarchy(query)
    end = time.time()
    time_hier_weight = end - start
    start = time.time()
    # hierarchy_results_similarity = manager.hierarchy.find_nodes_by_similarity(query, 0.5)
    end = time.time()
    time_hier_sim = end - start
    print(
        f"\n \n \n Hierarchy weight results for query '{query}': {hierarchy_results_weight} \n \n \n"
    )
    # print(f"\n \n \n Hierarchy similiarity results for query '{query}': {hierarchy_results_similarity} \n \n \n")
    print(
        f"\n \n \n Flat access results for query '{query}': {flat_access_results} \n \n \n"
    )
    time.sleep(2)
    # Write same to file
    with open("results.txt", "a") as file:
        file.write(f"Query: {query} \n \n")
        file.write(f"Hierarchy search time for edge weight: {time_hier_weight} \n")

        file.write("Hierarchy weight results: \n \n")
        for memory, weight in hierarchy_results_weight.items():
            file.write(f"{memory.description}, weight: {weight} \n")

        file.write(f"Flat access search time: {time_flat} \n")

        file.write("Flat access results: \n \n")
        for memory, weight in flat_access_results.items():
            file.write(f"{memory.description}, weight: {weight} \n")
        # file.write(json.dumps({"hierarchy_weight": hierarchy_results_weight, "hierarchy_similarity": hierarchy_results_similarity, "flat_access": flat_access_results}, default=lambda o: o.__dict__, sort_keys=True, indent=4))

    query = "What is the current state of the ebola outbreak?"

    time_hier_weight = 0
    time_hier_sim = 0
    time_flat = 0

    # Time each query
    start = time.time()
    flat_access_results = manager.flat_access.find_memories_by_query(
        MemoryQuery(
            query, query_time=datetime.now(), gametime_manager=tiny_time_manager
        )
    )
    end = time.time()
    time_flat = end - start
    start = time.time()
    hierarchy_results_weight = manager.retrieve_from_hierarchy(query)
    end = time.time()
    time_hier_weight = end - start
    start = time.time()
    # hierarchy_results_similarity = manager.hierarchy.find_nodes_by_similarity(query, 0.5)
    end = time.time()
    time_hier_sim = end - start
    print(
        f"\n \n \n Hierarchy weight results for query '{query}': {hierarchy_results_weight} \n \n \n"
    )
    # print(f"\n \n \n Hierarchy similiarity results for query '{query}': {hierarchy_results_similarity} \n \n \n")
    print(
        f"\n \n \n Flat access results for query '{query}': {flat_access_results} \n \n \n"
    )
    time.sleep(2)
    # Write same to file
    with open("results.txt", "a") as file:
        file.write(f"Query: {query} \n \n")
        file.write(f"Hierarchy search time for edge weight: {time_hier_weight} \n")

        file.write("Hierarchy weight results: \n \n")
        for memory, weight in hierarchy_results_weight.items():
            file.write(f"{memory.description}, weight: {weight} \n")

        file.write(f"Flat access search time: {time_flat} \n")

        file.write("Flat access results: \n \n")
        for memory, weight in flat_access_results.items():
            file.write(f"{memory.description}, weight: {weight} \n")
        # file.write(json.dumps({"hierarchy_weight": hierarchy_results_weight, "hierarchy_similarity": hierarchy_results_similarity, "flat_access": flat_access_results}, default=lambda o: o.__dict__, sort_keys=True, indent=4))

    query = "I am a farmer, what crop should I grow to make the most profit?"

    time_hier_weight = 0
    time_hier_sim = 0
    time_flat = 0

    # Time each query
    start = time.time()
    flat_access_results = manager.flat_access.find_memories_by_query(
        MemoryQuery(
            query, query_time=datetime.now(), gametime_manager=tiny_time_manager
        )
    )
    end = time.time()
    time_flat = end - start
    start = time.time()
    hierarchy_results_weight = manager.retrieve_from_hierarchy(query)
    end = time.time()
    time_hier_weight = end - start
    start = time.time()
    # hierarchy_results_similarity = manager.hierarchy.find_nodes_by_similarity(query, 0.5)
    end = time.time()
    time_hier_sim = end - start
    print(
        f"\n \n \n Hierarchy weight results for query '{query}': {hierarchy_results_weight} \n \n \n"
    )
    # print(f"\n \n \n Hierarchy similiarity results for query '{query}': {hierarchy_results_similarity} \n \n \n")
    print(
        f"\n \n \n Flat access results for query '{query}': {flat_access_results} \n \n \n"
    )
    time.sleep(2)
    # Write same to file
    with open("results.txt", "a") as file:
        file.write(f"Query: {query} \n \n")
        file.write(f"Hierarchy search time for edge weight: {time_hier_weight} \n")

        file.write("Hierarchy weight results: \n \n")
        for memory, weight in hierarchy_results_weight.items():
            file.write(f"{memory.description}, weight: {weight} \n")

        file.write(f"Flat access search time: {time_flat} \n")

        file.write("Flat access results: \n \n")
        for memory, weight in flat_access_results.items():
            file.write(f"{memory.description}, weight: {weight} \n")
        # file.write(json.dumps({"hierarchy_weight": hierarchy_results_weight, "hierarchy_similarity": hierarchy_results_similarity, "flat_access": flat_access_results}, default=lambda o: o.__dict__, sort_keys=True, indent=4))

    query = "Is the new iPhone worth buying?"

    time_hier_weight = 0
    time_hier_sim = 0
    time_flat = 0

    # Time each query
    start = time.time()
    flat_access_results = manager.flat_access.find_memories_by_query(
        MemoryQuery(
            query, query_time=datetime.now(), gametime_manager=tiny_time_manager
        )
    )
    end = time.time()
    time_flat = end - start
    start = time.time()
    hierarchy_results_weight = manager.retrieve_from_hierarchy(query)
    end = time.time()
    time_hier_weight = end - start
    start = time.time()
    # hierarchy_results_similarity = manager.hierarchy.find_nodes_by_similarity(query, 0.5)
    end = time.time()
    time_hier_sim = end - start
    print(
        f"\n \n \n Hierarchy weight results for query '{query}': {hierarchy_results_weight} \n \n \n"
    )
    # print(f"\n \n \n Hierarchy similiarity results for query '{query}': {hierarchy_results_similarity} \n \n \n")
    print(
        f"\n \n \n Flat access results for query '{query}': {flat_access_results} \n \n \n"
    )
    time.sleep(2)
    # Write same to file
    with open("results.txt", "a") as file:
        file.write(f"Query: {query} \n \n")
        file.write(f"Hierarchy search time for edge weight: {time_hier_weight} \n")

        file.write("Hierarchy weight results: \n \n")
        for memory, weight in hierarchy_results_weight.items():
            file.write(f"{memory.description}, weight: {weight} \n")

        file.write(f"Flat access search time: {time_flat} \n")

        file.write("Flat access results: \n \n")
        for memory, weight in flat_access_results.items():
            file.write(f"{memory.description}, weight: {weight} \n")
        # file.write(json.dumps({"hierarchy_weight": hierarchy_results_weight, "hierarchy_similarity": hierarchy_results_similarity, "flat_access": flat_access_results}, default=lambda o: o.__dict__, sort_keys=True, indent=4))

    query = "What vegetables are selling the most at the market?"

    time_hier_weight = 0
    time_hier_sim = 0
    time_flat = 0

    # Time each query
    start = time.time()
    flat_access_results = manager.flat_access.find_memories_by_query(
        MemoryQuery(
            query, query_time=datetime.now(), gametime_manager=tiny_time_manager
        )
    )
    end = time.time()
    time_flat = end - start
    start = time.time()
    hierarchy_results_weight = manager.retrieve_from_hierarchy(query)
    end = time.time()
    time_hier_weight = end - start
    start = time.time()
    # hierarchy_results_similarity = manager.hierarchy.find_nodes_by_similarity(query, 0.5)
    end = time.time()
    time_hier_sim = end - start
    print(
        f"\n \n \n Hierarchy weight results for query '{query}': {hierarchy_results_weight} \n \n \n"
    )
    # print(f"\n \n \n Hierarchy similiarity results for query '{query}': {hierarchy_results_similarity} \n \n \n")
    print(
        f"\n \n \n Flat access results for query '{query}': {flat_access_results} \n \n \n"
    )
    time.sleep(2)
    # Write same to file
    with open("results.txt", "a") as file:
        file.write(f"Query: {query} \n \n")
        file.write(f"Hierarchy search time for edge weight: {time_hier_weight} \n")

        file.write("Hierarchy weight results: \n \n")
        for memory, weight in hierarchy_results_weight.items():
            file.write(f"{memory.description}, weight: {weight} \n")

        file.write(f"Flat access search time: {time_flat} \n")

        file.write("Flat access results: \n \n")
        for memory, weight in flat_access_results.items():
            file.write(f"{memory.description}, weight: {weight} \n")
        # file.write(json.dumps({"hierarchy_weight": hierarchy_results_weight, "hierarchy_similarity": hierarchy_results_similarity, "flat_access": flat_access_results}, default=lambda o: o.__dict__, sort_keys=True, indent=4))

    # Visualize the Class Interaction Graph class_interaction_graph

    # # Draw the Class Interaction Graph
    # plt.figure(figsize=(8, 6))
    # plt.title('Class Interaction Graph')
    # pos = nx.spring_layout(class_interaction_graph)
    # nx.draw(class_interaction_graph, pos, with_labels=True, connectionstyle='arc3, rad = 0.1')
    # plt.show()

    # # Draw the Call Flow Diagram
    # plt.figure(figsize=(8, 6))
    # plt.title('Call Flow Diagram')
    # pos = nx.spring_layout(call_flow_diagram)
    # edge_labels = {(u, v): f"{d['num_calls']} calls, {d['total_time']:.2f} s" for u, v, d in call_flow_diagram.edges(data=True)}
    # nx.draw(call_flow_diagram, pos, with_labels=True, connectionstyle='arc3, rad = 0.1')
    # nx.draw_networkx_edge_labels(call_flow_diagram, pos, edge_labels=edge_labels)
    # plt.show()
    # df = pd.DataFrame(heat_map, columns=heat_map.keys(), index=heat_map.keys())

    # # Draw the Heat Map
    # plt.figure(figsize=(8, 6))
    # plt.title('Heat Map')
    # sns.heatmap(heat_map, annot=True, fmt="d")
    # plt.show()
