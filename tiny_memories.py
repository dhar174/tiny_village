import html
import json
import pickle
import random

# from locale import normalize
import math
import re
import time

from collections import deque
from networkx import node_link_data
import numpy as np
from datetime import datetime, timedelta
import pandas as pd
import scipy as sp
from sklearn import tree
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
from sympy import Q, comp, lex, per
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
import tiny_sr_mapping as tsm
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

os.environ["TRANSFORMERS_CACHE"] = "/mnt/d/transformers_cache"
remove_list = ["\)", "\(", "–", '"', "”", '"', "\[.*\]", ".*\|.*", "—"]
lda = LatentDirichletAllocation(n_components=3)
nlp = spacy.load("en_core_web_trf")
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


debug = True


def debug_print(text):
    if debug:
        print(f"{text}")


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


from sklearn.decomposition import NMF


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
        # print(f"Shape of outputs: {outputs.last_hidden_state.shape}")
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
        # print(f"Sentiment score: {sentiment_score}")
        return sentiment_score

    def get_emotion_classification(self, text):
        result = self.emo_classifier(text)
        # print(f"Emotion classification: {result[0]['label']}")
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
    # #print(type(model_output))
    # #print(type(attention_mask))
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
        # #print(self.description_embedding[0].shape)
        # #print(self.description_embedding[1].shape)

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
        # self.analyze_description()

    # def analyze_description(self):
    #     self.analysis = manager.analyze_query_context(self.description)
    #     self.sentiment_score = self.analysis["sentiment_score"]
    #     self.emotion_classification = self.analysis["emotion_classification"]
    #     self.keywords = self.analysis["keywords"]
    #     self.entities = self.analysis["named_entities"]
    #     self.main_subject = self.analysis["main_subject"]
    #     self.main_verb = self.analysis["main_verb"]
    #     self.main_object = self.analysis["main_object"]
    #     self.temporal_expressions = self.analysis["temporal_expressions"]
    #     self.verb_aspects = self.analysis["verb_aspects"]

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
        # #print(f"Specific memories: {[memory.description for memory in specific_memories]}")

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
        # Check if specific_memory.analysis is None, if so, analyze the description
        if specific_memory.analysis is None:
            specific_memory.analyze_description()
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
            manager.update_embeddings(specific_memory)
            manager.flat_access.faiss_index.add(
                specific_memory.embedding.cpu().detach().numpy()
            )
            manager.flat_access.index_id_to_node_id[
                manager.flat_access.faiss_index.ntotal - 1
            ] = specific_memory.description

            for fact_embed in specific_memory.get_facts_embeddings():

                manager.flat_access.faiss_index.add(fact_embed.cpu().detach().numpy())
                manager.flat_access.index_id_to_node_id[
                    manager.flat_access.faiss_index.ntotal - 1
                ] = specific_memory.description

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

        # #print(embeddings)
        self.faiss_index = faiss.IndexFlatL2(embeddings.shape[1])
        # for embedding in embeddings:
        # #print(embeddings[0].shape)
        self.faiss_index.add(embeddings.cpu().detach().numpy())
        return self.faiss_index


def is_question(sentence):
    # Check for question mark
    if sentence.strip().endswith("?"):
        return True
    doc = nlp(sentence)
    # Check if the sentence contains a question word and an auxiliary verb
    if doc[0].pos_ == "AUX" and doc[0].tag_ in ["VBZ", "VBP", "VBD", "VBG", "VBN"]:
        return True

    # Check for auxiliary presence and interrogative words (who, what, where, etc.)
    aux = any(token.dep_ == "aux" for token in doc)
    interrogative = any(token.tag_ in ["WP", "WRB", "WDT"] for token in doc)
    if aux and interrogative:
        return True

    return False


# @track_calls
class SpecificMemory(Memory):
    def __init__(
        self,
        description,
        parent_memory,
        importance_score,
        subject=None,
        normalized_embeddings=False,
    ):
        super().__init__(description)
        self.normalized_embeddings = normalized_embeddings
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
        self.facts = []
        self.facts_embeddings = None
        self.map_fact_embeddings = {}
        self.analyze_description()

    def __getstate__(self):
        state = self.__dict__.copy()
        # Remove the embedding attribute
        del state["embedding"]
        del state["att_mask"]
        del state["facts_embeddings"]
        return state

    def __setstate__(self, state):
        # Restore instance attributes
        self.__dict__.update(state)
        # Add the embedding attribute back so it doesn't cause issues
        self.embedding = None
        self.att_mask = None
        self.facts_embeddings = None

    def analyze_description(self):
        self.analysis = manager.analyze_query_context(self.description)
        self.sentiment_score = self.analysis["sentiment_score"]
        self.emotion_classification = self.analysis["emotion_classification"]
        self.keywords = self.analysis["keywords"]
        # self.entities = self.analysis["named_entities"]
        self.main_subject = self.analysis["main_subject"]
        self.main_verb = self.analysis["main_verb"]
        self.main_object = self.analysis["main_object"]
        # self.temporal_expressions = self.analysis["temporal_expressions"]
        self.verb_aspects = self.analysis["verb_aspects"]
        self.facts = self.analysis["facts"]
        self.facts_embeddings = self.get_facts_embeddings()

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

    def get_embedding(self, normalize=None):
        if normalize is not None and self.normalized_embeddings != normalize:
            self.normalized_embeddings = normalize
            self.embedding = None
        if self.embedding is None:
            self.embedding_and_mask = self.generate_embedding(
                normalize=self.normalized_embeddings
            )
            self.embedding = mean_pooling(
                self.embedding_and_mask[0], self.embedding_and_mask[1]
            )
        return self.embedding, self.att_mask

    def generate_embedding(self, string=None, normalize=None):
        if string is None:
            string = self.description
        description = [string.strip()]
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
        if normalize is not None:
            if normalize:
                outputs.last_hidden_state = faiss.normalize_L2(
                    outputs.last_hidden_state
                )
        return [outputs.last_hidden_state, input["attention_mask"]]

    def get_facts_embeddings(self):
        if self.facts_embeddings is None:
            facts = self.facts
            facts_embeddings = []
            for fact in facts:
                fact_embedding = self.generate_embedding(fact)
                fact_embedding = mean_pooling(fact_embedding[0], fact_embedding[1])
                facts_embeddings.append(fact_embedding)
                self.map_fact_embeddings[fact_embedding] = fact
            self.facts_embeddings = facts_embeddings

        return self.facts_embeddings


# @track_calls
class FlatMemoryAccess:
    def __init__(self, memory_embeddings={}, json_file=None, index_load_filename=None):
        self.index_load_filename = index_load_filename
        self.index_is_normalized = False
        self.recent_memories = deque(maxlen=50)
        self.common_memories = {}
        self.repetitive_memories = {}
        self.urgent_query_memories = {}
        self.most_importance_memories = {}
        self.index_id_to_node_id = {}
        self.faiss_index = None
        self.euclidean_threshold = 0.35
        self.index_build_count = 0
        self.memory_embeddings = memory_embeddings
        self.specific_memories = {}
        self.initialize_faiss_index(768, "ip", index_load_filename=index_load_filename)

        for key, value in vars(self).items():

            if (
                isinstance(value, spacy.tokens.token.Token)
                or isinstance(value, spacy.tokens.Doc)
                or isinstance(value, spacy.tokens.Span),
            ):
                if isinstance(value, spacy.tokens.token.Token):
                    print(
                        f"Token object found in flat_access at key: {key} with value: {value} and type: {type(value)} and text: {value.text}"
                    )
                elif isinstance(value, spacy.tokens.Doc):
                    print(
                        f"Doc object found in flat_access at key: {key} with value: {value} and type: {type(value)} and text: {value.text}"
                    )
                elif isinstance(value, spacy.tokens.Span):
                    print(
                        f"Span object found in flat_access at key: {key} with value: {value} and type: {type(value)} and text: {value.root.text}"
                    )
            elif isinstance(value, dict):
                for k, v in value.items():
                    if isinstance(v, spacy.tokens.token.Token):
                        print(
                            f"Token object found in dictionary at key: {k} in flat_access at key: {key}"
                        )
                    elif isinstance(v, spacy.tokens.Doc):
                        print(
                            f"Doc object found in dictionary at key: {k} in flat_access at key: {key}"
                        )
                    elif isinstance(v, spacy.tokens.Span):
                        print(
                            f"Span object found in dictionary at key: {k} in flat_access at key: {key}"
                        )
            elif isinstance(value, object) and not isinstance(
                value, (str, int, float, list, tuple)
            ):
                print(f"{name}.{key}, {type(value)}, {value}")

        # self.json_file = json_file
        # self.cache = retrieve_cache(self.json_file)

    def __getstate__(self):
        state = self.__dict__.copy()
        del state["faiss_index"]
        del state["memory_embeddings"]
        del state["recent_memories"]

        return state

    def __setstate__(self, state):
        # Restore instance attributes
        self.__dict__.update(state)
        # Add the embedding attribute back so it doesn't cause issues
        self.faiss_index = None
        self.memory_embeddings = {}
        self.recent_memories = deque(maxlen=50)
        # Convert string representations of Token objects back to Token objects
        # for key, value in vars(self).items():
        #     if isinstance(value, str):
        #         self.__dict__[key] = nlp(value)[0]
        #     elif isinstance(value, dict):
        #         for k, v in value.items():
        #             if isinstance(v, str):
        #                 self.__dict__[key][k] = nlp(v)[0]

    def save_all_specific_memories_embeddings_to_file(self, filename):
        # save all specific memories embeddings to file as a numpy array.
        # Save them individually, using the description as the key so they can later be loaded and assigned to the correct specific memory object
        self.set_all_memory_embeddings_to_normalized(self.index_is_normalized)
        # Create a dictionary with the description as the key and the tuple of the embedding and attention mask as the value
        memory_dict = {
            memory.description: memory.get_embedding()
            for memory in self.get_specific_memories()
        }

        # separate the embeddings and attention masks into two separate dictionaries
        embeddings_dict = {key: value[0] for key, value in memory_dict.items()}
        att_mask_dict = {key: value[1] for key, value in memory_dict.items()}
        # Save the dictionaries to separate files
        norm_str = "_normalized" if self.index_is_normalized else ""
        np.save(f"{filename}_embeddings{norm_str}.npy", embeddings_dict)
        np.save(f"{filename}_att_mask.npy", att_mask_dict)

    def load_all_specific_memories_embeddings_from_file(self, filename):
        # MUST be called after the specific memories have been created or loaded and before the specific memories are used
        # Load the embeddings and attention masks from file and assign them to the correct specific memory object
        norm_str = "_normalized" if self.index_is_normalized else ""
        embeddings_dict = np.load(
            f"{filename}_embeddings{norm_str}.npy", allow_pickle=True
        ).item()
        att_mask_dict = np.load(f"{filename}_att_mask.npy", allow_pickle=True).item()

        for key, value in embeddings_dict.items():
            specific_memory = self.get_specific_memory_by_description(key)
            if specific_memory is not None:
                specific_memory.embedding = value
                specific_memory.att_mask = att_mask_dict[key]

    def get_specific_memory_by_description(self, description):
        for memory in self.get_specific_memories():
            if memory.description == description:
                return memory
        return None

    def set_all_memory_embeddings_to_normalized(self, normalized=None):
        if normalized is None:
            normalized = not self.index_is_normalized
        if normalized == self.index_is_normalized:
            return
        self.index_is_normalized = normalized
        for memory in self.get_specific_memories():
            memory.normalized_embeddings = normalized

    def initialize_faiss_index(
        self, dimension, metric="l2", normalize=False, index_load_filename=None
    ):
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
        if (
            index_load_filename is not None
            and self.index_load_filename != index_load_filename
        ):
            self.index_load_filename = index_load_filename

        self.set_all_memory_embeddings_to_normalized(normalize)

        if self.index_load_filename is not None and os.path.exists(
            self.index_load_filename
        ):
            try:
                self.faiss_index = faiss.read_index(index_load_filename)
                print(f"\n Loaded index from file {index_load_filename} \n")
                # Type
                return None
            except:
                print(f"Could not load FAISS index from {index_load_filename}")
                self.faiss_index = None
                raise ValueError("Could not load FAISS index from file")

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

        # print(f"FAISS index initialized with dimension {self.faiss_index.d} and metric {metric}")
        embeddings = []
        index_key = 0
        # Check the specific memories dict is not empty
        if len(self.specific_memories) == 0:
            return None
        for specific_memory in self.get_specific_memories():
            if (
                isinstance(specific_memory, SpecificMemory)
                and specific_memory.embedding is not None
            ):
                embeddings.append(specific_memory.get_embedding()[0])
                for fact in specific_memory.get_facts_embeddings():
                    embeddings.append(fact)
                self.index_id_to_node_id[manager.flat_access.faiss_index.ntotal - 1] = (
                    specific_memory.description
                )
                index_key += 1
            elif (
                isinstance(specific_memory, SpecificMemory)
                and specific_memory.embedding is None
            ):
                # specific_memory.get_embedding()
                embeddings.append(specific_memory.get_embedding()[0], normalize)
                for fact in specific_memory.get_facts_embeddings():
                    embeddings.append(fact)
                self.index_id_to_node_id[manager.flat_access.faiss_index.ntotal - 1] = (
                    specific_memory.description
                )

                index_key += 1

        if embeddings:
            # print(f"Adding {len(embeddings)} embeddings to the FAISS index")
            # for em in embeddings:
            # print(f"Embedding shape: {em.shape}")
            # embeddings = np.array([e.cpu().detach() for e in embeddings])

            print(f"Embedding type: {type(embeddings)}")
            first_type = type(embeddings[0])
            for i in range(len(embeddings)):
                if type(embeddings[i]) != first_type:
                    raise ValueError(
                        f"Embedding type mismatch: {type(embeddings[i])} != {first_type}"
                    )

                embeddings[i] = embeddings[i].cpu().detach().numpy()
                # if normalize:
                #     faiss.normalize_L2(embeddings[i])
            if normalize:
                self.index_is_normalized = True
            else:
                self.index_is_normalized = False
            embeddings = np.concatenate(embeddings)
            # if normalize:
            #     faiss.normalize_L2(embeddings)
            # embeddings = torch.stack([e.squeeze(0) for e in embeddings]).cpu().detach().numpy()
            # print(f"Embedding shape: {embeddings.shape}")
            # print(f"Embedding type: {type(embeddings)}")
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

            # print(f"FAISS index populated with {self.faiss_index.ntotal} embeddings")
            self.index_build_count += 1
            # self.faiss_index.add(np.array(embeddings))

        else:
            print("No embeddings found to populate the FAISS index")
            time.sleep(5)
            raise ValueError("No embeddings found to populate the FAISS index")
        print(f"The current index type is: {type(self.faiss_index).__name__} \n")

        with open("results.txt", "a") as f:
            f.write(
                f"\n Index type: {type(self.faiss_index).__name__} build count: {self.index_build_count}\n \n \n"
            )

    def delete_index(self):
        self.faiss_index.reset()
        self.faiss_index = None
        self.index_is_normalized = False

    def save_index_to_file(self, filename):
        faiss.write_index(self.faiss_index, filename)

    def load_index_from_file(self, filename, normalize=False):
        self.faiss_index = faiss.read_index(filename)
        if normalize != self.index_is_normalized:
            self.set_all_memory_embeddings_to_normalized(normalize)
        self.index_is_normalized = normalize
        self.set_all_memory_embeddings_to_normalized(normalize)
        print(
            f"Loaded index from file. The current index type is: {type(manager.flat_access.faiss_index).__name__} \n \n"
        )

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
            # print(f"Error retrieving embedding for query {query}")
            return {}
        if self.faiss_index is None:
            raise ValueError("FAISS index is not initialized")
            self.initialize_faiss_index(query_embedding.shape[1], "ip")

        # self.faiss_index.nprobe = num_clusters
        query_vec = query_embedding.cpu().detach().numpy()
        distances, indices = self.faiss_index.search(x=query_vec, k=top_k)

        assert len(indices) == len(
            distances
        ), "Indices and distances are not the same length"

        if similarity_metric == "cosine":
            # Convert L2 distances to cosine similarity scores
            similarities = 1 - distances
        else:
            similarities = distances

        # print(f"Similarities: {similarities},\n Indices: {indices},\n Distances: {distances}\n")
        # print(self.get_graph().nodes)
        # print(self.index_id_to_node_id)

        # Convert indices to memory IDs
        # for idx in indices[0]:
        #     if idx != -1:
        # print(f"Memory ID: {self.index_id_to_node_id[idx]}")
        # print(f"Corresponding graph node: {manager.hierarchy.memory_graph.nodes[self.index_id_to_node_id[idx]]}")

        # print(f"Corresponding graph node: {self.get_graph().nodes[self.index_id_to_node_id[idx]]}")
        similar_memories = {
            self.specific_memories[self.index_id_to_node_id[idx]]: float(similarity)
            for idx, similarity in zip(indices[0], similarities[0])
            if (
                idx != -1
                and similarity > threshold
                and self.index_id_to_node_id[idx] in self.specific_memories
                and len(indices[0]) == len(similarities[0])
            )
        }
        assert isinstance(
            similar_memories, dict
        ), f"Similar memories is not a dictionary: {similar_memories}"

        end_time = time.time()
        # print(f"Time taken to find similar memories: {end_time - start_time:.4f} seconds")
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
                # print(f"Error retrieving embedding for memory {specific_memory_id}: {e}")
                return []

        similarities = {}
        for memory in self.get_graph().nodes:
            if memory == specific_memory_id:
                continue
            if self.get_graph().nodes[memory].embedding is None:
                try:
                    self.get_graph().nodes[memory].get_embedding()
                except Exception as e:
                    # print(f"Error retrieving embedding for memory {memory}: {e}")
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
                # print(f"Error retrieving embedding for memory {specific_memory_id}: {e}")
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
                    # print(f"Error retrieving embedding for memory {memory}: {e}")
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
        # print(f"Time taken to find similar memories: {end_time - start_time:.4f} seconds")
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
                # print(f"Error retrieving embedding for memory {specific_memory_id}: {e}")
                return []

        if self.faiss_index is None or self.faiss_index.ntotal == 0:
            # print(f"(Re)Initializing FAISS index with {query_memory.embedding.shape[0]} dimensions")
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
        # print(f"Time taken to find similar memories: {end_time - start_time:.4f} seconds")
        return similar_memories

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

    def get_some_memories(self):
        return list(
            self.recent_memories
            + list(self.common_memories.values())
            + list(self.repetitive_memories.values())
            + list(self.urgent_query_memories.values())
            + [mem[1] for mem in self.most_importance_memories]
        )

    def get_specific_memories(self):
        for memory in self.specific_memories.values():
            yield memory


# Global variable declarations - initialized here to prevent NameError in class definitions
# These will be properly initialized in the main section or when the module is imported
manager = None
model = None
sentiment_analysis = None


# @track_calls
class MemoryManager:
    def __init__(self, gametime_manager, index_load_filename=None):
        self.flat_access = FlatMemoryAccess(index_load_filename=index_load_filename)
        self.index_load_filename = index_load_filename
        self.memory_embeddings = {}
        self.complex_keywords = set()
        self.similarity_threshold_specific = 0.4
        self.similarity_threshold_general = 0.25
        self.recent_queries = deque(maxlen=50)
        self.faiss_index_recent_queries_flatl2 = None
        self.map_tags = {}
        self.general_memories = {}
        self.gametime_manager = gametime_manager
        self.lda_tokenizer = RegexpTokenizer(r"\w+")
        self.lda_vectorizer = CountVectorizer(stop_words="english", min_df=1)
        self.lda = LatentDirichletAllocation(
            n_components=3,
            max_iter=5,
            learning_method="online",
            learning_offset=50.0,
            random_state=0,
        )
        self.tfidf_vectorizer = TfidfVectorizer(
            stop_words=sentiment_analysis.stop_words
        )
        self.rake = Rake()
        assert self.gametime_manager is not None, "Game time manager is required"

    def init_memories(self, general_memories):
        # self.general_memories = general_memories
        for general_memory in general_memories:
            general_memory.index_memories()
            self.flat_access.add_memory(general_memory)

    def index_recent_queries_flatl2(self):
        # print(f"\n Indexing recent queries \n")
        if len(self.recent_queries) < 1:
            return None
        embeddings = [query.query_embedding for query in self.recent_queries]
        # print(f"Length of embeddings: {len(embeddings)}")

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
        # print(f"\n Indexing recent queries \n")
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

    def add_general_memory(self, general_memory: GeneralMemory):
        self.general_memories[general_memory.description] = general_memory
        self.update_embeddings(general_memory)
        return general_memory

    def update_embeddings(self, memory=None):
        if isinstance(memory, SpecificMemory):
            self.memory_embeddings.update({memory.description: memory.get_embedding()})
        elif isinstance(memory, GeneralMemory):
            for specific_memory in memory.get_specific_memories():
                if (
                    specific_memory != None
                    and specific_memory not in self.memory_embeddings
                ):
                    self.memory_embeddings[specific_memory.description] = (
                        specific_memory.get_embedding()
                    )
        elif isinstance(memory, list):
            for mem in memory:
                if mem != None and mem not in self.memory_embeddings:
                    self.memory_embeddings.update(
                        {mem.description: mem.get_embedding()}
                    )
        elif memory == None:
            for memory in self.flat_access.get_specific_memories():
                if memory not in self.memory_embeddings:
                    self.memory_embeddings.update(
                        {memory.description: memory.get_embedding()}
                    )
            for memory in self.recent_queries:
                if memory not in self.memory_embeddings:
                    self.memory_embeddings.update(
                        {memory.query: memory.get_embedding()}
                    )
        elif isinstance(memory, dict):
            self.memory_embeddings.update(memory)
        # Update the embeddings in flat access as well
        self.flat_access.memory_embeddings = self.memory_embeddings
        for gm in self.general_memories.values():
            for sm in gm.get_specific_memories():
                if sm.description not in self.flat_access.specific_memories.keys():
                    self.flat_access.specific_memories[sm.description] = sm

    def save_all_flat_access_memories_to_file(self, filename):
        try:
            for key, value in vars(self.flat_access).items():
                if isinstance(
                    value, spacy.tokens.token.Token or isinstance(v, spacy.tokens.Doc)
                ):
                    print(f"Token object found in flat_access at key: {key}")
                elif isinstance(value, dict):
                    for k, v in value.items():
                        if isinstance(v, spacy.tokens.token.Token):
                            print(
                                f"Token object found in dictionary at key: {k} in flat_access at key: {key}"
                            )
                elif isinstance(value, object) and not isinstance(
                    value, (str, int, float, list, tuple)
                ):
                    print(f"{name}.{key}, {type(value)}, {value}")
            with open(filename, "wb") as f:
                pickle.dump(self.flat_access, f)
        except (pickle.PicklingError, IOError) as e:
            print(f"Error while saving memories: {e}")

    def load_all_flat_access_memories_from_file(self, filename):
        try:
            with open(filename, "rb") as f:
                loaded_memories = pickle.load(f)
                self.flat_access = loaded_memories
            assert isinstance(
                self.flat_access, FlatMemoryAccess
            ), "Loaded object is not an instance of FlatMemoryAccess"
            self.flat_access.index_load_filename = None
        except (pickle.UnpicklingError, IOError) as e:
            print(f"Error while loading memories: {e}")

    def add_memory(self, memory):
        self.flat_access.add_memory(memory)

    def extract_entities(self, text):
        if isinstance(text, list):
            text = " ".join(text)
        doc = nlp(text)
        return [ent.text for ent in doc.ents]

    # Function to perform LDA topic modeling

    def extract_lda_keywords(self, docs, num_topics=3, num_words=3):
        tokenizer = self.lda_tokenizer
        docs = [
            re.sub(r"|".join(map(re.escape, remove_list)), "", docs) for docs in docs
        ]

        doc_tokens = [tokenizer.tokenize(doc.lower()) for doc in docs]
        docs = [" ".join(tokens) for tokens in doc_tokens]

        vectorizer = self.lda_vectorizer
        X = vectorizer.fit_transform(docs)

        self.lda.fit(X)

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

        X = self.tfidf_vectorizer.fit_transform(docs)
        feature_array = self.tfidf_vectorizer.get_feature_names_out()
        tfidf_sorting = [
            X.toarray().argsort(axis=1)[:, -top_n:] for _ in range(X.shape[0])
        ]
        tfidf_keywords = [feature_array[i] for row in tfidf_sorting for i in row[0]]
        return tfidf_keywords

    # Function to extract keywords using RAKE
    def extract_rake_keywords(self, docs, top_n=2):
        # print(type(docs))

        if isinstance(docs, list):
            docs = " ".join(docs)
        self.rake.extract_keywords_from_text(docs)
        rake_keywords = set(self.rake.get_ranked_phrases()[:top_n])
        return rake_keywords

    def get_query_embedding(self, query):
        input = model.tokenizer(
            query, return_tensors="pt", padding=True, truncation=True
        )
        input = input.to(model.device)
        outputs = model.forward(input["input_ids"], input["attention_mask"])
        # print(f"\n Shape of query embedding: {outputs.last_hidden_state.shape}")
        query_embedding = mean_pooling(
            outputs.last_hidden_state, input["attention_mask"]
        )
        return query_embedding

    def retrieve_memories_bst(self, general_memory, query):
        # print(f"\n Retrieving memories from BST for general memory: {general_memory.description} and query: {query.query}")
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
        # print(f"\n Matching memories BST: {matching_memories}")
        return matching_memories

    def retrieve_from_flat_access(self, query_features):
        # print(f"\n Retrieving from flat access: {query_features.query}")
        # rel_mems = [memory for memory in self.flat_access.get_specific_memories() if self.is_relevant_flat_memory(memory, query_features)]

        rel_mems = self.flat_access.find_memories_by_query(
            query_features, threshold=0.5
        )

        # print(f"\n Relevant memories and scores, flat access: {rel_mems}")
        return rel_mems

    def get_sentiment_score(self, text):
        return sentiment_analysis.get_sentiment_score(text)

    def get_emotion_classification(self, text):
        return sentiment_analysis.get_emotion_classification(text)

    def extract_relationships(self, semantic_roles, text):
        # print(f"\n Extracting relationships from semantic roles: {semantic_roles}")
        # Extract relationships from the semantic roles and text
        # Remember, who is doing what to whom is what this function extracts
        relationships = []

        return relationships

    def find_root_verb(self, token):
        """Recursively find the root verb related to the current token."""
        if token.dep_ == "ROOT":
            return token
        elif token.head.pos_ in [
            "VERB",
            "VB",
            "VBD",
            "VBG",
            "VBN",
            "VBP",
            "VBZ",
            "ROOT",
        ]:
            return self.find_root_verb(token.head)
        return None

    def extract_facts(self, token, explored=set()):
        """Recursively extract facts from nested clauses and conjunctions."""
        facts = []
        if token in explored:
            return facts
        explored.add(token)

        # Handle nested clauses and implications
        for child in token.children:
            if child.dep_ in [
                "acl",
                "relcl",
                "advcl",
                "ccomp",
                "xcomp",
            ] or child.pos_ in [
                "VERB",
                "VB",
                "VBD",
                "VBG",
                "VBN",
                "VBP",
                "VBZ",
                "ROOT",
            ]:
                fact = " ".join([child.text for child in child.subtree])
                facts.append(fact)
                facts.extend(self.extract_facts(child, explored))

            # Handle conjunctions by identifying related verbs or entities
            elif child.dep_ in [
                "conj",
                "cconj",
                "sconj",
                "cc",
                "mark",
            ] or child.pos_ in [
                "SCONJ",
                "CCONJ",
                "CONJ",
            ]:
                facts.extend(self.extract_facts(child, explored))

        return facts

    def analyze_query_context(self, query):
        # print(f"\n Analyzing query: {query}")
        # Linguistic Analysis
        docs = nlp(query)
        # print(docs)
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
        # #print(f"Outputs: {outputs}")
        # inputs_decoded = tiny_brain_io.tokenizer.decode(outputs[1][0], skip_special_tokens=True)
        # #print(f"Input ids decoded: {inputs_decoded}")
        # outputs_decodedb = tiny_brain_io.tokenizer.decode(outputs[0].logits.argmax(2)[0], skip_special_tokens=True)

        # #print(f"Outputs decodedb: {outputs_decodedb}")

        # perplexity = torch.exp(outputs[0])
        # #print(f"Perplexity: {perplexity}")
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
        # print(f"Flesch-Kincaid Grade Level: {flesch_kincaid_grade_level}")

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
        # print(f"Gunning Fog Index: {gunning_fog_index}")

        # Calculate the complexity of the sentence using the Coleman-Liau Index
        # https://en.wikipedia.org/wiki/Coleman%E2%80%93Liau_index
        # Range of coleman_liau_index: 1-12
        # 1-6: 6th grade
        # 7-8: 7th & 8th grade
        # 9-12: 9th to 12th grade
        coleman_liau_index = 5.89 * (num_words / 1) - 29.6 * (1 / 1) - 15.8
        # print(f"Coleman-Liau Index: {coleman_liau_index}")

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
        # print(f"Automated Readability Index: {automated_readability_index}")

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
        # print(f"SMOG Index: {smog_index}")

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
        # print(f"Dale-Chall Readability Score: {dale_chall_readability_score}")

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
        # print(f"Linsear Write Formula: {linsear_write_formula}")

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
        # print(f"Spache Readability Formula: {spache_readability_formula}")

        # Determine the depth of the parse tree for the sentence
        parse_tree = [list(sent.subtree) for sent in docs.sents]
        parse_tree_depth = max([len(tree) for tree in parse_tree])
        print(f"Parse Tree: {parse_tree}")
        print(f"Parse Tree Depth: {parse_tree_depth}")

        # Determine the number of entities in the sentence
        num_entities = len([ent for ent in docs.ents])
        print(f"Number of entities: {num_entities}")

        # Determine the number of noun phrases in the sentence
        noun_phrases = [chunk for chunk in docs.noun_chunks]
        num_noun_phrases = len(noun_phrases)
        print(f"Noun Phrases: {noun_phrases}")
        print(f"Number of noun phrases: {num_noun_phrases}")

        # Determine the number of sentences in the input
        num_sentences = len([sent for sent in docs.sents])

        # Determine the number of subordinate clauses in the input
        subordinate_clauses = [token for token in docs if token.dep_ == "mark"]
        num_subordinate_clauses = len(subordinate_clauses)
        print(f"Subordinate clauses: {subordinate_clauses}")
        print(f"Number of subordinate clauses: {num_subordinate_clauses}")

        # Determine the number of T-units in the input
        # A t-unit is an independent clause and all its dependent clauses
        num_t_units = sum(
            1
            for sent in docs.sents
            for token in sent
            if token.tag_ in ["VBD", "VBZ", "VBP", "VBG", "VBN"]
        )
        print(f"Number of T-units: {num_t_units}")
        t_units = [
            sent
            for sent in docs.sents
            if any(token.tag_ in ["VBD", "VBZ", "VBP", "VBG", "VBN"] for token in sent)
        ]
        t_units = [sent.text for sent in t_units]
        for t_unit in t_units:
            print(f"T-unit: {t_unit} \n")
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
                "root",
                "dep",
                "mark",
                "punct",
                "cc",
                "root",
                "dep",
                "punct",
            ]
        ]
        num_constituents = len(constituents)
        constituent_dependencies = [constituent.dep_ for constituent in constituents]
        print(f"Number of constituents: {num_constituents}")
        if len(constituents) != len(constituent_dependencies):
            print(f"Constituents: {constituents}")
        print(f"Constituent dependencies: {constituent_dependencies}")

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
        print(f"Number of crossing dependencies: {num_crossing_dependencies}")

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
        # print(f"\n Lexical Density: {lexical_density}")

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
        # print(f"\n Lexical Diversity: {lexical_diversity}")

        # Calculate the use of passive voice in the input
        passive_voice = len([token for token in docs if token.dep_ == "nsubjpass"])
        # Use passive_voice to determine the subject and object of the passive voice
        passive_voice_details = {}
        # print(f"\n Passive Voice: {passive_voice}")
        for token in docs:
            if token.dep_ == "nsubjpass":
                subject = token.text
                object = token.head.text
                # print(f"Passive Voice: Subject: {subject}, Object: {object} \n")
                passive_voice_details[subject] = object

        # Calculate the use of active voice in the input
        active_voice = len([token for token in docs if token.dep_ == "nsubj"])
        # Use active_voice to determine the subject and object of the active voice
        active_voice_details = {}
        # print(f"\n Active Voice: {active_voice}")
        for token in docs:
            if token.dep_ == "nsubj":
                subject = token.text
                object = token.head.text
                # print(f"Active Voice: Subject: {subject}, Object: {object} \n")
                active_voice_details[subject] = object

        # Calculate the use of modals in the input
        modals = [token for token in docs if token.tag_ == "MD"]
        # modal_len = len(modals)
        # modal_frequency = sum([token.prob for token in modals if token.prob != -1])
        print(f"\n Modals: {modals}")

        # Calculate the use of adverbs in the input
        adverbs = [token for token in docs if token.pos_ == "ADV"]
        print(f"\n Adverbs: {adverbs}")
        # adverb_len = len(adverbs)
        # adverb_frequency = sum([token.prob for token in adverbs if token.prob != -1])

        # Calculate the use of adjectives in the input
        adjectives = [token for token in docs if token.pos_ == "ADJ"]
        print(f"\n Adjectives: {adjectives}")
        # adjective_len = len(adjectives)
        # adjective_frequency = sum([token.prob for token in adjectives if token.prob != -1])

        # Calculate the use of pronouns in the input
        pronouns = [token for token in docs if token.pos_ == "PRON"]
        print(f"\n Pronouns: {pronouns}")
        # pronoun_len = len(pronouns)
        # pronoun_frequency = sum([token.prob for token in pronouns if token.prob != -1])

        # # Calculate the use of conjunctions in the input
        # conjunctions = [token for token in docs if token.pos_ == "CCONJ"]
        # print(f"\n Conjunctions: {conjunctions}")
        # conjunction_len = len(conjunctions)
        # conjunction_frequency = sum([token.prob for token in conjunctions if token.prob != -1])

        # Calculate the use of determiners in the input
        # determiners = [token for token in docs if token.pos_ == "DET"]

        determiners = []
        # print(f"\n Determiners: {determiners}")
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
                determiners.append(
                    {
                        "associated_noun": associated_noun,
                        "definiteness": definiteness,
                        "quantity": quantity,
                        "possession": possession,
                    }
                )

        for determiner in determiners:
            for key, value in determiner.items():
                print(f"\n Determiner: {key}, Details: {value}")

        for token in docs:

            if token.pos_ == "ADV" and token.text.lower() in urgent_words:
                is_urgent = True
            if token.pos_ == "VERB" and token.tag_ == "VBZ":  # Imperative verbs
                is_urgent = True
            if token.pos_ in ["INTJ", "ADV"] or token.dep_ in ["prt"]:
                if token.text.lower() in urgent_words:
                    is_urgent = True
            if token.pos_ == "INTJ":  # Interjections are strong indicators
                is_urgent = True
        print(f"\n Is urgent: {is_urgent}")

        # Calculate the use of prepositions in the input
        # Example of how to structure prepositions
        prepositions = [
            (token, token.dep_, token.head) for token in docs if token.pos_ == "ADP"
        ]
        print(f"\n Prepositions: {prepositions}")
        # preposition_len = len(prepositions)

        # Calculate the use of interjections in the input
        interjections = [token for token in docs if token.pos_ == "INTJ"]
        # interjection_len = len(interjections)
        # interjection_frequency = sum([token.prob for token in interjections if token.prob != -1])
        print(f"\n Interjections: {interjections}")

        # Calculate the use of particles in the input
        particles = [token for token in docs if token.pos_ == "PART"]
        print(f"\n Particles: {particles}")
        # particle_len = len(particles)
        # particle_frequency = sum([token.prob for token in particles if token.prob != -1])

        # Calculate the use of punctuations in the input
        punctuations = [token for token in docs if token.pos_ == "PUNCT"]
        print(f"\n Punctuations: {punctuations}")
        # punctuation_len = len(punctuations)
        # punctuation_frequency = sum([token.prob for token in punctuations if token.prob != -1])

        # Calculate the use of symbols in the input
        symbols = [token for token in docs if token.pos_ == "SYM"]
        print(f"\n Symbols: {symbols}")
        # symbol_len = len(symbols)
        # symbol_frequency = sum([token.prob for token in symbols if token.prob != -1])

        # Calculate the use of numbers in the input
        numbers = [token for token in docs if token.pos_ == "NUM"]
        print(f"\n Numbers: {numbers}")
        # number_len = len(numbers)
        # number_frequency = sum([token.prob for token in numbers if token.prob != -1])

        # Calculate the use of foreign words in the input
        foreign_words = [token for token in docs if token.pos_ == "X"]
        print(f"\n Foreign Words: {foreign_words}")
        # foreign_word_len = len(foreign_words)
        # foreign_word_frequency = sum([token.prob for token in foreign_words if token.prob != -1])

        # Calculate the use of proper nouns in the input
        proper_nouns = [token for token in docs if token.pos_ == "PROPN"]
        print(f"\n Proper Nouns: {proper_nouns}")
        # proper_noun_len = len(proper_nouns)
        # proper_noun_frequency = sum([token.prob for token in proper_nouns if token.prob != -1])

        # Calculate the use of common nouns in the input
        common_nouns = [token for token in docs if token.pos_ == "NOUN"]
        print(f"\n Common Nouns: {common_nouns}")
        # common_noun_len = len(common_nouns)
        # common_noun_frequency = sum([token.prob for token in common_nouns if token.prob != -1])

        # Calculate the use of verbs in the input
        verbs = [
            token
            for token in docs
            if token.pos_ in ["VERB", "VB", "VBD", "VBG", "VBN", "VBP", "VBZ", "AUX"]
        ]
        if not verbs:
            verbs = [
                token
                for token in docs
                if token.dep_ in ["aux", "ccomp", "xcomp", "acomp"]
            ]

        sentence_fragment = False if verbs else True
        if sentence_fragment:
            print("Sentence is a fragment")
            return None
        print(f"\n Verbs: {verbs}")
        # verb_len = len(verbs)
        # verb_frequency = sum([token.prob for token in verbs if token.prob != -1])

        # Calculate the use of adpositions in the input
        adpositions = [token for token in docs if token.pos_ == "ADP"]
        print(f"\n Adpositions: {adpositions}")
        # adposition_len = len(adpositions)
        # adposition_frequency = sum([token.prob for token in adpositions if token.prob != -1])

        # Calculate the use of adverbs in the input
        adverbs = [token for token in docs if token.pos_ == "ADV"]
        print(f"\n Adverbs: {adverbs}")
        # adverb_len = len(adverbs)
        # adverb_frequency = sum([token.prob for token in adverbs if token.prob != -1])

        # Calculate the use of auxiliaries in the input
        auxiliaries = [token for token in docs if token.pos_ == "AUX"]
        print(f"\n Auxiliaries: {auxiliaries}")
        # auxiliary_len = len(auxiliaries)
        # auxiliary_frequency = sum([token.prob for token in auxiliaries if token.prob != -1])

        # Calculate the use of conjunctions in the input
        conjunctions = [
            token
            for token in docs
            if (token.pos_ in ["CCONJ", "SCONJ"] or token.dep_ in ["cc", "mark"])
        ]
        print(f"\n Conjunctions: {conjunctions}")
        # conjunction_len = len(conjunctions)
        # conjunction_frequency = sum([token.prob for token in conjunctions if token.prob != -1])

        # Calculate the use of advanced vocabulary in the input
        advanced_vocabulary = [token for token in docs if token.prob > -15]
        # advanced_vocabulary_len = len(advanced_vocabulary)
        # advanced_vocabulary_frequency = sum([token.prob for token in advanced_vocabulary if token.prob != -1])

        # Calculate the use of simple vocabulary in the input
        simple_vocabulary = [token for token in docs if token.prob < -15]
        # simple_vocabulary_len = len(simple_vocabulary)
        # simple_vocabulary_frequency = sum([token.prob for token in simple_vocabulary if token.prob != -1])
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
            print(f"Modal: {modal.text}, Type: {modality_type}")

        # 2. Subject-Verb Relationship, subject_verb_relationships = {subject: [verb1, verb2, ...]}
        # subject_verb_relationships = {}
        # for key, value in modal_details.items():
        #     if value["subject"] not in subject_verb_relationships:
        #         subject_verb_relationships[value["subject"]] = []
        #     subject_verb_relationships[value["subject"]].append(value["verb"])
        #     print(
        #         f"Subject-Verb Relationship: {value['subject']}, Verb: {value['verb']}"
        #     )

        # # 3. Verb-Object Relationship
        # verb_object_relationships = {}
        # for key, value in modal_details.items():
        #     if value["verb"] not in verb_object_relationships:
        #         verb_object_relationships[value["verb"]] = []
        #     verb_object_relationships[value["verb"]].append(value["object"])
        #     print(
        #         f"Verb-Object Relationship: {value['verb']}, Object: {value['object']}"
        #     )

        # # 4. Subject-Object Relationship
        # subject_object_relationships = {}
        # for key, value in modal_details.items():
        #     if value["subject"] not in subject_object_relationships:
        #         subject_object_relationships[value["subject"]] = []
        #     subject_object_relationships[value["subject"]].append(value["object"])
        #     # print(
        #     #     f"Subject-Object Relationship: {value['subject']}, Object: {value['object']}"
        #     # )

        # Check for negation
        negations = [token for token in docs if token.dep_ == "neg"]

        # 4. Semantic Role Labeling

        # for token in docs:
        #     if token.dep_ != "ROOT":
        #         if token.head.text not in semantic_roles:
        #             semantic_roles[token.head.text] = []
        #         semantic_roles[token.head.text].append({token.text: token.dep_})
        #         print(f"Semantic Role: {token.text}, Role: {token.dep_}")
        # print(f"\n Semantic Roles: {semantic_roles}")
        semantic_roles = {}

        for token in docs:
            if token.dep_ != "ROOT":
                # Using the entire subtree for each token to capture phrases
                subtree_span = docs[token.left_edge.i : token.right_edge.i + 1]
                role_info = {
                    "text": subtree_span.text,  # The whole phrase
                    "dep": token.dep_,  # Dependency relation to the head
                    "pos": token.pos_,  # Part of Speech tag
                    "token": token,  # The token itself
                    "tag": token.tag_,  # Fine-grained part of speech tag
                    "index": token.i,  # Index of the token in the sentence
                }

                head = token.head
                if head not in semantic_roles:
                    semantic_roles[head] = []
                semantic_roles[head].append(role_info)

        for head, roles in semantic_roles.items():
            print(
                f"Head: {head.text} (POS: {head.pos_}) (dep: {head.dep_}) (tag: {head.tag_}) (index: {head.i})"
            )

            # Sort roles based on dependency relation

            roles.append(
                {
                    "text": head.text,
                    "dep": head.dep_,
                    "pos": head.pos_,
                    "token": head,
                    "tag": head.tag_,
                    "index": head.i,
                }
            )
            sorted_roles = sorted(roles, key=lambda role: role["token"].i)

            composite = " ".join([role["text"] for role in sorted_roles])

            print(f"  Composite: {composite}")

            for role in sorted_roles:
                print(f"  Role: {role}")

        # Initialize structure to hold semantic roles
        semantic_rolesb = {
            "who": [],
            "did": [],
            "whom/what": [],
            "when": [],
            "where": [],
            "why": [],
            "how": [],
        }

        # Mapping of dependency tags to semantic roles
        dep_to_role = {
            "nsubj": "who",
            "csubj": "who",
            "csubjpass": "who",
            "nsubjpass": "who",  # Subjects
            # "ROOT": "did",  # Verbs
            # "aux": "did",
            # "xcomp": "did",
            "ccomp": "did",
            "advcl": "did",
            "dobj": "whom/what",
            "pobj": "whom/what",  # Objects
            "acomp": "how",
            "oprd": "how",  # Manner
            "pcomp": "where",  # Place
            "npadvmod": "when",
            "advmod": "when",  # Time
            "relcl": "did",  # Relative clauses
        }

        # Iterate over tokens to assign roles
        actions = []
        facts = []
        root_verb = None

        for token in docs:
            role = (
                dep_to_role.get(token.dep_, None)
                if token.dep_ != "ROOT"
                else (
                    "did"
                    if token.tag_ in ["VERB", "VB", "VBD", "VBG", "VBN", "VBP", "VBZ"]
                    else "who"
                )
            )

            if token.dep_ == "conj" and token.tag_ in [
                "VERB",
                "VB",
                "VBD",
                "VBG",
                "VBN",
                "VBP",
                "VBZ",
            ]:
                role = "did"

            if token.dep_ == "ROOT":
                root_verb = token
            if token.dep_ == "ROOT" and token.pos_ == "AUX":
                if token not in semantic_rolesb["did"]:
                    semantic_rolesb["did"].append(token)

            # if token.dep_ == "nsubj" and token.head.dep_ == "ROOT":
            #     semantic_rolesb["did"].append(token.text)
            if token.head.dep_ in ["aux", "auxpass"]:
                head = token.head
                rel_pos = token.i - head.i
                # Find 'head' in one of the lists from semantic_rolesb values
                if rel_pos != -1:
                    for key, value in semantic_rolesb.items():
                        if head.text in value:
                            # Change the value of the key to the new value
                            semantic_rolesb[key].remove(head)

            if role and role not in ["who", "whom/what"]:
                if token.i not in [tk.i for tk in semantic_rolesb[role]]:
                    semantic_rolesb[role].append(token)

            elif role in ["who", "whom/what"]:
                for noun_chunk in noun_phrases:
                    if token.text in noun_chunk.text:
                        semantic_rolesb[role].append(noun_chunk.root)
            elif token.dep_ == "prep" and token.head.dep_ in [
                "ROOT",
                "advcl",
            ]:  # Handle prepositions for why/where
                for child in token.children:
                    if child.dep_ == "pobj":
                        semantic_rolesb[
                            "where" if token.head.dep_ == "ROOT" else "why"
                        ].append(child)
            if role == "how":

                if token.dep_ == "acomp":
                    for child in token.children:
                        if child.dep_ == "prep":
                            semantic_rolesb["how"].append(child)

            if role == "when":
                if token.dep_ == "npadvmod":
                    for child in token.children:
                        if child.dep_ == "prep":
                            semantic_rolesb["where"].append(child)
                if token.dep_ == "advmod":
                    for child in token.children:
                        if child.dep_ == "prep":
                            semantic_rolesb["why"].append(child)
            if role == "where":
                if token.dep_ == "pcomp":
                    for child in token.children:
                        if child.dep_ == "prep":
                            semantic_rolesb["where"].append(child)
                if token.dep_ == "advmod":
                    for child in token.children:
                        if child.dep_ == "prep":
                            semantic_rolesb["where"].append(child)

            # Find main actions and subjects
            if token.dep_ == "ROOT" or (
                token.pos_ == "VERB"
                and "subj" in {child.dep_ for child in token.children}
            ):
                actiona = " ".join(
                    [child.text for child in token.subtree if child.dep_ != "punct"]
                )
                actions.append(actiona)

            # Extract facts about specific subjects, considering nested and complex sentences
            if (
                token.pos_ == "NOUN" or token.pos_ == "PROPN" or token.pos_ == "PRON"
            ) and token.head.pos_ in [
                "VERB",
                "VB",
                "VBD",
                "VBG",
                "VBN",
                "VBP",
                "VBZ",
            ]:
                rv = self.find_root_verb(token.head)

                if rv:
                    subject_actions = " ".join(
                        [rv.text]
                        + [
                            child.text
                            for child in rv.children
                            if (child.dep_ != "nsubj" or child.dep_ != "nsubjpass")
                            and child.pos_ != "PUNCT"
                        ]
                    )
                    facts.append((token.text, subject_actions))
                facts.extend([(token.text, fact) for fact in self.extract_facts(token)])
            # if role == "why":
            #     if token.dep_ == "advcl":
            #         for child in token.children:
            #             if child.dep_ == "prep":
            #                 semantic_rolesb["why"].append(child.text)

        # print(f"Actions: {actions}")
        # print(f"Facts: {facts}")
        print(semantic_rolesb)
        # Check if the number of subordinate clauses is equal to the number of actions minus 1
        assert (
            num_subordinate_clauses == len(semantic_rolesb["did"]) - 1
            if num_subordinate_clauses > 0
            else True
        )

        # check if the ROOT verb has an auxiliary verb
        subj_compound = []
        obj_compound = []
        action_compound = []
        compound = []
        dobj_text = ""
        templates = []
        for action in semantic_rolesb["did"]:
            # if action.dep_ == "advcl":
            #     print(f"Action: {action.text}")
            #     exit(0)
            # Find the subject of the action in the original docs object
            for token in docs:
                if token.text == action.text and token.dep_ == action.dep_:
                    subj_compound = []
                    subj = [
                        tk
                        for tk in token.children
                        if tk.dep_ == "nsubj" or tk.dep_ == "nsubjpass"
                    ]
                    if len(subj) == 0:
                        subj = [
                            tk
                            for tk in token.head.children
                            if tk.dep_ == "nsubj" or tk.dep_ == "nsubjpass"
                        ]
                    obj_compound = []
                    subj_text = subj[0].text if subj else ""
                    compound = (
                        [
                            tk
                            for tk in subj[0].children
                            if tk.dep_ == "compound"
                            or tk.dep_ == "poss"
                            or tk.dep_ == "amod"
                            or tk.dep_ == "det"
                            or tk.dep_ == "aux"
                        ]
                        if subj
                        else []
                    )
                    if (
                        token.dep_ == "conj"
                        and tok.pos_ in ["NOUN", "PROPN"]
                        and compound == []
                    ):
                        for sbj in token.head.children:
                            if sbj.dep_ == "cc" and sbj.head == token.head:
                                direction_from_conj = (
                                    "left" if tok.i > sbj.i else "right"
                                )

                                connected_word = (
                                    docs[sbj.i - 1]
                                    if direction_from_conj == "left"
                                    else docs[sbj.i + 1]
                                )
                                while connected_word.dep_ in [
                                    "compound",
                                    "amod",
                                    "poss",
                                    "det",
                                    "aux",
                                ]:
                                    connected_word = (
                                        docs[connected_word.i - 1]
                                        if direction_from_conj == "left"
                                        else docs[connected_word.i + 1]
                                    )
                                if connected_word.dep_ in ["dobj", "pobj"]:
                                    break
                                if connected_word.dep_ in [
                                    "nsubj",
                                    "nsubjpass",
                                    "csubj",
                                    "csubjpass",
                                ]:
                                    subj.append(connected_word)
                                    compound.append(connected_word)
                                    compound.append(sbj)
                                    if token not in subj:
                                        subj.append(token)
                                    if token not in compound:
                                        compound.append(token)

                    elif (
                        token.dep_ == "conj"
                        and token.pos_
                        in [
                            "VERB",
                            "VB",
                            "VBD",
                            "VBG",
                            "VBN",
                            "VBP",
                            "VBZ",
                        ]
                        and len(subj) == 0
                        and len(compound) == 0
                    ):
                        if token.head.dep_ == "ROOT":

                            cw_subject = [
                                tk
                                for tk in token.head.children
                                if tk.dep_ == "nsubj" or tk.dep_ == "nsubjpass"
                            ]
                            if len(cw_subject) == 0:
                                cw_subject = [
                                    tk
                                    for tk in token.head.head.children
                                    if tk.dep_ == "nsubj" or tk.dep_ == "nsubjpass"
                                ]
                            print(f"cw_subject: {cw_subject}")
                            compound.extend(cw_subject)
                            subj.extend(cw_subject)

                            for tk in cw_subject:
                                for tkk in tk.children:
                                    if tkk.dep_ in [
                                        "compound",
                                        "poss",
                                        "amod",
                                        "det",
                                        "aux",
                                    ]:
                                        compound.append(tkk)
                                for tkk in tk.head.children:
                                    if tkk.dep_ in [
                                        "compound",
                                        "poss",
                                        "amod",
                                        "det",
                                        "aux",
                                    ]:
                                        compound.append(tkk)
                        else:
                            assert root_verb is not None
                            cw_subject = [
                                tk
                                for tk in root_verb.children
                                if tk.dep_ == "nsubj" or tk.dep_ == "nsubjpass"
                            ]
                            if len(cw_subject) == 0:
                                cw_subject = [
                                    tk
                                    for tk in root_verb.head.children
                                    if tk.dep_ == "nsubj" or tk.dep_ == "nsubjpass"
                                ]
                            print(f"cw_subject: {cw_subject}")
                            compound.extend(cw_subject)
                            if len(subj) == 0:
                                subj.extend(cw_subject)

                            for tk in cw_subject:
                                for tkk in tk.children:
                                    if tkk.dep_ in [
                                        "compound",
                                        "poss",
                                        "amod",
                                        "det",
                                        "aux",
                                    ]:
                                        if tkk not in compound:
                                            compound.append(tkk)
                                for tkk in tk.head.children:
                                    if (
                                        tkk.dep_
                                        in [
                                            "compound",
                                            "poss",
                                            "amod",
                                            "det",
                                            "aux",
                                            "acomp",
                                            "advmod",
                                            "npadvmod",
                                            "attr",
                                        ]
                                        and token.i < tkk.i
                                    ):
                                        # print(f"FART1: {tkk}")
                                        # exit(0)
                                        if tkk not in compound:
                                            if (
                                                tkk.dep_ != "acomp"
                                                or (
                                                    tkk.dep_ == "acomp"
                                                    and (
                                                        token.tag_ == "VBP"
                                                        or token.tag_ == "VBZ"
                                                    )
                                                    and token == tkk.head
                                                )
                                                and tkk.dep_ != "npadvmod"
                                            ):
                                                # compound.append(tkk)
                                                # subj_compound.append(tkk)
                                                if tkk not in obj_compound:
                                                    obj_compound.append(tkk)
                                                for tkki in tkk.children:
                                                    if tkki.dep_ in [
                                                        "compound",
                                                        "poss",
                                                        "amod",
                                                        "det",
                                                        "aux",
                                                    ]:
                                                        if tkki not in obj_compound:
                                                            # compound.append(tkki)
                                                            obj_compound.append(tkki)
                                        if (
                                            tkk.dep_ == "npadvmod" or tkk.dep_ == "attr"
                                        ) and (
                                            token == tkk.head
                                            if tkk.head.pos_ == "VERB"
                                            else (
                                                True
                                                if token == tkk.head.head
                                                else False
                                            )
                                        ):

                                            if tkk not in obj_compound:
                                                obj_compound.append(tkk)
                                            if tkk.dep_ == "attr":
                                                for tkki in tkk.children:
                                                    if tkki not in obj_compound:
                                                        obj_compound.append(tkki)
                                            for tkki in tkk.children:
                                                if tkki.dep_ in [
                                                    "compound",
                                                    "poss",
                                                    "amod",
                                                    "det",
                                                    "aux",
                                                ]:
                                                    if tkki not in obj_compound:
                                                        obj_compound.append(tkki)
                                                elif "WP" or "WDT" in [
                                                    tki.tag_ for tki in tkk.children
                                                ]:
                                                    for tki in tkk.children:
                                                        if (
                                                            tki.tag_ == "WP"
                                                            or tki.tag_ == "WDT"
                                                        ):
                                                            if tki not in subj_compound:
                                                                subj_compound.append(
                                                                    tki
                                                                )
                                                            for (
                                                                tkki
                                                            ) in tki.head.children:

                                                                if (
                                                                    tkki
                                                                    not in subj_compound
                                                                    and tkk.dep_
                                                                    != "npadvmod"
                                                                ):
                                                                    subj_compound.append(
                                                                        tkki
                                                                    )
                    else:
                        if len(subj) == 0:
                            print(f"\n Why is subj empty? {subj} {token.text} \n")
                            print(f"\nsemantic_rolesb: {semantic_rolesb}")
                            print(
                                f"\nroot_verb.children: {[tk.text for tk in root_verb.children]}"
                            )
                            subj = [
                                tk
                                for tk in root_verb.children
                                if tk.dep_ == "nsubj" or tk.dep_ == "nsubjpass"
                            ]
                        for tkk in subj[0].head.children:
                            if (
                                tkk.dep_
                                in [
                                    "compound",
                                    "poss",
                                    "amod",
                                    "det",
                                    "aux",
                                    "acomp",
                                    "advmod",
                                    "npadvmod",
                                ]
                                and token.i < tkk.i
                            ):
                                # print(f"FART2: {tkk}")
                                # exit(0)
                                if tkk not in compound:
                                    if (
                                        tkk.dep_ != "acomp"
                                        or (
                                            tkk.dep_ == "acomp"
                                            and (
                                                token.tag_ == "VBP"
                                                or token.tag_ == "VBZ"
                                            )
                                            and token == tkk.head
                                        )
                                        and tkk.dep_ != "npadvmod"
                                    ):
                                        # compound.append(tkk)
                                        # subj_compound.append(tkk)
                                        if tkk not in obj_compound:
                                            obj_compound.append(tkk)
                                        for tkki in tkk.children:
                                            if tkki.dep_ in [
                                                "compound",
                                                "poss",
                                                "amod",
                                                "det",
                                                "aux",
                                                "npadvmod",
                                            ]:
                                                if tkki not in obj_compound:
                                                    # compound.append(tkki)
                                                    obj_compound.append(tkki)

                                if tkk.dep_ == "npadvmod" and token == tkk.head:

                                    if tkk not in obj_compound:
                                        obj_compound.append(tkk)
                                    for tkki in tkk.children:
                                        if tkki.dep_ in [
                                            "compound",
                                            "poss",
                                            "amod",
                                            "det",
                                            "aux",
                                            "npadvmod",
                                        ]:
                                            if tkki not in obj_compound:
                                                obj_compound.append(tkki)
                                    print(f"obj_compound npadvmod: {obj_compound}")

                            elif tkk.tag_ == "WP" or tkk.tag_ == "WDT":
                                if tkk not in subj_compound:
                                    subj_compound.append(tkk)
                                for tki in tkk.children:

                                    if tki not in subj_compound:
                                        subj_compound.append(tki)
                            elif "WP" or "WDT" in [tki.tag_ for tki in tkk.children]:
                                for tki in tkk.children:
                                    if tki.tag_ == "WP" or tki.tag_ == "WDT":
                                        if (
                                            tki not in subj_compound
                                            and tkk.dep_ != "npadvmod"
                                        ):
                                            subj_compound.append(tki)
                                        for tkki in tki.head.children:

                                            if (
                                                tkki not in subj_compound
                                                and tkk.dep_ != "npadvmod"
                                            ):
                                                subj_compound.append(tkki)
                        print(f"subj_compound: {subj_compound}")

                    subj_text = subj[0].text if subj else ""
                    print(f"Compound: {compound}")
                    # if compound and len(compound) > 0:
                    #     obj_compound.extend(compound)

                    for tk in subj[0].children:

                        if len(compound) > 0:
                            if len(subj_compound) == 0:
                                subj_compound = [subj[0]]
                            for tkk in tk.head.children:
                                if (
                                    tkk.dep_
                                    in [
                                        "compound",
                                        "poss",
                                        "amod",
                                        "det",
                                        "aux",
                                    ]
                                    and tkk not in subj_compound
                                    and tkk not in compound
                                ):
                                    subj_compound.append(tkk)
                            for tkk in tk.children:
                                if (
                                    tkk.dep_
                                    in [
                                        "compound",
                                        "poss",
                                        "amod",
                                        "det",
                                        "aux",
                                    ]
                                    and tkk not in subj_compound
                                    and tkk not in compound
                                ):
                                    subj_compound.append(tkk)
                            if tk not in subj_compound:
                                subj_compound.append(tk)
                            subj_compound.extend(compound)
                            # remove tokens with duplicate indexes from token.i
                            for sub in subj_compound:
                                if subj_compound.count(sub) > 1:
                                    subj_compound.remove(sub)

                            subj_compound = sorted(subj_compound, key=lambda x: x.i)
                            print(f"subj_compound: {subj_compound}")
                            # create a string that joins each word in token.sent with a space but only if the word is in subj_compound in order of each word's index
                            subj_text = " ".join(
                                [
                                    token.text
                                    for token in token.sent
                                    if token in subj_compound
                                ]
                            )
                            print(f"Subject: {subj_text}")
                            if tk.dep_ == "prep":
                                for tkk in tk.children:
                                    if tkk.dep_ == "pobj":
                                        if tkk not in subj_compound:
                                            subj_compound.append(tkk)
                                        for tkkk in tkk.children:
                                            if tkkk not in subj_compound:
                                                subj_compound.append(tkkk)
                        else:
                            if tk.dep_ in [
                                "compound",
                                "poss",
                                "amod",
                                "det",
                                "aux",
                            ]:
                                subj_compound.append(tk)
                            subj_compound.append(tk)
                            subj_compound = sorted(subj_compound, key=lambda x: x.i)
                            subj_text = " ".join(
                                [
                                    token.text
                                    for token in token.sent
                                    if token in subj_compound
                                ]
                            )
                            print(f"Subject: {subj_text}")
                            if tk.dep_ == "prep":
                                for tkk in tk.children:
                                    if tkk.dep_ == "pobj":
                                        if tkk not in subj_compound:
                                            subj_compound.append(tkk)
                                        for tkkk in tkk.children:
                                            if tkkk not in subj_compound:
                                                subj_compound.append(tkkk)
                    subj_compound = sorted(subj_compound, key=lambda x: x.i)
                    print(f"subj_compound: {subj_compound}")
                    for tk in subj_compound:
                        if tk.text not in subj_text.split():
                            print(
                                f"Fixing missing subj compound token at token: {tk.text}"
                            )
                            temp = subj_compound
                            temp.append(subj[0])
                            temp = sorted(temp, key=lambda x: x.i)
                            subj_text = " ".join(
                                [token.text for token in token.sent if token in temp]
                            )
                    if subj[0].text not in subj_text.split():
                        print("Adding subject to subj_compound and subj_text")
                        if subj[0] not in subj_compound:
                            subj_compound.append(subj[0])
                        subj_text = " ".join(
                            [
                                token.text
                                for token in token.sent
                                if token in subj_compound
                            ]
                        )

                    if subj_text == "" or subj_text == " ":
                        subj_text = subj[0].text
                        print(f"Subject: {subj_text}")
                    print(f"Subject: {subj_text}")
                    print(f"Tokens children: {[tk.text for tk in token.children]}")

                    dobj = [
                        tk
                        for tk in token.children
                        if tk.dep_ == "dobj" or tk.dep_ == "pobj"
                    ]
                    if (
                        len(dobj) == 0
                        and token.dep_ == "conj"
                        and token.tag_
                        in ["VERB", "VB", "VBD", "VBG", "VBN", "VBP", "VBZ"]
                    ):
                        for tk in token.children:
                            if tk.dep_ in ["dobj", "pobj"]:
                                dobj = [tk]
                        else:
                            dobj = [token]
                    if len(dobj) == 0:
                        dobj = [tk for tk in token.children if tk.dep_ == "prep"]
                        for tk in dobj:
                            for tkk in tk.children:
                                if tkk.dep_ == "pobj":
                                    dobj = [tkk]
                                    for tkkk in tkk.children:
                                        if tkkk not in dobj:
                                            dobj.append(tkkk)
                    if len(dobj) == 0:

                        dobj = [
                            tk
                            for tk in token.head.children
                            if tk.dep_ == "attr"
                            and (subj[0].head == tk.head or token == tk.head)
                        ]
                        print(f"Last resort dobj: {dobj}")
                    else:
                        dobj = sorted(dobj, key=lambda x: x.i)
                        obj_compound.extend(ob for ob in dobj)
                    # if len(dobj) == 0:
                    dobj = sorted(dobj, key=lambda x: x.i)
                    for tok in token.children:

                        if tok.dep_ == "prep":
                            dobj = [tok]
                            for tkk in tok.children:
                                if tkk.dep_ == "pobj":
                                    if tkk not in dobj:
                                        dobj.append(tkk)
                                    for tkkk in tkk.children:
                                        if tkkk not in dobj and tkkk.dep_ in [
                                            "compound",
                                            "amod",
                                            "poss",
                                            "det",
                                        ]:
                                            dobj.append(tkkk)

                        elif tok.dep_ == "xcomp":

                            # Find main clause via the ROOT verb of the whole sentence
                            root_verb = None
                            for tk in docs:
                                if tk.dep_ == "ROOT":
                                    root_verb = tk
                                    break

                            assert root_verb is not None
                            ob = [
                                tkk
                                for tkk in tok.children
                                if tkk.dep_ == "pobj" or tkk.dep_ == "dobj"
                            ]

                            if len(ob) > 0 and len(dobj) == 0:
                                dobj = ob
                            elif len(dobj) > 0 and len(ob) > 0:
                                dobj.extend(ob)
                            elif len(dobj) == 0 and len(ob) == 0:
                                if tok.head == root_verb or tok.head.tag_ == "VBG":
                                    if len(dobj) == 0:
                                        dobj = [
                                            tk
                                            for tk in tok.children
                                            if tk.dep_ == "dobj"
                                        ]

                    dobj = sorted(dobj, key=lambda x: x.i)
                    for objj in dobj:
                        if objj.dep_ not in ["conj", "cc"] and objj not in obj_compound:
                            obj_compound.append(objj)
                        for objjj in objj.children:
                            if objjj not in obj_compound and objjj.dep_ in [
                                "compound",
                                "amod",
                                "poss",
                                "det",
                            ]:
                                obj_compound.append(objjj)
                            if objjj.dep_ == "prep":
                                for objjjj in objjj.children:
                                    if objjjj.dep_ in [
                                        "pobj",
                                        "det",
                                        "amod",
                                        "poss",
                                        "compound",
                                    ]:
                                        if objjjj not in obj_compound:
                                            obj_compound.append(objjjj)
                                        for objjjjj in objjjj.children:
                                            if objjjjj not in obj_compound:
                                                obj_compound.append(objjjjj)

                        if objj.dep_ == "conj":
                            for objjj in objj.head.children:
                                if (
                                    objjj.dep_
                                    in [
                                        "compound",
                                        "amod",
                                        "poss",
                                        "det",
                                    ]
                                    and objjj not in obj_compound
                                    and objjj not in subj_compound
                                ):

                                    obj_compound.append(objjj)

                    break
            dobj = sorted(dobj, key=lambda x: x.i)
            obj_compound = sorted(obj_compound, key=lambda x: x.i)
            if len(dobj) < 1:
                print(f"DOBJ still seems to be empty. Token: {token}")
                if docs[token.i - 1].text == "to":  # or token.tag_ == "VBG":
                    if token.head.pos_ == "VERB":
                        dobj = [token.head]
                else:
                    for tkn in token.children:
                        if tkn.dep_ == "xcomp" or tkn.dep_ == "advcl":
                            if docs[tkn.i - 1].text == "to" or tkn.tag_ == "VBG":
                                if tkn.head.pos_ == "VERB":
                                    dobj = [tkn.head]
            dobj = sorted(dobj, key=lambda x: x.i)
            if len(dobj) < 1:

                if token.dep_ == "conj":
                    print(f"DOBJ still seems to be empty. Token: {token}")
                    for tk in token.children:
                        print(f"Token: {tk.text}, Dependency: {tk.dep_}")
                        if tk.dep_ in ["dobj", "pobj"]:
                            dobj = [tk]
                    if len(dobj) == 0:
                        dobj = [token]
                elif len(dobj) == 0:
                    print(
                        f"dobj seems to still be empty so we will make it the token: {token}"
                    )
                    dobj = [token]
                # elif len(dobj) == 0:
                #     dobj = [token.head]
            for tk in dobj:
                if (
                    tk.text == action.text
                    and action.dep_ == "ROOT"
                    and "mark" in {child.dep_ for child in token.children}
                ):
                    dobj.remove(tk)
                    dobj.append(nlp("something")[0])
            dobj = sorted(dobj, key=lambda x: x.i)
            print(f"Subject: {subj}, Action: {action}, Object: {dobj}")
            if len(dobj) > 1:
                # Since dobj is more than 1, we must determine which object in dobj is correctly associated with the action by checking the dependency relation in a more complex way than simply checking the dep tag
                # We can do this by checking the dependency relation of each token in dobj to the action token and selecting the token with the closest dependency relation to the action token
                # We can also check the dependency relation of each token in dobj to the ROOT verb to ensure that the object is correctly associated with the action
                # We can also check the dependency relation of each token in dobj to the subject token to ensure that the object is correctly associated with the action
                # We can also check the dependency relation of each token in dobj to the auxiliaries to ensure that the object is correctly associated with the action
                # We can also check the dependency relation of each token in dobj to the verb's children to ensure that the object is correctly associated with the action
                # We can also check the dependency relation of each token in dobj to the verb's head to ensure that the object is correctly associated with the action
                # The actions are typically the roots of the sentence, but they can also be other verbs
                action_tokens = [
                    tok
                    for tok in docs
                    if tok.dep_ in {"ROOT", "relcl", "xcomp", "ccomp"}
                ]

                # The verbs are typically the tokens with part-of-speech "VERB", but they can also be other tokens with verb-like behavior
                verb_tokens = [tok for tok in docs if tok.pos_ in {"VERB", "AUX"}]

                # The subjects are typically the tokens with a direct dependency to a verb, but they can also be other tokens
                subject_tokens = [
                    tok
                    for tok in docs
                    if any(
                        sub.dep_ in {"nsubj", "nsubjpass", "csubj", "csubjpass"}
                        for sub in tok.children
                    )
                ]

                # The auxiliaries are typically the tokens with dependency "aux", but they can also be other tokens
                auxiliaries_tokens = [
                    tok for tok in docs if tok.dep_ in {"aux", "auxpass"}
                ]

                # The children of the verbs, including indirect children
                verbs_children_tokens = [
                    child for verb in verb_tokens for child in verb.subtree
                ]

                # The heads of the verbs, including indirect heads
                verbs_head_tokens = [
                    ancestor for verb in verb_tokens for ancestor in verb.ancestors
                ]

                for tk in dobj:
                    print(
                        f"Token: {tk.text}, Dependency to Action: {[tkk.dep_ for tkk in action_tokens if tkk.head == tk]}"
                    )
                    print(
                        f"Token: {tk.text}, Dependency to ROOT Verb: {[tkk.dep_ for tkk in verb_tokens if tkk.head == tk and tkk.dep_ == 'ROOT']}"
                    )
                    print(
                        f"Token: {tk.text}, Dependency to Subject: {[tkk.dep_ for tkk in subject_tokens if tkk.head == tk]}"
                    )
                    print(
                        f"Token: {tk.text}, Dependency to Auxiliaries: {', '.join([aux.dep_ for aux in auxiliaries_tokens if tk.head == aux]) or 'N/A'}"
                    )
                    print(
                        f"Token: {tk.text}, Dependency to Verb's Children: {', '.join([child.dep_ for child in verbs_children_tokens if tk.head == child]) or 'N/A'}"
                    )
                    print(
                        f"Token: {tk.text}, Dependency to Verb's Head: {', '.join([ancestor.dep_ for ancestor in verbs_head_tokens if tk.head == ancestor]) or 'N/A'}"
                    )
                    if tk not in obj_compound:
                        obj_compound.append(tk)
                    for tko in tk.children:
                        if (
                            tko.dep_ == "compound"
                            or tko.dep_ == "poss"
                            or tko.dep_ == "amod"
                            or tko.dep_ == "det"
                            or tko.dep_ == "attr"
                        ):
                            if len(obj_compound) == 0:
                                obj_compound = [tk]
                            else:
                                if tko not in obj_compound:
                                    obj_compound.append(tk)
                            if tko.dep_ == "attr":
                                for tkk in tko.children:
                                    if tkk.i not in [tkk.i for tkk in obj_compound]:
                                        obj_compound.append(tkk)
                        elif tko.dep_ == "prep":
                            obj_compound = [tk]
                            for tkk in tko.children:
                                if tkk.dep_ == "pobj":
                                    if tkk not in obj_compound:
                                        obj_compound.append(tkk)
                                    for tkkk in tkk.children:
                                        if tkkk not in obj_compound:
                                            obj_compound.append(tkkk)

            for tk in dobj[0].children:
                print(f"dobjs children: Token: {tk.text}, Dependency: {tk.dep_}")
                if (
                    tk.dep_ == "compound"
                    or tk.dep_ == "poss"
                    or tk.dep_ == "amod"
                    or tk.dep_ == "det"
                    or tk.dep_ == "attr"
                ):
                    if tk not in obj_compound:
                        obj_compound.append(tk)
                        # print(f"added to obj_compound: {tk.text}")
                    if tk.dep_ == "attr":
                        for tkk in tk.children:
                            if tkk not in obj_compound:
                                obj_compound.append(tkk)
                            for tkkk in tkk.children:
                                if tkkk not in obj_compound:
                                    obj_compound.append(tkkk)
                elif tk.dep_ == "prep":
                    if len(obj_compound) == 0:
                        obj_compound = [tk]
                    else:
                        obj_compound.append(tk)
                    for tkk in tk.children:
                        if tkk.dep_ == "pobj":
                            if tkk not in obj_compound:
                                obj_compound.append(tkk)
                            for tkkk in tkk.children:
                                if tkkk not in obj_compound:
                                    obj_compound.append(tkkk)
                        elif tkk.dep_ == "prep":
                            if len(obj_compound) == 0:
                                obj_compound = [tk]
                            else:
                                obj_compound.append(tk)
                            for tkkk in tkk.children:
                                if tkkk.dep_ == "pobj":
                                    if tkkk not in obj_compound:
                                        obj_compound.append(tkkk)
                                    for tkkkk in tkkk.children:
                                        if tkkkk not in obj_compound:
                                            obj_compound.append(tkkk)
            obj_compound = sorted(obj_compound, key=lambda x: x.i)
            print(f"obj_compound: {obj_compound}")
            if len(obj_compound) > 0 and "prep" not in {
                tk.dep_ for tk in action.children
            }:
                if dobj[0] not in obj_compound:
                    obj_compound.append(dobj[0])
                # Ensure no token is repeated in obj_compound that was in subj_compound (unless the token exists twice in the sentence or had a different index)
                obj_compound = [
                    tk
                    for tk in obj_compound
                    if tk not in subj_compound
                    or (
                        tk in subj_compound
                        and sum(tk.text == t.text for t in subj_compound) > 1
                    )
                    or tk.i != [t.i for t in subj_compound if t.text == tk.text][0]
                ]

            elif len(obj_compound) == 0:
                print(f"For some reason, obj_compound is empty. Token: {token}")
                obj_compound = dobj
            elif len(obj_compound) > 0 and "prep" in {
                tk.dep_ for tk in action.children
            }:
                for tk in action.children:
                    if tk.dep_ in ["det", "amod", "poss", "compound", "attr"]:
                        if tk not in obj_compound:
                            obj_compound.append(tk)
                    if tk.dep_ == "prep":
                        if len(obj_compound) == 0:
                            obj_compound = [tk]
                        else:
                            if tk not in obj_compound:
                                obj_compound.append(tk)
                        for tkk in tk.children:
                            if tkk.dep_ == "pobj":
                                if tkk not in obj_compound:
                                    obj_compound.append(tkk)
                                for tkkk in tkk.children:
                                    if tkkk not in obj_compound:
                                        obj_compound.append(tkkk)
                            elif tkk.dep_ == "prep":
                                if len(obj_compound) == 0:
                                    obj_compound = [tk]
                                else:
                                    obj_compound.append(tk)
                                for tkkk in tkk.children:
                                    if tkkk.dep_ == "pobj":
                                        if tkkk not in obj_compound:
                                            obj_compound.append(tkkk)
                                        for tkkkk in tkkk.children:
                                            if tkkkk not in obj_compound:
                                                obj_compound.append(tkkk)
            obj_compound = sorted(obj_compound, key=lambda x: x.i)

            action_compound = [action]
            action_compound.extend(
                action_tok
                for action_tok in token.children
                if action_tok.dep_
                in [
                    "aux",
                    "auxpass",
                    "neg",
                    "prt",
                    "det",
                    "amod",
                    "poss",
                    "compound",
                    "pos",
                    # "mark",
                    "xcomp",
                ]
            )
            if token.head.dep_ == "ROOT":
                action_compound.extend(
                    action_tok
                    for action_tok in token.head.children
                    if action_tok.dep_
                    in [
                        "aux",
                        "auxpass",
                        "neg",
                        "prt",
                        "det",
                        "amod",
                        "poss",
                        "compound",
                        "pos",
                        # "mark",
                        "xcomp",
                    ]
                    if action_tok.i > token.head.i
                )
            if (
                token.tag_
                in [
                    "VERB",
                    "VB",
                    "VBD",
                    "VBG",
                    "VBN",
                    "VBP",
                    "VBZ",
                ]
                and token.head.dep_ != "xcomp"
            ):
                if token.text == dobj[0].text:
                    if dobj[0] in obj_compound:
                        dobj_text = ""
                    else:
                        print(f"obj_compound: {obj_compound}")
                        print(f"Token: {token.text}, dobj: {dobj[0].text}")
            if (
                token.dep_ == "conj"
                and token.head.dep_ == "xcomp"
                and token.tag_
                in [
                    "VERB",
                    "VB",
                    "VBD",
                    "VBG",
                    "VBN",
                    "VBP",
                    "VBZ",
                ]
            ):
                action_compound.extend(
                    action_tok
                    for action_tok in token.head.children
                    if action_tok.dep_
                    in [
                        "aux",
                        "auxpass",
                        "neg",
                        "prt",
                        "det",
                        "amod",
                        "poss",
                        "compound",
                        "pos",
                        # "mark",
                        "xcomp",
                    ]
                )
                action_compound.remove(action)
                action_compound.append(root_verb)
                if root_verb in obj_compound:
                    obj_compound.remove(root_verb)
                    print(f"obj_compound: {obj_compound}")
            for action_tok in action_compound:
                if action_tok in obj_compound:
                    for obj in obj_compound:
                        if action_tok.i == obj.i:
                            obj_compound.remove(obj)
                            print(f"obj_compound: {obj_compound}")
            obj_compound = sorted(obj_compound, key=lambda x: x.i)
            for tk in obj_compound:
                print(f"obj_compound: {tk.text}, index: {tk.i}")
            # create a string that joins each word in token.sent with a space but only if the word is in subj_compound in order of each word's index
            dobj_text = " ".join([token.text for token in obj_compound])
            print(f"obj_compound: {obj_compound}")
            print(f"dobj_text: {dobj_text}")
            # Ensure no token is repeated in action_compound that was in subj_compound or obj_compound (unless the token exists twice in the sentence or had a different index)
            # if "mark" in {tk.dep_ for tk in action.children}:
            #         # Make the action compund consist of only mark token, its associated verb and the verb's children
            #         action_compound = [
            #             tk
            #             for tk in action.children
            #         ]
            #         action_compound.append(action)

            if "xcomp" in {tk.dep_ for tk in action_compound}:
                for tk in action_compound:
                    if tk.dep_ == "xcomp":
                        action_compound.extend(
                            tkk
                            for tkk in tk.children
                            if tkk.dep_
                            in [
                                "aux",
                                "auxpass",
                                "neg",
                                "prt",
                                "det",
                                "amod",
                                "poss",
                            ]
                        )
                        break

            if docs[subj[0].i + 1].dep_ == "aux":
                action_compound.append(docs[subj[0].i + 1])

            action_compound = [
                tk
                for tk in action_compound
                if tk not in subj_compound
                or (
                    tk in subj_compound
                    and sum(tk.text == t.text for t in subj_compound) > 1
                )
                or tk.i != [t.i for t in subj_compound if t.text == tk.text][0]
            ]

            # action_compound = [
            #     tk
            #     for tk in action_compound
            #     if tk not in obj_compound
            #     or (
            #         tk in obj_compound
            #         and sum(tk.text == t.text for t in obj_compound) > 1
            #     )
            #     or tk.i != [t.i for t in obj_compound if t.text == tk.text][0]
            # ]
            for tk in subj_compound:
                if tk.dep_ == "acomp" and token.tag_ == "VBP":
                    action_compound.append(tk)
                    if tk in subj_compound:
                        subj_compound.remove(tk)
                    subj_text = subj[0].text if subj else ""
                    print(f"subj_compound: {subj_compound}")
                    print(f"obj_compound: {obj_compound}")
                    print(f"action_compound: {action_compound}")
                    print(f"Subject: {subj_text}")
                    subj_compound = sorted(subj_compound, key=lambda x: x.i)
                    subj_text = " ".join(
                        [token.text for token in token.sent if token in subj_compound]
                    )
                    print(f"Subject: {subj_text}")
            action_compound = sorted(action_compound, key=lambda x: x.i)
            action_text = " ".join(
                [token.text for token in token.sent if token in action_compound]
            )
            print(f"Action Compound: {action_compound}")
            template = ""
            if subj[0].i < dobj[0].i:
                template = f"{subj_text} {action_text} {dobj_text}"
            elif subj[0].i > dobj[0].i:
                template = f"{dobj_text} {subj_text} {action_text}"
            print(f"\n \n Template: {template}\n \n")
            templates.append(template)
            for tk in docs:
                if (
                    tk.dep_ == "conj"
                    and tk.pos_ in ["NOUN", "PROPN"]
                    and (tk.head.text == action.text or tk.head.dep_ == "ROOT")
                    and tk not in subj_compound
                    and tk not in obj_compound
                ):
                    dobj = [tk]
                    obj_compound = dobj

                    for tkk in tk.children:
                        if tkk.dep_ in ["compound", "amod", "poss", "det", "attr"]:
                            obj_compound.append(tkk)
                        if tkk.dep_ == "prep":
                            for tkkk in tkk.children:
                                if tkkk.dep_ == "pobj":
                                    if tkkk not in obj_compound:
                                        obj_compound.append(tkkk)
                                    for tkkkk in tkkk.children:
                                        if tkkkk not in obj_compound:
                                            obj_compound.append(tkkk)
                                if tkkk.dep_ == "prep":
                                    for tkkk in tkk.children:
                                        if tkkk.dep_ == "pobj":
                                            if tkkk not in obj_compound:
                                                obj_compound.append(tkkk)
                                            for tkkkk in tkkk.children:
                                                if tkkkk not in obj_compound:
                                                    obj_compound.append(tkkk)

                    dobj_text = " ".join(
                        [token.text for token in token.sent if token in obj_compound]
                    )
                    print(f"Subject: {subj}, Action: {action}, Object: {dobj}")
                    print(f"Action Compound: {action_compound}")
                    template = f"{subj_text} {action_text} {dobj_text}"
                    print(f"\n \n Template (C1): {template}\n \n")
                    templates.append(template)
                elif (
                    tk.dep_ == "conj"
                    and tk.pos_ in ["NOUN", "PROPN"]
                    and dobj[0].head.text == action.text
                    and dobj[0] != action
                    and (
                        tk.head.head.dep_ == "ROOT" or tk.head.head.text == action.text
                    )
                    and tk not in subj_compound
                    and tk not in obj_compound
                    and docs[tk.i - 2].dep_ != "nsubj"
                ):
                    dobj = [tk]
                    obj_compound = dobj
                    for tkk in tk.children:
                        if tkk.dep_ in ["compound", "amod", "poss", "det", "attr"]:
                            obj_compound.append(tkk)
                        if tkk.dep_ == "prep":
                            for tkkk in tkk.children:
                                if tkkk.dep_ == "pobj":
                                    if tkkk not in obj_compound:
                                        obj_compound.append(tkkk)
                                    for tkkkk in tkkk.children:
                                        if tkkkk not in obj_compound:
                                            obj_compound.append(tkkk)
                                if tkkk.dep_ == "prep":
                                    for tkkk in tkk.children:
                                        if tkkk.dep_ == "pobj":
                                            if tkkk not in obj_compound:
                                                obj_compound.append(tkkk)
                                            for tkkkk in tkkk.children:
                                                if tkkkk not in obj_compound:
                                                    obj_compound.append(tkkk)
                    dobj_text = " ".join(
                        [token.text for token in token.sent if token in obj_compound]
                    )
                    print(f"dobj: {dobj}")

                    print(f"Subject: {subj}, Action: {action}, Object: {dobj}")
                    print(f"Action Compound: {action_compound}")
                    print(f"Object Compound: {obj_compound}")
                    print(f"dobj_text: {dobj_text}")
                    template = f"{subj_text} {action_text} {dobj_text}"
                    print(f"\n \n Template (C2): {template}\n \n")
                    templates.append(template)
                elif (
                    tk.dep_ == "conj"
                    and tk.pos_ in ["NOUN", "PROPN"]
                    and (tk.head.dep_ == "xcomp" or tk.head.head.dep_ == "xcomp")
                ):

                    dobj = [tk]
                    obj_compound = dobj
                    for tkk in tk.children:
                        if tkk.dep_ in ["compound", "amod", "poss", "det", "attr"]:
                            obj_compound.append(tkk)
                        if tkk.dep_ == "prep":
                            for tkkk in tkk.children:
                                if tkkk.dep_ == "pobj":
                                    if tkkk not in obj_compound:
                                        obj_compound.append(tkkk)
                                    for tkkkk in tkkk.children:
                                        if tkkkk not in obj_compound:
                                            obj_compound.append(tkkk)
                                if tkkk.dep_ == "prep":
                                    for tkkk in tkk.children:
                                        if tkkk.dep_ == "pobj":
                                            if tkkk not in obj_compound:
                                                obj_compound.append(tkkk)
                                            for tkkkk in tkkk.children:
                                                if tkkkk not in obj_compound:
                                                    obj_compound.append(tkkk)
                    dobj_text = " ".join(
                        [token.text for token in token.sent if token in obj_compound]
                    )
                    # print(f"Subject: {subj}, Action: {action}, Object: {dobj}")
                    # print(f"Action Compound: {action_compound}")
                    template = f"{subj_text} {action_text} {dobj_text}"
                    print(f"\n \n Template (C3): {template}\n \n")
                    templates.append(template)
                elif (
                    tk.dep_ == "conj"
                    and tk.pos_ in ["NOUN", "PROPN"]
                    and tk.head.dep_ == "conj"
                    and tk.head.text == action.text
                    and dobj[0] != action
                ):

                    dobj = [tk]
                    obj_compound = dobj
                    for tkk in tk.children:
                        if tkk.dep_ in ["compound", "amod", "poss", "det", "attr"]:
                            obj_compound.append(tkk)
                        if tkk.dep_ == "prep":
                            for tkkk in tkk.children:
                                if tkkk.dep_ == "pobj":
                                    if tkkk not in obj_compound:
                                        obj_compound.append(tkkk)
                                    for tkkkk in tkkk.children:
                                        if tkkkk not in obj_compound:
                                            obj_compound.append(tkkk)
                                if tkkk.dep_ == "prep":
                                    for tkkk in tkk.children:
                                        if tkkk.dep_ == "pobj":
                                            if tkkk not in obj_compound:
                                                obj_compound.append(tkkk)
                                            for tkkkk in tkkk.children:
                                                if tkkkk not in obj_compound:
                                                    obj_compound.append(tkkk)

                    dobj_text = " ".join(
                        [token.text for token in token.sent if token in obj_compound]
                    )
                    # print(f"Subject: {subj}, Action: {action}, Object: {dobj}")
                    # print(f"Action Compound: {action_compound}")
                    template = f"{subj_text} {action_text} {dobj_text}"
                    print(f"\n \n Template (C4): {template}\n \n")
                    templates.append(template)
                elif (
                    tk.dep_ == "conj"
                    and tk.pos_ in ["NOUN", "PROPN"]
                    and tk.head.text == action.text
                    and dobj[0] != action
                ):

                    dobj = [tk]
                    obj_compound = dobj
                    for tkk in tk.children:
                        if tkk.dep_ in ["compound", "amod", "poss", "det", "attr"]:
                            obj_compound.append(tkk)
                        if tkk.dep_ == "prep":
                            for tkkk in tkk.children:
                                if tkkk.dep_ == "pobj":
                                    if tkkk not in obj_compound:
                                        obj_compound.append(tkkk)
                                    for tkkkk in tkkk.children:
                                        if tkkkk not in obj_compound:
                                            obj_compound.append(tkkk)
                                if tkkk.dep_ == "prep":
                                    for tkkk in tkk.children:
                                        if tkkk.dep_ == "pobj":
                                            if tkkk not in obj_compound:
                                                obj_compound.append(tkkk)
                                            for tkkkk in tkkk.children:
                                                if tkkkk not in obj_compound:
                                                    obj_compound.append(tkkk)

                    dobj_text = " ".join(
                        [token.text for token in token.sent if token in obj_compound]
                    )
                    # print(f"Subject: {subj}, Action: {action}, Object: {dobj}")
                    # print(f"Action Compound: {action_compound}")
                    # print(f"Object Compound: {obj_compound}")
                    template = f"{subj_text} {action_text} {dobj_text}"
                    print(f"\n \n Template (C5): {template}\n \n")
                    templates.append(template)
                elif (
                    tk.dep_ == "conj"
                    and tk.pos_ in ["NOUN", "PROPN"]
                    and tk.head.dep_ == "conj"
                    and (
                        tk.head.head.text == action.text
                        or tk.head.head.head.text == action.text
                    )
                    and dobj[0] != action
                ):

                    dobj = [tk]
                    obj_compound = dobj
                    for tkk in tk.children:
                        if tkk.dep_ in ["compound", "amod", "poss", "det", "attr"]:
                            obj_compound.append(tkk)
                        if tkk.dep_ == "prep":
                            for tkkk in tkk.children:
                                if tkkk.dep_ == "pobj":
                                    if tkkk not in obj_compound:
                                        obj_compound.append(tkkk)
                                    for tkkkk in tkkk.children:
                                        if tkkkk not in obj_compound:
                                            obj_compound.append(tkkk)
                                if tkkk.dep_ == "prep":
                                    for tkkk in tkk.children:
                                        if tkkk.dep_ == "pobj":
                                            if tkkk not in obj_compound:
                                                obj_compound.append(tkkk)
                                            for tkkkk in tkkk.children:
                                                if tkkkk not in obj_compound:
                                                    obj_compound.append(tkkk)

                    dobj_text = " ".join(
                        [token.text for token in token.sent if token in obj_compound]
                    )
                    # print(f"Subject: {subj}, Action: {action}, Object: {dobj}")
                    # print(f"Action Compound: {action_compound}")
                    template = f"{subj_text} {action_text} {dobj_text}"
                    print(f"\n \n Template (C6): {template}\n \n")
                    templates.append(template)

                elif (
                    tk.dep_ == "conj"
                    and tk.pos_ in ["NOUN", "PROPN"]
                    and tk.head.dep_ == "conj"
                    and action.text in [tk.text for tk in tk.ancestors]
                    and dobj[0] != action
                ):
                    other_verb = False
                    for right_token in docs[token.i + 1 :]:
                        if right_token.dep_ == "conj":

                            if right_token.pos_ == "VERB":
                                print(f"Token: {tk.text}, Right: {tk.rights}")
                                other_verb = True
                                break
                    if not other_verb:
                        dobj = [tk]
                        obj_compound = dobj
                        for tkk in tk.children:
                            if tkk.dep_ in [
                                "compound",
                                "amod",
                                "poss",
                                "det",
                                "attr",
                            ]:
                                obj_compound.append(tkk)
                            if tkk.dep_ == "prep":
                                for tkkk in tkk.children:
                                    if tkkk.dep_ == "pobj":
                                        if tkkk not in obj_compound:
                                            obj_compound.append(tkkk)
                                        for tkkkk in tkkk.children:
                                            if tkkkk not in obj_compound:
                                                obj_compound.append(tkkk)
                                    if tkkk.dep_ == "prep":
                                        for tkkk in tkk.children:
                                            if tkkk.dep_ == "pobj":
                                                if tkkk not in obj_compound:
                                                    obj_compound.append(tkkk)
                                                for tkkkk in tkkk.children:
                                                    if tkkkk not in obj_compound:
                                                        obj_compound.append(tkkk)

                        dobj_text = " ".join(
                            [
                                token.text
                                for token in token.sent
                                if token in obj_compound
                            ]
                        )
                        # print(f"Subject: {subj}, Action: {action}, Object: {dobj}")
                        # print(f"Action Compound: {action_compound}")
                        template = f"{subj_text} {action_text} {dobj_text}"
                        print(f"\n \n Template (C7): {template}\n \n")
                        templates.append(template)

            assert len(subj) > 0 and len(action_compound) > 0

        fa = tsm.main(docs, nlp)

        # assert fa is equivalent to templates
        assert fa == templates
        print(f"\nTemplates: {templates}")
        print(f"FA: {fa}\n")

        # 5. Named Entity Recognition
        named_entities = {}

        for ent in docs.ents:
            if ent.label_ not in named_entities:
                named_entities[ent.label_] = []
            named_entities[ent.label_].append(ent.text)
            # print(f"Named Entity: {ent.text}, Label: {ent.label_}")
        # print(f"\n Named Entities: {named_entities}")

        # # 6. Coreference Resolution
        # coreferences = {}
        # for cluster in docs._.coref_clusters:
        #     coreferences[cluster.main.text] = [mention.text for mention in cluster.mentions]
        #     #print(f"Coreference: {cluster.main.text}, Mentions: {[mention.text for mention in cluster.mentions]}")

        # 7 Dependency Parsing
        dependency_tree = {}
        for token in docs:
            if token.dep_ != "ROOT":
                if token.head.text not in dependency_tree:
                    dependency_tree[token.head.text] = []
                dependency_tree[token.head.text].append({token.text: token.dep_})
                # print(f"Dependency: {token.text}, Relation: {token.dep_}")

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

        # Iterate over matches and #print results
        spans = {}
        for match_id, start, end in matches:
            span = Span(docs, start, end)
            # print(f"Temporal expression found: {span.text}")
            spans[span.text] = span
        # print(f"\n Temporal Expressions: {spans}")

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
        # #     #Use verb and aspect to determine the aspect of the input

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
            # print(f"\n Preposition: {prep.text}, Relation: {relation}, Head: {head.text}")
            if relation == "pobj":  # Object of preposition
                if "nsubj" in head.dep_:  # Subject of the verb
                    entities.append(head.text)  # Subject of the verb
                    relationships[head.text] = (
                        prep.text
                    )  # head.text is the possessor, prep.text is the relationship
                elif "dobj" in head.dep_:  # Direct object of the verb
                    entities.append(head.text)  # Direct object of the verb
                    relationships[head.text] = (
                        prep.text
                    )  # head.text is the object, prep.text is the relationship

            elif relation == "prep":  # Covers cases beyond just objects
                # Example: '...belonging to Sarah' -> Sarah (possessor), belonging to (relationship)
                entities.append(head.text)
        if entities:
            entities = list(set(entities))
            print(f"\n Entities: {entities}, Relationships: {relationships} \n")

        main_subject = (
            next(iter(named_entities.values()), " ")[0] if named_entities else " "
        )
        main_object = None
        main_verb = None

        if (
            len(main_subject) <= 1
            or not main_subject
            or len(named_entities.keys()) == 0
        ):
            print(
                f"\n No named entities found, using the first noun phrase {next(iter(docs.noun_chunks))} of type {next(iter(docs.noun_chunks)).root.dep_} and data type {next(iter(docs.noun_chunks)).root.head.pos_} with text {next(iter(docs.noun_chunks)).root.head.text} \n"
            )
            main_subject = next(iter(docs.noun_chunks)).root.text
        print(f"\n Named Entities: {named_entities}")
        # print(f"\n Main Entity: {main_subject if main_subject else 'None'}")
        main_entity = main_subject
        previous_token = None
        for token in docs:
            if token.dep_ == "ROOT":
                main_verb = token.text
            if "nsubj" in token.dep_ and (
                previous_token and previous_token.dep_ == "compound"
            ):
                # print(f"token text: {token.text}")
                main_subject = f"{previous_token.text} {token.text}"
            elif token.dep_ == "nsubj":  # Subject of the verb
                main_subject = token.text
            if token.dep_ == "dobj":
                main_object = token.text
            previous_token = token if token.dep_ == "compound" else None
        # print(f"\n Main Subject: {main_subject}")
        print(
            f"\n Main Subject: {main_subject}, Main Verb: {main_verb}, Main Object: {main_object} \n"
        )

        # Create a string of the template: "Who is doing what and to whom or what?"
        if main_entity:
            template = f"{main_entity} is doing {main_verb} to {main_object}"
            if len(named_entities) > 1:
                template += f" and {', '.join(named_entities.keys())} are also involved"

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
        # print(f"\n Definite Main Subject: {main_subject}")

        for head, roles in semantic_roles.items():
            print(
                f"Head: {head.text} (POS: {head.pos_}) (dep: {head.dep_}) (tag: {head.tag_}) (index: {head.i})"
            )
            for role in roles:
                print(f"  Role: {role}")

        # 5. Verb-Adjective Relationship

        # 6. Verb-Preposition Relationship

        # 7. Verb-Particle Relationship

        # 8. Verb-Adposition Relationship

        # Use the lexical_density, type_token_ratio, word_frequency, word_length, lexical_diversity, passive_voice, active_voice, modals, adverbs, adjectives, pronouns, conjunctions, determiners, prepositions, interjections, particles, punctuations, symbols, numbers, foreign_words, proper_nouns, common_nouns, verbs, adpositions, adverbs, auxiliaries, conjunctions, advanced_vocabulary, and simple_vocabulary to determine the complexity of the input
        complexity += lexical_density * 10
        # complexity += type_token_ratio * 10
        complexity += int(avg_word_length)

        complexity += lexical_diversity * 10

        keywords = self.extract_keywords(query)

        query_embedding = outputs
        # print(f"Shape of query embedding: {query_embedding.shape}")
        # kmeans = KMeans(n_clusters=2)

        # kmeans.fit(outputs[0][-1].cpu().detach().numpy())
        # #print(f"Kmeans labels: {kmeans.labels_}")
        # #print(f"Kmeans cluster centers: {kmeans.cluster_centers_}")
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
        # #print(f"Complexity: {complexity}")
        # for token in docs:
        #     if token.dep_ == "xcomp":
        #         print(f"Xcomp: {token.text}")
        #         exit(0)
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

                # for dist, index in zip(dists, indices):
                #     #print(f"Distance: {dist}, Index: {index}")

                if dists[0][0] < 0.1:
                    # print(f"Query is similar to recent queries \n")
                    is_common_query = True

        # Extract themes with the SRL data using the semantic_roles and extract_themes method
        # themes = self.extract_themes(semantic_roles)

        # Who is doing what and to whom or what? Extract relationships with the SRL data using the semantic_roles and extract_relationships method
        # relationships = self.extract_relationships(semantic_roles, text)

        # reading_level =

        ambiguity_score = complexity / 10
        sentiment_score = self.get_sentiment_score(query)
        emotion_classification = self.get_emotion_classification(query)
        # if (
        #     "Oliver told me that ginger root crop has the highest profit margin"
        #     in query
        # ):
        #     exit(0)
        return {
            # "features": features,
            "embedding": query_embedding,
            "sentiment_score": sentiment_score,
            "emotion_classification": emotion_classification,
            "ambiguity_score": ambiguity_score,
            "keywords": keywords,
            # "themes": themes,
            "text": query,
            "main_subject": main_subject,
            "main_verb": main_verb,
            "main_object": main_object,
            # "named_entities": named_entities,
            # "temporal_expressions": spans,
            "verb_aspects": verb_aspects,
            "complexity": complexity,
            "is_common_query": is_common_query,
            "is_urgent": is_urgent,
            "lexical_density": lexical_density,
            "type_token_ratio": type_token_ratio,
            "avg_word_length": avg_word_length,
            "lexical_diversity": lexical_diversity,
            # "passive_voice": passive_voice,
            # "active_voice": active_voice,
            # "modals": modals,
            # "determiners": determiners,
            # "semantic_roles": semantic_roles,
            # "dependencies": dependency_tree,
            # "relationships": relationships,
            # "proper_nouns": proper_nouns,
            # "common_nouns": common_nouns,
            # "verbs": verbs,
            # "adpositions": adpositions,
            # "adverbs": adverbs,
            # "auxiliaries": auxiliaries,
            # "conjunctions": conjunctions,
            # "advanced_vocabulary": advanced_vocabulary,
            # "simple_vocabulary": simple_vocabulary,
            # "numbers": numbers,
            # "symbols": symbols,
            # "punctuations": punctuations,
            # "particles": particles,
            # "interjections": interjections,
            # "prepositions": prepositions,
            # "conjunctions": conjunctions,
            # "pronouns": pronouns,
            # "adjectives": adjectives,
            # "adverbs": adverbs,
            # "word_frequency": word_frequencies,
            "facts": templates,
        }

    def is_relevant_flat_memory(self, flat_memory, query_features):
        # print(f"\n Checking relevance of flat memory: {flat_memory.description} to query: {query_features.text} \n")
        # Simpler logic compared to hierarchical/graph-based layers
        # Example: Check for keyword or category match, or use a basic text similarity measure

        # Extract keywords or categories from the memory and the query
        if flat_memory.keywords is None or len(flat_memory.keywords) == 0:
            # print(f"\n Extracting keywords for flat memory: {flat_memory.description} \n")
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
            # print(f"\n Analyzing query: {query.query}")
            if is_question(query.query):
                print(f"\n Query is a question: {query.query}")
                query.query_embedding = self.get_query_embedding(query.query)
            else:
                print(f"\n Query is not a question: {query.query}")
                query.analysis = self.analyze_query_context(query.query)
            if query.query_embedding is None:
                query.query_embedding = query.analysis["embedding"]
            # for k, value in query.analysis.items():
            #     if k != 'embedding' and value is not None and value != {} and value != [] and value != '':
            # print(f"Key: {k}, Value: {value}")
        self.recent_queries.append(query)
        if urgent or common:
            # print(f"\n Retrieving urgent memories: {urgent}, common memories: {common}")
            return (
                self.flat_access.get_urgent_query_memories()
                if urgent
                else (
                    self.flat_access.get_common_memories(key)
                    if common and key is not None
                    else self.retrieve_from_flat_access(query)
                )
            )
        return self.retrieve_from_flat_access(query)

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
                # print(f"Query is similar to recent queries")
                return False

        is_long = len(query_features.analysis["text"].split()) > length_threshold
        if is_long:
            # print(f"Query is long")
            return True
        has_specific_keywords = any(
            self.keyword_specificity(keyword) > specificity_threshold
            for keyword in query_features.analysis["keywords"]
        )
        if has_specific_keywords:
            # print(f"Query has specific keywords")
            return True
        is_ambiguous = query_features.analysis["ambiguity_score"] > ambiguity_threshold
        if is_ambiguous:
            # print(f"Query is ambiguous {query_features.analysis['ambiguity_score']}")
            return True

        is_polarized = (
            query_features.analysis["sentiment_score"]["polarity"] > polarity_threshold
        )
        if is_polarized:
            # print(f"Query is polarized")
            return True

        is_subjective = (
            query_features.analysis["sentiment_score"]["subjectivity"]
            > subjectivity_threshold
        )

        if is_subjective:
            # print(f"Query is subjective")
            return True

        is_emotional = query_features.analysis["emotion_classification"] != "others"
        if is_emotional:
            # print(f"Query is emotional")
            return True
        # print(f"Query is not complex")
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
        # print(f"Rake keywords: {rake_keywords}")
        # print(f"Type of text: {type(text)}")
        # print(f"Text: {text}")
        # if isinstance(text, list):
        #     lda_keywords = self.extract_lda_keywords(text)
        # else:
        #     lda_keywords = self.extract_lda_keywords([text])
        # #remove duplicates from lda_keywords, keep the order
        # lda_keywords = list(dict.fromkeys(lda_keywords))

        # #print(f"LDA keywords: {lda_keywords}")
        tfidf_keywords = self.extract_tfidf_keywords(text)
        # print(f"TF-IDF keywords: {tfidf_keywords}")
        # print(f"Type of tfidf_keywords: {type(tfidf_keywords)}")
        entity_keywords = self.extract_entities(text)
        # print(f"Entity keywords: {entity_keywords}") if entity_keywords else #print(f"No entity keywords found")
        # print(f"Type of entity_keywords: {type(entity_keywords)}")

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

    def extract_themes(self, srl_features):
        # Apply NMF for theme extraction
        tfidf = self.tfidf_vectorizer.fit_transform(srl_features)

        n_components = 5  # Number of themes
        nmf = NMF(n_components=n_components, random_state=1).fit(tfidf)

        # Extract and display themes
        feature_names = self.tfidf_vectorizer.get_feature_names_out()
        for topic_idx, topic in enumerate(nmf.components_):
            print(f"Theme #{topic_idx+1}:")
            print(" ".join([feature_names[i] for i in topic.argsort()[: -10 - 1 : -1]]))
        return [feature_names[i] for i in topic.argsort()[: -10 - 1 : -1]]

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

    # def search_by_tag(self, tag):
    #     res = []
    #     for mem in self.hierarchy.general_memories:
    #         if tag in mem.tags:
    #             res.append(mem)
    #     return res
    def search_by_index(self, general_memory, k=10):
        return general_memory.faiss_index.search(
            general_memory.description_embedding.cpu().detach().numpy(), k
        )

    def calculate_recency_score(self, memory, current_time, decay_factor=0.995):
        time_since_last_access = (
            current_time - memory.last_access_time
        ).total_seconds() / 3600  # Convert to hours
        return decay_factor**time_since_last_access


# Function to manage the index and search
def manage_index_and_search(index_type, normalization, filename, memory_dict, queries):
    # Check if the index file exists and initialize accordingly
    matching_type = {
        "ip": "IndexFlatIP",
        "l2": "IndexFlatL2",
    }

    if (
        manager.flat_access.faiss_index is not None
        and manager.flat_access.faiss_index.__class__.__name__
        == matching_type[index_type]
        and manager.flat_access.index_is_normalized == normalization
    ):
        # do nothing
        print(
            f"\n Index already exists. Type: {index_type}, Normalized: {normalization}, Filename: {filename} \n"
        )
        pass
    elif not os.path.exists(filename):
        # Delete index to free up memory
        manager.flat_access.delete_index()
        manager.flat_access.initialize_faiss_index(
            768, index_type, normalize=normalization, index_load_filename=filename
        )
    else:
        # Delete index to free up memory
        manager.flat_access.delete_index()
        manager.flat_access.load_index_from_file(filename, normalize=normalization)
        manager.flat_access.load_all_specific_memories_embeddings_from_file(
            "test_specific_memories"
        )

    # Perform the search for each query and store results
    for query in queries:

        memory_dict[query] = manager.search_memories(query)
    # Save the index after processing
    manager.flat_access.save_index_to_file(filename)


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
#     #print(memory.description)
if __name__ == "__main__":
    # Declare global variables to ensure we're modifying the module-level variables
    global manager, model, sentiment_analysis
    
    tiny_calendar = ttm.GameCalendar()
    tiny_time_manager = ttm.GameTimeManager(tiny_calendar)
    manager = MemoryManager(tiny_time_manager, "ip_no_norm.bin")
    tiny_brain_io = tbi.TinyBrainIO("alexredna/TinyLlama-1.1B-Chat-v1.0-reasoning-v2")

    model = EmbeddingModel()
    sentiment_analysis = SentimentAnalysis()

    # Determine whether there is a saved flat_access_memories file
    if os.path.exists("flat_access_memories.pkl"):
        manager.load_all_flat_access_memories_from_file("flat_access_memories.pkl")

        if os.path.exists("ip_no_norm.bin"):
            manager.flat_access.load_index_from_file("ip_no_norm.bin", normalize=False)
        elif os.path.exists("ip_norm.bin"):
            manager.flat_access.load_index_from_file("ip_norm.bin", normalize=True)
        elif os.path.exists("l2.bin"):
            manager.flat_access.load_index_from_file("l2.bin", normalize=False)
        else:

            manager.flat_access.initialize_faiss_index(768)
        manager.flat_access.load_all_specific_memories_embeddings_from_file(
            "test_specific_memories"
        )
        print("Loaded flat access memories from file")
    else:
        print("No flat access memories file found")

    # NOTE: The descriptions should be "Memories about [type of nounn] and [specific topic]" OR "Memories about [type of noun] and [adjectives about the noun]"

    # Assuming GeneralMemory and manager classes are defined elsewhere

    # Loop over the list of memories and add them to the manager
    from collections import defaultdict

    memory_dict = defaultdict(list)
    # Load JSON data from a file after checking it exists
    memories = []

    if os.path.exists("test_memories.json"):
        with open("test_memories.json", "r") as file:
            # Load the memories list from the JSON file
            memories = json.load(file)
    else:
        memories = [
            {
                "category": "Test Memories",
                "memory": "This memory is a test for the game.",
                "priority": 6,
            },
        ]
        print("No test memories file found")
        raise FileNotFoundError

    random_index = random.randint(0, len(memories) - 1)

    if (
        manager.flat_access.get_specific_memory_by_description(
            memories[random_index]["memory"]
        )
        is None
    ):
        print(
            f"\n Test specific memory: {memories[random_index]['memory']} not found in flat access memories, so adding test file memories"
        )

        # Process each memory entry
        for memory in memories:
            category = memory["category"]
            specific_memory = memory["memory"]
            priority = memory["priority"]

            # Append the memory and priority to the dictionary under the appropriate category
            memory_dict[category].append((specific_memory, priority))

        memory_cache = {}
        for category, entries in memory_dict.items():
            if category not in memory_cache:
                memory_cache[category] = manager.add_general_memory(
                    GeneralMemory(category)
                )
            for specific_memory, priority in entries:
                try:
                    memory_cache[category].add_specific_memory(
                        specific_memory, priority
                    )
                except Exception as e:
                    print(
                        f"Failed to add memory '{specific_memory}' in category {category}: {e}"
                    )
    else:
        print(
            f"\n Test specific memory: {memories[random_index]['memory']} found in flat access memories, so not adding test file memories"
        )

    # print(f"\n \n \n Keywords for each memory: {[(mem.description, mem.keywords) for mem in manager.hierarchy.general_memories]} \n \n \n")

    # time.sleep(2)

    # print(f"\n \n \n List of specific memories for each general memory: {[(mem.description, [specific_memory.description for specific_memory in mem.get_specific_memories()]) for mem in manager.hierarchy.general_memories]} \n \n \n")

    # time.sleep(2)

    # print(f"\n \n \n General Memories: {[mem.description for mem in manager.hierarchy.general_memories]} \n \n \n")

    # Display several views of the initial networkx graph in the hierarchy using networkx and matplotlib
    # Create a new graph that contains only the structure of the original graph
    # visualization_graph = nx.Graph()

    # #print(f"Memory graph: {manager.hierarchy.memory_graph.nodes()}")
    # node_mapping = {node.description: node for node in manager.hierarchy.memory_graph.nodes()}
    # #print(f"Node mapping: {node_mapping}")

    # edge_colors = []
    # edge_list  = []
    # edge_widths = []

    # node_colors = []
    # node_sizes = []
    # descriptions = []
    # # Add nodes with integer labels
    # for node in manager.hierarchy.memory_graph.nodes():
    #     #print(f"Node: {node}")
    #     #print(f"Node description: {node.description}")
    #     # if node.description in descriptions:
    #         #print(f"\n Node description: {node.description} already in descriptions! \n ")
    #     descriptions.append(node.description)
    #     visualization_graph.add_node(node.description)
    #     if isinstance(node, GeneralMemory):
    #         node_colors.append('green')
    #         node_sizes.append(50)
    #     elif isinstance(node, SpecificMemory):
    #         node_colors.append('yellow')
    #         node_sizes.append(node.importance_score * 10)
    #     else:
    #         #print(f"\n atypical Node type: {type(node)} \n")
    #         node_colors.append('black')  # default color
    #         node_sizes.append(50)
    #     visualization_graph.remove_edges_from(list(visualization_graph.edges()))
    #     # Add only the edges from the original graph to the new graph

    # # Map old nodes to new nodes
    # for (node1, node2, keys, data) in manager.hierarchy.memory_graph.edges(data = "weight", keys=True, default=0):
    #     if node1.description == node2.description:
    #         continue
    #     # if node1.description not in descriptions or node2.description not in descriptions:
    #         #print(f"\n Node1: {node1.description}, Node2: {node2.description} not in descriptions! \n ")
    #     if visualization_graph.has_node(node1.description) and visualization_graph.has_node(node2.description) and not visualization_graph.has_edge(node1.description, node2.description) and not visualization_graph.has_edge(node2.description, node1.description) and node1.description != node2.description and data > 0:
    #         visualization_graph.add_edge(node1.description, node2.description, weight=data)
    #         edge_list.append((node1.description, node2.description))
    # print(f"Edge added: {node1.description} to {node2.description} with weight: {data}")
    # elif not visualization_graph.has_node(node1.description):
    #     #print(f"\n Node1: {node1.description} not in visualization graph! \n ")
    # elif not visualization_graph.has_node(node2.description):
    #     #print(f"\n Node2: {node2.description} not in visualization graph! \n ")
    # elif visualization_graph.has_edge(node1.description, node2.description):
    #     #print(f"\n Edge between {node1.description} and {node2.description} already exists! \n ")
    # elif visualization_graph.has_edge(node2.description, node1.description):
    #     #print(f"\n Edge between {node2.description} and {node1.description} already exists! \n ")
    # elif node1.description == node2.description:
    #     #print(f"\n Node1: {node1.description} is the same as Node2: {node2.description}! \n ")
    # elif data <= 0:
    # print(f"\n Edge weight is 0 for {node1.description} and {node2.description}! \n ")

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
    # print(f"Number of nodes: {visualization_graph.number_of_nodes()}")
    # print(f"Number of node sizes: {len(node_sizes)}")

    # print(f"Number of edges: {visualization_graph.number_of_edges()}")

    # nx.draw_networkx(visualization_graph, with_labels=True, node_color=node_colors, node_size=node_sizes,  font_size=8, font_color='blue', font_weight='bold', pos=nx.random_layout(visualization_graph))
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
    #                 #print(f"Error with layout: {layout}, {e} for graph for general memory {general_memory.description}")

    # Create 2 subgraphs that contains only the specific memories that are relevant to a query, one just cosine and the other the score passed from the manager
    #
    # query = MemoryQuery(query,query_time=datetime.now(), gametime_manager=tiny_time_manager)
    # query_embedding = query.get_embedding()
    # relevant_specific_memories = manager.search_memories(query)
    # #print(f"\n \n Relevant specific memories for query '{query}': {relevant_specific_memories} \n \n ")
    # query = "Where was the World Cup held in 2018?"

    # relevant_specific_memories = manager.search_memories(query)
    # #print(f"\n \n Relevant specific memories for query '{query}': {relevant_specific_memories} \n \n ")
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
    #             #print(f"Error with layout: {layout}, {e} for subgraph1 cosine query similarity and subgraph2 score from manager")

    # Create a subgraph that splits the specific memories into clusters based on their similarity
    # subgraph1 = nx.Graph()
    # subgraph2 = nx.Graph()
    # for general_memory in manager.hierarchy.general_memories:
    #     for specific_memory in general_memory.get_specific_memories():
    #         subgraph1.add_node(specific_memory.description)
    #         subgraph2.add_node(specific_memory.description )
    #         node_colors.append('yellow')
    # for general_memory in manager.hierarchy.general_memories:
    #     for specific_memory in general_memory.get_specific_memories():
    #         for specific_memory2 in general_memory.get_specific_memories():
    #             if specific_memory != specific_memory2 and not subgraph1.has_edge(specific_memory.description, specific_memory2.description):
    #                 subgraph1.add_edge(specific_memory.description, specific_memory2.description, weight=cosine_similarity(specific_memory.embedding.cpu().detach().numpy(), specific_memory2.embedding.cpu().detach().numpy()))
    #                 subgraph2.add_edge(specific_memory.description, specific_memory2.description, weight=manager.hierarchy.memory_graph.get_edge_data(specific_memory, specific_memory2).get('weight', 0))

    # if len(subgraph1.nodes()) > 1:
    #     for lname, layout in {"random": nx.random_layout, "spring": nx.spring_layout, "spectral": nx.spectral_layout, "shell": nx.shell_layout, "circular": nx.circular_layout, "kamada_kawai": nx.kamada_kawai_layout, "planar": nx.planar_layout, "fruchterman_reingold": nx.fruchterman_reingold_layout, "spiral": nx.spiral_layout, "bipartite": nx.bipartite_layout, "multipartite": nx.multipartite_layout}.items():

    #         try:
    #             nx.draw_networkx(subgraph1, with_labels=True, font_weight='bold', pos=layout(subgraph1), node_color=node_colors)
    #             plt.title(f"Graph for subgraph1 cosine similarity with layout: {lname}")
    #             plt.show()
    #             nx.draw_networkx(subgraph2, with_labels=True, font_weight='bold', pos=layout(subgraph2), node_color=node_colors)
    #             plt.title(f"Graph for subgraph2 weight from manager graph with layout: {lname}")
    #             plt.show()
    #         except Exception as e:
    # print(f"Error with layout: {layout}, {e} for subgraph1 cosine similarity and subgraph2 weight from manager graph")

    # Define the queries
    queries = [
        "Where was the World Cup held in 2018?",
        "When will the 2022 Winter Olympics be held?",
        "I think someone is planning a surprise party",
        "I need to think of a popular tourist attraction",
        "I am planning a trip to Europe",
        "Who is learning to play the guitar?",
        "I think someone is studying for a Chemistry test",
        "What product should I sell in my electronics store?",
        "What should I eat at the French restaurant?",
        "What book should I read?",
        "Where should I go for a night out to have drinks and meet someone?",
        "What bar should I avoid?",
        "What fashion accessory should I wear to the party?",
        "What is the future of transportation?",
        "What is the current state of the ebola outbreak?",
        "I am a farmer, what crop should I grow to make the most profit?",
        "Is the new iPhone worth buying?",
        "Should I purchase a new iPhone?",
        "As a farmer, what crop would make the most money?",
        "What can I grow on my farm to get the most bang from my buck?",
        "What farm vegetable is selling the most these days?",
        "What farm vegetable is selling the best these days?",
        "What did Billy buy at the store?",
        "Who bought cheese at the shop?",
        "Who is both learning to sing and play the guitar?",
    ]

    # Initialize dictionaries to store results
    memories_ip = {}
    memories_ip_norm = {}
    memories_l2 = {}

    # Manage and search memories for IP without normalization
    manage_index_and_search("ip", False, "ip_no_norm.bin", memories_ip, queries)

    print(f"\n About to save \n \n")
    manager.save_all_flat_access_memories_to_file("flat_access_memories.pkl")
    manager.flat_access.save_all_specific_memories_embeddings_to_file(
        "test_specific_memories"
    )

    # Manage and search memories for IP with normalization
    manage_index_and_search("ip", True, "ip_norm.bin", memories_ip_norm, queries)
    manager.flat_access.save_all_specific_memories_embeddings_to_file(
        "test_specific_memories"
    )
    # Manage and search memories for L2 similarity
    manage_index_and_search("l2", False, "l2.bin", memories_l2, queries)
    print(f"\n\n\n")
    for query in queries:
        print(f"\n Query: {query} \n")
        print(f"\n IP No Norm: {memories_ip[query]} \n")
        established_facts = []
        memory = None
        i = 0
        for mem in memories_ip[query]:
            print(
                f"IP No Norm: Facts: {mem.facts}, Description: {mem.description} Weight: {memories_ip[query][mem]} \n"
            )
            i += 1
            if i == 1:
                established_facts += mem.facts
                memory = mem
        i = 0

        for mem in memories_ip_norm[query]:
            print(
                f"IP Norm: {mem.facts}, Description: {mem.description} Weight: {memories_ip_norm[query][mem]} \n"
            )
        #     i+=1
        #     if i == 1:
        #         established_facts.append(mem.facts)
        # i = 0

        for mem in memories_l2[query]:
            print(
                f"L2: {mem.facts}, Description: {mem.description} Weight: {memories_l2[query][mem]} \n"
            )
        #     i += 1
        #     if i == 1:
        #         established_facts.append(mem.facts)
        # i = 0

        for fact in tsm.main(memory.description, nlp):
            print(f"From SR Mapper: {fact} \n")
            if fact not in established_facts:
                print(
                    f"\nFact not in established facts: {fact}, \n{established_facts}\n"
                )

    exit()

    # Write results to a file in a tabulated format
    with open("results.txt", "w") as file:
        file.write("Query Results Comparison\n")
        file.write("Query | IP No Norm | IP Norm | L2\n")
        file.write("-" * 70 + "\n")
        for query in queries:
            ip_results = ", ".join(
                [
                    f"{mem.description}, weight: {weight}"
                    for mem, weight in memories_ip[query].items()
                ]
            )
            ip_norm_results = ", ".join(
                [
                    f"{mem.description}, weight: {weight}"
                    for mem, weight in memories_ip_norm[query].items()
                ]
            )
            l2_results = ", ".join(
                [
                    f"{mem.description}, weight: {weight}"
                    for mem, weight in memories_l2[query].items()
                ]
            )
            file.write(f"{query} | {ip_results} | {ip_norm_results} | {l2_results}\n")

    # Writing results to an HTML file
    with open("results_comparison.html", "w") as htmlfile:
        htmlfile.write(
            """
        <html>
        <head>
            <title>Memory Search Results Comparison</title>
            <style>
                table {
                    border-collapse: collapse;
                    width: 100%;
                }
                th, td {
                    padding: 15px;
                    text-align: left;
                }
                tr:nth-child(even) {
                    background-color: #f2f2f2;
                }
                th {
                    background-color: #4CAF50;
                    color: white;
                }
            </style>
        </head>
        <body>
            <table border='1'>
                <tr>
                    <th>Query</th>
                    <th>IP No Norm</th>
                    <th>IP Norm</th>
                    <th>L2</th>
                </tr>
        """
        )

        for query in queries:
            ip_results = "<br>".join(
                [
                    f"{mem.description}, weight: {weight}"
                    for mem, weight in memories_ip[query].items()
                ]
            )
            ip_norm_results = "<br>".join(
                [
                    f"{mem.description}, weight: {weight}"
                    for mem, weight in memories_ip_norm[query].items()
                ]
            )
            l2_results = "<br>".join(
                [
                    f"{mem.description}, weight: {weight}"
                    for mem, weight in memories_l2[query].items()
                ]
            )

            htmlfile.write(
                f"""
                <tr>
                    <td>{html.escape(query)}</td>
                    <td>{ip_results}</td>
                    <td>{ip_norm_results}</td>
                    <td>{l2_results}</td>
                </tr>
            """
            )

        htmlfile.write(
            """
            </table>
        </body>
        </html>
        """
        )

    # for mem in memory:
    #     #print(f"Memory lower: {mem} \n")
    # print(f"Memory description: {mem.description} \n")
    # print(f"Memory parent memory: {mem.parent_memory} \n")
    # if not isinstance(mem.parent_memory, GeneralMemory) and isinstance(mem.parent_memory, str):
    #     #print(f"Memory parent memory description: {[m.description for m in manager.hierarchy.general_memories if m.description == mem.parent_memory]} \n")
    #     #print(f"Memory parent general memory keywords: {[m.keywords for m in manager.hierarchy.general_memories if m.description == mem.parent_memory]} \n")
    # elif isinstance(mem.parent_memory, GeneralMemory):
    #     #print(f"Memory parent memory description: {mem.parent_memory.description} \n")
    #     #print(f"Memory parent general memory keywords: {mem.parent_memory.keywords} \n")
    # #print(f"Memory related memories: {mem.related_memories} \n")
    # #print(f"Memory keywords: {mem.keywords} \n")
    # print(f"Memory tags: {mem.tags} \n")
    # print(f"Memory importance score: {mem.importance_score} \n")
    # print(f"Memory sentiment score: {mem.sentiment_score} \n")
    # print(f"Memory emotion classification: {mem.emotion_classification} \n")
    # print(f"Memory last access time: {mem.last_access_time} \n")
    # print(f"Memory recency index: {index} \n")

    exit()
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
    # hierarchy_results_weight = {}
    # hierarchy_results_similarity = {}
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
        ),
    )
    end = time.time()
    time_flat = end - start
    start = time.time()
    # #hierarchy_results_weight = manager.retrieve_from_hierarchy(query)
    end = time.time()
    time_hier_weight = end - start
    start = time.time()
    # hierarchy_results_similarity = manager.hierarchy.find_nodes_by_similarity(query, 0.5)
    end = time.time()
    time_hier_sim = end - start
    # print(f"\n \n \n Hierarchy weight results for query '{query}': {hierarchy_results_weight} \n \n \n")
    ##print(f"\n \n \n Hierarchy similiarity results for query '{query}': {hierarchy_results_similarity} \n \n \n")
    # print(f"\n \n \n Flat access results for query '{query}': {flat_access_results} \n \n \n")
    # time.sleep(2)
    # Write same to file
    with open("results.txt", "a") as file:
        file.write(f"Query: {query} \n \n")
        # file.write(f"Hierarchy search time for edge weight: {time_hier_weight} \n")

        file.write("Facts: \n \n")
        for memory, weight in flat_access_results.items():
            file.write(f"{memory.facts} \n")

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
    # hierarchy_results_weight = manager.retrieve_from_hierarchy(query)
    end = time.time()
    time_hier_weight = end - start
    start = time.time()
    # hierarchy_results_similarity = manager.hierarchy.find_nodes_by_similarity(query, 0.5)
    end = time.time()
    time_hier_sim = end - start
    # print(f"\n \n \n Hierarchy weight results for query '{query}': {hierarchy_results_weight} \n \n \n")
    ##print(f"\n \n \n Hierarchy similiarity results for query '{query}': {hierarchy_results_similarity} \n \n \n")
    # print(f"\n \n \n Flat access results for query '{query}': {flat_access_results} \n \n \n")
    # time.sleep(2)
    # Write same to file
    with open("results.txt", "a") as file:
        file.write(f"Query: {query} \n \n")
        # file.write(f"Hierarchy search time for edge weight: {time_hier_weight} \n")
        # file.write("Semantic Roles: \n \n")
        # for memory, weight in flat_access_results.items():
        #     file.write(f"{memory.analysis['semantic_roles']} \n")

        file.write("Facts: \n \n")
        for memory, weight in flat_access_results.items():
            file.write(f"{memory.facts} \n")

        file.write(f"\n Flat access search time: {time_flat} \n")

        file.write("\n Flat access results: \n \n")
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
    # hierarchy_results_weight = manager.retrieve_from_hierarchy(query)
    end = time.time()
    time_hier_weight = end - start
    start = time.time()
    # hierarchy_results_similarity = manager.hierarchy.find_nodes_by_similarity(query, 0.5)
    end = time.time()
    time_hier_sim = end - start
    # print(f"\n \n \n Hierarchy weight results for query '{query}': {hierarchy_results_weight} \n \n \n")
    ##print(f"\n \n \n Hierarchy similiarity results for query '{query}': {hierarchy_results_similarity} \n \n \n")
    # print(f"\n \n \n Flat access results for query '{query}': {flat_access_results} \n \n \n")
    # time.sleep(2)
    # Write same to file
    with open("results.txt", "a") as file:
        file.write(f"Query: {query} \n \n")
        # file.write(f"Hierarchy search time for edge weight: {time_hier_weight} \n")

        file.write("Facts: \n \n")
        for memory, weight in flat_access_results.items():
            file.write(f"{memory.facts} \n")

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
    # hierarchy_results_weight = manager.retrieve_from_hierarchy(query)
    end = time.time()
    time_hier_weight = end - start
    start = time.time()
    # hierarchy_results_similarity = manager.hierarchy.find_nodes_by_similarity(query, 0.5)
    end = time.time()
    time_hier_sim = end - start
    # print(f"\n \n \n Hierarchy weight results for query '{query}': {hierarchy_results_weight} \n \n \n")
    ##print(f"\n \n \n Hierarchy similiarity results for query '{query}': {hierarchy_results_similarity} \n \n \n")
    # print(f"\n \n \n Flat access results for query '{query}': {flat_access_results} \n \n \n")
    # time.sleep(2)
    # Write same to file
    with open("results.txt", "a") as file:
        file.write(f"Query: {query} \n \n")
        # file.write(f"Hierarchy search time for edge weight: {time_hier_weight} \n")

        file.write("Facts: \n \n")
        for memory, weight in flat_access_results.items():
            file.write(f"{memory.facts} \n")

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
    # hierarchy_results_weight = manager.retrieve_from_hierarchy(query)
    end = time.time()
    time_hier_weight = end - start
    start = time.time()
    # hierarchy_results_similarity = manager.hierarchy.find_nodes_by_similarity(query, 0.5)
    end = time.time()
    time_hier_sim = end - start
    # print(f"\n \n \n Hierarchy weight results for query '{query}': {hierarchy_results_weight} \n \n \n")
    ##print(f"\n \n \n Hierarchy similiarity results for query '{query}': {hierarchy_results_similarity} \n \n \n")
    # print(f"\n \n \n Flat access results for query '{query}': {flat_access_results} \n \n \n")
    # time.sleep(2)
    # Write same to file
    with open("results.txt", "a") as file:
        file.write(f"Query: {query} \n \n")
        # file.write(f"Hierarchy search time for edge weight: {time_hier_weight} \n")

        file.write("Facts: \n \n")
        for memory, weight in flat_access_results.items():
            file.write(f"{memory.facts} \n")

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
    # hierarchy_results_weight = manager.retrieve_from_hierarchy(query)
    end = time.time()
    time_hier_weight = end - start
    start = time.time()
    # hierarchy_results_similarity = manager.hierarchy.find_nodes_by_similarity(query, 0.5)
    end = time.time()
    time_hier_sim = end - start
    # print(f"\n \n \n Hierarchy weight results for query '{query}': {hierarchy_results_weight} \n \n \n")
    ##print(f"\n \n \n Hierarchy similiarity results for query '{query}': {hierarchy_results_similarity} \n \n \n")
    # print(f"\n \n \n Flat access results for query '{query}': {flat_access_results} \n \n \n")
    # time.sleep(2)
    # Write same to file
    with open("results.txt", "a") as file:
        file.write(f"Query: {query} \n \n")
        # file.write(f"Hierarchy search time for edge weight: {time_hier_weight} \n")

        file.write("Facts: \n \n")
        for memory, weight in flat_access_results.items():
            file.write(f"{memory.facts} \n")

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
    # hierarchy_results_weight = manager.retrieve_from_hierarchy(query)
    end = time.time()
    time_hier_weight = end - start
    start = time.time()
    # hierarchy_results_similarity = manager.hierarchy.find_nodes_by_similarity(query, 0.5)
    end = time.time()
    time_hier_sim = end - start
    # print(f"\n \n \n Hierarchy weight results for query '{query}': {hierarchy_results_weight} \n \n \n")
    ##print(f"\n \n \n Hierarchy similiarity results for query '{query}': {hierarchy_results_similarity} \n \n \n")
    # print(f"\n \n \n Flat access results for query '{query}': {flat_access_results} \n \n \n")
    # time.sleep(2)
    # Write same to file
    with open("results.txt", "a") as file:
        file.write(f"Query: {query} \n \n")
        # file.write(f"Hierarchy search time for edge weight: {time_hier_weight} \n")

        file.write("Facts: \n \n")
        for memory, weight in flat_access_results.items():
            file.write(f"{memory.facts} \n")

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
    # hierarchy_results_weight = manager.retrieve_from_hierarchy(query)
    end = time.time()
    time_hier_weight = end - start
    start = time.time()
    # hierarchy_results_similarity = manager.hierarchy.find_nodes_by_similarity(query, 0.5)
    end = time.time()
    time_hier_sim = end - start
    # print(f"\n \n \n Hierarchy weight results for query '{query}': {hierarchy_results_weight} \n \n \n")
    ##print(f"\n \n \n Hierarchy similiarity results for query '{query}': {hierarchy_results_similarity} \n \n \n")
    # print(f"\n \n \n Flat access results for query '{query}': {flat_access_results} \n \n \n")
    # time.sleep(2)
    # Write same to file
    with open("results.txt", "a") as file:
        file.write(f"Query: {query} \n \n")
        # file.write(f"Hierarchy search time for edge weight: {time_hier_weight} \n")

        file.write("Facts: \n \n")
        for memory, weight in flat_access_results.items():
            file.write(f"{memory.facts} \n")

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
    # hierarchy_results_weight = manager.retrieve_from_hierarchy(query)
    end = time.time()
    time_hier_weight = end - start
    start = time.time()
    # hierarchy_results_similarity = manager.hierarchy.find_nodes_by_similarity(query, 0.5)
    end = time.time()
    time_hier_sim = end - start
    # print(f"\n \n \n Hierarchy weight results for query '{query}': {hierarchy_results_weight} \n \n \n")
    ##print(f"\n \n \n Hierarchy similiarity results for query '{query}': {hierarchy_results_similarity} \n \n \n")
    # print(f"\n \n \n Flat access results for query '{query}': {flat_access_results} \n \n \n")
    # time.sleep(2)
    # Write same to file
    with open("results.txt", "a") as file:
        file.write(f"Query: {query} \n \n")
        # file.write(f"Hierarchy search time for edge weight: {time_hier_weight} \n")

        file.write("Facts: \n \n")
        for memory, weight in flat_access_results.items():
            file.write(f"{memory.facts} \n")

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
    # hierarchy_results_weight = manager.retrieve_from_hierarchy(query)
    end = time.time()
    time_hier_weight = end - start
    start = time.time()
    # hierarchy_results_similarity = manager.hierarchy.find_nodes_by_similarity(query, 0.5)
    end = time.time()
    time_hier_sim = end - start
    # print(f"\n \n \n Hierarchy weight results for query '{query}': {hierarchy_results_weight} \n \n \n")
    ##print(f"\n \n \n Hierarchy similiarity results for query '{query}': {hierarchy_results_similarity} \n \n \n")
    # print(f"\n \n \n Flat access results for query '{query}': {flat_access_results} \n \n \n")
    # time.sleep(2)
    # Write same to file
    with open("results.txt", "a") as file:
        file.write(f"Query: {query} \n \n")
        # file.write(f"Hierarchy search time for edge weight: {time_hier_weight} \n")

        file.write("Facts: \n \n")
        for memory, weight in flat_access_results.items():
            file.write(f"{memory.facts} \n")

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
    # hierarchy_results_weight = manager.retrieve_from_hierarchy(query)
    end = time.time()
    time_hier_weight = end - start
    start = time.time()
    # hierarchy_results_similarity = manager.hierarchy.find_nodes_by_similarity(query, 0.5)
    end = time.time()
    time_hier_sim = end - start
    # print(f"\n \n \n Hierarchy weight results for query '{query}': {hierarchy_results_weight} \n \n \n")
    ##print(f"\n \n \n Hierarchy similiarity results for query '{query}': {hierarchy_results_similarity} \n \n \n")
    # print(f"\n \n \n Flat access results for query '{query}': {flat_access_results} \n \n \n")
    # time.sleep(2)
    # Write same to file
    with open("results.txt", "a") as file:
        file.write(f"Query: {query} \n \n")
        # file.write(f"Hierarchy search time for edge weight: {time_hier_weight} \n")

        file.write("Facts: \n \n")
        for memory, weight in flat_access_results.items():
            file.write(f"{memory.facts} \n")

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
    # hierarchy_results_weight = manager.retrieve_from_hierarchy(query)
    end = time.time()
    time_hier_weight = end - start
    start = time.time()
    # hierarchy_results_similarity = manager.hierarchy.find_nodes_by_similarity(query, 0.5)
    end = time.time()
    time_hier_sim = end - start
    # print(f"\n \n \n Hierarchy weight results for query '{query}': {hierarchy_results_weight} \n \n \n")
    ##print(f"\n \n \n Hierarchy similiarity results for query '{query}': {hierarchy_results_similarity} \n \n \n")
    # print(f"\n \n \n Flat access results for query '{query}': {flat_access_results} \n \n \n")
    # time.sleep(2)
    # Write same to file
    with open("results.txt", "a") as file:
        file.write(f"Query: {query} \n \n")
        # file.write(f"Hierarchy search time for edge weight: {time_hier_weight} \n")

        file.write("Facts: \n \n")
        for memory, weight in flat_access_results.items():
            file.write(f"{memory.facts} \n")

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
    # hierarchy_results_weight = manager.retrieve_from_hierarchy(query)
    end = time.time()
    time_hier_weight = end - start
    start = time.time()
    # hierarchy_results_similarity = manager.hierarchy.find_nodes_by_similarity(query, 0.5)
    end = time.time()
    time_hier_sim = end - start
    # print(f"\n \n \n Hierarchy weight results for query '{query}': {hierarchy_results_weight} \n \n \n")
    ##print(f"\n \n \n Hierarchy similiarity results for query '{query}': {hierarchy_results_similarity} \n \n \n")
    # print(f"\n \n \n Flat access results for query '{query}': {flat_access_results} \n \n \n")
    # time.sleep(2)
    # Write same to file
    with open("results.txt", "a") as file:
        file.write(f"Query: {query} \n \n")
        # file.write(f"Hierarchy search time for edge weight: {time_hier_weight} \n")

        file.write("Facts: \n \n")
        for memory, weight in flat_access_results.items():
            file.write(f"{memory.facts} \n")

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
    # hierarchy_results_weight = manager.retrieve_from_hierarchy(query)
    end = time.time()
    time_hier_weight = end - start
    start = time.time()
    # hierarchy_results_similarity = manager.hierarchy.find_nodes_by_similarity(query, 0.5)
    end = time.time()
    time_hier_sim = end - start
    # print(f"\n \n \n Hierarchy weight results for query '{query}': {hierarchy_results_weight} \n \n \n")
    ##print(f"\n \n \n Hierarchy similiarity results for query '{query}': {hierarchy_results_similarity} \n \n \n")
    # print(f"\n \n \n Flat access results for query '{query}': {flat_access_results} \n \n \n")
    # time.sleep(2)
    # Write same to file
    with open("results.txt", "a") as file:
        file.write(f"Query: {query} \n \n")
        # file.write(f"Hierarchy search time for edge weight: {time_hier_weight} \n")

        file.write("Facts: \n \n")
        for memory, weight in flat_access_results.items():
            file.write(f"{memory.facts} \n")

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
    # hierarchy_results_weight = manager.retrieve_from_hierarchy(query)
    end = time.time()
    time_hier_weight = end - start
    start = time.time()
    # hierarchy_results_similarity = manager.hierarchy.find_nodes_by_similarity(query, 0.5)
    end = time.time()
    time_hier_sim = end - start
    # print(f"\n \n \n Hierarchy weight results for query '{query}': {hierarchy_results_weight} \n \n \n")
    ##print(f"\n \n \n Hierarchy similiarity results for query '{query}': {hierarchy_results_similarity} \n \n \n")
    # print(f"\n \n \n Flat access results for query '{query}': {flat_access_results} \n \n \n")
    # time.sleep(2)
    # Write same to file
    with open("results.txt", "a") as file:
        file.write(f"Query: {query} \n \n")
        # file.write(f"Hierarchy search time for edge weight: {time_hier_weight} \n")

        file.write("Facts: \n \n")
        for memory, weight in flat_access_results.items():
            file.write(f"{memory.facts} \n")

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
    # hierarchy_results_weight = manager.retrieve_from_hierarchy(query)
    end = time.time()
    time_hier_weight = end - start
    start = time.time()
    # hierarchy_results_similarity = manager.hierarchy.find_nodes_by_similarity(query, 0.5)
    end = time.time()
    time_hier_sim = end - start
    # print(f"\n \n \n Hierarchy weight results for query '{query}': {hierarchy_results_weight} \n \n \n")
    ##print(f"\n \n \n Hierarchy similiarity results for query '{query}': {hierarchy_results_similarity} \n \n \n")
    # print(f"\n \n \n Flat access results for query '{query}': {flat_access_results} \n \n \n")
    # time.sleep(2)
    # Write same to file
    with open("results.txt", "a") as file:
        file.write(f"Query: {query} \n \n")
        # file.write(f"Hierarchy search time for edge weight: {time_hier_weight} \n")

        file.write("Facts: \n \n")
        for memory, weight in flat_access_results.items():
            file.write(f"{memory.facts} \n")

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
    # hierarchy_results_weight = manager.retrieve_from_hierarchy(query)
    end = time.time()
    time_hier_weight = end - start
    start = time.time()
    # hierarchy_results_similarity = manager.hierarchy.find_nodes_by_similarity(query, 0.5)
    end = time.time()
    time_hier_sim = end - start
    # print(f"\n \n \n Hierarchy weight results for query '{query}': {hierarchy_results_weight} \n \n \n")
    ##print(f"\n \n \n Hierarchy similiarity results for query '{query}': {hierarchy_results_similarity} \n \n \n")
    # print(f"\n \n \n Flat access results for query '{query}': {flat_access_results} \n \n \n")
    # time.sleep(2)
    # Write same to file
    with open("results.txt", "a") as file:
        file.write(f"Query: {query} \n \n")
        # file.write(f"Hierarchy search time for edge weight: {time_hier_weight} \n")

        file.write("Facts: \n \n")
        for memory, weight in flat_access_results.items():
            file.write(f"{memory.facts} \n")

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
    # hierarchy_results_weight = manager.retrieve_from_hierarchy(query)
    end = time.time()
    time_hier_weight = end - start
    start = time.time()
    # hierarchy_results_similarity = manager.hierarchy.find_nodes_by_similarity(query, 0.5)
    end = time.time()
    time_hier_sim = end - start
    # print(f"\n \n \n Hierarchy weight results for query '{query}': {hierarchy_results_weight} \n \n \n")
    ##print(f"\n \n \n Hierarchy similiarity results for query '{query}': {hierarchy_results_similarity} \n \n \n")
    # print(f"\n \n \n Flat access results for query '{query}': {flat_access_results} \n \n \n")
    # time.sleep(2)
    # Write same to file
    with open("results.txt", "a") as file:
        file.write(f"Query: {query} \n \n")
        # file.write(f"Hierarchy search time for edge weight: {time_hier_weight} \n")

        file.write("Facts: \n \n")
        for memory, weight in flat_access_results.items():
            file.write(f"{memory.facts} \n")

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
