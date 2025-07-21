import json
import re
from unittest import result
# Graceful fallback for numpy
try:
    from numpy import where
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    def where(condition):
        return condition  # Simple fallback

# Optional requests import
try:
    from requests import get
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False
    def get(*args, **kwargs):
        raise ImportError("requests not available")

# Optional transformers import
try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    AutoModelForCausalLM = AutoTokenizer = None
# Conditional torch import - skip functionality if not available
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None

import time

# Optional sklearn imports
try:
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    CountVectorizer = None
    def cosine_similarity(a, b):
        return [[0.5]]  # Neutral similarity

import re
from collections import defaultdict
import os

os.environ["TRANSFORMERS_CACHE"] = "/mnt/d/transformers_cache"

# Optional llama_cpp import
try:
    from llama_cpp import Llama
    LLAMA_CPP_AVAILABLE = True
except ImportError:
    LLAMA_CPP_AVAILABLE = False
    Llama = None

remove_list = [r"\)", r"\(", "–", '"', '"', '"', r"\[.*\]", r".*\|.*", "—"]


class TinyBrainIO:
    def __init__(self, model_name, model_special_args={}):
        self.model_name = model_name
        self.model_special_args = model_special_args
        self.model = None
        self.tokenizer = None
        self.special_args = model_special_args.get(model_name, {})
        self.model_path = None
        self.n_ctx = 512
        self.n_threads = 6
        self.n_gpu_layers = 0
        self.offload_kqv = False
        self.use_mlock = True
        self.device = "cuda" if (TORCH_AVAILABLE and torch.cuda.is_available()) else "cpu"
        self.load_model()

    def load_model(self, model_name="alexredna/TinyLlama-1.1B-Chat-v1.0-reasoning-v2"):
        if model_name is not None:
            self.model_name = model_name
        if "gguf" in self.model_name.lower():
            if not LLAMA_CPP_AVAILABLE:
                print("Warning: llama_cpp not available, cannot load GGUF models")
                return
            self.model = Llama(
                model_path="./" + self.model_name,
                n_ctx=self.n_ctx,
                n_threads=self.n_threads,
                n_gpu_layers=self.n_gpu_layers,
                offload_kqv=self.offload_kqv,
                use_mlock=self.use_mlock,
            )
        else:
            if not TRANSFORMERS_AVAILABLE:
                print("Warning: transformers not available, cannot load HuggingFace models")
                return
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                **self.special_args,
                cache_dir="/mnt/d/transformers_cache",
            ).to(self.device)
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                cache_dir="/mnt/d/transformers_cache",
                trust_remote_code=True,
            )

    def input_to_model(self, prompts, reset_model=True):
        if not self.model:
            print("Warning: No model loaded. Cannot process prompts.")
            return [(f"Model not available: {prompt}", "0.0") for prompt in (prompts if isinstance(prompts, list) else [prompts])]
            
        print(f"Testing model: {self.model_name}")
        special_args = self.model_special_args.get(self.model_name, {})
        print("Type of prompts: ", type(prompts))
        results = []
        for text in prompts if isinstance(prompts, list) else [prompts]:
            print(f"\nPrompt: {text} \n")
            print(f"---------------------------------------------- \n")
            print(f"Type: {type(text)} \n")
            text = re.sub(r"|".join(map(re.escape, remove_list)), "", text)
            text = text.replace("  ", " ")
            # Measure the processing time
            if "gguf" in self.model_name.lower():
                if not LLAMA_CPP_AVAILABLE:
                    results.append(("GGUF model not available", "0.0"))
                    continue
                start_time = time.time()
                output = self.model(text, max_tokens=256, stop="</s>", echo=True)
                end_time = time.time()
                print(output)
                generated_text = output["choices"][0]["text"]
            else:
                if not TRANSFORMERS_AVAILABLE or not self.tokenizer:
                    results.append(("HuggingFace model not available", "0.0"))
                    continue
                # Encode the input text
                input_ids = self.tokenizer.encode(text, return_tensors="pt").to(
                    self.device
                )
                start_time = time.time()
                output = self.model.generate(
                    input_ids, max_new_tokens=20, no_repeat_ngram_size=2
                )
                print(output)
                end_time = time.time()

                # Decode the generated text
                generated_text = self.tokenizer.decode(
                    output[0], skip_special_tokens=True
                )
                print(f"\nGenerated text pre-clean: {generated_text} \n")
            try:
                if "I choose" in generated_text and "I choose" in text:
                    generated_text = generated_text.split("I choose ")[1]
                else:
                    generated_text = generated_text.replace(text, "")
            except:
                generated_text = generated_text
            print(f"\nGenerated text: {generated_text} \n")
            # Calculate processing time
            processing_time = end_time - start_time
            print(f"Processing time: {processing_time} seconds \n")
            results.append(tuple((generated_text, str(processing_time))))

        return results
