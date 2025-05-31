import json
import re
from unittest import result
from numpy import where
from requests import get

# Conditional imports for transformers and torch
_TRANSFORMERS_AVAILABLE = False
_TORCH_AVAILABLE = False

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    _TRANSFORMERS_AVAILABLE = True
    print("Successfully imported transformers in tiny_brain_io.")
except ImportError:
    AutoModelForCausalLM = None
    AutoTokenizer = None
    print("WARNING: Transformers library not found in tiny_brain_io. HuggingFace model features will be unavailable.")

try:
    import torch
    _TORCH_AVAILABLE = True
    print("Successfully imported torch in tiny_brain_io.")
except ImportError:
    torch = None
    print("WARNING: PyTorch (torch) not found in tiny_brain_io. Torch-dependent features will be unavailable.")

import time

_SKLEARN_AVAILABLE_TBI = False # TBI for TinyBrainIO specific
try:
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    _SKLEARN_AVAILABLE_TBI = True
    print("Successfully imported sklearn components in tiny_brain_io.")
except ImportError:
    CountVectorizer = None
    cosine_similarity = None
    print("WARNING: scikit-learn components not found in tiny_brain_io. Some features will be unavailable.")

import re
from collections import defaultdict
import os

os.environ["TRANSFORMERS_CACHE"] = "/mnt/d/transformers_cache"

_LLAMA_CPP_AVAILABLE = False
try:
    from llama_cpp import Llama
    _LLAMA_CPP_AVAILABLE = True
    print("Successfully imported llama_cpp in tiny_brain_io.")
except ImportError:
    Llama = None
    print("WARNING: llama_cpp library not found in tiny_brain_io. GGUF model features will be unavailable.")

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
        self.device = "cuda" if _TORCH_AVAILABLE and torch and torch.cuda.is_available() else "cpu"
        self.load_model()

    def load_model(self, model_name="alexredna/TinyLlama-1.1B-Chat-v1.0-reasoning-v2"):
        if model_name is not None:
            self.model_name = model_name

        if "gguf" in self.model_name.lower():
            if _LLAMA_CPP_AVAILABLE and Llama is not None:
                try:
                    self.model = Llama(
                        model_path="./" + self.model_name,
                        n_ctx=self.n_ctx, n_threads=self.n_threads, n_gpu_layers=self.n_gpu_layers,
                        offload_kqv=self.offload_kqv, use_mlock=self.use_mlock,
                    )
                except Exception as e:
                    print(f"Failed to load Llama GGUF model: {e}")
                    self.model = None
            else:
                print("Llama CPP library not available. Cannot load GGUF model.")
                self.model = None
        elif _TRANSFORMERS_AVAILABLE and AutoModelForCausalLM is not None and AutoTokenizer is not None:
            try:
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name, trust_remote_code=True, **self.special_args, cache_dir="/mnt/d/transformers_cache"
                )
                if _TORCH_AVAILABLE and torch and self.model: # Ensure model is loaded before .to(device)
                    self.model.to(self.device)

                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.model_name, cache_dir="/mnt/d/transformers_cache", trust_remote_code=True
                )
            except Exception as e:
                print(f"Failed to load HuggingFace model/tokenizer: {e}")
                self.model = None; self.tokenizer = None
        else:
            print("Transformers library not available. Cannot load HuggingFace model.")
            self.model = None
            self.tokenizer = None

    def input_to_model(self, prompts, reset_model=True): # reset_model not used
        if not self.model:
            print(f"Model {self.model_name} is not loaded. Cannot process input.")
            return [("Error: Model not loaded.", "0.0")] * (len(prompts) if isinstance(prompts, list) else 1)

        print(f"Testing model: {self.model_name}")
        # special_args = self.model_special_args.get(self.model_name, {}) # Not used
        results = []
        for text in prompts if isinstance(prompts, list) else [prompts]:
            print(f"\nPrompt: {text} \n---------------------------------------------- \nType: {type(text)} \n")
            text = re.sub(r"|".join(map(re.escape, remove_list)), "", text)
            text = text.replace("  ", " ")

            generated_text = "Error: Inference failed."
            processing_time_str = "0.0"

            if "gguf" in self.model_name.lower() and _LLAMA_CPP_AVAILABLE and isinstance(self.model, Llama):
                try:
                    start_time = time.time()
                    output = self.model(text, max_tokens=256, stop="</s>", echo=True)
                    end_time = time.time()
                    print(output)
                    generated_text = output["choices"][0]["text"]
                    processing_time_str = str(end_time - start_time)
                except Exception as e:
                    print(f"Error during GGUF model inference: {e}")
            elif _TRANSFORMERS_AVAILABLE and self.tokenizer and self.model and _TORCH_AVAILABLE and torch: # Ensure tokenizer and torch also available
                try:
                    input_ids = self.tokenizer.encode(text, return_tensors="pt").to(self.device)
                    start_time = time.time()
                    output_tokens = self.model.generate(input_ids, max_new_tokens=20, no_repeat_ngram_size=2)
                    end_time = time.time()
                    print(output_tokens)
                    generated_text = self.tokenizer.decode(output_tokens[0], skip_special_tokens=True)
                    processing_time_str = str(end_time - start_time)
                except Exception as e:
                    print(f"Error during HuggingFace model inference: {e}")
            else:
                print("Model type not supported or dependencies missing (Transformers/Torch).")
                generated_text = "Error: Model type not supported or deps missing."

            print(f"\nGenerated text pre-clean: {generated_text} \n")
            # Removed duplicate, incorrectly indented line below
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
