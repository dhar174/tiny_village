import json
import re
from unittest import result
from numpy import where
from requests import get
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import time
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
from collections import defaultdict
import statistic
import os

os.environ["TRANSFORMERS_CACHE"] = "/mnt/d/transformers_cache"

from llama_cpp import Llama

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
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.load_model()

    def load_model(self, model_name="alexredna/TinyLlama-1.1B-Chat-v1.0-reasoning-v2"):
        if model_name is not None:
            self.model_name = model_name
        if "gguf" in self.model_name.lower():
            self.model = Llama(
                model_path="./" + self.model_name,
                n_ctx=self.n_ctx,
                n_threads=self.n_threads,
                n_gpu_layers=self.n_gpu_layers,
                offload_kqv=self.offload_kqv,
                use_mlock=self.use_mlock,
            )
        else:
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
                start_time = time.time()
                output = self.model(text, max_tokens=256, stop="</s>", echo=True)
                end_time = time.time()
                print(output)
                generated_text = output["choices"][0]["text"]
            else:
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
