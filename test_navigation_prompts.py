from calendar import c
from http.client import responses
from json import load
from math import e
from mimetypes import init
import sys
import tokenize
import transformers
import numpy as np
import random
from transformers import Conversation


import torch


class ModelLoader:
    def __init__(
        self,
        model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        initial_conversation=None,
        initial_prompt=None,
        system="conversational",
        system_prompt=None,
    ):
        self.tokenizer = None
        self.system = system
        self.system_prompt = system_prompt
        self.conversation = None

        self.model = self.load_model(model)
        if initial_conversation:
            self.conversation = initial_conversation
        elif initial_prompt:
            self.conversation = self.begin_conversation(initial_prompt)
        self.latest_response = None
        self.response = None

    def __call__(self, conversation=None):
        if conversation:
            return self.generate_response(conversation)
        else:
            return self.generate_response(self.conversation)

    def __str__(self):
        return f"ModelLoader(model={self.model}, conversation={self.conversation}, latest_response={self.latest_response}, response={self.response})"

    def __repr__(self):
        return f"ModelLoader(model={self.model}, conversation={self.conversation}, latest_response={self.latest_response}, response={self.response})"

    def __getitem__(self, key):
        if key == "model":
            return self.model
        elif key == "conversation":
            return self.conversation
        elif key == "latest_response":
            return self.latest_response
        elif key == "response":
            return self.response
        else:
            raise KeyError(f"Invalid key: {key}")

    def __setitem__(self, key, value):
        if key == "model":
            self.model = value
        elif key == "conversation":
            self.conversation = value
        elif key == "latest_response":
            self.latest_response = value
        elif key == "response":
            self.response = value
        else:
            raise KeyError(f"Invalid key: {key}")

    def __delitem__(self, key):
        if key == "model":
            self.model = None
        elif key == "conversation":
            self.conversation = None
        elif key == "latest_response":
            self.latest_response = None
        elif key == "response":
            self.response = None
        else:
            raise KeyError(f"Invalid key: {key}")

    def __contains__(self, key):
        return key in ["model", "conversation", "latest_response", "response"]

    def forward(self, conversation):
        return self.generate_response(conversation)

    def load_model(self, model):
        # Check if a GPU is available and if not, use the CPU
        device = 0 if torch.cuda.is_available() else -1

        # Load the model
        print("Loading model...")
        if isinstance(model, str):
            self.tokenizer = transformers.AutoTokenizer.from_pretrained(model)
            print(f"Tokenizer loaded for model {model}")
        elif isinstance(
            model, transformers.models.llama.modeling_llama.LlamaForCausalLM
        ):
            self.tokenizer = transformers.AutoTokenizer.from_pretrained(
                model.model.name_or_path
            )
            print(f"Tokenizer loaded for model {model.model.name_or_path}")
        elif model.model:
            self.tokenizer = transformers.AutoTokenizer.from_pretrained(
                model.model.name_or_path
            )
            print(f"Tokenizer loaded for model {model.model.name_or_path}")
        assert self.tokenizer is not None, "Tokenizer not loaded."
        model = transformers.pipeline(
            self.system,
            model=model,
            device=device,
            trust_remote_code=True,
            tokenizer=self.tokenizer,
        )
        print("Model loaded.")
        return model

    def begin_conversation(self, prompt):
        conversation = Conversation()
        if self.system_prompt:
            conversation.add_message(
                {"role": "assistant", "content": self.system_prompt}
            )
        conversation.add_message({"role": "user", "content": prompt})
        initial_response = self.model(conversation)
        self.latest_response = self.get_latest_response(initial_response)
        self.response = self.get_latest_response(initial_response)
        conversation.add_message({"role": "assistant", "content": self.latest_response})
        self.conversation = conversation

        return initial_response, conversation

    def batch_single_response(self, messages):
        if isinstance(messages, str):
            conversation = Conversation()
            if self.system_prompt:
                conversation.add_message(
                    {"role": "assistant", "content": self.system_prompt}
                )
            conversation.add_message({"role": "user", "content": prompt})
            initial_response = self.model(conversation)
            conversation.add_message(
                {"role": "assistant", "content": self.latest_response}
            )
            return [(initial_response, messages)]

        elif isinstance(messages, list):

            responses = []

            for message in messages:
                conversation = Conversation()
                if self.system_prompt:
                    conversation.add_message(
                        {"role": "assistant", "content": self.system_prompt}
                    )
                response, conversation = self.begin_conversation(message)
                self.conversation.add_message(
                    {
                        "role": "assistant",
                        "content": self.get_latest_response(response),
                    }
                )
                responses.append((self.get_latest_response(response), message))

            return responses

        elif isinstance(messages, Conversation):
            responses = []
            if "system" not in messages.messages[0] and self.system_prompt:
                messages.add_message(
                    {"role": "assistant", "content": self.system_prompt}
                )
            for message in messages.messages:

                if message["role"] == "user":
                    response = self.model(messages)
                    messages.add_message(
                        {
                            "role": "assistant",
                            "content": self.get_latest_response(response),
                        }
                    )
                    responses.append((self.get_latest_response(response), message))
            return responses
        else:
            raise ValueError(
                "Invalid input. Please provide a string, a list of strings, or a Conversation object."
            )

    def generate_response(self, conversation):
        if isinstance(conversation, str):
            if not self.conversation:
                response, self.conversation = self.begin_conversation(conversation)
                self.conversation.add_message(
                    {"role": "assistant", "content": self.get_latest_response(response)}
                )
                return self.get_latest_response(response)
            else:
                self.conversation.add_message({"role": "user", "content": conversation})
        elif isinstance(conversation, list):

            responses = []
            for message in conversation:
                if not self.conversation:
                    response, self.conversation = self.begin_conversation(message)
                    self.conversation.add_message(
                        {
                            "role": "assistant",
                            "content": self.get_latest_response(response),
                        }
                    )
                else:
                    self.conversation.add_message({"role": "user", "content": message})
                    response = self.model(self.conversation)
                    self.conversation.add_message(
                        {
                            "role": "assistant",
                            "content": self.get_latest_response(response),
                        }
                    )
                responses.append((message, self.get_latest_response(response)))
            return responses

        elif isinstance(conversation, Conversation):
            if not self.conversation:
                if (
                    conversation.messages is not []
                    and conversation.messages is not None
                ):
                    # If the conversation has messages, use the last message from the user as the prompt.

                    if conversation.messages[-1]["role"] == "user":
                        response, self.conversation = self.begin_conversation(
                            conversation.messages[-1]["content"]
                        )
                    elif (
                        len(conversation.messages) > 1
                        and conversation.messages[-2]["role"] == "user"
                    ):
                        response, self.conversation = self.begin_conversation(
                            conversation.messages[-2]["content"]
                        )
                    else:
                        response = conversation.messages[-1]["content"]
                    self.conversation = conversation
                    self.conversation.add_message(
                        {
                            "role": "assistant",
                            "content": self.get_latest_response(response),
                        }
                    )
                    return self.get_latest_response(response)
        self.conversation = conversation
        response = self.model(self.conversation)
        self.conversation.add_message(
            {"role": "assistant", "content": self.get_latest_response(response)}
        )
        self.latest_response = self.get_latest_response(response)
        self.response = response
        return self.get_latest_response(response)

    def get_latest_response(self, response):
        return response.generated_responses[-1]


if __name__ == "__main__":
    # Check if a GPU is available and if not, use the CPU
    device = 0 if torch.cuda.is_available() else -1

    # Load the TinyLlama/TinyLlama-1.1B-Chat-v1.0 model
    print("Loading model...")
    # tokenizer = transformers.AutoTokenizer.from_pretrained(
    #     "microsoft/Phi-3-mini-4k-instruct"
    # )
    # model = transformers.pipeline(
    #     "conversational",
    #     model="microsoft/Phi-3-mini-4k-instruct",
    #     device=device,
    #     trust_remote_code=True,
    #     tokenizer=tokenizer,
    # )

    # ... rest of the code ...
    # Define the prompt
    environments = ["forest", "desert", "mountains", "swamp"]
    enemy_types = ["goblin", "ogre", "troll", "dragon"]
    obstacle_types = ["boulder", "trap", "river", "cliff"]
    environment = random.choice(environments)
    # Create a grid environment of random size between 5x5 and 30x30
    length = np.random.randint(5, 15)
    width = np.random.randint(5, 15)
    grid = np.zeros((length, width))
    enemies = [
        (
            random.choice(enemy_types),
            (np.random.randint(0, length), np.random.randint(0, width)),
        )
        for _ in range(3)
    ]
    obstacles = [
        (
            random.choice(obstacle_types),
            (np.random.randint(0, length), np.random.randint(0, width)),
        )
        for _ in range(3)
    ]

    # Randomly place the character, treasure, enemies, and obstacles
    character_pos = (np.random.randint(0, length), np.random.randint(0, width))
    treasure_pos = (np.random.randint(0, length), np.random.randint(0, width))
    task = f"You are in a {environment} environment. Your character is at position {character_pos}. The treasure is at position {treasure_pos}. There are enemies at positions {enemies}. There are obstacles at positions {obstacles}. Plan a step-by-step route to reach the treasure, avoiding the enemies and obstacles.\n"
    example = f"""You are a video game character searching for treasure in an environment represented by a grid filled with enemies and obstacles. Your job is to plan a step-by-step route to reach the treasure, avoiding the enemies and navigating around obstacles.
        \n
        Rules:\n
        1. A cell with either an obstacle, an enemy, or that is at the edges of the grid cannot be passed through. Ending up on the same cell as an enemy or obstacle, or being outside the grid, results in the death of the game character.
        \n2. If adjacent (any direction) to an enemy or obstacle, you can move past them to the cell on the opposite side using "avoid enemy at (x,y)" or "navigate around obstacle at (x,y)", which will place you in the cell on the opposite side to the named enemy or obstacle. "Avoid" can only be used with enemies and "navigate around" can only be used with obstacles.
        \n
        Example:\n
        Plan a route for the given environment: You are in a forest environment of grid size (10,10). Your character is at position (2,2). The treasure is at position (4,4). There are enemies at positions [(3, 3), (1, 1), (4, 3)]. There are obstacles at positions [(2, 3), (3, 2), (1, 4)]. Plan a step-by-step route to reach the treasure, avoiding the enemies and obstacles.
        \n\n
        Route:\n
        1. move east to (3, 2)\n
        2. move east to (4, 2)\n
        3. move south to (4, 3)\n
        4. avoid enemy at (4, 3)\n
        5. move south to (4, 4)\n

        Now, plan a route for the given environment:\n"""
    # prompt = f"{example}\n{task}"
    conversation = Conversation()
    conversation.add_message({"role": "system", "content": example})
    conversation.add_message({"role": "user", "content": task})

    model = ModelLoader(
        model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        initial_conversation=conversation,
    )
    print("Model loaded.")
    # Generate a response
    print("Generating response...")
    response = model(conversation)

    print(response)
