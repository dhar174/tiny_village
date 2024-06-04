# wavelet_train_test


from calendar import c
from itertools import count
from math import e, log
import os
from re import A
import re
from typing import Any, Optional, Tuple, Union
import torch
import torch.nn as nn
import pywt
from datasets import load_metric
from tqdm import tqdm
from typing import Optional, Union, Any

from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling

from transformers import GPT2LMHeadModel, GPT2Tokenizer, LlamaConfig
from transformers.models.llama.modeling_llama import LlamaDecoderLayer
from test_navigation_prompts import ModelLoader
import pandas as pd
import numpy as np
import torch.optim as optim


# Wavelet Embedding Module
class WaveletEmbedding(nn.Module):
    def __init__(self, input_dim, embed_dim, wavelet_name="db1", mode="symmetric"):
        super(WaveletEmbedding, self).__init__()
        self.wavelet = pywt.Wavelet(wavelet_name)
        self.mode = mode
        self.linear = nn.Linear(input_dim, embed_dim)

    def forward(self, x):
        batch_size, seq_len, input_d = x.size()
        assert (
            input_d == self.linear.in_features
        ), "Input dimension does not match the expected input dimension"

        # Apply wavelet decomposition to each sequence element in the batch
        wavelet_coeffs = []
        for i in range(batch_size):
            coeffs = []
            for j in range(seq_len):
                c = pywt.wavedec(
                    x[i, j].detach().cpu().numpy(), self.wavelet, mode=self.mode
                )
                c_flat = np.concatenate(c)  # Flatten the coefficients
                coeffs.append(torch.tensor(c_flat, dtype=torch.float32).to(x.device))
            wavelet_coeffs.append(torch.stack(coeffs))

        # Stack wavelet coefficients and reshape
        wavelet_coeffs = torch.stack(wavelet_coeffs)

        # Ensure the final shape matches the input dimension
        wavelet_coeffs = wavelet_coeffs.view(batch_size, seq_len, -1)

        # If the flattened wavelet coefficients do not match input_d, pad/truncate them
        if wavelet_coeffs.size(-1) != input_d:
            if wavelet_coeffs.size(-1) > input_d:
                wavelet_coeffs = wavelet_coeffs[:, :, :input_d]
            else:
                padding = input_d - wavelet_coeffs.size(-1)
                wavelet_coeffs = torch.nn.functional.pad(wavelet_coeffs, (0, padding))

        return self.linear(wavelet_coeffs)


class TransformerLayerWithWaveletEmbeddings(nn.Module):
    def __init__(
        self,
        original_layer,
        config: LlamaConfig,
        layer_idx: int,
        wavelet_name="db1",
        mode="symmetric",
    ):
        super(TransformerLayerWithWaveletEmbeddings, self).__init__(config, layer_idx)
        self.hidden_size = original_layer.hidden_size

        self.self_attn = original_layer.self_attn

        self.mlp = original_layer.mlp
        self.input_layernorm = original_layer.input_layernorm
        self.post_attention_layernorm = original_layer.post_attention_layernorm
        self.wavelet_embedding = WaveletEmbedding(
            config.hidden_size, config.hidden_size, wavelet_name, mode
        )
        self.current_wav_embed = None
        self.current_regular_embed = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Tuple[
        torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]
    ]:
        residual = hidden_states

        # Apply wavelet embedding
        hidden_states = self.wavelet_embedding(hidden_states)

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        self.current_wav_embed = self.wavelet_embedding(hidden_states)

        self.current_regular_embed = outputs[0]
        return outputs


class WaveletFeatureAugmentation(nn.Module):
    def __init__(self, input_dim, wavelet_name="db1", mode="symmetric"):
        super(WaveletFeatureAugmentation, self).__init__()
        self.wavelet = pywt.Wavelet(wavelet_name)
        self.mode = mode
        self.linear = nn.Linear(input_dim, input_dim)

    def forward(self, x):
        coeffs = pywt.wavedec(x.detach().cpu().numpy(), self.wavelet, mode=self.mode)
        wavelet_features = torch.tensor(coeffs[0]).to(x.device)
        augmented_features = self.linear(wavelet_features)
        return torch.cat((x, augmented_features), dim=-1)


class TransformerLayerWithWaveletAugmentation(nn.Module):
    def __init__(
        self, input_dim, d_model, nhead, d_ff, wavelet_name="db1", mode="symmetric"
    ):
        super(TransformerLayerWithWaveletAugmentation, self).__init__()
        self.wavelet_augmentation = WaveletFeatureAugmentation(
            input_dim, wavelet_name, mode
        )
        self.self_attn = nn.MultiheadAttention(d_model * 2, nhead)
        self.feedforward = nn.Sequential(
            nn.Linear(d_model * 2, d_ff), nn.ReLU(), nn.Linear(d_ff, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model * 2)
        self.norm2 = nn.LayerNorm(d_model * 2)
        self.dropout = nn.Dropout(0.1)

    def forward(
        self,
        x,
        src_mask=None,
        attention_mask: Optional[Any] = None,
        position_ids: Optional[Any] = None,
        past_key_values: Union[Any, None] = None,
        inputs_embeds: Optional[Any] = None,
        use_cache: Optional[Any] = None,
        output_attentions: Optional[Any] = None,
        output_hidden_states: Optional[Any] = None,
        return_dict: Optional[Any] = None,
        cache_position: Optional[Any] = None,
        **kwargs,
    ):
        augmented_features = self.wavelet_augmentation(x)
        attn_output, _ = self.self_attn(
            augmented_features,
            augmented_features,
            augmented_features,
            attn_mask=src_mask,
        )
        x = augmented_features + self.dropout(attn_output)
        x = self.norm1(x)
        ff_output = self.feedforward(x)
        x = x + self.dropout(ff_output)
        x = self.norm2(x)
        return x


class WaveletActivation(nn.Module):
    def __init__(self, wavelet_name="db1", mode="symmetric"):
        super(WaveletActivation, self).__init__()
        self.wavelet = pywt.Wavelet(wavelet_name)
        self.mode = mode

    def forward(self, x):
        coeffs = pywt.wavedec(x.detach().cpu().numpy(), self.wavelet, mode=self.mode)
        return torch.tensor(coeffs[0]).to(x.device)


class HiddenStateWaveletLayer(nn.Module):
    def __init__(self, d_model, wavelet_name="db1", mode="symmetric"):
        super(HiddenStateWaveletLayer, self).__init__()
        self.wavelet_activation = WaveletActivation(wavelet_name, mode)
        self.mode = mode
        self.linear = nn.Linear(d_model, d_model)

    def forward(self, hidden_states):
        transformed_hidden = self.wavelet_activation(hidden_states)
        return self.linear(transformed_hidden)


class TransformerLayerWithWaveletHiddenStates(nn.Module):
    def __init__(self, d_model, nhead, d_ff, wavelet_name="db1", mode="symmetric"):
        super(TransformerLayerWithWaveletHiddenStates, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead)
        self.hidden_wavelet = HiddenStateWaveletLayer(d_model, wavelet_name, mode)
        self.feedforward = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.ReLU(), nn.Linear(d_ff, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(0.1)

    def forward(
        self,
        x,
        src_mask=None,
        attention_mask: Optional[Any] = None,
        position_ids: Optional[Any] = None,
        past_key_values: Union[Any, None] = None,
        inputs_embeds: Optional[Any] = None,
        use_cache: Optional[Any] = None,
        output_attentions: Optional[Any] = None,
        output_hidden_states: Optional[Any] = None,
        return_dict: Optional[Any] = None,
        cache_position: Optional[Any] = None,
        **kwargs,
    ):
        attn_output, _ = self.self_attn(x, x, x, attn_mask=src_mask)
        x = x + self.dropout(attn_output)
        x = self.norm1(x)
        x_hidden_wavelet = self.hidden_wavelet(x)
        ff_output = self.feedforward(x_hidden_wavelet)
        x = x + self.dropout(ff_output)
        x = self.norm2(x)
        return x


from transformers import Trainer, TrainingArguments, AutoTokenizer, AutoModelForCausalLM
from datasets import Dataset, load_dataset
from transformers import DataCollatorForLanguageModeling
import random
import json

from sklearn.metrics import accuracy_score, f1_score


def load_synthetic_data(file_path="synthetic_data_eval.json"):
    with open(file_path, "r") as f:
        data = json.load(f)

    dataset = Dataset.from_dict(
        {
            "input": [item["task"] for item in data],
            "output": [item["planned_route"] for item in data],
        }
    )

    return dataset


class WaveletModelTrainer:
    def __init__(self, model_name, layer, num_samples=1000, test_size=0.2):
        # self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # self.model = AutoModelForCausalLM.from_pretrained(model_name)
        # print(f"Model loaded: {self.model.model.name_or_path}")
        self.layer = layer
        self.dataset = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # self.model.to(self.device)
        self.num_samples = num_samples
        self.test_size = test_size
        # self.data_collator = DataCollatorForLanguageModeling(
        #     tokenizer=self.tokenizer,
        #     mlm=False,
        # )
        self.example = f"""You are a video game character searching for treasure in an environment represented by a grid filled with enemies and obstacles. Your job is to plan a step-by-step route to reach the treasure, avoiding the enemies and navigating around obstacles.
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
        self.train_dataset = None
        self.eval_dataset = None
        self.trainer = None
        self.model_loader = ModelLoader(model_name, system_prompt=self.example)
        config = self.model_loader.model.model.config
        print(f"Model config: {config}")
        # print(f"Model layers: {self.model_loader.model.model.model.layers}")
        # print(f"Model layer 0: {self.model_loader.model.model.model.layers[0]}")
        # self.model_loader.model.model.model.layers[0] = self.layer(config, 0).to(
        #     self.device
        # )
        # print(
        #     f"Model layer 0 after replacement: {self.model_loader.model.model.model.layers[0]}"
        # )
        original_layer = self.model_loader.model.model.model.layers[0]
        if isinstance(self.layer, TransformerLayerWithWaveletEmbeddings):

            self.model_loader.model.model.model.layers[0] = self.layer(config, 0).to(
                self.device
            )
            # self.model_loader.model.model.model.layers[0].load_state_dict(
            #     original_layer.state_dict()
            # )
        elif isinstance(self.layer, TransformerLayerWithWaveletAugmentation):
            self.model_loader.model.model.model.layers[0] = self.layer(
                config.hidden_size, config.nhead, config.intermediate_size
            ).to(self.device)
            # self.model_loader.model.model.model.layers[0].load_state_dict(
            #     original_layer.state_dict()
            # )
        elif isinstance(self.layer, TransformerLayerWithWaveletHiddenStates):
            self.model_loader.model.model.model.layers[0] = self.layer(
                config.hidden_size, config.nhead, config.intermediate_size
            ).to(self.device)
            # self.model_loader.model.model.model.layers[0].load_state_dict(
            #     original_layer.state_dict()
            # )
        if type(layer) != LlamaDecoderLayer:
            print(
                f"Model layer 0 after replacement with {layer}: {self.model_loader.model.model.model.layers[0]}"
            )

    def freeze_pretrained_params(self):
        for param in self.model_loader.model.model.parameters():
            param.requires_grad = False
        for param in self.model_loader.model.model.model.layers[0].parameters():
            param.requires_grad = True

    def generate_synthetic_data(
        self, num_samples=1000, file_path="synthetic_data.json"
    ):
        data = []
        environments = ["forest", "desert", "mountains", "swamp"]
        enemy_types = ["goblin", "ogre", "troll", "dragon"]
        obstacle_types = ["boulder", "trap", "river", "cliff"]

        for _ in range(num_samples):
            environment = random.choice(environments)
            # Create a grid environment of random size between 5x5 and 30x30
            length = np.random.randint(5, 15)
            width = np.random.randint(5, 15)
            grid = np.zeros((length, width))
            # Randomly place the character, treasure, enemies, and obstacles
            character_pos = (np.random.randint(0, length), np.random.randint(0, width))
            treasure_pos = (np.random.randint(0, length), np.random.randint(0, width))
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

            task = f"You are in a {environment} environment with grid size ({length}, {width}). Your character is at position {character_pos}. The treasure is at position {treasure_pos}. There are enemies at positions {enemies}. There are obstacles at positions {obstacles}. Plan a step-by-step route to reach the treasure, avoiding the enemies and obstacles.\n"

            # Example environment description
            environment_description = f"Environment: {environment}. Character at {character_pos}. Treasure at {treasure_pos}. Enemies at {enemies}. Obstacles at {obstacles}.\n "

            planned_route = "Move east to avoid the goblin. Turn north to bypass the boulder. Navigate around the ogre to the west and continue north to the treasure."

            data_point = {
                "environment": environment,
                "character_position": character_pos,
                "treasure_position": treasure_pos,
                "enemies": enemies,
                "obstacles": obstacles,
                "task": task,
                "planned_route": planned_route,
                "example": self.example,
                "environment_description": environment_description,
                "input": f"{self.example}\n{task}",
            }
            data.append(data_point)

        with open(file_path, "w") as f:
            json.dump(data, f, indent=4)

    def convert_to_columnar(self, data):
        columnar_data = {key: [] for key in data[0].keys()}
        for item in data:
            for key, value in item.items():
                columnar_data[key].append(value)
        return columnar_data

    def load_synthetic_data(self, file_path="synthetic_data.json"):
        if not os.path.exists(file_path):
            self.generate_synthetic_data(self.num_samples)

        with open(file_path, "r") as f:
            data = json.load(f)
            # Convert data to columnar format
            data = self.convert_to_columnar(data)
            # Convert data to Hugging Face Dataset format
            dataset = Dataset.from_dict(
                {
                    "input": data["input"],
                    "environment": data["environment"],
                    "task": data["task"],
                }
            )

        return dataset

    def prepare_datasets(self):
        self.dataset = self.load_synthetic_data()

        # dataset = dataset.train_test_split(test_size=self.test_size)
        # self.train_dataset = dataset["train"]
        # self.eval_dataset = dataset["test"]

    def train(self, dataset, num_epochs=5, batch_size=4, learning_rate=1e-4):
        self.freeze_pretrained_params()
        if dataset is None:
            dataset = self.dataset
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=True
        )
        optimizer = optim.Adam(
            filter(
                lambda p: p.requires_grad,
                self.model_loader.model.model.model.parameters(),
            ),
            lr=learning_rate,
        )

        for epoch in range(num_epochs):
            total_loss = 0
            samples = []
            pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1}")

            for batch in pbar:
                results = []
                # inputs = self.tokenizer(
                #     batch["input"],
                #     return_tensors="pt",
                #     padding=True,
                #     truncation=True,
                # ).to(self.device)
                inputs = batch["task"]
                # print(f"Tokenized inputs shape: {inputs.input_ids.shape}")

                optimizer.zero_grad()
                # Generate predictions
                predictions = []
                # Convert token IDs to embeddings
                # inputs_embeds = self.model.model.embed_tokens(inputs.input_ids)
                # print(f"Inputs embeddings shape: {inputs_embeds.shape}")

                # outputs = self.model.generate(
                #     inputs.input_ids,
                #     attention_mask=inputs.attention_mask,
                #     max_new_tokens=50,
                # )
                outputs = self.model_loader.batch_single_response(inputs)

                print(f"\n\nOutputs: {outputs}\n\n")

                losses = []
                # outputs is a list of tuples
                for output, input in outputs:
                    # extract character position, treasure position, enemies, and obstacles from the first line of the input. And example is: "You are in a desert environment. Your character is at position (2, 7). The treasure is at position (8, 7). There are enemies at positions [('ogre', (2, 3)), ('troll', (4, 9)), ('troll', (3, 0))]. There are obstacles at positions [('cliff', (2, 9)), ('river', (5, 10)), ('river', (8, 6))]. Plan a step-by-step route to reach the treasure, avoiding the enemies and obstacles.\n",
                    loss = None

                    character_pos = tuple(
                        map(
                            int,
                            input.split("Your character is at position (")[1]
                            .split(")")[0]
                            .split(","),
                        )
                    )
                    treasure_pos = tuple(
                        map(
                            int,
                            input.split("The treasure is at position (")[1]
                            .split(")")[0]
                            .split(","),
                        )
                    )
                    # Extract enemies and obstacles from the environment description
                    # Enemies are in the format [('enemy_type', (x, y)), ('enemy_type', (x, y)), ...]

                    # Extract enemies and obstacles from the environment description
                    # Enemies are in the format [('enemy_type', (x, y)), ('enemy_type', (x, y)), ...]
                    enemies_string = input.split("There are enemies at positions ")[
                        1
                    ].split(". There")[0]
                    enemy_info = re.findall(
                        r"\('(.+?)', \((\d+), (\d+)\)\)", enemies_string
                    )
                    enemies = [(name, (int(x), int(y))) for name, x, y in enemy_info]
                    print(f"Enemies: {enemies}\n")

                    # Obstacles are in the format [('obstacle_type', (x, y)), ('obstacle_type', (x, y)), ...]
                    obstacles_string = input.split("There are obstacles at positions ")[
                        1
                    ].split(". Plan")[0]
                    obstacle_info = re.findall(
                        r"\('(.+?)', \((\d+), (\d+)\)\)", obstacles_string
                    )
                    obstacles = [
                        (name, (int(x), int(y))) for name, x, y in obstacle_info
                    ]
                    print(f"Obstacles: {obstacles}\n")

                    print(
                        f"Character position: {character_pos}, Treasure position: {treasure_pos}, Enemies: {enemies}, Obstacles: {obstacles}\n\n"
                    )

                    # print(f"Output : {output}")
                    # print(f"Output type: {type(output)}")
                    # prediction = self.tokenizer.decode(output, skip_special_tokens=True)
                    prediction = output
                    print(f"Prediction: {output}\n\n")
                    predictions.append(
                        {
                            "prediction": prediction,
                            "input": input,
                            "character_position": character_pos,
                            "treasure_position": treasure_pos,
                            "enemies": enemies,
                            "obstacles": obstacles,
                        }
                    )

                    # Calculate rewards

                    # Extract grid size from the input
                    grid_size = (
                        int(input.split("grid size (")[1].split(")")[0].split(",")[0]),
                        int(input.split("grid size (")[1].split(")")[0].split(",")[1]),
                    )
                    prediction = prediction
                    env = {
                        "character_pos": character_pos,
                        "treasure_pos": treasure_pos,
                        "enemies": enemies,
                        "obstacles": obstacles,
                        "grid_size": grid_size,
                    }
                    input = input
                    print(f"Environment: {env}")
                    parsed_route = parse_route(
                        prediction,
                        env["character_pos"],
                        env["enemies"],
                        env["obstacles"],
                    )
                    optimal = optimal_steps(env)
                    validity = evaluate_route_validity(parsed_route, env)
                    steps_taken, _, efficiency = evaluate_route_efficiency(
                        parsed_route, optimal
                    )
                    reward = 0
                    reward += efficiency if validity else -1
                    reward = torch.tensor(
                        reward, dtype=torch.float32, requires_grad=True
                    ).to(self.device)
                    if self.layer == TransformerLayerWithWaveletEmbeddings:
                        loss = torch.nn.functional.mse_loss(
                            self.model_loader.model.model.model.layers[
                                0
                            ].current_wav_embed,
                            self.model_loader.model.model.model.layers[
                                0
                            ].current_regular_embed,
                        )
                    else:
                        # Use the efficiency to calculate the loss
                        loss = torch.nn.functional.mse_loss(
                            reward,
                            torch.tensor(
                                1.0, dtype=torch.float32, requires_grad=True
                            ).to(self.device),
                        )
                    if loss.requires_grad == False:
                        loss.requires_grad = True
                    print(f"Steps: {steps_taken}")
                    print(f"Reward: {reward}")
                    print(f"Loss: {loss}")
                    print(f"Optimal steps: {optimal}")
                    print(f"Parsed route: {parsed_route}")
                    print(f"Validity: {validity}")
                    print(f"Efficiency: {efficiency}")
                    print(f"Loss type: {type(loss)}")
                    print(f"Loss: {loss}")
                    losses.append(loss)
                    results.append(
                        {
                            "input": input,
                            "prediction": prediction,
                            "parsed_route": parsed_route,
                            "validity": validity,
                            "efficiency": efficiency,
                            "reward": reward,
                            "loss": loss,
                        }
                    )
                    loss.backward()
                    optimizer.step()
                    pbar.update(1)

                loss = torch.stack(losses).mean()

                pbar.set_postfix({"Batch Loss": loss.item()})
                total_loss += loss.item()
                # Log loss
                print(f"Epoch {epoch + 1}, Batch Loss: {loss.item()}")
                # Save a sample of the data from the best score and the worst score from the results
                best_score = max(results, key=lambda x: x["reward"])
                worst_score = min(results, key=lambda x: x["reward"])
                samples.append({"best": best_score, "worst": worst_score})

            avg_loss = total_loss / len(dataloader)
            print(f"Epoch {epoch + 1} Average Loss: {avg_loss:.4f}")
            # Print the best and worst samples from the epoch
            best_score = max(samples, key=lambda x: x["best"]["reward"])
            worst_score = min(samples, key=lambda x: x["worst"]["reward"])
            print(f"Best Sample of epoch {epoch + 1}: {best_score['best']}")
            print(f"Worst Sample of epoch {epoch + 1}: {worst_score['worst']}")

    def generate(self, prompt, max_length=50):
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        outputs = self.model.generate(**inputs, max_length=max_length)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def save_wavelet_model(self, path):
        # Save the entire model's state_dict
        torch.save(self.model.state_dict(), path)
        print(f"Model saved to {path}")

    def load_wavelet_model(self, path):
        # Load the state_dict into the model
        self.model.load_state_dict(torch.load(path))
        self.model.to(self.device)
        print(f"Model loaded from {path}")


def compute_accuracy_f1(predictions, references):
    correct_predictions = 0
    total_predictions = len(predictions)

    for pred, ref in zip(predictions, references):
        if pred.strip() == ref.strip():
            correct_predictions += 1

    accuracy = correct_predictions / total_predictions
    f1 = f1_score(
        [ref.strip() for ref in references],
        [pred.strip() for pred in predictions],
        average="macro",
    )

    return {"accuracy": accuracy, "f1": f1}


def compute_perplexity(model, dataset):
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir="./results",
        per_device_eval_batch_size=8,
        logging_dir="./logs",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        eval_dataset=dataset,
    )

    eval_results = trainer.evaluate()
    return eval_results["eval_loss"]


def compute_bleu_rouge(predictions, references):
    predictions = [pred.strip() for pred in predictions]
    references = [[ref.strip()] for ref in references]

    bleu_result = bleu.compute(predictions=predictions, references=references)
    rouge_result = rouge.compute(predictions=predictions, references=references)

    return {"bleu": bleu_result["bleu"], "rouge": rouge_result["rouge1"].mid.fmeasure}


# Generate predictions for each model
def generate_predictions(model, dataset, tokenizer):
    inputs = [item["input"] for item in dataset]
    predictions = []

    for input_text in inputs:
        prompt = f"Environment: {input_text}\nWhat is the best route for the character to reach the treasure?"

        input_ids = tokenizer(prompt, return_tensors="pt").input_ids
        output = model.generate(input_ids, max_length=50)
        predicted_text = tokenizer.decode(output[0], skip_special_tokens=True)
        predictions.append(predicted_text)
        print(f"Input: {input_text}")
        print(f"Predicted: {predicted_text}")

    return predictions


def parse_route(response, character_pos, enemies, obstacles):
    # Define possible actions and their effects on position with synonyms
    actions = {
        "move east": (1, 0),
        "move west": (-1, 0),
        "move north": (0, 1),
        "move south": (0, -1),
        "go east": (1, 0),
        "go west": (-1, 0),
        "go north": (0, 1),
        "go south": (0, -1),
        "head east": (1, 0),
        "head west": (-1, 0),
        "head north": (0, 1),
        "head south": (0, -1),
        "proceed east": (1, 0),
        "proceed west": (-1, 0),
        "proceed north": (0, 1),
        "proceed south": (0, -1),
    }

    # Function to parse directions relative to the character's position
    def infer_position(direction, reference_pos):
        if direction == "north":
            return (reference_pos[0], reference_pos[1] - 1)
        elif direction == "south":
            return (reference_pos[0], reference_pos[1] + 1)
        elif direction == "east":
            return (reference_pos[0] + 1, reference_pos[1])
        elif direction == "west":
            return (reference_pos[0] - 1, reference_pos[1])
        else:
            return reference_pos

    # Create dictionaries for quick lookup
    enemy_dict = {enemy[0]: enemy[1] for enemy in enemies}
    obstacle_dict = {obstacle[0]: obstacle[1] for obstacle in obstacles}

    # Clean and split the response into individual actions, accounting for numbered steps
    steps = response.lower().replace(".", "").replace(",", "").split()
    parsed_route = [character_pos]

    current_position = character_pos

    i = 0
    while i < len(steps):
        if steps[i].isdigit():
            i += 1
            continue

        action = ""
        if steps[i] in ["move", "go", "head", "proceed"]:
            if i + 1 < len(steps):
                action = steps[i] + " " + steps[i + 1]
        elif (
            steps[i] in ["avoid", "dodge", "circumvent"] and steps[i + 1] in enemy_dict
        ):
            if i + 2 < len(steps):
                direction = steps[i + 3] if i + 3 < len(steps) else ""
                reference_type = steps[i + 1]
                reference_pos = enemy_dict[reference_type]
                inferred_pos = infer_position(direction, reference_pos)
                parsed_route.append(inferred_pos)
                current_position = inferred_pos
                i += 4
                continue
        elif (
            steps[i] in ["navigate", "go", "move"]
            and steps[i + 1] == "around"
            and steps[i + 2] in obstacle_dict
        ):
            if i + 3 < len(steps):
                direction = steps[i + 4] if i + 4 < len(steps) else ""
                reference_type = steps[i + 2]
                reference_pos = obstacle_dict[reference_type]
                inferred_pos = infer_position(direction, reference_pos)
                parsed_route.append(inferred_pos)
                current_position = inferred_pos
                i += 5
                continue

        if action in actions:
            dx, dy = actions[action]
            current_position = (current_position[0] + dx, current_position[1] + dy)
            parsed_route.append(current_position)
            i += len(action.split())
        else:
            i += 1

    return parsed_route


def optimal_steps(env):
    # Calculate the optimal number of steps based on the environment, character, and treasure positions as well as obstacles and enemies
    character_pos = env["character_pos"]
    treasure_pos = env["treasure_pos"]
    obstacles = env["obstacles"]
    enemies = env["enemies"]

    # Create a grid environment
    grid = np.zeros(env["grid_size"])
    grid[character_pos] = 1
    # grid[treasure_pos] = 1
    for obstacle in obstacles:
        # print(obstacle)
        # print(type(obstacle))
        # print(obstacle[1])
        # print(type(obstacle[1]))
        grid[obstacle[1]] = 1

    for enemy in enemies:
        grid[enemy[1]] = 1

    def is_valid_jump(grid, visited, current, direction):
        new_i, new_j = current[0] + 2 * direction[0], current[1] + 2 * direction[1]
        mid_i, mid_j = current[0] + direction[0], current[1] + direction[1]
        if 0 <= new_i < grid.shape[0] and 0 <= new_j < grid.shape[1]:
            if (
                (grid[mid_i, mid_j] in [1, 2])
                and (grid[new_i, new_j] == 0)
                and not visited[new_i, new_j]
            ):
                return True
        return False

    def dijkstra(grid, start, end):
        rows, cols = grid.shape
        dist = np.full((rows, cols), np.inf)
        visited = np.zeros((rows, cols))
        dist[start] = 0
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]

        while True:
            min_dist = np.inf
            current = None
            for i in range(rows):
                for j in range(cols):
                    if visited[i, j] == 0 and dist[i, j] < min_dist:
                        min_dist = dist[i, j]
                        current = (i, j)

            if current is None or current == end:
                break

            visited[current] = 1

            for direction in directions:
                new_i, new_j = current[0] + direction[0], current[1] + direction[1]
                if 0 <= new_i < rows and 0 <= new_j < cols:
                    new_dist = dist[current] + 1
                    if (
                        grid[new_i, new_j] == 0
                        and not visited[new_i, new_j]
                        and new_dist < dist[new_i, new_j]
                    ):
                        dist[new_i, new_j] = new_dist
                    elif is_valid_jump(grid, visited, current, direction):
                        jump_i, jump_j = (
                            current[0] + 2 * direction[0],
                            current[1] + 2 * direction[1],
                        )
                        if dist[current] + 1 < dist[jump_i, jump_j]:
                            dist[jump_i, jump_j] = dist[current] + 1

        return dist[end]

    return dijkstra(grid, character_pos, treasure_pos)


def evaluate_route_validity(route, environment):
    character_pos = environment["character_pos"]
    for step in route:
        if step in environment["obstacles"] or step in environment["enemies"]:
            return False
        character_pos = step
    return character_pos == environment["treasure_pos"]


def evaluate_route_efficiency(route, optimal_steps):
    steps_taken = len(route)
    return (
        steps_taken,
        optimal_steps,
        max(min(1 - log(steps_taken + e) / log(optimal_steps + e), 1), 0),
    )


def evaluate_response_coherence(response, environment):
    parsed_route = parse_route(
        response,
        environment["character_position"],
        environment["enemies"],
        environment["obstacles"],
    )
    optimal = optimal_steps(environment)
    validity = evaluate_route_validity(parsed_route, environment)
    efficiency = evaluate_route_efficiency(parsed_route, optimal)
    return validity and efficiency


# def evaluate_model_output(model, environment, optimal_route_length):
#     """
#     Evaluates the model's output based on multiple criteria.
#     """
#     response = model.generate(environment)
#     route = parse_route_from_response(response)

#     # Evaluate validity
#     validity = evaluate_route_validity(route, environment)

#     # Evaluate efficiency
#     steps_taken, optimal_steps, efficiency = evaluate_route_efficiency(
#         route, optimal_route_length
#     )

#     # Evaluate coherence
#     coherence = evaluate_response_coherence(response, environment)

#     return {
#         "validity": validity,
#         "steps_taken": steps_taken,
#         "optimal_steps": optimal_steps,
#         "efficiency": efficiency,
#         "coherence": coherence,
#     }


if __name__ == "__main__":

    # Usage:
    # training_args = TrainingArguments(
    #     output_dir="./results",
    #     num_train_epochs=30,
    #     per_device_train_batch_size=8,
    #     per_device_eval_batch_size=16,
    #     warmup_steps=500,
    #     weight_decay=0.01,
    #     logging_dir="./logs",
    #     logging_steps=10,
    # )

    layer_classes = [
        TransformerLayerWithWaveletEmbeddings,
        TransformerLayerWithWaveletAugmentation,
        TransformerLayerWithWaveletHiddenStates,
    ]
    wavelet_embedding_model = None
    wavelet_augmentation_model = None
    wavelet_hidden_states_model = None
    combined_wavelet_model = None
    for layer_class in layer_classes:
        trainer = WaveletModelTrainer(
            "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            layer_class,
        )
        # check if synthetic_data.json exists
        if not os.path.exists("synthetic_data.json"):
            trainer.generate_synthetic_data(2000)
        if not os.path.exists("synthetic_data_eval.json"):
            trainer.generate_synthetic_data(1000, "synthetic_data_eval.json")
        trainer.prepare_datasets()
        trainer.train(None, num_epochs=5, batch_size=4, learning_rate=1e-4)
        trainer.save_wavelet_model(f"./saved_{layer_class.__name__}")

    # Load the tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")

    # Load the base TinyLlama model
    base_model = GPT2LMHeadModel.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")

    # Load the wavelet-augmented models
    wavelet_embedding_model = AutoModelForCausalLM.from_pretrained(
        "./saved_TransformerLayerWithWaveletEmbeddings"
    )
    wavelet_augmentation_model = AutoModelForCausalLM.from_pretrained(
        "./saved_TransformerLayerWithWaveletAugmentation"
    )
    wavelet_hidden_states_model = AutoModelForCausalLM.from_pretrained(
        "./saved_TransformerLayerWithWaveletHiddenStates"
    )
    # combined_wavelet_model = GPT2LMHeadModel.from_pretrained(
    #     "path_to_combined_wavelet_model"
    # )

    dataset = load_synthetic_data()
    test_dataset = dataset.train_test_split(test_size=0.2)["test"]

    print(f"Train Dataset Size: {len(dataset)}")
    print(f"Eval Dataset Size: {len(test_dataset)}")

    print("Train Dataset Example:", dataset[0])
    print("Eval Dataset Example:", test_dataset[0])

    base_perplexity = compute_perplexity()
    wavelet_embedding_perplexity = compute_perplexity(
        wavelet_embedding_model, test_dataset
    )
    wavelet_augmentation_perplexity = compute_perplexity(
        wavelet_augmentation_model, test_dataset
    )
    wavelet_hidden_states_perplexity = compute_perplexity(
        wavelet_hidden_states_model, test_dataset
    )
    # combined_wavelet_perplexity = compute_perplexity(
    #     combined_wavelet_model, test_dataset
    # )

    print(f"Base Model Perplexity: {base_perplexity}")
    print(f"Wavelet Embedding Model Perplexity: {wavelet_embedding_perplexity}")
    print(f"Wavelet Augmentation Model Perplexity: {wavelet_augmentation_perplexity}")
    print(f"Wavelet Hidden States Model Perplexity: {wavelet_hidden_states_perplexity}")
    # print(f"Combined Wavelet Model Perplexity: {combined_wavelet_perplexity}")

    bleu = load_metric("bleu")
    rouge = load_metric("rouge")

    base_predictions = generate_predictions(base_model, test_dataset, tokenizer)
    wavelet_embedding_predictions = generate_predictions(
        wavelet_embedding_model, test_dataset
    )
    wavelet_augmentation_predictions = generate_predictions(
        wavelet_augmentation_model, test_dataset, tokenizer
    )
    wavelet_hidden_states_predictions = generate_predictions(
        wavelet_hidden_states_model, test_dataset, tokenizer
    )
    # combined_wavelet_predictions = generate_predictions(
    #     combined_wavelet_model, test_dataset
    # )

    references = [item["output"] for item in test_dataset]

    base_bleu_rouge = compute_bleu_rouge(base_predictions, references)
    wavelet_embedding_bleu_rouge = compute_bleu_rouge(
        wavelet_embedding_predictions, references
    )
    wavelet_augmentation_bleu_rouge = compute_bleu_rouge(
        wavelet_augmentation_predictions, references
    )
    wavelet_hidden_states_bleu_rouge = compute_bleu_rouge(
        wavelet_hidden_states_predictions, references
    )
    # combined_wavelet_bleu_rouge = compute_bleu_rouge(
    #     combined_wavelet_predictions, references
    # )

    print(
        f"Base Model BLEU: {base_bleu_rouge['bleu']}, ROUGE: {base_bleu_rouge['rouge']}"
    )
    print(
        f"Wavelet Embedding Model BLEU: {wavelet_embedding_bleu_rouge['bleu']}, ROUGE: {wavelet_embedding_bleu_rouge['rouge']}"
    )
    print(
        f"Wavelet Augmentation Model BLEU: {wavelet_augmentation_bleu_rouge['bleu']}, ROUGE: {wavelet_augmentation_bleu_rouge['rouge']}"
    )
    print(
        f"Wavelet Hidden States Model BLEU: {wavelet_hidden_states_bleu_rouge['bleu']}, ROUGE: {wavelet_hidden_states_bleu_rouge['rouge']}"
    )
    # print(
    #     f"Combined Wavelet Model BLEU: {combined_wavelet_bleu_rouge['bleu']}, ROUGE: {combined_wavelet_bleu_rouge['rouge']}"
    # )

    base_accuracy_f1 = compute_accuracy_f1(base_predictions, references)
    wavelet_embedding_accuracy_f1 = compute_accuracy_f1(
        wavelet_embedding_predictions, references
    )
    wavelet_augmentation_accuracy_f1 = compute_accuracy_f1(
        wavelet_augmentation_predictions, references
    )
    wavelet_hidden_states_accuracy_f1 = compute_accuracy_f1(
        wavelet_hidden_states_predictions, references
    )
    # combined_wavelet_accuracy_f1 = compute_accuracy_f1(
    #     combined_wavelet_predictions, references
    # )

    print(
        f"Base Model Accuracy: {base_accuracy_f1['accuracy']}, F1: {base_accuracy_f1['f1']}"
    )
    print(
        f"Wavelet Embedding Model Accuracy: {wavelet_embedding_accuracy_f1['accuracy']}, F1: {wavelet_embedding_accuracy_f1['f1']}"
    )
    print(
        f"Wavelet Augmentation Model Accuracy: {wavelet_augmentation_accuracy_f1['accuracy']}, F1: {wavelet_augmentation_accuracy_f1['f1']}"
    )
    print(
        f"Wavelet Hidden States Model Accuracy: {wavelet_hidden_states_accuracy_f1['accuracy']}, F1: {wavelet_hidden_states_accuracy_f1['f1']}"
    )
    # print(
    #     f"Combined Wavelet Model Accuracy: {combined_wavelet_accuracy_f1['accuracy']}, F1: {combined_wavelet_accuracy_f1['f1']}"
    # )

    results = {
        "Model": [
            "Base Model",
            "Wavelet Embedding",
            "Wavelet Augmentation",
            "Wavelet Hidden States",
        ],
        "Perplexity": [
            base_perplexity,
            wavelet_embedding_perplexity,
            wavelet_augmentation_perplexity,
            wavelet_hidden_states_perplexity,
        ],
        "BLEU": [
            base_bleu_rouge["bleu"],
            wavelet_embedding_bleu_rouge["bleu"],
            wavelet_augmentation_bleu_rouge["bleu"],
            wavelet_hidden_states_bleu_rouge["bleu"],
        ],
        "ROUGE": [
            base_bleu_rouge["rouge"],
            wavelet_embedding_bleu_rouge["rouge"],
            wavelet_augmentation_bleu_rouge["rouge"],
            wavelet_hidden_states_bleu_rouge["rouge"],
        ],
        "Accuracy": [
            base_accuracy_f1["accuracy"],
            wavelet_embedding_accuracy_f1["accuracy"],
            wavelet_augmentation_accuracy_f1["accuracy"],
            wavelet_hidden_states_accuracy_f1["accuracy"],
        ],
        "F1": [
            base_accuracy_f1["f1"],
            wavelet_embedding_accuracy_f1["f1"],
            wavelet_augmentation_accuracy_f1["f1"],
            wavelet_hidden_states_accuracy_f1["f1"],
        ],
    }

    df_results = pd.DataFrame(results)
    print(df_results)
    # Save the results to a CSV file
    df_results.to_csv("wavelet_results.csv", index=False)
