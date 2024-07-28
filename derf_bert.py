import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SymmetricRandomFeatures(nn.Module):
    def __init__(self, d_k, M):
        super(SymmetricRandomFeatures, self).__init__()
        self.d_k = d_k
        self.M = M

        # Initialize random matrix and orthogonalize using eigen-decomposition
        random_matrix = torch.randn(d_k, M)
        eig_values, eig_vectors = torch.linalg.eigh(random_matrix @ random_matrix.T)
        self.random_matrix = nn.Parameter(eig_vectors, requires_grad=False)

        # Initialize A, B, C, D parameters as learnable with more informative priors
        self.A = nn.Parameter(torch.diag(1 / 16 * (1 - 2 * eig_values - torch.sqrt((2 * eig_values + 1)**2 + 8 * eig_values))), requires_grad=True)
        self.B = nn.Parameter(torch.zeros(d_k, d_k), requires_grad=True)
        self.C = nn.Parameter(-0.5 * torch.eye(d_k), requires_grad=True)
        self.D = nn.Parameter(torch.tensor(1.0), requires_grad=True)

    def forward(self, q, k):
        batch_size, seq_length, d_k = q.size()
        M1 = (1 / seq_length) * torch.matmul(q.transpose(-2, -1), q)
        M2 = (1 / seq_length) * torch.matmul(k.transpose(-2, -1), k)
        mu4 = q.mean(dim=1)
        mu5 = k.mean(dim=1)

        matrix = M1 + torch.outer(mu4, mu5) + torch.outer(mu5, mu4) + M2
        Q3, Lambda3, _ = torch.linalg.svd(matrix)
        Lambda3_diag = torch.diag(Lambda3)
        A_diag = 1 / 16 * (1 - 2 * Lambda3_diag - torch.sqrt((2 * Lambda3_diag + 1)**2 + 8 * Lambda3_diag))

        self.A.data = torch.diag_embed(A_diag)
        self.B.data = (torch.eye(self.d_k) - 4 * self.A).sqrt() @ Q3.T
        self.C.data = -0.5 * torch.eye(self.d_k)
        self.D.data = torch.det(torch.eye(self.d_k) - 4 * self.A).pow(1/4)

        omega = torch.matmul(self.random_matrix, torch.randn(self.M, self.d_k).t())
        exp_term_q = torch.exp(torch.matmul(q, self.B.t()) + 0.5 * torch.sum(omega**2, dim=0) + self.D)
        exp_term_k = torch.exp(torch.matmul(k, self.B.t()) + 0.5 * torch.sum(omega**2, dim=0) + self.D)
        return exp_term_q, exp_term_k

class DERFAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, num_random_features=64):
        super(DERFAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_random_features = num_random_features

        self.query_proj = nn.Linear(embed_dim, embed_dim)
        self.key_proj = nn.Linear(embed_dim, embed_dim)
        self.value_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self.random_features_qk = SymmetricRandomFeatures(embed_dim // num_heads, num_random_features)

    def random_features(self, q, k):
        return self.random_features_qk(q, k)

    def forward(self, x, mask=None):
        batch_size, seq_length, embed_dim = x.size()
        assert embed_dim == self.embed_dim

        q = self.query_proj(x).view(batch_size, seq_length, self.num_heads, embed_dim // self.num_heads)
        k = self.key_proj(x).view(batch_size, seq_length, self.num_heads, embed_dim // self.num_heads)
        v = self.value_proj(x).view(batch_size, seq_length, self.num_heads, embed_dim // self.num_heads)

        q = q / math.sqrt(self.embed_dim // self.num_heads)
        k = k / math.sqrt(self.embed_dim // self.num_heads)

        q_features, k_features = self.random_features(q, k)

        q_features = q_features / torch.norm(q_features, p=2, dim=-1, keepdim=True)
        k_features = k_features / torch.norm(k_features, p=2, dim=-1, keepdim=True)

        attention_scores = torch.matmul(q_features, k_features.transpose(-2, -1))
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, -1e9)
        attention_probs = F.softmax(attention_scores, dim=-1)

        context = torch.matmul(attention_probs, v)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_length, embed_dim)
        out = self.out_proj(context)

        return out
from transformers import BertModel, BertConfig, BertTokenizer

import unittest
import torch
import torch.nn as nn
from transformers import BertConfig, BertTokenizer
from transformers.modeling_outputs import BaseModelOutput

class CustomBertSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = DERFAttention(config.hidden_size, config.num_attention_heads)

    def forward(self, hidden_states, attention_mask=None, head_mask=None, output_attentions=False):
        attention_output = self.attention(hidden_states)
        return attention_output, None

class CustomBertLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = CustomBertSelfAttention(config)
        self.intermediate = nn.Linear(config.hidden_size, config.intermediate_size)
        self.output = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, attention_mask=None, head_mask=None):
        self_attention_outputs = self.attention(hidden_states, attention_mask, head_mask)
        attention_output = self_attention_outputs[0]
        hidden_states = self.LayerNorm(attention_output + hidden_states)
        
        intermediate_output = self.intermediate(hidden_states)
        layer_output = self.output(intermediate_output)
        layer_output = self.dropout(layer_output)
        layer_output = self.LayerNorm(layer_output + hidden_states)
        
        return layer_output

class CustomBertEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layer = nn.ModuleList([CustomBertLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(self, hidden_states, attention_mask=None, head_mask=None, output_attentions=False, output_hidden_states=False, return_dict=True):
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        for i, layer_module in enumerate(self.layer):
            hidden_states = layer_module(hidden_states, attention_mask, head_mask)

            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

        return hidden_states

class CustomBertModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.encoder = CustomBertEncoder(config)

    def forward(self, input_ids):
        hidden_states = self.encoder(input_ids)
        return BaseModelOutput(last_hidden_state=hidden_states)

class TestCustomBertModel(unittest.TestCase):
    def setUp(self):
        self.config = BertConfig(
            hidden_size=64,
            num_attention_heads=8,
            num_hidden_layers=2,
            intermediate_size=128,
            hidden_dropout_prob=0.1,
            layer_norm_eps=1e-12,
        )
        self.model = CustomBertModel(self.config)
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    def test_initialization(self):
        self.assertIsInstance(self.model.encoder, CustomBertEncoder)
        for layer in self.model.encoder.layer:
            self.assertIsInstance(layer.attention, CustomBertSelfAttention)
            self.assertIsInstance(layer.intermediate, nn.Linear)
            self.assertIsInstance(layer.output, nn.Linear)
            self.assertIsInstance(layer.LayerNorm, nn.LayerNorm)
            self.assertIsInstance(layer.dropout, nn.Dropout)

    def test_forward_pass_shape(self):
        inputs = self.tokenizer("Hello, my dog is cute", return_tensors="pt")
        input_ids = inputs["input_ids"]
        outputs = self.model(input_ids)
        self.assertEqual(outputs.last_hidden_state.shape, (1, input_ids.shape[1], self.config.hidden_size))

    def test_attention_mechanism(self):
        inputs = self.tokenizer("Hello, my dog is cute", return_tensors="pt")
        input_ids = inputs["input_ids"]
        hidden_states = torch.randn(1, input_ids.shape[1], self.config.hidden_size)
        attention = CustomBertSelfAttention(self.config)
        outputs, _ = attention(hidden_states)
        self.assertEqual(outputs.shape, hidden_states.shape)

    def test_training_step(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        criterion = nn.MSELoss()

        inputs = self.tokenizer("Hello, my dog is cute", return_tensors="pt")
        input_ids = inputs["input_ids"]
        labels = torch.randn(1, input_ids.shape[1], self.config.hidden_size)

        for _ in range(5):  # small number of steps for testing
            optimizer.zero_grad()
            outputs = self.model(input_ids)
            loss = criterion(outputs.last_hidden_state, labels)
            loss.backward()
            optimizer.step()

            self.assertTrue(loss.item() > 0)

    def test_gradient_flow(self):
        inputs = self.tokenizer("Hello, my dog is cute", return_tensors="pt")
        input_ids = inputs["input_ids"]
        outputs = self.model(input_ids)

        outputs.last_hidden_state.mean().backward()

        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.assertIsNotNone(param.grad)
                self.assertTrue(torch.any(param.grad != 0))

if __name__ == '__main__':
    unittest.main()

