from audioop import avg
from functools import lru_cache
from hmac import new
import math
import random
from turtle import st

import numpy as np
from sympy import comp
from torch import normal, rand
import torch
import torch.nn as nn
import torch.optim as optim
import xformers as xf
import logging
from xformers.components import MultiHeadDispatch
from torch.utils.data import DataLoader, TensorDataset

logging.basicConfig(level=logging.DEBUG)
torch.device("cuda" if torch.cuda.is_available() else "cpu")
xf.torch.device("cuda" if torch.cuda.is_available() else "cpu")


def cached_sigmoid_motive_scale_approx_optimized(
    motive_value, max_value=10.0, steepness=0.01
):
    motive_value = max(0, motive_value)
    mid_point = max_value / 2
    x = steepness * (motive_value - mid_point)
    sigmoid_value = 0.5 * (x / (1 + abs(x)) + 1)
    return min(sigmoid_value * max_value, max_value)


def tanh_scaling(x, data_max, data_min, data_avg, data_std):
    a = data_std
    centered_value = x - data_avg
    scaled_value = math.tanh(centered_value / a)
    return scaled_value * (data_max - data_min) / 2.0 + (data_max + data_min) / 2.0


data_original = [104, 42, 292, 146]
max_in_data = 292
min_in_data = 42
avg_in_data = np.mean(data_original)
manual_avg = (104 + 42 + 292 + 146) / 4

simple_calc_a = cached_sigmoid_motive_scale_approx_optimized(manual_avg, 100.0)
logging.info(f"Simple calculation a: {simple_calc_a}")


logging.info(f"Manual average: {manual_avg}")

std_in_data = np.std(data_original)

scaled_data = [
    cached_sigmoid_motive_scale_approx_optimized(data, 100.0) for data in data_original
]
normalized_data = [
    (data - min_in_data) / (max_in_data - min_in_data) for data in data_original
]

avg_of_scaled = np.mean(scaled_data)
logging.info(f"Average of scaled data: {avg_of_scaled}")


logging.info(f"Max in data: {max_in_data}")
logging.info(f"Min in data: {min_in_data}")
logging.info(f"Avg in data: {avg_in_data}")
logging.info(f"Std in data: {std_in_data}")
logging.info(f"Scaled data: {scaled_data}")
logging.info(f"Normalized data: {normalized_data}")


# Using the above data, we can experiment with different ways of calculating the final variable var_*

# Test case 1
# Here, we calculate the final variable var_simple_sig using the sum of the data points in data_original
var_simple_sum = sum(data_original)
logging.info(f"Sum of data points: {var_simple_sum}")
var_simple_scaled = tanh_scaling(
    var_simple_sum, max_in_data, min_in_data, avg_in_data, std_in_data
)
logging.info(f"Scaled sum: {var_simple_scaled}")
var_simple_sig = cached_sigmoid_motive_scale_approx_optimized(var_simple_scaled, 100.0)
logging.info(f"Sigmoided scaled sum: {var_simple_sig}")

# Test case 2
# Here, we calculate the final variable var_a using the sum of the data points in data_original and a random gaussian value with std equal to the std of the data points in data_original
# and mean equal to the output of the function cached_sigmoid_motive_scale_approx applied individually to each data point in data_original with a max value of 25.0, added together, and then passed through the function cached_sigmoid_motive_scale_approx with a max value of 100.0
var_a = random.gauss(
    cached_sigmoid_motive_scale_approx_optimized(
        (
            (
                cached_sigmoid_motive_scale_approx_optimized(104, 25.0)
                + cached_sigmoid_motive_scale_approx_optimized(42, 25.0)
                + cached_sigmoid_motive_scale_approx_optimized(292, 25.0)
            )
            + cached_sigmoid_motive_scale_approx_optimized(146, 25.0)
            if 146 > 0.0
            else 1.0
        )
        + tanh_scaling(
            ((random.randrange(-4, 4) + random.randrange(-4, 4)) * 10.0),
            10.0,
            -10.0,
            0.0,
            np.std([random.randrange(-4, 4) for _ in range(100)]),
        ),
        100.0,
    ),
    std_in_data,
)
logging.info(f"Random gaussian value: {var_a}")

# Test case 3
# For this test case, we calculate the final variable var_nogauss using the sum of the data points in data_original and the output of the function cached_sigmoid_motive_scale_approx applied individually to each data point in data_original with a max value of 25.0, added together, and then passed through the function cached_sigmoid_motive_scale_approx with a max value of 100.0
var_nogauss = cached_sigmoid_motive_scale_approx_optimized(
    (
        (
            cached_sigmoid_motive_scale_approx_optimized(104, 25.0)
            + cached_sigmoid_motive_scale_approx_optimized(42, 25.0)
            + cached_sigmoid_motive_scale_approx_optimized(292, 25.0)
        )
        + cached_sigmoid_motive_scale_approx_optimized(146, 25.0)
        if 146 > 0.0
        else 1.0
    ),
    +tanh_scaling(
        ((random.randrange(-4, 4) + random.randrange(-4, 4)) * 10.0),
        10.0,
        -10.0,
        0.0,
        np.std([random.randrange(-4, 4) for _ in range(100)]),
    ),
    100.0,
)
logging.info(f"Final variable without gaussian value: {var_nogauss}")

# Test case 4
# For this test case, we calculate the final variable var_one_pass_sig using the sum of the data points in data_original and the output of the function cached_sigmoid_motive_scale_approx applied to the sum of the data points in data_original and the output of the function tanh_scaling applied to a random value between -4 and 4, multiplied by 10.0, with a max value of 10.0, min value of -10.0, avg value of 0.0, and std value equal to the std of 100 random values between -4 and 4
var_one_pass_sig = cached_sigmoid_motive_scale_approx_optimized(
    (104 + 42 + 292 + 146)
    + tanh_scaling(
        ((random.randrange(-4, 4) + random.randrange(-4, 4)) * 10.0),
        10.0,
        -10.0,
        0.0,
        np.std([random.randrange(-4, 4) for _ in range(100)]),
    ),
    100.0,
)
logging.info(f"Final variable with one pass sigmoid: {var_one_pass_sig}")
# Test case 5
# This one is a little different, as we are calculating the gaussian value first and then passing it through the function cached_sigmoid_motive_scale_approx
var_gauss_first = random.gauss(avg_in_data, std_in_data)
var_gauss_first = cached_sigmoid_motive_scale_approx_optimized(var_gauss_first, 100.0)

# Test case 6
# This time we calculate the gauss second, adding it to the sigmoided sum of the data points in data_original and then sigging the result

var_gauss_second_double_sig = cached_sigmoid_motive_scale_approx_optimized(
    cached_sigmoid_motive_scale_approx_optimized(
        (104 + 42 + 292 + 146)
        + tanh_scaling(
            ((random.randrange(-4, 4) + random.randrange(-4, 4)) * 10.0),
            10.0,
            -10.0,
            0.0,
            np.std([random.randrange(-4, 4) for _ in range(100)]),
        ),
        100.0,
    )
    + cached_sigmoid_motive_scale_approx_optimized(
        random.gauss(avg_in_data, std_in_data), 100.0
    ),
    100.0,
)
logging.info(f"Final variable with double sigmoid: {var_gauss_second_double_sig}")
# Test case 7
# This time we calculate the gauss second, adding it to the sigmoided sum of the data points in data_original, then average the two sigmoided values

var_gauss_second_onesig_avgd = cached_sigmoid_motive_scale_approx_optimized(
    (104 + 42 + 292 + 146)
    + tanh_scaling(
        ((random.randrange(-4, 4) + random.randrange(-4, 4)) * 10.0),
        10.0,
        -10.0,
        0.0,
        np.std([random.randrange(-4, 4) for _ in range(100)]),
    ),
    100.0,
) + cached_sigmoid_motive_scale_approx_optimized(
    random.gauss(avg_in_data, std_in_data), 100.0
)
var_gauss_second_onesig_avgd = var_gauss_second_onesig_avgd / 2
logging.info(
    f"Final variable with one sigmoid averaged: {var_gauss_second_onesig_avgd}"
)

# Test case 8
# This one we simply calculate the median of the data points in data_original and pass it through the function cached_sigmoid_motive_scale_approx, then use that value to calculate the gaussian value
median_data = cached_sigmoid_motive_scale_approx_optimized(
    (104 + 42 + 292 + 146)
    + tanh_scaling(
        ((random.randrange(-4, 4) + random.randrange(-4, 4)) * 10.0),
        10.0,
        -10.0,
        0.0,
        np.std([random.randrange(-4, 4) for _ in range(100)]),
    ),
    100.0,
)
indiv_scaled_data = [
    cached_sigmoid_motive_scale_approx_optimized(data, 100.0) for data in data_original
]
logging.info(f"Individual scaled data: {indiv_scaled_data}")
scaled_std = np.std(indiv_scaled_data)
logging.info(f"Scaled std: {scaled_std}")
var_use_median_for_gauss = random.gauss(median_data, scaled_std)
var_use_median_for_gauss = cached_sigmoid_motive_scale_approx_optimized(
    var_use_median_for_gauss, 100.0
)
logging.info(f"Final variable using median for gaussian: {var_use_median_for_gauss}")


# Test cases 9 and 10
# These two test cases are similar to the previous one, but instead of using the median, we use the average of the data points in data_original to calculate the gaussian value.
# In test case 9, we use the average of the data points in data_original to calculate the gaussian value, but each value was passed through the function cached_sigmoid_motive_scale_approx before calculating the average
# In test case 10, we use the average of the data points in data_original to calculate the gaussian value directly without passing each value through the function cached_sigmoid_motive_scale_approx first
avg_data_indiv = np.mean(indiv_scaled_data)
var_use_avg_for_gauss_indiv = random.gauss(avg_data_indiv, scaled_std)
logging.info(f"Average of individual scaled data: {avg_data_indiv}")
var_use_avg_for_gauss_unscaled = cached_sigmoid_motive_scale_approx_optimized(
    random.gauss(avg_in_data, std_in_data), 100.0
)
logging.info(
    f"Final variable using average for gaussian (individual scaled): {var_use_avg_for_gauss_indiv}"
)
logging.info(
    f"Final variable using average for gaussian (unscaled): {var_use_avg_for_gauss_unscaled}"
)

# Other test cases experimenting with different ways of calculating the final variable
# These ones get more creative, and some of them are not very practical, but they are interesting to explore. Others are either variations of the previous test cases or completely new approaches.
# Brainstorm new ideas step-by-step, ensuring they are unique and not repetitive.

# Test case 11 (new approach)
# Different approach. This time, instead of using the sum of the data points in data_original, we use the product of the data points in data_original and pass it through the function cached_sigmoid_motive_scale_approx with a max value of 100.0
var_product = cached_sigmoid_motive_scale_approx_optimized(104 * 42 * 292 * 146, 100.0)
logging.info(f"Product of data points: {104 * 42 * 292 * 146}")
# Test case 12 (variation of test case 11)
# Similar to test case 11, but we add a random gaussian value with std equal to the std of the data points in data_original and mean equal to the output of the function cached_sigmoid_motive_scale_approx applied individually to each data point in data_original with a max value of 25.0, added together, and then passed through the function cached_sigmoid_motive_scale_approx with a max value of 100.0
var_product_gauss = cached_sigmoid_motive_scale_approx_optimized(
    (104 * 42 * 292 * 146)
    + tanh_scaling(
        (
            cached_sigmoid_motive_scale_approx_optimized(104, 25.0)
            + cached_sigmoid_motive_scale_approx_optimized(42, 25.0)
            + cached_sigmoid_motive_scale_approx_optimized(292, 25.0)
        )
        + cached_sigmoid_motive_scale_approx_optimized(146, 25.0),
        100.0,
        0.0,
        cached_sigmoid_motive_scale_approx_optimized(avg_in_data, 100.0),
        std_in_data,
    )
    / 2,
    100.0,
)
var_product_gauss = random.gauss(var_product_gauss, std_in_data)
logging.info(f"Final variable with product and gaussian: {var_product_gauss}")

# Test case 13 (variation of test case 12)
# Similar to test case 12, but we add the output of the function cached_sigmoid_motive_scale_approx applied to the sum of the data points in data_original and the output of the function tanh_scaling applied to a random value between -4 and 4, multiplied by 10.0, with a max value of 10.0, min value of -10.0, avg value of 0.0, and std value equal to the std of 100 random values between -4 and 4
var_product_gauss_tanh = cached_sigmoid_motive_scale_approx_optimized(
    var_simple_sum,
    100.0
    + tanh_scaling(
        ((random.randrange(-4, 4) + random.randrange(-4, 4)) * 10.0),
        10.0,
        -10.0,
        0.0,
        np.std([random.randrange(-4, 4) for _ in range(100)]),
    ),
    100.0,
)
logging.info(
    f"Final variable with product, gaussian, and tanh: {var_product_gauss_tanh}"
)
# Test case 14 (variation of test case 13)
# Here's how this one is different and pretty creative and unlike the rest.
# This one is based aqw@RE EWRQ3W
# ON THE idea of some random stuff, including a technique thats more similiar to the previous test cases, but with a twist
# Here, we calculate the final variable var_product_gauss_tanh_gauss using the sum of the data points in data_original and the output of the function cached_sigmoid_motive_scale_approx applied individually to each data point in data_original with a max value of 25.0, added together, and then passed through the function cached_sigmoid_motive_scale_approx with a max value of 100.0
# We then add the output of the function tanh_scaling applied to a random value between -4 and 4, multiplied by 10.0, with a max value of 10.0, min value of -10.0, avg value of 0.0, and std value equal to the std of 100 random values between -4 and 4 to the sum
# Finally, we add a random gaussian value with std equal to the std of the data points in data_original and mean equal to the output of the function cached_sigmoid_motive_scale_approx applied individually to each data point in data_original with a max value of 25.0, added together, and then passed through the function cached_sigmoid_motive_scale_approx with a max value of 100.0
# This is then used to calculate the final variable var_product_gauss_tanh_gauss
var_product_gauss_tanh_gauss = (
    cached_sigmoid_motive_scale_approx_optimized(
        (104 + 42 + 292 + 146)
        + cached_sigmoid_motive_scale_approx_optimized(104, 25.0)
        + cached_sigmoid_motive_scale_approx_optimized(42, 25.0)
        + cached_sigmoid_motive_scale_approx_optimized(292, 25.0)
        + cached_sigmoid_motive_scale_approx_optimized(146, 25.0),
        100.0,
    )
    + tanh_scaling(
        ((random.randrange(-4, 4) + random.randrange(-4, 4)) * 10.0),
        10.0,
        -10.0,
        0.0,
        np.std([random.randrange(-4, 4) for _ in range(100)]),
    )
    + random.gauss(
        cached_sigmoid_motive_scale_approx_optimized(104, 25.0)
        + cached_sigmoid_motive_scale_approx_optimized(42, 25.0)
        + cached_sigmoid_motive_scale_approx_optimized(292, 25.0)
        + cached_sigmoid_motive_scale_approx_optimized(146, 25.0),
        std_in_data,
    )
)
var_product_gauss_tanh_gauss = cached_sigmoid_motive_scale_approx_optimized(
    random.gauss(var_product_gauss_tanh_gauss, std_in_data), 100.0
)
logging.info(
    f"Final variable with product, tanh, gaussian, and tanh: {var_product_gauss_tanh_gauss}"
)

compiled_var_a = {
    "var_simple_sum": var_simple_sum,
    "var_simple_scaled": var_simple_scaled,
    "var_simple_sig": var_simple_sig,
    "var_a": var_a,
    "var_nogauss": var_nogauss,
    "var_one_pass_sig": var_one_pass_sig,
    "var_gauss_first": var_gauss_first,
    "var_gauss_second_double_sig": var_gauss_second_double_sig,
    "var_gauss_second_onesig_avgd": var_gauss_second_onesig_avgd,
    "var_use_median_for_gauss": var_use_median_for_gauss,
    "var_product": var_product,
    "var_product_gauss": var_product_gauss,
    "var_product_gauss_tanh": var_product_gauss_tanh,
    "var_product_gauss_tanh_gauss": var_product_gauss_tanh_gauss,
    "var_use_avg_for_gauss_indiv": var_use_avg_for_gauss_indiv,
    "var_use_avg_for_gauss_unscaled": var_use_avg_for_gauss_unscaled,
}

for var, value in compiled_var_a.items():
    logging.info(f"Compiled variable {var}: {value}")
compiled_var = [val for _, val in compiled_var_a.items() if val > 0.0 and val < 100.0]

# generate random data using avg_of_scaled, simple_calc_a, and var_gauss_second_double_sig:
torch.manual_seed(0)
np.random.seed(0)
data = np.random.normal(avg_of_scaled, np.std([avg_of_scaled, simple_calc_a]), 10000)
data = np.append(data, np.random.normal(var_gauss_second_double_sig, scaled_std, 10000))

# calculate the mean and std of the data
mean_data = np.mean(data)
std_data = np.std(data)
logging.info(f"Mean of the data: {mean_data}")
logging.info(f"Std of the data: {std_data}")


# Show random selections of data
logging.info(f"Random selections of data: {data[:10]}")

# remove outliers
compiled_var = [
    val
    for val in compiled_var
    if val
    > np.mean([avg_of_scaled, simple_calc_a, var_gauss_second_double_sig])
    - 2 * np.std([avg_of_scaled, simple_calc_a, var_gauss_second_double_sig])
    and val
    < np.mean([avg_of_scaled, simple_calc_a, var_gauss_second_double_sig])
    + 2 * np.std([avg_of_scaled, simple_calc_a, var_gauss_second_double_sig])
]

logging.info(f"Final Compiled variables: \n")
for var in compiled_var:
    logging.info(f"{var}")

# Test case 15 (complete variation)
# This one is a complete variation of the previous test cases, using a completely different approach
# its more akin to RNNs and LSTMs


# Define the SimpleLSTMModel
class SimpleLSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, device="cuda"):
        super(SimpleLSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = xf.torch.nn.LSTM(input_size, hidden_size, device=device)
        self.fc = xf.torch.nn.Linear(hidden_size, 1, device=device)
        # self.sigmoid = xf.torch.nn.Sigmoid()
        self.device = device

    def forward(self, x):
        h_0 = torch.zeros(1, x.size(1), self.hidden_size).to(self.device)
        c_0 = torch.zeros(1, x.size(1), self.hidden_size).to(self.device)

        # LSTM layer
        lstm_out, _ = self.lstm(x, (h_0, c_0))
        lstm_out = lstm_out[:, -1, :]  # Get the last output

        # Fully connected layer
        output = self.fc(lstm_out)

        # Apply sigmoid activation
        # output = self.sigmoid(output)
        return output


# Input data
data_original = [104, 42, 292, 146]

# Normalize input data
data_normalized = [
    (x - min(data_original)) / (max(data_original) - min(data_original))
    for x in data_original
]

# Convert data to tensor and reshape
input_tensor = (
    torch.tensor(data_normalized, dtype=torch.float32).view(1, -1, 1).to("cuda")
)


data_tensor = [
    torch.tensor(dt, dtype=torch.float32).view(1, -1).to("cuda") for dt in data
]

logging.info(f"Data tensor: {data_tensor[12]} of shape {data_tensor[12].shape}")

# Define model
input_size = 1
hidden_size = 10
model = SimpleLSTMModel(input_size, hidden_size).to("cuda")

# Forward pass
initial_output = model(input_tensor.to("cuda"))
logging.info(f"Initial output of LSTM: {initial_output}")
# Hyperparameters
learning_rate = 0.001
num_epochs = 10000

# Dummy training data and targets (in practice, use your real data)
train_data = [
    torch.tensor(data_normalized, dtype=torch.float32).view(1, -1, 1).to("cuda")
]
train_targets = torch.tensor(data, dtype=torch.float32).view(1, -1, 1).to("cuda")

# Loss function and optimizer
criterion = nn.MSELoss().to("cuda")
optimizer = xf.torch.optim.AdamW(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(train_data[0])
    loss = criterion(outputs, data_tensor[epoch].to("cuda"))
    loss.backward()
    optimizer.step()

    # if (epoch + 1) % 10 == 0:
    #     # logging.info(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item()}")
    #     logging.info(f"Output after epoch {epoch + 1}: {model(train_data[0])}")

# Final output after training
final_output = model(train_data[0])
logging.info(f"Final output after training: {final_output}")


class FeedForwardLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(FeedForwardLayer, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = xf.torch.nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


# Define the Transformer-based Model
class TransformerScaler(nn.Module):
    def __init__(self, input_dim, embed_dim, num_heads, ff_dim):
        super(TransformerScaler, self).__init__()
        self.embedding = nn.Linear(input_dim, embed_dim)
        self.attention = nn.modules.MultiheadAttention(embed_dim, num_heads)
        self.feed_forward = FeedForwardLayer(embed_dim, ff_dim, embed_dim)
        self.output_layer = nn.Linear(embed_dim, 1)
        # self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.embedding(x)
        x = self.attention(x.unsqueeze(0), x.unsqueeze(0), x.unsqueeze(0))[0]
        x = self.feed_forward(x)
        x = self.output_layer(x.mean(dim=1))
        return x


# Initialize the Model, Loss Function, and Optimizer
input_dim = 4  # Length of the input list
embed_dim = 16
num_heads = 2
ff_dim = 32

model = TransformerScaler(input_dim, embed_dim, num_heads, ff_dim).to("cuda")
criterion = nn.MSELoss().to("cuda")
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Example Data
data_original = (
    torch.tensor([104, 42, 292, 146], dtype=torch.float32).view(1, -1).to("cuda")
)

# Forward Pass
output = model(data_original)
logging.info(f"Initial Output TransformerScaler: {output}")
print(output)  # Output should be between 0 and 1

# Example Training Loop

for epoch in range(10000):
    model.train()
    optimizer.zero_grad()
    output = model(data_original)
    loss = criterion(output, data_tensor[epoch].to("cuda"))
    loss.backward()
    optimizer.step()

    # if (epoch + 1) % 10 == 0:
    #     logging.info(f"Epoch [{epoch + 1}/100], Loss: {loss.item()}")

# Final Output
output = model(data_original)

logging.info(f"Final Output TransformerScaler: {output}")


class ConcreteFeedforward(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ConcreteFeedforward, self).__init__()
        self.layer1 = nn.Linear(input_dim, output_dim)
        self.relu = xf.torch.nn.ReLU()
        self.layer2 = nn.Linear(output_dim, output_dim)

    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        return x


class TransformerModel(nn.Module):
    def __init__(self, input_dim, embed_dim, num_heads):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Linear(input_dim, embed_dim)
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, device="cuda")
        self.memory = nn.GRU(embed_dim, embed_dim, batch_first=True)
        self.feedforward = ConcreteFeedforward(embed_dim, embed_dim)
        self.output_layer = nn.Linear(embed_dim, 1)

    def forward(self, x):
        x = self.embedding(x)  # Shape: (batch_size, seq_len, embed_dim)
        attn_output, _ = self.attention(x, x, x)  # Self-attention
        memory_output, _ = self.memory(attn_output)  # Memory via GRU

        final_output = self.feedforward(memory_output)  # Feedforward layer
        # final_output = torch.mean(memory_output, dim=1)  # Aggregate along seq_len

        # final_output = torch.sigmoid(
        self.output_layer(final_output)
        # )  # Output in range [0, 1]
        return final_output


# Example usage
data_original = torch.tensor([104, 42, 292, 146], dtype=torch.float32).unsqueeze(
    0
)  # Shape: (1, 4)
data_original = data_original.to("cuda")
model = TransformerModel(input_dim=4, embed_dim=16, num_heads=4).to("cuda")
output = model(data_original)

logging.info(f"Initial Output TransformerModel: {output}")

logging.info(f"Data Tensor: {data_tensor[42]}")
logging.info(f"Data Tensor Dims: {data_tensor[42].size()}")
tensor_of_mean_values = [train_targets.mean(dim=1).squeeze().to("cuda")]
logging.info(f"Tensor of Mean Values: {tensor_of_mean_values}")
tensor_of_mean_values = torch.tensor(tensor_of_mean_values, dtype=torch.float32)
logging.info(
    f"Tensor of Mean Values: {tensor_of_mean_values} of type {tensor_of_mean_values.dtype} and size {tensor_of_mean_values.size()}"
)
# Create a dataset and DataLoader
dataset = TensorDataset(data_original, tensor_of_mean_values.unsqueeze(0).to("cuda"))
data_loader = DataLoader(dataset, batch_size=4, shuffle=True)

# Define model, loss function, and optimizer
criterion = nn.MSELoss().to("cuda")
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 10000
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for inputs, labels in data_loader:
        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    # logging.info(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss}")
# Evaluation (Optional)
model.eval()
with torch.no_grad():
    test_input = torch.tensor([[104, 42, 292, 146]], dtype=torch.float32).to("cuda")
    test_output = model(test_input)
    logging.info(f"Test Output TransformerModel: {test_output}")

final_output = model(data_original)
logging.info(f"Final Output TransformerModel: {final_output}")


class ScalerModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_heads):
        super(ScalerModel, self).__init__()
        self.embedding = nn.Linear(
            input_dim, hidden_dim, device="cuda" if torch.cuda.is_available() else "cpu"
        )
        self.attention = xf.torch.nn.MultiheadAttention(
            hidden_dim,
            num_heads,
            device="cuda" if xf.torch.cuda.is_available() else "cpu",
        )
        self.memory = nn.LSTM(
            hidden_dim,
            hidden_dim,
            batch_first=True,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
        self.output = nn.Linear(
            hidden_dim, 1, device="cuda" if torch.cuda.is_available() else "cpu"
        )
        # self.sigmoid = nn.Sigmoid()

    def forward(self, x, hidden_state=None):
        x = self.embedding(x)
        attn_output, _ = self.attention(x.unsqueeze(1), x.unsqueeze(1), x.unsqueeze(1))
        if hidden_state is None:
            hidden_state = (
                torch.zeros(1, x.size(0), x.size(1)).to(
                    "cuda" if torch.cuda.is_available() else "cpu"
                ),
                torch.zeros(1, x.size(0), x.size(1)).to(
                    "cuda" if torch.cuda.is_available() else "cpu"
                ),
            )
        memory_output, hidden_state = self.memory(attn_output, hidden_state)
        output = self.output(memory_output[:, -1, :])
        # output = self.sigmoid(output)
        return output, hidden_state


# Example usage
data_original = torch.tensor([[104, 42, 292, 146]], dtype=torch.float32).to("cuda")
model = ScalerModel(input_dim=4, hidden_dim=16, num_heads=4).to("cuda")
output, hidden_state = model(data_original)
logging.info(f"Initial Output ScalerModel 2: {output}")

train_targets = torch.tensor(data, dtype=torch.float32).view(1, -1, 1).to("cuda")
# Initialize the model, loss function, and optimizer
criterion = nn.MSELoss().to("cuda")
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 10000
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()

    # Forward pass
    output, _ = model(data_original)

    # Compute loss
    loss = criterion(output, data_tensor[epoch].to("cuda"))

    # Backward pass and optimization
    loss.backward()
    optimizer.step()

    # if (epoch + 1) % 100 == 0:
    # logging.info(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item()}")
# Example usage after training
model.eval()
with torch.no_grad():
    output, _ = model(data_original)
    logging.info(f"Final Output ScalerModel 2: {output}")
