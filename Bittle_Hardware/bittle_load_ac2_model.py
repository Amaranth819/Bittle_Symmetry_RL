import torch
import torch.nn as nn
from collections import OrderedDict
import numpy as np

# Option to add safe globals if using weights_only=True
def add_safe_globals():
    torch.serialization.add_safe_globals([np.core.multiarray.scalar])

# Function to remove keys that are not part of the model
def filter_state_dict(state_dict, allowed_keys):
    filtered_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if k in allowed_keys:
            filtered_state_dict[k] = v
    return filtered_state_dict

# Define the model architecture
class CustomLSTM(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(CustomLSTM, self).__init__()
        self.rnn = nn.LSTM(input_size, hidden_size, batch_first=True)

    def forward(self, x, hx=None):
        if hx is None:
            hx = (torch.zeros(1, x.size(0), self.rnn.hidden_size).to(x.device),
                  torch.zeros(1, x.size(0), self.rnn.hidden_size).to(x.device))
        return self.rnn(x, hx)

class A2CNetwork(nn.Module):
    def __init__(self, input_dim=42, hidden_dim=400, rnn_hidden_dim=256, output_dim=9):
        super(A2CNetwork, self).__init__()

        # Actor MLP
        self.actor_mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # Critic MLP
        self.critic_mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # Actor RNN (LSTM wrapped in CustomLSTM)
        self.a_rnn = CustomLSTM(input_size=hidden_dim + input_dim, hidden_size=rnn_hidden_dim)

        # Critic RNN (LSTM wrapped in CustomLSTM)
        self.c_rnn = CustomLSTM(input_size=hidden_dim + input_dim, hidden_size=rnn_hidden_dim)

        # Value (Critic output)
        self.value = nn.Linear(rnn_hidden_dim, 1)  # Single value output for critic

        # Policy Mean (Actor output)
        self.mu = nn.Linear(rnn_hidden_dim, output_dim)  # 9-dimensional output

        # Policy Log Std (Actor output)
        self.sigma = nn.Linear(rnn_hidden_dim, output_dim)  # 9-dimensional output

    def forward(self, x):
        hx_actor = torch.zeros(1, x.size(0), self.a_rnn.rnn.hidden_size).to(x.device)
        cx_actor = torch.zeros(1, x.size(0), self.a_rnn.rnn.hidden_size).to(x.device)
        hx_critic = torch.zeros(1, x.size(0), self.c_rnn.rnn.hidden_size).to(x.device)
        cx_critic = torch.zeros(1, x.size(0), self.c_rnn.rnn.hidden_size).to(x.device)

        # Actor path
        actor_out = self.actor_mlp(x)
        rnn_input_actor = torch.cat([actor_out, x], dim=-1)
        rnn_actor_out, (hx_actor, cx_actor) = self.a_rnn(rnn_input_actor, (hx_actor, cx_actor))
        
        mu = self.mu(rnn_actor_out)
        sigma = self.sigma(rnn_actor_out)

        # Critic path
        critic_out = self.critic_mlp(x)
        rnn_input_critic = torch.cat([critic_out, x], dim=-1)
        rnn_critic_out, (hx_critic, cx_critic) = self.c_rnn(rnn_input_critic, (hx_critic, cx_critic))
        
        value = self.value(rnn_critic_out)

        return mu, sigma, value, (hx_actor, cx_actor), (hx_critic, cx_critic)

class WrappedA2CNetwork(nn.Module):
    def __init__(self):
        super(WrappedA2CNetwork, self).__init__()
        self.a2c_network = A2CNetwork()

    def forward(self, *args, **kwargs):
        return self.a2c_network(*args, **kwargs)

# Function to load the model
def load_a2c_model(checkpoint_path):
    model = WrappedA2CNetwork()

    # Attempt to load the checkpoint with weights_only=True
    try:
        add_safe_globals()
        checkpoint = torch.load(checkpoint_path, weights_only=True)
    except Exception as e:
        print(f"Loading with weights_only=True failed: {e}")
        print("Retrying with weights_only=False...")
        checkpoint = torch.load(checkpoint_path, weights_only=False)

    model_state_dict = checkpoint['model']
    allowed_keys = model.state_dict().keys()
    filtered_state_dict = filter_state_dict(model_state_dict, allowed_keys)

    # Load the filtered state dictionary into the model
    model.load_state_dict(filtered_state_dict)
    model.eval()
    
    return model

# Example usage:
# checkpoint_path = "path_to_checkpoint.pth"
# model = load_a2c_model(checkpoint_path)
