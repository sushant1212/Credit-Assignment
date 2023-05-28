import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
import numpy as np
import torch.optim as optim
from math import sqrt
from typing import List
import os
import json
from torch.utils.data import Dataset

class MLP(nn.Module):
    """
    Class for a Multi Layered Perceptron. LeakyReLU activations would be applied between each layer.
    Args:
    input_layer_size (int): The size of the input layer
    hidden_layers (list): A list containing the sizes of the hidden layers
    last_relu (bool): If True, then a LeakyReLU would be applied after the last hidden layer
    """
    def __init__(self, input_layer_size:int, hidden_layers:list, last_relu=False) -> None:
        super().__init__()
        self.layers = []
        self.layers.append(nn.Linear(input_layer_size, hidden_layers[0]))
        self.layers.append(nn.LeakyReLU())
        gain = nn.init.calculate_gain('leaky_relu')
        nn.init.xavier_uniform_(self.layers[-2].weight, gain=gain)
        for i in range(len(hidden_layers)-1):
            if i != (len(hidden_layers)-2):
                self.layers.append(nn.Linear(hidden_layers[i], hidden_layers[i+1]))
                self.layers.append(nn.LeakyReLU())
                nn.init.xavier_uniform_(self.layers[-2].weight, gain=gain)
            elif last_relu:
                self.layers.append(nn.Linear(hidden_layers[i], hidden_layers[i+1]))
                self.layers.append(nn.LeakyReLU())
                nn.init.xavier_uniform_(self.layers[-2].weight, gain=gain)
            else:
                self.layers.append(nn.Linear(hidden_layers[i], hidden_layers[i+1]))
        self.network = nn.Sequential(*self.layers)
    
    def forward(self, x):
        x = self.network(x)
        return x
    
class Actor(nn.Module):
    def __init__(self, obs_dim:int, hidden_layers:List[int], last_layer_dim:int) -> None:
        super(Actor, self).__init__()
        self.obs_dim = obs_dim
        self.hidden_layers = hidden_layers
        self.last_layer_dim = last_layer_dim
        self.net = MLP(self.obs_dim, self.hidden_layers + [self.last_layer_dim])

    def forward(self, x):
        x = self.net(x)
        return x
    
class Critic(nn.Module):
    def __init__(self, input_dim:int, hidden_layers:List[int], last_layer_dim:int) -> None:
        super(Critic, self).__init__()
        self.input_dim = input_dim
        self.hidden_layers = hidden_layers
        self.last_layer_dim = last_layer_dim
        self.net = MLP(self.input_dim, self.hidden_layers + [self.last_layer_dim])

    def forward(self, x):
        x = self.net(x)
        return x
    
class TransformerRewardPredictor(nn.Module):
    def __init__(self, e_dim, d_k, mlp_hidden_layers) -> None:
        super(TransformerRewardPredictor, self).__init__()
        self.key_net = MLP(e_dim, [d_k])
        self.query_net = MLP(e_dim, [d_k])
        self.value_net = MLP(e_dim, [d_k])
        self.mlp = MLP(d_k, mlp_hidden_layers)
        self.attention_weights = None
        self.d_k = d_k

    def forward(self, state_actions):
        # obtaining keys queries and values
        self.query = self.query_net(torch.sum(state_actions, dim=1, keepdim=True))
        self.key = self.key_net(state_actions)
        self.value = self.value_net(state_actions)

        # self attention layer
        self.attention_weights = F.softmax((self.query @ self.key.permute(0, 2, 1) / sqrt(self.d_k)), dim=-1)
        self.attention_values =  (self.attention_weights @ self.value).squeeze(1)

        # MLP for predicting reward
        y_hat = self.mlp(self.attention_values)
        return y_hat, self.attention_weights
    
class MLP_RewardPredictor(nn.Module):
    def __init__(self, input_dim:int, hidden_layers:list) -> None:
        super(MLP_RewardPredictor, self).__init__()
        self.mlp = MLP(input_dim, hidden_layers)
    
    def forward(self, x):
        return self.mlp(x)
    
class RewardDataset(Dataset):
    def __init__(self, json_files:list, json_base_dir:str, episode_length=25) -> None:
        self.json_files = json_files
        self.json_base_dir = json_base_dir
        self.episode_length = episode_length
        assert(os.path.exists(json_base_dir)), f"{json_base_dir} does not exist!"
    
    def __len__(self):
        return len(self.json_files) * self.episode_length
    
    def __getitem__(self, index):
        json_file_index = index // self.episode_length
        data_index = index % self.episode_length
        json_file = self.json_files[json_file_index]
        assert(os.path.exists(os.path.join(self.json_base_dir, json_file))), f"{os.path.join(self.json_base_dir, json_file)} does not exist!"
        with open(os.path.join(self.json_base_dir, json_file), "r+") as f:
            d = json.load(f)
        return torch.tensor(d["state_actions"][data_index]), torch.tensor(d["global_rewards"][data_index]), torch.tensor(d["agent_rewards"][data_index])
