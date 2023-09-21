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

## Basic NN building blocks

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
    
## Custom Reward Models
    
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

def init(module, weight_init, bias_init, gain=1):
	weight_init(module.weight.data, gain=gain)
	if module.bias is not None:
		bias_init(module.bias.data)
	return module

def init_(m, gain=0.01, activate=False):
	if activate:
		gain = nn.init.calculate_gain('relu')
	return init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), gain=gain)


class TransformerRewardPredictor_v2(nn.Module):
    def __init__(self, obs_input_dim, num_heads) -> None:
        super(TransformerRewardPredictor, self).__init__()
        self.num_heads = num_heads
        
        self.embedding_net = nn.Sequential(
            nn.LayerNorm(obs_input_dim)
            init_(nn.Linear(obs_input_dim+action_dim, 64), activate=True),
            nn.GELU(),
        )
        self.key_net = init_(nn.Linear(64, 64))
        self.query_net = nn.Sequential(
            nn.LayerNorm(64),
            init_(nn.Linear(64, 64))
        )
        self.value_net = init_(nn.Linear(64, 64))

        self.value_norm = nn.LayerNorm(64)
        self.value_linear = nn.Sequential(
            init_(nn.Linear(64, 1024), activate=True),
            nn.GELU(),
            init_(nn.Linear(1024, 64)),
        )
        self.value_linear_norm = nn.LayerNorm(64)
        self.mlp = nn.Sequential(
            init_(nn.Linear(64, 64), activate=True),
            nn.GELU(),
            init_(nn.Linear(64, 1)),
        )
        # self.attention_weights = None
        self.d_k = 64//self.num_heads
        

    def forward(self, state):
        # obtaining keys queries and values
        state_embeddings = self.embedding_net(state) # Batch, Trajectory Length, Embedding Dim
        batch, trajectory_len, emd = state_embeddings.shape
        assert emd//self.num_heads == 0
        query = self.query_net(torch.sum(state_embeddings, dim=1, keepdim=True)).reshape(batch, 1, self.num_heads, emd//self.num_heads).permute(0, 3, 1, 2) # Batch, Num Heads, 1, Embedding Dim
        key = self.key_net(state_embeddings).reshape(batch, trajectory_len, self.num_heads, emd//self.num_heads).permute(0, 3, 1, 2) # Batch, Num Heads, Trajectory Len, Embedding Dim
        value = self.value_net(state_embeddings).reshape(batch, trajectory_len, self.num_heads, emd//self.num_heads).permute(0, 3, 1, 2) # Batch, Num Heads, Trajectory Len, Embedding Dim

        # self attention layer
        attention_weights = F.softmax((query @ key.transpose(-1, -2) / sqrt(self.d_k)), dim=-1) # Batch, Num Heads, 1, Trajectory Len
        attention_values =  (attention_weights @ value).squeeze(-2).view(batch, -1) # Batch, Embedding Dim

        attention_values_ = self.value_norm(attention_values + torch.mean(state_embeddings, dim=1))
        attention_values = self.value_linear(attention_values_)
        attention_values = self.value_linear_norm(attention_values+attention_values_)

        # MLP for predicting reward
        episodic_reward = self.mlp(attention_values)
        
        return episodic_reward, attention_weights
        
    
class TransformerRewardPredictorNew(nn.Module):
    def __init__(self, e_dim, d_k, mlp_hidden_layers) -> None:
        super(TransformerRewardPredictorNew, self).__init__()
        self.d_k = d_k
        self.e_dim = e_dim

        # Key, Query and Value Nets
        self.key_net = MLP(e_dim, [d_k])
        self.query_net = MLP(e_dim, [d_k])
        self.value_net = MLP(e_dim, [d_k])
        
        # Feed forward network
        self.ff = nn.Sequential(
            nn.Linear(self.d_k, self.d_k * 4),
            nn.ReLU(),
            nn.Linear(self.d_k * 4, self.d_k)
        )

        # layer norms
        self.ln1 = nn.LayerNorm(self.d_k)
        self.ln2 = nn.LayerNorm(self.d_k)

        # Final MLP to predict global reward

        
        self.mlp = MLP(d_k, mlp_hidden_layers)
        self.attention_weights = None

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

## Custom Pytorch Datasets

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

class RewardDatasetList(Dataset):
    def __init__(self, state_actions_list:list, global_reward_list:list) -> None:
        assert len(state_actions_list) == len(global_reward_list)
        self.state_actions = torch.tensor(state_actions_list)
        self.global_rewards = torch.tensor(global_reward_list)

    def __len__(self):
        return self.state_actions.shape[0]
    
    def __getitem__(self, index):
        return self.state_actions[index], self.global_rewards[index]
    
## Custom Replay Buffer class

class ReplayBuffer:
    def __init__(self, n_agents) -> None:
        self.n_agents = n_agents
        self.agent_buffers = [{} for _ in range(self.n_agents)]
        self.global_buffer = {}
        self.agent_allowed_keys = set([
                "state", 
                "state_combined", 
                "action",
                "reward",
                "next_state",
                "next_state_combined",
                "done",
            ])
        self.global_keys = set([
            "state_action"
        ])

    def add_values(self, agent_id=None, **kwargs):
        for k, v in kwargs.items():
            assert(k in self.agent_allowed_keys or k in self.global_keys), f"Unknown key \"{k}\" found"
            # converting value to torch tensor
            if type(v) == list:
                v = torch.from_numpy(np.array(v))
            elif type(v) == np.ndarray:
                v = torch.from_numpy(v)
            elif type(v) == torch.Tensor:
                pass
            else:
                raise TypeError(f"Inappropriate type {type(v)} found")

            # adding to the corresponding dictionary
            if k in self.agent_allowed_keys:
                assert(agent_id is not None), "Please provide agent id while adding to the buffer"
                agent_dict = self.agent_buffers[agent_id]
                if k not in agent_dict:
                    agent_dict[k] = v
                else:
                    agent_dict[k] = torch.cat((agent_dict[k], v), dim=0)
            else:
                if k not in self.global_buffer:
                    self.global_buffer[k] = v
                else:
                    self.global_buffer[k] = torch.cat((self.global_buffer[k], v), dim=0)

    def update_values(self, agent_id, **kwargs):
        agent_dict = self.agent_buffers[agent_id]
        for k, v in kwargs.items():
            assert(k in self.agent_allowed_keys), f"Unknown key \"{k}\" found"
            # converting value to torch tensor
            if type(v) == list:
                v = torch.from_numpy(np.array(v))
            elif type(v) == np.ndarray:
                v = torch.from_numpy(v)
            elif type(v) == torch.Tensor:
                pass
            else:
                raise TypeError(f"Inappropriate type {type(v)} found")
            
            assert k in agent_dict, f"Unknown Key {k}"
            assert agent_dict[k].shape == v.shape, f"Shapes do not match: {agent_dict[k].shape} and {v.shape}"

            agent_dict[k] = v
    
    def clear(self):
        del self.agent_buffers
        del self.global_buffer
        self.agent_buffers = [{} for _ in range(self.n_agents)]
        self.global_buffer = {}

    def get(self, key:str, agent_index=None):
        assert (key in self.agent_allowed_keys) or (key in self.global_keys), f"Unknown key {key} found"
        if key in self.agent_allowed_keys:
            assert agent_index is not None, "Please provide agent_index"
            return self.agent_buffers[agent_index][key]
        else:
            return self.global_buffer[key]
    
    @torch.no_grad()
    def predict_and_update_rewards(self, reward_model:nn.Module, device):
        reward_model.eval()
        reward_model.to(device)
        if (isinstance(reward_model, AgentSelfAttentionReward_AREL) or isinstance(reward_model, AgentTransformerReward_AREL)): 
            state_actions = self.global_buffer["state_action"]
            state_actions = state_actions.to(device)
            agent_rewards, _ = reward_model(state_actions)
            assert(len(agent_rewards.shape) == 2)
            agent_rewards = torch.permute(agent_rewards, (1, 0))
            agent_rewards_list = list(agent_rewards)
            for ind, reward_tensor in enumerate(agent_rewards_list):
                self.update_values(ind, reward=reward_tensor.detach().cpu())
            return agent_rewards
        elif isinstance(reward_model, TransformerRewardPredictor):
            pass

    def __getitem__(self, key:str):
        assert (key in self.agent_allowed_keys or key in self.global_keys), f"Unknown key {key} given"
        if key in self.agent_allowed_keys: return [self.agent_buffers[ind][key] for ind in range(self.n_agents)]
        else: return self.global_buffer[key]

## AREL models

class AgentSelfAttentionReward_AREL(nn.Module):
    def __init__(self, e_dim, d_k, mlp_hidden_layers) -> None:
        super(AgentSelfAttentionReward_AREL, self).__init__()
        self.e_dim = e_dim
        self.d_k = d_k
        
        # Key Query and Value nets
        self.key_net = MLP(self.e_dim, [self.d_k])
        self.query_net = MLP(self.e_dim, [self.d_k])
        self.value_net = MLP(self.e_dim, [self.d_k])

        # Final MLP to predict agent rewards
        assert(mlp_hidden_layers[-1] == 1)
        self.mlp = MLP(self.d_k, mlp_hidden_layers)

    def forward(self, x):
        self.query = self.query_net(x)
        self.key = self.key_net(x)
        self.value = self.value_net(x)

        # scaled dot product attention
        self.attention_weights = F.softmax((self.query @ self.key.permute(0, 2, 1) / sqrt(self.d_k)), dim=-1)
        attention_values = self.attention_weights @ self.value
        reward_values = self.mlp(attention_values).squeeze(-1)

        return reward_values, self.attention_weights


class AgentTransformerReward_AREL(nn.Module):
    def __init__(self, e_dim, d_k, mlp_hidden_layers) -> None:
        super(AgentTransformerReward_AREL, self).__init__()
        self.e_dim = e_dim
        self.d_k = d_k
        
        # Key Query and Value nets
        self.key_net = MLP(self.e_dim, [self.d_k])
        self.query_net = MLP(self.e_dim, [self.d_k])
        self.value_net = MLP(self.e_dim, [self.d_k])

        # Feed forward network
        self.ff = nn.Sequential(
            nn.Linear(self.d_k, self.d_k * 4),
            nn.ReLU(),
            nn.Linear(self.d_k * 4, self.d_k)
        )

        # layer norms
        self.ln1 = nn.LayerNorm(self.d_k)
        self.ln2 = nn.LayerNorm(self.d_k)

        # Final MLP to predict agent rewards
        assert(mlp_hidden_layers[-1] == 1)
        self.mlp = MLP(self.d_k, mlp_hidden_layers)

    def forward(self, x):
        self.query = self.query_net(x)
        self.key = self.key_net(x)
        self.value = self.value_net(x)

        # scaled dot product attention
        self.attention_weights = F.softmax((self.query @ self.key.permute(0, 2, 1) / sqrt(self.d_k)), dim=-1)
        attention_values = self.attention_weights @ self.value

        # add + norm
        x = self.ln1(attention_values + x)

        # pass through feed forward network
        feedforward = self.ff(x)

        # add + norm
        x = self.ln2(feedforward + x)

        # pass through MLP
        reward_values = self.mlp(x).squeeze(-1)

        return reward_values, self.attention_weights
        
if __name__ == "__main__":
    d = {
        "e_dim" : 10,
        "d_k" : 10,
        "mlp_hidden_layers": [10, 10, 1]        
    }

    a = AgentTransformerReward_AREL(**d)
    print(a.e_dim)
    
