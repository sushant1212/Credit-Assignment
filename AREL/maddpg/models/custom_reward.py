import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from math import sqrt

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


class CustomTimeRewardNet(nn.Module):
    def __init__(
            self, 
            state_dim:int, 
            num_agents:int,
            state_embedding_dim:int, 
            episode_length:int,
            episode_embedding_dim:int,
            mlp1_hidden_layers:list, 
            mlp2_hidden_layers:list, 
            mlp3_hidden_layers:list,
            reward_mlp_hidden_layers:list
        ) -> None:

        super(CustomTimeRewardNet, self).__init__()
        self.state_dim = state_dim
        self.num_agents = num_agents
        self.state_embedding_dim = state_embedding_dim
        self.episode_embedding_dim = episode_embedding_dim
        self.episode_length = episode_length
        self.mlp1_hidden_layers = mlp1_hidden_layers + [self.state_embedding_dim]
        self.mlp1 = MLP(self.state_dim * self.num_agents, self.mlp1_hidden_layers)
        self.mlp2_hidden_layers = mlp2_hidden_layers + [1]
        self.mlp2 = MLP(self.state_embedding_dim * 2, self.mlp2_hidden_layers)
        self.mlp3_hidden_layers = mlp3_hidden_layers + [self.episode_embedding_dim]
        self.mlp3 = MLP(self.state_embedding_dim, self.mlp3_hidden_layers)
        self.reward_mlp_hidden_layers = reward_mlp_hidden_layers + [1]
        self.reward_mlp = MLP(self.episode_embedding_dim, self.reward_mlp_hidden_layers)

    def predict_reward(self, state_tensor, reward_tensor):
        # state_tensor.shape = (batch, self.episode_length, num_agents * state_dim)
        # reward_tensor.shape = (batch, 1)
        _, weights = self(state_tensor)
        predicted_rewards = (weights * reward_tensor)
        return predicted_rewards
    
    def forward(self, x):
        # x.shape = (batch, self.episode_length, num_agents * state_dim)
        assert (x.shape[-1] == (self.state_dim * self.num_agents)), "Unexpected input shape found"
        assert (x.shape[-2] == (self.episode_length)), "Unexpected input shape found"

        embedding_tensor = self.mlp1(x)  # (batch, episode_length, state_embedding_dim)
        embedding_mean = torch.mean(embedding_tensor, dim=-2)  # embedding_mean.shape = (batch, state_embedding_dim)
        embedding_mean_repeat = torch.repeat_interleave(embedding_mean, self.episode_length, dim=0).reshape(-1, self.episode_length, self.state_embedding_dim)
        logits = self.mlp2(torch.cat((embedding_tensor, embedding_mean_repeat), dim=-1)).squeeze(-1)
        weights = F.softmax(logits, dim=-1)  # (batch, self.episode_length)
        assert((len(weights.shape) == 2) and (weights.shape[-1] == self.episode_length))
        episode_embeddings = self.mlp3(embedding_tensor)  # (batch, episode_length, self.episode_embedding_dim)
        assert(episode_embeddings.shape[-1] == self.episode_embedding_dim)

        mean_episode_embedding = (weights.unsqueeze(1) @ episode_embeddings).squeeze(1)
        assert(mean_episode_embedding.shape[-1] == self.episode_embedding_dim)

        episode_reward = self.reward_mlp(mean_episode_embedding).squeeze(-1)
        return episode_reward, weights


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
    def __init__(self, obs_input_dim, num_heads, query_reduction="sum") -> None:
        super(TransformerRewardPredictor_v2, self).__init__()
        self.num_heads = num_heads
        
        self.embedding_net = nn.Sequential(
            nn.LayerNorm(obs_input_dim),
            init_(nn.Linear(obs_input_dim, 64), activate=True),
            nn.GELU(),
            nn.LayerNorm(64),
        )
        self.key_net = init_(nn.Linear(64, 64))
        self.query_net = nn.Sequential(
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
        self.query_reduction = query_reduction
        assert(self.query_reduction == "sum" or self.query_reduction == "mean")

    def forward(self, state):
        # obtaining keys queries and values
        state_embeddings = self.embedding_net(state) # Batch, Trajectory Length, Embedding Dim
        batch, trajectory_len, emd = state_embeddings.shape
        assert emd % self.num_heads == 0
        if self.query_reduction == "sum":
            query = self.query_net(torch.sum(state_embeddings, dim=1, keepdim=True)).reshape(batch, 1, self.num_heads, emd//self.num_heads).permute(0, 2, 1, 3) # Batch, Num Heads, 1, Embedding Dim
        elif self.query_reduction == "mean":
            query = self.query_net(torch.mean(state_embeddings, dim=1, keepdim=True)).reshape(batch, 1, self.num_heads, emd//self.num_heads).permute(0, 2, 1, 3) # Batch, Num Heads, 1, Embedding Dim
        key = self.key_net(state_embeddings).reshape(batch, trajectory_len, self.num_heads, emd//self.num_heads).permute(0, 2, 1, 3) # Batch, Num Heads, Trajectory Len, Embedding Dim
        value = self.value_net(state_embeddings).reshape(batch, trajectory_len, self.num_heads, emd//self.num_heads).permute(0, 2, 1, 3) # Batch, Num Heads, Trajectory Len, Embedding Dim

        # self attention layer
        attention_weights = F.softmax((query @ key.transpose(-1, -2) / sqrt(self.d_k)), dim=-1) # Batch, Num Heads, 1, Trajectory Len
        attention_values =  (attention_weights @ value).squeeze(-2).view(batch, -1) # Batch, Embedding Dim

        attention_values_ = self.value_norm(attention_values + torch.mean(state_embeddings, dim=1))
        attention_values = self.value_linear(attention_values_)
        attention_values = self.value_linear_norm(attention_values+attention_values_)

        # MLP for predicting reward
        episodic_reward = self.mlp(attention_values)
        
        return episodic_reward, attention_weights
    
class TransformerRewardPredictor_v3(nn.Module):
    def __init__(self, n_agents, obs_input_dim, num_heads, query_reduction="sum") -> None:
        super(TransformerRewardPredictor_v3, self).__init__()
        self.num_heads = num_heads
        self.n_agents = n_agents
        self.embedding_net = nn.Sequential(
            nn.LayerNorm(obs_input_dim),
            init_(nn.Linear(obs_input_dim, 64), activate=True),
            nn.GELU(),
        )
        self.key_net = init_(nn.Linear(64, 64))
        self.query_net = nn.Sequential(
            nn.LayerNorm(64),
            init_(nn.Linear(64, 64))
        )
        self.value_norm = nn.LayerNorm(64)
        self.value_net = init_(nn.Linear(64, 64))
        self.value_linear = nn.Sequential(
            init_(nn.Linear(64, 1024), activate=True),
            nn.GELU(),
            init_(nn.Linear(1024, 64)),
        )
        self.value_linear_norm = nn.LayerNorm(64)
        self.time_transformer = TransformerRewardPredictor_v2(64, num_heads, query_reduction=query_reduction)
        self.d_k = 64//self.num_heads
        self.query_reduction = query_reduction
        assert(self.query_reduction == "sum" or self.query_reduction == "mean")

    def forward(self, states):
        batch_size, num_steps, n_agents, obs_dim = states.shape
        assert n_agents == self.n_agents
        state_embeddings = self.embedding_net(states)  # (batch, num_steps, n_agents, emd)
        emb = state_embeddings.shape[-1]
        assert emb % self.num_heads == 0

        if self.query_reduction == "sum":
            query = self.query_net(torch.sum(state_embeddings, dim=2, keepdim=True)).reshape(batch_size, num_steps, 1, self.num_heads, emb//self.num_heads).permute(0, 1, 3, 2, 4) # Batch, num steps, Num Heads, 1, Embedding Dim
        elif self.query_reduction == "mean":
            query = self.query_net(torch.mean(state_embeddings, dim=2, keepdim=True)).reshape(batch_size, num_steps, 1, self.num_heads, emb//self.num_heads).permute(0, 1, 3, 2, 4) # Batch, num steps, Num Heads, 1, Embedding Dim

        assert query.shape == (batch_size, num_steps, self.num_heads, 1, emb//self.num_heads)
        key = self.key_net(state_embeddings).reshape(batch_size, num_steps, n_agents, self.num_heads, emb//self.num_heads).permute(0, 1, 3, 2, 4) # Batch, Num Heads, num_steps, n_agents, emb
        assert key.shape == (batch_size, num_steps, self.num_heads, n_agents, emb//self.num_heads)
        value = self.value_net(state_embeddings).reshape(batch_size, num_steps, n_agents, self.num_heads, emb//self.num_heads).permute(0, 1, 3, 2, 4) # Batch, Num Heads, num_steps, n_agents, emb
        assert value.shape == (batch_size, num_steps, self.num_heads, n_agents, emb//self.num_heads)

        # self attention layer
        attention_weights = F.softmax((query @ key.transpose(-1, -2) / sqrt(self.d_k)), dim=-1) # Batch, num_steps, Num Heads, 1, n_agents
        assert attention_weights.shape == (batch_size, num_steps, self.num_heads, 1, n_agents)
        attention_values =  (attention_weights @ value).squeeze(-2).view(batch_size, num_steps, -1) # Batch, num_steps, 64
        assert attention_values.shape == (batch_size, num_steps, emb)

        attention_values_ = self.value_norm(attention_values + torch.mean(state_embeddings, dim=2))
        attention_values = self.value_linear(attention_values_)
        attention_values = self.value_linear_norm(attention_values+attention_values_)

        # pass through time transformer
        ep_reward, attn_weights = self.time_transformer(attention_values)
        return ep_reward, attn_weights 

class TransformerRewardPredictor_v4(nn.Module):
    def __init__(self, n_agents, obs_input_dim, num_heads, query_reduction="sum") -> None:
        super(TransformerRewardPredictor_v4, self).__init__()
        self.num_heads = num_heads
        self.n_agents = n_agents
        self.embedding_net = nn.Sequential(
            nn.LayerNorm(obs_input_dim),
            init_(nn.Linear(obs_input_dim, 64), activate=True),
            nn.GELU(),
            nn.LayerNorm(64),
        )
        self.key_net = init_(nn.Linear(64, 64))
        self.query_net = nn.Sequential(
            init_(nn.Linear(64, 64))
        )
        self.value_norm = nn.LayerNorm(64)
        self.value_net = init_(nn.Linear(64, 64))
        self.value_linear = nn.Sequential(
            init_(nn.Linear(64, 1024), activate=True),
            nn.GELU(),
            init_(nn.Linear(1024, 64)),
        )
        self.value_linear_norm = nn.LayerNorm(64)
        self.time_transformer = TransformerRewardPredictor_v2(64, num_heads, query_reduction=query_reduction)
        self.d_k = 64//self.num_heads
        self.query_reduction = query_reduction
        assert(self.query_reduction == "sum" or self.query_reduction == "mean")

    def forward(self, states):
        batch_size, num_steps, n_agents, obs_dim = states.shape
        assert n_agents == self.n_agents
        state_embeddings = self.embedding_net(states)  # (batch, num_steps, n_agents, emd)
        emb = state_embeddings.shape[-1]
        assert emb % self.num_heads == 0

        if self.query_reduction == "sum":
            query = self.query_net(torch.sum(state_embeddings, dim=2, keepdim=True)).reshape(batch_size, num_steps, 1, self.num_heads, emb//self.num_heads).permute(0, 1, 3, 2, 4) # Batch, num steps, Num Heads, 1, Embedding Dim
        elif self.query_reduction == "mean":
            query = self.query_net(torch.mean(state_embeddings, dim=2, keepdim=True)).reshape(batch_size, num_steps, 1, self.num_heads, emb//self.num_heads).permute(0, 1, 3, 2, 4) # Batch, num steps, Num Heads, 1, Embedding Dim

        assert query.shape == (batch_size, num_steps, self.num_heads, 1, emb//self.num_heads)
        key = self.key_net(state_embeddings).reshape(batch_size, num_steps, n_agents, self.num_heads, emb//self.num_heads).permute(0, 1, 3, 2, 4) # Batch, Num Heads, num_steps, n_agents, emb
        assert key.shape == (batch_size, num_steps, self.num_heads, n_agents, emb//self.num_heads)
        value = self.value_net(state_embeddings).reshape(batch_size, num_steps, n_agents, self.num_heads, emb//self.num_heads).permute(0, 1, 3, 2, 4) # Batch, Num Heads, num_steps, n_agents, emb
        assert value.shape == (batch_size, num_steps, self.num_heads, n_agents, emb//self.num_heads)

        # self attention layer
        agent_attention_weights = F.softmax((query @ key.transpose(-1, -2) / sqrt(self.d_k)), dim=-1) # Batch, num_steps, Num Heads, 1, n_agents
        assert agent_attention_weights.shape == (batch_size, num_steps, self.num_heads, 1, n_agents)
        attention_values =  (agent_attention_weights @ value).squeeze(-2).view(batch_size, num_steps, -1) # Batch, num_steps, 64
        assert attention_values.shape == (batch_size, num_steps, emb)

        attention_values_ = self.value_norm(attention_values + torch.mean(state_embeddings, dim=2))
        attention_values = self.value_linear(attention_values_)
        attention_values = self.value_linear_norm(attention_values+attention_values_)

        # pass through time transformer
        ep_reward, temporal_attention_weights = self.time_transformer(attention_values)
        return ep_reward, temporal_attention_weights, agent_attention_weights