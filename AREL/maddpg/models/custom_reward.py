import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

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
