import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
import numpy as np
import torch.optim as optim
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