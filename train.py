import torch
import torch.nn as nn
import torch.nn.functional as F
from agents.actor_critic_v import ActorCriticAgent



agent = ActorCriticAgent(
    n_agents=4,
    actor_hidden_layers=[64, 64, 32],
    critic_hidden_layers=[128, 128, 32],
    actor_lr=1e-4,
    critic_lr=1e-4,
    run_name="actor_critic"
)

agent.train(100000, "actor_critic")