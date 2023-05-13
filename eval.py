import torch
import torch.nn as nn
import torch.nn.functional as F
from agents.actor_critic_v import ActorCriticAgent

model_weight_path = "actor_episode_00099500.pth"
agent = ActorCriticAgent(
    4,
    actor_hidden_layers=[64, 64, 32],
    critic_hidden_layers=[128, 128, 32],
    actor_lr=1e-4,
    critic_lr=1e-4,
    run_name="actor_critic",
    max_cycles=25
)

agent.eval(model_weight_path, 10, True)
