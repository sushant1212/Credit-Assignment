import torch
import torch.nn as nn
import torch.nn.functional as F
# from agents.actor_critic_v import ActorCriticAgent
from agents.actor_critic_v_reward import ActorCriticRewardAgent



# agent = ActorCriticAgent(
#     n_agents=4,
#     actor_hidden_layers=[64, 64, 32],
#     critic_hidden_layers=[128, 128, 32],
#     actor_lr=1e-4,
#     critic_lr=1e-4,
#     run_name="actor_critic"
# )


agent = ActorCriticRewardAgent(
    n_agents=4,
    actor_hidden_layers=[64, 64, 32],
    critic_hidden_layers=[128, 128, 32],
    actor_lr=1e-4,
    critic_lr=1e-4,
    run_name="actor_critic_with_reward_model",
    reward_model_wt="epoch_00099.pth"
)
agent.train(100000, "actor_critic_")