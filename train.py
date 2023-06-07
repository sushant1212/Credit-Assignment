import torch
import torch.nn as nn
import torch.nn.functional as F
from agents.actor_critic_v import ActorCriticAgent
from agents.actor_critic_v_reward import ActorCriticRewardAgent
from agents.actor_critic_reward_train import ActorCriticRewardTrainerAgent
from agents.arel import AREL_RewardTrainer
from argparse import ArgumentParser


if __name__ == "__main__":
    ap = ArgumentParser()
    ap.add_argument("-t", "--type", type=str, required=True, help="actor_critic / actor_critic_reward / actor_critic_with_reward_training / arel")
    ap.add_argument("-r", "--run_name", type=str, required=True)
    ap.add_argument("-s", "--save_path", type=str, required=True)
    ap.add_argument("-d", "--device", type=int, required=False, default=0)
    ap.add_argument("-n", "--num_episodes", type=int, required=False, default=100000)

    args = vars(ap.parse_args())

    if args["type"] == "actor_critic":
        agent =  ActorCriticAgent(
            n_agents=4,
            actor_hidden_layers=[64, 64, 32],
            critic_hidden_layers=[128, 128, 32],
            actor_lr=1e-4,
            critic_lr=1e-4,
            run_name=args["run_name"],
            device_id=args["device"]
        )
    elif args["type"] == "actor_critic_reward":
        agent = ActorCriticRewardAgent(
            n_agents=4,
            actor_hidden_layers=[64, 64, 32],
            critic_hidden_layers=[128, 128, 32],
            actor_lr=1e-4,
            critic_lr=1e-4,
            run_name=args["run_name"],
            reward_model_wt="models_transformer/epoch_00099.pth",
            device_id=args["device"]
        )
    elif args["type"] == "actor_critic_with_reward_training":
        agent = ActorCriticRewardTrainerAgent(
            n_agents=4,
            actor_hidden_layers=[64, 64, 32],
            critic_hidden_layers=[128, 128, 32],
            actor_lr=1e-4,
            critic_lr=1e-4,
            run_name=args["run_name"],
            reward_model_wt="models_transformer/epoch_00099.pth",
            device_id=args["device"],
            update_freq=4
        )

    elif args["type"] == "arel":
        agent = AREL_RewardTrainer(
            n_agents=4,
            actor_hidden_layers=[64, 64, 32],
            critic_hidden_layers=[128, 128, 32],
            actor_lr=1e-4,
            critic_lr=1e-4,
            run_name=args["run_name"],
            reward_net_kwargs={
                "e_dim": 15,
                "d_k": 15,
                "mlp_hidden_layers": [128, 64, 4, 1]
            },
            reward_net_type="transformer",
            device_id=args["device"],
        )

    else:
        raise NotImplementedError
    
    agent.train(args["num_episodes"], args["save_path"])