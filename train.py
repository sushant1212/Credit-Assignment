import torch
import torch.nn as nn
import torch.nn.functional as F
from agents.actor_critic_v import ActorCriticAgent
from agents.actor_critic_v_reward import ActorCriticRewardAgent
from argparse import ArgumentParser


if __name__ == "__main__":
    ap = ArgumentParser()
    ap.add_argument("-t", "--type", type=str, required=True, help="actor_critic or actor_critic_reward")
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

    else:
        raise NotImplementedError
    
    agent.train(args["nume_episodes"], args["save_path"])