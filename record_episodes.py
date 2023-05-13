import torch
import torch.nn as nn
import torch.nn.functional as F
import simple_spread_custom
from agents.actor_critic_v import ActorCriticAgent
from argparse import ArgumentParser
import numpy as np
import sys
import json
import os
from tqdm import tqdm


def record(n_agents, wt_path, num_episodes, data_dir, max_cycles, start_index=0):
    env = simple_spread_custom.env(max_cycles=max_cycles, N=n_agents)
    
    action_dim = env.action_space("agent_0").n
    
    agent = ActorCriticAgent(
        n_agents=4,
        actor_hidden_layers=[64, 64, 32],
        critic_hidden_layers=[128, 128, 32],
        actor_lr=1e-4,
        critic_lr=1e-4,
        run_name="actor_critic"
    )
    try:
        agent.actor.load_state_dict(torch.load(wt_path, map_location=torch.device(agent.device)))
        print("Loaded model successfully")

    except Exception as e:
        print(e)
        sys.exit(0)

    if not os.path.exists(data_dir):
        os.makedirs(data_dir)


    for i in tqdm(range(num_episodes)):
        env.reset()
        d = {}
        d["state_actions"] = []
        d["global_rewards"] = []
        d["agent_rewards"] = []


        # index to keep track of the current agent
        curr_agent_index = 0

        state = None
        actions = []
        global_reward = 0
        agent_rewards = []

        for _ in env.agent_iter():
            # this reward is R_t and not R_t+1. Also terminated and truncated are current states
            obs, rew, terminated, truncated, info = env.last()

            # global state of the env is the concatenation of individual states of agents
            global_state = env.state()

            state = global_state.reshape(n_agents, -1)

            global_reward += rew
            agent_rewards.append(rew)

            # making feature vector for current agent that encodes other agent states as well
            agent_obs = agent.combine_observations(obs, global_state, curr_agent_index)

            # sampling action using current policy
            action = agent.get_deterministic_action(agent_obs)

            actions.append(action)

            # the agent has already terminated or truncated
            if terminated or truncated:
                env.step(None)
            
            else:
                env.step(action)

            curr_agent_index += 1
            curr_agent_index %= n_agents

            if curr_agent_index == 0:
                actions = torch.tensor(actions)
                actions = F.one_hot(actions, action_dim)
                state = torch.from_numpy(state)

                state_action = torch.cat((state, actions), dim=1)

                assert(state_action.shape[0] == n_agents)

                d["state_actions"].append(state_action.cpu().numpy().tolist())
                d["agent_rewards"].append(agent_rewards)
                d["global_rewards"].append(global_reward)

                state = None
                actions = []
                global_reward = 0
                agent_rewards = []
        d["state_actions"] = d["state_actions"][0:-1]
        d["agent_rewards"] = d["agent_rewards"][1:]
        d["global_rewards"] = d["global_rewards"][1:]

        assert(len(d["state_actions"]) == len(d["global_rewards"]) == len(d["agent_rewards"]) == 25)

        with open(os.path.join(data_dir, str(i + start_index).zfill(5) + ".json"), "w+") as f:
            json.dump(d, f)


if __name__ == "__main__":
    ap = ArgumentParser()
    ap.add_argument("-n", "--num_episodes", required=True, type=int, help="Number of episodes to record")
    ap.add_argument("-w", "--wt_path", required=True, type=str, help="Path to the weight file")
    ap.add_argument("-a", "--num_agents", required=True, type=int, help="Number of agents in the enviroment")
    ap.add_argument("-d", "--data_dir", required=True, type=str, help="Path to the directory where the dataset should be stored")
    ap.add_argument("-m", "--max_cycles", required=True, type=int, help="Episode length in the environment")
    ap.add_argument("-s", "--start_index", required=False, default=0, type=int, help="Starting index of the json file")

    args = vars(ap.parse_args())
    record(
        args["num_agents"], 
        args["wt_path"], 
        args["num_episodes"], 
        args["data_dir"], 
        args["max_cycles"], 
        args["start_index"]
    )