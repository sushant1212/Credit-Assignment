import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import simple_spread_custom
from agents.models import MLP
from typing import List
import numpy as np
import torch.optim as optim
from torch.distributions import Categorical
from agents.utils import MetricMonitor
import os
from comet_ml import Experiment
from tqdm import tqdm
from agents.models import (
    Actor,
    Critic,
    AgentSelfAttentionReward_AREL,
    AgentTransformerReward_AREL,
    ReplayBuffer,
    RewardDatasetList,
)
from torch.utils.data import Dataset, DataLoader

class AREL_RewardTrainer:
    def __init__(
            self,
            n_agents:int, 
            actor_hidden_layers:List[int],
            critic_hidden_layers:List[int],
            run_name,
            actor_lr, 
            critic_lr,
            reward_net_kwargs:dict,
            gamma=0.99,
            lambd=0.95,
            entropy_penalty=1e-3,
            max_cycles:int=100,
            update_freq:int=500,  # number of episodes per update (reward model)
            reward_net_epochs=100,  # number of epochs to train the reward network for
            reward_net_lr=1e-3,  # learning rate of the reward network
            reward_net_type="transformer",
            device_id=0,
        ) -> None:
        self.n_agents = n_agents
        self.max_cycles = max_cycles
        self.update_freq = update_freq

        assert(self.n_agents >= 1), "Number of agents should be greater than 0"
        self.env = simple_spread_custom.env(max_cycles=self.max_cycles, N=self.n_agents)
        
        # assuming all agents have the same action_space and observation_space
        self.env_action_dim = self.env.action_space("agent_0").n
        self.env_obs_dim = self.env.observation_space("agent_0").shape[0]

        # setting torch device
        self.device = "cuda:"+str(device_id) if torch.cuda.is_available() else "cpu"

        # creating actor and critic networks
        self.actor = Actor(self.env_obs_dim + (self.n_agents-1) * 4, actor_hidden_layers, self.env_action_dim).to(self.device)  # obs of current agent concatenated with relative px, py, vx, vy of other agents 
        self.critic = Critic(self.env_obs_dim + (self.n_agents-1) * 4, critic_hidden_layers, 1).to(self.device)

        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.gamma = gamma
        self.gae_lambda = lambd
        self.entropy_penalty = entropy_penalty
        self.run_name = run_name

        assert(reward_net_type == "transformer" or reward_net_type == "self-attention"), "parameter reward_net_type should be \"transformer\" or \"self-attention\"" 
        if reward_net_type == "transformer":
            try:
                self.reward_model = AgentTransformerReward_AREL(**reward_net_kwargs).to(self.device)
            except Exception as e:
                print(e)
                sys.exit(0)
        else:
            try:
                self.reward_model = AgentSelfAttentionReward_AREL(**reward_net_kwargs).to(self.device)
            except Exception as e:
                print(e)
                sys.exit(0)
        self.reward_net_lr = reward_net_lr
        self.reward_net_epochs = reward_net_epochs

        # initialising replay buffer
        self.replay_buffer = ReplayBuffer(self.n_agents)

    def get_action(self, state):
        with torch.no_grad():
            state = torch.FloatTensor(state).to(self.device)
            logits = self.actor(state)
            dist = F.softmax(logits, dim=0)
            probs = Categorical(dist)
            return probs.sample().cpu().detach().item()
        
    def get_deterministic_action(self, state):
        with torch.no_grad():
            state = torch.FloatTensor(state).to(self.device)
            logits = self.actor(state)
            return torch.argmax(logits, dim=0).cpu().detach().item()
        
    def combine_observations(self, agent_obs:np.ndarray, global_obs:np.ndarray, curr_agent_index=None):
        """Combines the observation of current agent with the global state.
        It computes [obs_agent +  [(agent.rel_px, agent.rel_py, agent.rel_vx, agent.rel_vy) for all agents]]

        Args:
            agent_obs (np.ndarray): observation received for the current agent
            global_obs (np.ndarray): observation of the global state
        """
        if curr_agent_index is None:
            index = self.curr_agent_index
        else:
            index = curr_agent_index
        
        global_states = []
        for i in range(self.n_agents):
            if i == index:
                continue
            other_agent_obs = global_obs[self.env_obs_dim*i: self.env_obs_dim*(i+1)]
            global_states.append(np.array([
                other_agent_obs[self.n_agents] - agent_obs[self.n_agents],          # rel px
                other_agent_obs[self.n_agents + 1] - agent_obs[self.n_agents + 1],  # rel py
                other_agent_obs[self.n_agents + 2] - agent_obs[self.n_agents + 2],  # rel vx
                other_agent_obs[self.n_agents + 3] - agent_obs[self.n_agents + 3]   # rel vy
            ], dtype=np.float32))
        return np.concatenate([agent_obs] + global_states, dtype=np.float32)
    
    def update_reward_function(self, state_action_list, global_reward_list):
        self.reward_model.train(True)
        reward_dataset = RewardDatasetList(state_action_list, global_reward_list)

        train_loader = DataLoader(
            reward_dataset,
            32,  # hard-coded for now
            shuffle=True
        )

        criterion = nn.HuberLoss()
        optimizer = optim.Adam(self.reward_model.parameters(), lr=self.reward_net_lr)

        for epoch in range(self.reward_net_epochs):
            for i, (X, y) in enumerate(train_loader, start=1):
                X = X.float().to(self.device)
                y = y.float().to(self.device)

                agent_rewards, _ = self.reward_model(X)
                y_hat = torch.sum(agent_rewards, dim=1)

                loss = criterion(y_hat, y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                self.reward_net_loss += loss.item()

        self.reward_net_loss /= self.reward_net_epochs
        return

    def update(self):
        agent_states = []
        agent_states_combined = []
        agent_actions = []
        agent_rewards = []
        agent_next_states_combined = []
        agent_dones = []

        for agent_index in range(self.n_agents ):
            states_agent = self.replay_buffer.get("state", agent_index).float().to(self.device)
            states_combined = self.replay_buffer.get("state_combined", agent_index).float().to(self.device)
            actions = self.replay_buffer.get("action", agent_index).long().view(-1, 1).to(self.device)
            rewards = self.replay_buffer.get("reward", agent_index).float().view(-1, 1).to(self.device)
            next_states_combined = self.replay_buffer.get("next_state_combined", agent_index).float().to(self.device)
            dones = self.replay_buffer.get("done", agent_index).float().view(-1, 1).to(self.device)
      
            agent_states.append(states_agent)
            agent_states_combined.append(states_combined)
            agent_actions.append(actions)
            agent_rewards.append(rewards)
            agent_next_states_combined.append(next_states_combined)
            agent_dones.append(dones)

        # calculating policy and value losses
        actor_loss = 0
        critic_loss = 0
        entropy_loss = 0
        for agent_index in range(self.n_agents):
            critic_input = agent_states_combined[agent_index]
            V_s = self.critic(critic_input)
            V_s_prime = self.critic(agent_next_states_combined[agent_index]) * (1 - agent_dones[agent_index])
            critic_target = (agent_rewards[agent_index] + self.gamma * V_s_prime)
            td_errors = critic_target - V_s
            assert(agent_states_combined[agent_index].shape[0] == (self.max_cycles)), "something wrong"
            advantages = torch.zeros(agent_states_combined[agent_index].shape[0]).to(self.device)
            advantages[self.max_cycles - 1] = td_errors[self.max_cycles - 1][0]
            for j in reversed(range(0, (self.max_cycles - 1))):
                advantages[j] = td_errors[j][0]
                advantages[j] += (self.gae_lambda * self.gamma * advantages[j+1])
            
            logits = self.actor(agent_states_combined[agent_index])
            dists = F.softmax(logits, dim=1)
            probs = Categorical(dists)

            # computing critic loss
            critic_loss += F.mse_loss(V_s, critic_target.detach())

            # computing entropy bonus
            entropy = -torch.mean(torch.sum(dists * torch.log(torch.clamp(dists, 1e-10,1.0)), dim=1))
            
            policy_loss = -probs.log_prob(agent_actions[agent_index].view(agent_actions[agent_index].size(0))).view(-1, 1) * advantages.view(-1, 1).detach()
            actor_loss += policy_loss.mean()
            entropy_loss += entropy

        # total loss
        loss = actor_loss + critic_loss - self.entropy_penalty * entropy_loss
        self.actor_loss += actor_loss.item()
        self.critic_loss += critic_loss.item()
        self.entropy_loss += -(self.entropy_penalty * entropy_loss).item()

        # backpropagation
        self.actor_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()
        loss.backward()

        # gradient clipping
        total_grad_norm = 0.0
        total_grad_norm += torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=0.5)
        total_grad_norm += torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=0.5)
        self.total_grad_norm = total_grad_norm.item()

        # updating parameters
        self.actor_optimizer.step()
        self.critic_optimizer.step()

    def plot(self, model_save_path, episode):
        if not os.path.isdir(os.path.join(model_save_path, "plots")):
            os.makedirs(os.path.join(os.path.join(model_save_path, "plots")))
        
        if episode % self.update_freq == 0:
            self.train_metric_monitor.update("reward_net_loss", self.reward_net_loss, os.path.join(model_save_path, "plots"))
        self.train_metric_monitor.update("actor_loss", self.actor_loss, os.path.join(model_save_path, "plots"))
        self.train_metric_monitor.update("critic_loss", self.critic_loss, os.path.join(model_save_path, "plots"))
        self.train_metric_monitor.update("entropy_loss", self.entropy_loss, os.path.join(model_save_path, "plots"))
        self.train_metric_monitor.update("total_loss", self.actor_loss + self.critic_loss + self.entropy_loss, os.path.join(model_save_path, "plots"))
        self.train_metric_monitor.update("total_grad_norm", self.total_grad_norm, os.path.join(model_save_path, "plots"))

        self.agent_metric_monitor.update("episode_reward", self.episode_reward, os.path.join(model_save_path, "plots"))
        for i in range(self.n_agents):
            self.agent_metric_monitor.update(f"agent_{i}_reward", self.agent_rewards[i], os.path.join(model_save_path, "plots"))
            self.agent_metric_monitor.update(f"agent_{i}_given_reward", self.agent_given_rewards[i], os.path.join(model_save_path, "plots"))
        
        # plot using comet_ml
        metrics = {"episode_reward": self.episode_reward}

        if episode % self.update_freq == 0:
            metrics["reward_net_loss"] = self.reward_net_loss
        
        metrics["actor_loss"] =  self.actor_loss
        metrics["critic_loss"] =  self.critic_loss
        metrics["entropy_loss"] =  self.entropy_loss
        metrics["total_loss"] =  self.actor_loss + self.critic_loss + self.entropy_loss
        metrics["total_grad_norm"] =  self.total_grad_norm

        for i in range(self.n_agents):
            metrics[f"agent_{i}_reward"] = self.agent_rewards[i]
            metrics[f"agent_{i}_given_reward"] = self.agent_given_rewards[i]
        self.experiment.log_metrics(metrics, epoch=episode)

    def save(self, model_save_path, episode):
        if not os.path.exists(os.path.join(model_save_path, "models")):
            os.makedirs(os.path.join(model_save_path, "models"))
        
        actor_save_path = os.path.join(model_save_path, "models", "actor_episode_" + str(episode).zfill(8) + ".pth")
        critic_save_path = os.path.join(model_save_path, "models", "critic_episode_" + str(episode).zfill(8) + ".pth")
        reward_save_path = os.path.join(model_save_path, "models", "reward_episode_" + str(episode).zfill(8) + ".pth")

        torch.save(self.actor.state_dict(), actor_save_path)
        torch.save(self.critic.state_dict(), critic_save_path)
        torch.save(self.reward_model.state_dict(), reward_save_path)


    def train(self, n_episodes:int, model_save_path:str, plot:bool=True):
        if plot:
            # setting up Comet
            self.experiment = Experiment(
                api_key="8U8V63x4zSaEk4vDrtwppe8Vg",
                project_name="credit-assignment",
                parse_args=False
            )
            self.experiment.set_name(self.run_name)
            # logging hparams to comet_ml
            hyperparams = {
                "actor_lr": self.actor_lr,
                "critic_lr" : self.critic_lr,
                "gamma" : self.gamma,
                "reward_lr": self.reward_net_lr,
                "reward_net_epochs": self.reward_net_epochs,
                "update_freq": self.update_freq
            }
            self.experiment.log_parameters(hyperparams)

        # optimizers for actor and critic
        self.actor_optimizer = optim.Adam(self.actor.parameters(), self.actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), self.critic_lr)

        # metric monitors to store metric values
        self.train_metric_monitor = MetricMonitor()
        self.agent_metric_monitor = MetricMonitor()

        for episode in tqdm(range(n_episodes)):
            # reset the environment
            self.env.reset()

            # index to keep track of the current agent
            self.curr_agent_index = 0

            # reward net related variables
            if episode % self.update_freq == 0:
                state_action_list = []
                global_reward_list = []
                self.reward_net_loss = 0.0
            
            # clearing buffer
            self.replay_buffer.clear()  # on-policy

            # initialising episode related metrics
            self.agent_given_rewards = [0.0 for _ in range(self.n_agents)]
            self.actor_loss = 0.0
            self.critic_loss = 0.0
            self.entropy_loss = 0.0
            self.total_grad_norm = 0.0
            self.episode_reward = 0.0
            self.agent_rewards = [0.0 for _ in range(self.n_agents)]

            # initialising episode related variables / flags
            global_reward = 0.0
            state = None  # stores the global state
            actions = []
            ep_start = True  # flag that indicates the first step of the episode

            for agent in self.env.agent_iter():
                # this reward is R_t and not R_t+1. Also terminated and truncated are current states
                obs, rew, terminated, truncated, info = self.env.last()

                # global state of the env is the concatenation of individual states of agents
                global_state = self.env.state()

                # making feature vector for current agent that encodes other agent states as well
                agent_obs = self.combine_observations(obs, global_state)

                # sampling action using current policy
                action = self.get_action(agent_obs)

                # taking step in the environment
                if terminated or truncated:  # don't take any action when the agent has already terminated or truncated
                    self.env.step(None)
                else:
                    self.env.step(action)

                done = terminated or truncated

                # storing transition in buffer
                if not ep_start:
                    self.replay_buffer.add_values(
                        self.curr_agent_index,
                        reward=[rew],
                        next_state=np.expand_dims(obs, axis=0),
                        next_state_combined=np.expand_dims(agent_obs, axis=0),
                        done=[done]                      
                    )
                
                if not done:
                    self.replay_buffer.add_values(
                        self.curr_agent_index,
                        state=np.expand_dims(obs, axis=0),
                        state_combined=np.expand_dims(agent_obs, axis=0),
                        action=[action]
                    )

                self.episode_reward += rew
                self.agent_rewards[self.curr_agent_index] += rew
                global_reward += rew
                actions.append(action)
                state = global_state.reshape(self.n_agents, -1)

                # incrementing the current index
                self.curr_agent_index += 1
                self.curr_agent_index %= self.n_agents

                if self.curr_agent_index == 0:  # if cycled through all agents
                    actions = torch.tensor(actions)
                    actions = F.one_hot(actions, self.env_action_dim)
                    state = torch.from_numpy(state)
                    state_action = torch.cat((state, actions), dim=1)
                    assert(state_action.shape[0] == self.n_agents)
                    
                    if not ep_start: global_reward_list.append(global_reward)
                    if not done: 
                        self.replay_buffer.add_values(state_action=state_action.unsqueeze(0))
                        state_action_list.append(state_action.cpu().numpy().tolist())

                    if ep_start == True: ep_start = False
                    global_reward = 0.0
                    actions = []
                    state = None

            # first, update the buffer with predicted rewards using the reward model
            predicted_rewards = self.replay_buffer.predict_and_update_rewards(self.reward_model, self.device)
            predicted_rewards = torch.sum(predicted_rewards, dim=1)

            assert predicted_rewards.shape[0] == self.n_agents
            for i, rew in enumerate(list(predicted_rewards)):
                self.agent_given_rewards[i] = rew.item()

            # second, update the actor and critic networks
            self.update()

            # update the reward function every `update_every` episodes
            if (episode + 1) % self.update_freq == 0:
                # update reward function
                assert(len(state_action_list) == len(global_reward_list))
                self.update_reward_function(state_action_list, global_reward_list)

            if plot:
                self.plot(model_save_path, episode+1)

            # save model
            if((episode+1) % 500 == 0): self.save(model_save_path, episode)