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
import sys
from agents.actor_critic_v import Actor, Critic
from reward_regressor import TransformerRewardPredictor

class ActorCriticRewardAgent:
    def __init__(
            self,
            n_agents:int, 
            actor_hidden_layers:List[int],
            critic_hidden_layers:List[int],
            run_name,
            reward_model_wt,
            actor_lr, 
            critic_lr,
            gamma=0.99,
            lambd=0.95,
            entropy_penalty=1e-3,
            max_cycles:int=100,
            device_id=0,
        ) -> None:
        self.n_agents = n_agents
        self.max_cycles = max_cycles

        assert(self.n_agents >= 1), "Number of agents should be greater than 0"
        self.env = simple_spread_custom.env(max_cycles=max_cycles, N=self.n_agents)
        
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

        # reward model
        self.reward_model = TransformerRewardPredictor(15, 15, [128, 64, 4, 1]).to(self.device)
        self.reward_model.load_state_dict(torch.load(reward_model_wt, map_location=torch.device(self.device)))
    
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
    
    def update(self):
        agent_states = []
        agent_states_combined = []
        agent_actions = []
        agent_rewards = []
        agent_next_states_combined = []
        agent_dones = []

        for agent_index in range(self.n_agents ):
            states_agent = torch.FloatTensor(np.array([sars[0] for sars in self.trajectory[agent_index]])).to(self.device)
            states_combined = torch.FloatTensor(np.array([sars[1] for sars in self.trajectory[agent_index]])).to(self.device)
            actions = torch.LongTensor(np.array([sars[2] for sars in self.trajectory[agent_index]])).view(-1, 1).to(self.device)
            rewards = torch.FloatTensor(np.array([sars[3] for sars in self.trajectory[agent_index]])).view(-1, 1).to(self.device)
            next_states_agent = torch.FloatTensor(np.array([sars[4] for sars in self.trajectory[agent_index]])).to(self.device)
            next_states_combined = torch.FloatTensor(np.array([sars[5] for sars in self.trajectory[agent_index]])).to(self.device)
            dones = torch.FloatTensor(np.array([sars[6] for sars in self.trajectory[agent_index]])).view(-1, 1).to(self.device)
      
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
            advantages = torch.zeros(agent_states_combined[agent_index].shape[0]).to(self.device)
            advantages[-1] = td_errors[-1][0]
            for i in reversed(range(agent_states_combined[agent_index].shape[0]-1)):
                advantages[i] = td_errors[i][0]
                advantages[i] += (self.gae_lambda * self.gamma * advantages[i+1])
            
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
        metrics = {
            "actor_loss": self.actor_loss,
            "critic_loss": self.critic_loss,
            "entropy_loss": self.entropy_loss,
            "total_loss": self.actor_loss + self.critic_loss + self.entropy_loss,
            "total_grad_norm": self.total_grad_norm,
            "episode_reward": self.episode_reward,
        }
        for i in range(self.n_agents):
            metrics[f"agent_{i}_reward"] = self.agent_rewards[i]
            metrics[f"agent_{i}_given_reward"] = self.agent_given_rewards[i]
        self.experiment.log_metrics(metrics, epoch=episode)

    def save(self, model_save_path, episode):
        if not os.path.exists(os.path.join(model_save_path, "models")):
            os.makedirs(os.path.join(model_save_path, "models"))
        
        actor_save_path = os.path.join(model_save_path, "models", "actor_episode_" + str(episode).zfill(8) + ".pth")
        critic_save_path = os.path.join(model_save_path, "models", "critic_episode_" + str(episode).zfill(8) + ".pth")

        torch.save(self.actor.state_dict(), actor_save_path)
        torch.save(self.critic.state_dict(), critic_save_path)
    
    @torch.no_grad()
    def calculate_agent_rewards(self, global_reward, state_actions):
        if type(state_actions) == np.ndarray:
            state_actions = torch.from_numpy(state_actions)
        
        assert(type(global_reward) == float)
        state_actions = state_actions.to(self.device)
        _, attention_weights = self.reward_model(state_actions.unsqueeze(0))
        attention_weights = attention_weights.squeeze()
        attention_weights *= global_reward
        return attention_weights.cpu().detach().numpy()


    def train(self, n_episodes:int, model_save_path:str):
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
        }
        self.experiment.log_parameters(hyperparams)

        # optimizers for actor and critic
        self.actor_optimizer = optim.Adam(self.actor.parameters(), self.actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), self.critic_lr)

        # metric monitors to store metric values
        self.train_metric_monitor = MetricMonitor()
        self.agent_metric_monitor = MetricMonitor()
        
        stream = tqdm(range(n_episodes))
        for episode in stream:
            self.env.reset()

            # index to keep track of the current agent
            self.curr_agent_index = 0

            # episode related variables
            self.trajectory = [ [] for _ in range(self.n_agents)]

            self.episode_reward = 0
            self.agent_rewards = [0.0 for _ in range(self.n_agents)]
            self.agent_given_rewards = [0.0 for _ in range(self.n_agents)]
            self.actor_loss = 0.0
            self.critic_loss = 0.0
            self.entropy_loss = 0.0
            self.total_grad_norm = 0.0
            prev_global_state = None

            for agent in self.env.agent_iter():
                # storing variables
                global_reward = 0.0

                # this reward is R_t and not R_t+1. Also terminated and truncated are current states
                obs, rew, terminated, truncated, info = self.env.last()

                # global state of the env is the concatenation of individual states of agents
                global_state = self.env.state()

                # making feature vector for current agent that encodes other agent states as well
                agent_obs = self.combine_observations(obs, global_state)

                # sampling action using current policy
                action = self.get_action(agent_obs)

                # the agent has already terminated or truncated
                if terminated or truncated:
                    self.env.step(None)
                
                else:
                    self.env.step(action)

                global_reward += rew
                
                done = terminated or truncated

                # storing the trajectory in the format [s_agent, s_combined, a, r, s'_agent, s'_combined, done]
                if len(self.trajectory[self.curr_agent_index]) != 0:
                    self.trajectory[self.curr_agent_index][-1].append(rew)
                    self.trajectory[self.curr_agent_index][-1].append(obs)
                    self.trajectory[self.curr_agent_index][-1].append(agent_obs)
                    self.trajectory[self.curr_agent_index][-1].append(done)
                
                if not done:
                    self.trajectory[self.curr_agent_index].append([obs, agent_obs, action])
                
                self.curr_agent_index += 1
                self.curr_agent_index %= self.n_agents
                
                # redistributing the reward using learnt reward model
                if (self.curr_agent_index == 0) and (prev_global_state is not None):
                    actions = []
                    index = -2 if not done else -1
                    for i in range(self.n_agents):
                        assert(len(self.trajectory[i][index]) == 7)
                        actions.append(self.trajectory[i][index][2])

                    actions = torch.tensor(actions)
                    actions = F.one_hot(actions, self.env_action_dim)

                    state = torch.from_numpy(prev_global_state)
                    state_action = torch.cat((state, actions), dim=1)
                    assert(state_action.shape[0] == self.n_agents)

                    agent_rewards = self.calculate_agent_rewards(global_reward, state_action)
                    assert(agent_rewards.shape == (self.n_agents, ))
                    for i in range(self.n_agents):
                        self.trajectory[i][index][3] = agent_rewards[i]
                        self.agent_given_rewards[i] += agent_rewards[i]

                if self.curr_agent_index == 0:
                    prev_global_state = global_state.reshape(self.n_agents, -1)
                    global_reward = 0.0

                self.episode_reward += rew
                self.agent_rewards[self.curr_agent_index] += rew
            
            # update networks
            self.update()
            # update plots
            self.plot(model_save_path, episode+1)
            # save model
            if(episode % 500 == 0): self.save(model_save_path, episode)