import numpy as np
import random
from gymnasium.utils import EzPickle

from pettingzoo.utils.conversions import parallel_wrapper_fn

from pettingzoo.mpe._mpe_utils.core import Agent, Landmark, World
from pettingzoo.mpe._mpe_utils.scenario import BaseScenario
from pettingzoo.mpe._mpe_utils.simple_env import SimpleEnv, make_env
import os

import gymnasium
import numpy as np
import pygame
from gymnasium import spaces
from gymnasium.utils import seeding

from pettingzoo import AECEnv
from pettingzoo.mpe._mpe_utils.core import Agent
from pettingzoo.utils import wrappers
from pettingzoo.utils.agent_selector import agent_selector
alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


class raw_env(SimpleEnv, EzPickle):
    def __init__(
        self,
        N=3,
        local_ratio=1.0,
        max_cycles=25,
        continuous_actions=False,
        render_mode=None,
    ):
        EzPickle.__init__(
            self, N, local_ratio, max_cycles, continuous_actions, render_mode
        )
        assert (
            0.0 <= local_ratio <= 1.0
        ), "local_ratio is a proportion. Must be between 0 and 1."
        scenario = Scenario()
        world = scenario.make_world(N)
        super().__init__(
            scenario=scenario,
            world=world,
            render_mode=render_mode,
            max_cycles=max_cycles,
            continuous_actions=continuous_actions,
            local_ratio=local_ratio,
        )
        self.metadata["name"] = "simple_spread_custom"

    def draw(self):
        # clear screen
        self.screen.fill((255, 255, 255))

        # update bounds to center around agent
        all_poses = [entity.state.p_pos for entity in self.world.entities]
        # cam_range = np.max(np.abs(np.array(all_poses)))
        cam_range = 1

        # update geometry and text positions
        text_line = 0
        for e, entity in enumerate(self.world.entities):
            # geometry
            x, y = entity.state.p_pos
            y *= (
                -1
            )  # this makes the display mimic the old pyglet setup (ie. flips image)
            x = (
                (x / cam_range) * self.width // 2 * 0.9
            )  # the .9 is just to keep entities from appearing "too" out-of-bounds
            y = (y / cam_range) * self.height // 2 * 0.9
            x += self.width // 2
            y += self.height // 2
            pygame.draw.circle(
                self.screen, entity.color * 200, (x, y), entity.size * 350
            )  # 350 is an arbitrary scale factor to get pygame to render similar sizes as pyglet
            pygame.draw.circle(
                self.screen, (0, 0, 0), (x, y), entity.size * 350, 1
            )  # borders
            # assert (
            #     0 < x < self.width and 0 < y < self.height
            # ), f"Coordinates {(x, y)} are out of bounds."

            # text
            if isinstance(entity, Agent):
                if entity.silent:
                    continue
                if np.all(entity.state.c == 0):
                    word = "_"
                elif self.continuous_actions:
                    word = (
                        "[" + ",".join([f"{comm:.2f}" for comm in entity.state.c]) + "]"
                    )
                else:
                    word = alphabet[np.argmax(entity.state.c)]

                message = entity.name + " sends " + word + "   "
                message_x_pos = self.width * 0.05
                message_y_pos = self.height * 0.95 - (self.height * 0.05 * text_line)
                self.game_font.render_to(
                    self.screen, (message_x_pos, message_y_pos), message, (0, 0, 0)
                )
                text_line += 1


env = make_env(raw_env)
parallel_env = parallel_wrapper_fn(env)


class Scenario(BaseScenario):
    def __init__(self) -> None:
        super().__init__()
        self.agent_to_landmark_map = {}
        self.agent_name_to_index = {}
        self.l2_factor = 0.1
        self.collision_reward = -0.2
    
    def create_agent_to_landmark_map(self, world:World):
        # create a random mapping of landmark to each agent
        n_agents = len(world.agents)
        l = list(range(n_agents))
        random.shuffle(l)

        # set the agent_idx : landmark_idx mapping
        for agent_idx, landmark_idx in enumerate(l):
            self.agent_to_landmark_map[agent_idx] = landmark_idx

        # creating a map of agent name to index in list
        for idx, agent in enumerate(world.agents):
            self.agent_name_to_index[agent.name] = idx
        
    def make_world(self, N=3):
        world = World()
        # set any world properties first
        world.dim_c = 2
        num_agents = N
        num_landmarks = N
        world.collaborative = True
        # add agents
        world.agents = [Agent() for i in range(num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = f"agent_{i}"
            agent.collide = True
            agent.silent = True
            agent.size = 0.15
        # add landmarks
        world.landmarks = [Landmark() for i in range(num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = "landmark %d" % i
            landmark.collide = False
            landmark.movable = False
        return world

    def reset_world(self, world:World, np_random):
        # set random initial states
        for agent in world.agents:
            agent.state.p_pos = np_random.uniform(-1, +1, world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)
        for i, landmark in enumerate(world.landmarks):
            landmark.state.p_pos = np_random.uniform(-1, +1, world.dim_p)
            landmark.state.p_vel = np.zeros(world.dim_p)
        
        self.create_agent_to_landmark_map(world)

        
        for i, agent in enumerate(world.agents):
            color = np.array([np.random.random(), np.random.random(), np.random.random()])
            agent.color = color
            world.landmarks[self.agent_to_landmark_map[i]].color = color

    def benchmark_data(self, agent, world):
        rew = 0
        collisions = 0
        occupied_landmarks = 0
        min_dists = 0
        for lm in world.landmarks:
            dists = [
                np.sqrt(np.sum(np.square(a.state.p_pos - lm.state.p_pos)))
                for a in world.agents
            ]
            min_dists += min(dists)
            rew -= min(dists)
            if min(dists) < 0.1:
                occupied_landmarks += 1
        if agent.collide:
            for a in world.agents:
                if self.is_collision(a, agent):
                    rew -= 1
                    collisions += 1
        return (rew, collisions, min_dists, occupied_landmarks)

    def is_collision(self, agent1, agent2):
        delta_pos = agent1.state.p_pos - agent2.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        dist_min = agent1.size + agent2.size
        return True if dist < dist_min else False

    def reward(self, agent:Agent, world:World):
        # Agents are penalized for collisions, and rewarded based on how close they are to their goal
        rew = 0
        if agent.collide:
            for a in world.agents:
                if(a == agent):
                    continue
                if self.is_collision(a, agent):
                    rew += self.collision_reward

        # if agent collides, return collision reward
        if rew != 0:
            return rew
        
        # calculate L2 reward
        agent_name = agent.name
        agent_idx = self.agent_name_to_index[agent_name]
        landmark_idx = self.agent_to_landmark_map[agent_idx]

        landmark = world.landmarks[landmark_idx]
        l2_distance = np.linalg.norm(agent.state.p_pos-landmark.state.p_pos)

        rew = self.l2_factor * (-1 * l2_distance)
        return rew

    def global_reward(self, world):
        rew = 0
        for lm in world.landmarks:
            dists = [
                np.sqrt(np.sum(np.square(a.state.p_pos - lm.state.p_pos)))
                for a in world.agents
            ]
            rew -= min(dists)
        return rew

    def observation(self, agent, world):
        # [one_hot_encoding, px, py, vx, vy, g_x, g_y] (all in world frame)
        n_agents = len(world.agents)
        encoding = np.zeros(n_agents)
        encoding[self.agent_name_to_index[agent.name]] = 1

        entity_pos = agent.state.p_pos
        entity_vel = agent.state.p_vel
        goal_landmark_idx = self.agent_to_landmark_map[self.agent_name_to_index[agent.name]]
        goal_pos = world.landmarks[goal_landmark_idx].state.p_pos

        return np.concatenate(
            [encoding] + [entity_pos] + [entity_vel] + [goal_pos]
        )