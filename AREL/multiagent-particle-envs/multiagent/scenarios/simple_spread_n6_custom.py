import numpy as np
import random
from multiagent.core import World, Agent, Landmark
from multiagent.scenario import BaseScenario


class Scenario(BaseScenario):
    def make_world(self, sort_obs=False):
        world = World()
        self.np_rnd = np.random.RandomState(0)
        self.sort_obs = sort_obs
        # set any world properties first
        world.dim_c = 2
        num_agents = 6
        num_landmarks = 6
        self.world_radius = 1.5
        # add agents
        world.agents = [Agent() for i in range(num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.collide = True
            agent.silent = True
            agent.size = 0.15
        # add landmarks
        world.landmarks = [Landmark() for i in range(num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = 'landmark %d' % i
            landmark.collide = False
            landmark.movable = False
        # make initial conditions
        self.reset_world(world)
        world.agent_landmark_mapping = {}
        for i, agent in enumerate(world.agents):
            world.agent_landmark_mapping[agent] = world.landmarks[i]
        return world

    def reset_world(self, world):
        # random properties for agents
        for i, agent in enumerate(world.agents):
            agent.color = np.array([0.35, 0.35, 0.85])
        # random properties for landmarks
        for i, landmark in enumerate(world.landmarks):
            landmark.color = np.array([0.25, 0.25, 0.25])
        # set random initial states
        for agent in world.agents:
            agent.state.p_pos = self.np_rnd.uniform(-1.5, +1.5, world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)
        for i, landmark in enumerate(world.landmarks):
            landmark.state.p_pos = self.np_rnd.uniform(-1.5, +1.5, world.dim_p)
            landmark.state.p_vel = np.zeros(world.dim_p)

    def benchmark_data(self, agent, world):
        rew = 0
        collisions = 0
        occupied_landmarks = 0
        min_dists = 0
        for l in world.landmarks:
            dists = [np.sqrt(np.sum(np.square(a.state.p_pos - l.state.p_pos))) for a in world.agents]
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
        rew = 0
        assert hasattr(world, "agent_landmark_mapping")
        landmark:Landmark = world.agent_landmark_mapping[agent]
        rew -= np.sqrt(np.sum(np.square(agent.state.p_pos - landmark.state.p_pos)))
        if agent.collide:
            for a in world.agents:
                if a != agent and self.is_collision(a, agent) :
                    rew -= 1
        return rew

    def observation(self, agent, world):
        """
        :param agent: an agent
        :param world: the current world
        :return: obs: [18] np array,
        [0-1] self_agent velocity
        [2-3] self_agent location
        [4-9] landmarks location
        [10-11] agent_i's relative location
        [12-13] agent_j's relative location
        Note that i < j
        """
        # get positions of all entities in this agent's reference frame
        entity_pos = []
        for entity in world.landmarks:  # world.entities:
            entity_pos.append(entity.state.p_pos - agent.state.p_pos)
        # entity colors
        entity_color = []
        for entity in world.landmarks:  # world.entities:
            entity_color.append(entity.color)
        # communication of all other agents
        other_pos = []
        landmark_pos = []
        for i, other in enumerate(world.agents):
            if other is agent: 
                agent_index = i
                continue
            other_pos.append(other.state.p_pos - agent.state.p_pos)
            landmark_pos.append(entity_pos[i])

        obs = np.concatenate([agent.state.p_vel] + [agent.state.p_pos] + [entity_pos[agent_index]] + landmark_pos + other_pos)
        return obs

    def seed(self, seed=None):
        self.np_rnd.seed(seed)
