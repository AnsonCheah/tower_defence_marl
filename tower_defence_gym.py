import numpy as np
import gymnasium as gym
from pettingzoo import ParallelEnv
from gymnasium import spaces

class DefenderEnv(ParallelEnv):
    def __init__(self, num_defenders=3, num_attackers=3, map_size=(10, 10)):
        self.num_defenders = num_defenders
        self.num_attackers = num_attackers
        self.map_size = map_size
        self.agents = [f"defender_{i}" for i in range(num_defenders)]
        self.pos = {agent: np.random.randint(0, map_size[0], size=2) for agent in self.agents}
        self.attackers = {f"attacker_{i}": np.random.randint(0, map_size[0], size=2) for i in range(num_attackers)}

        # Observation: positions of all agents
        obs_size = (num_defenders + num_attackers) * 2  # x, y for each agent
        self.observation_spaces = {agent: spaces.Box(low=0, high=map_size[0], shape=(obs_size,), dtype=np.float32) for agent in self.agents}
        self.action_spaces = {agent: spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32) for agent in self.agents}

    def reset(self, seed=None):
        self.pos = {agent: np.random.uniform(0, self.map_size[0], size=2) for agent in self.agents}
        self.attackers = {f"attacker_{i}": np.random.uniform(0, self.map_size[0], size=2) for i in range(self.num_attackers)}
        observations = {agent: self._get_obs(agent) for agent in self.agents}
        return observations

    def _get_obs(self, agent):
        obs = []
        for pos in self.pos.values():
            obs.extend(pos)
        for pos in self.attackers.values():
            obs.extend(pos)
        return np.array(obs, dtype=np.float32)

    def step(self, actions):
        rewards = {agent: -0.1 for agent in self.agents}

        for agent, action in actions.items():
            dx, dy = np.clip(action, -1, 1)  # Ensure actions stay within bounds
            move_vector = np.array([dx, dy])
            move_vector = (move_vector / np.linalg.norm(move_vector)) * self.max_speed if np.linalg.norm(move_vector) > 0 else np.array([0, 0])
            
            self.pos[agent] = np.clip(self.pos[agent] + move_vector, 0, self.map_size[0])

        # Check if attackers are intercepted
        for attacker, att_pos in self.attackers.items():
            for agent, def_pos in self.pos.items():
                if np.linalg.norm(def_pos - att_pos) < 1.5:  # Within capture range
                    rewards[agent] += 1  

        observations = {agent: self._get_obs(agent) for agent in self.agents}
        dones = {agent: False for agent in self.agents}
        infos = {agent: {} for agent in self.agents}

        return observations, rewards, dones, infos

if __name__ == "__main__":
    env = DefenderEnv()
