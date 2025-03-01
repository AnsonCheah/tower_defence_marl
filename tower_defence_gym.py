import numpy as np
import gymnasium as gym
from pettingzoo import ParallelEnv
from gymnasium import spaces
import pygame
import logging
from pettingzoo.utils.env import ParallelEnv

logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")
class DefenderEnv(ParallelEnv):
    metadata = {"render_modes": ["human", "rgb_array"], "is_parallelizable": True}
    def __init__(self, num_defenders=3, num_attackers=3, map_size=(10, 10), render_mode=None):
        super().__init__()
        self.num_defenders = num_defenders
        self.num_attackers = num_attackers
        self.map_size = map_size
        self.render_mode = render_mode
        self.agents = [f"defender_{i}" for i in range(num_defenders)]
        self.max_speed = 1.0
        self.tower_position = (map_size[0]/2, map_size[1]/2)
        self.initialize_variables()
        if self.render_mode:
            self.window_size = 500  # Pixels
            self.window = None
            self.clock = None

        # Observation: positions of all agents
        # dxy, vxy, axy for each for each defender and attacker
        # Add observation size
        obs_size = (num_defenders + num_attackers) * 2
        self.observation_spaces = {agent: spaces.Box(low=0, high=map_size[0], shape=(obs_size,), dtype=np.float32) for agent in self.agents}
        self.action_spaces = {agent: spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32) for agent in self.agents}

    def reset(self, seed=None, options=None):
        self.initialize_variables()
        observations = {agent: self._get_obs(agent) for agent in self.agents}
        return observations

    def initialize_variables(self):
        self.defenders_pos = {f"defender_{i}": np.random.uniform(self.tower_position[0]-self.map_size[0]/2, self.tower_position[0] + self.map_size[0]/2, size=2) for i in range(self.num_defenders)}
        self.attackers = {f"attacker_{i}": np.random.uniform(0, self.map_size[0], size=2) for i in range(self.num_attackers)}
        
    def _get_obs(self, agent):
        # Add different observation calculations like displacement, velocities, accelerations, time, etc
        obs = []
        for pos in self.defenders_pos.values():
            obs.extend(pos)
        for pos in self.attackers.values():
            obs.extend(pos)
        return np.array(obs, dtype=np.float32)

    def calculate_rewards():
        # Add more rewards like neutralizing, approaching, surrounding time
        # Add more penalties like idle cost, 
        pass

    def step(self, actions):
        rewards = {agent: -0.1 for agent in self.agents}

        for agent, action in actions.items():
            dx, dy = np.clip(action, -1, 1)  # Ensure actions stay within bounds
            move_vector = np.array([dx, dy])
            move_vector = (move_vector / np.linalg.norm(move_vector)) * self.max_speed if np.linalg.norm(move_vector) > 0 else np.array([0, 0])
            
            self.defenders_pos[agent] = np.clip(self.defenders_pos[agent] + move_vector, 0, self.map_size[0])

        # Add attacker trajectories
        # Check if attackers are intercepted
        for attacker, att_pos in self.attackers.items():
            for agent, def_pos in self.defenders_pos.items():
                if np.linalg.norm(def_pos - att_pos) < 1.0:  # Within capture range
                    rewards[agent] += 1  

        observations = {agent: self._get_obs(agent) for agent in self.agents}
        dones = {agent: False for agent in self.agents}
        infos = {agent: {} for agent in self.agents}
        logging.debug(observations)
        logging.debug(dones)
        logging.debug(infos)
        if self.render_mode == "human":
            self.render()
        return observations, rewards, dones, infos

    def render(self):
        if self.window is None:
            pygame.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
            self.clock = pygame.time.Clock()

        self.window.fill((255, 255, 255))  # White background
        def scale_pos(pos):
            return (int(pos[0] / self.map_size[0] * self.window_size),
                    int(pos[1] / self.map_size[1] * self.window_size))
        
        # Draw Tower
        pygame.draw.circle(self.window, (0, 255, 0), scale_pos(self.tower_position), 20)
        # Draw defenders (blue)
        for pos in self.defenders_pos.values():
            pygame.draw.circle(self.window, (0, 0, 255), scale_pos(pos), 8)

        # Draw attackers (red)
        for pos in self.attackers.values():
            pygame.draw.circle(self.window, (255, 0, 0), scale_pos(pos), 8)

        pygame.display.flip()
        self.clock.tick(30)  # Limit FPS

    def close(self):
        if self.window is not None:
            pygame.quit()
            self.window = None
            self.clock = None

if __name__ == "__main__":
    env = DefenderEnv(render_mode="human")
    for _ in range(100):  # Simulate 100 steps
        actions = {agent: np.random.uniform(-1, 1, size=2) for agent in env.agents}
        env.step(actions)
        env.render()

    env.close()

