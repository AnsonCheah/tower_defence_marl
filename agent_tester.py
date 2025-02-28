import time
import os
from tower_defence_gym import DefenderEnv
from stable_baselines3 import PPO
import pygame

CHECKPOINT_DIR = "saves/"
MODEL_NAME = "sample_sb_model.zip"  # Change this if you need to load different model
MODEL_PATH = os.path.join(CHECKPOINT_DIR, MODEL_NAME)
model = PPO.load(MODEL_PATH)

env = DefenderEnv()
obs = env.reset()
for _ in range(100):
    actions = {agent: model.predict(obs[agent], deterministic=True)[0] for agent in env.agents}
    obs, rewards, dones, infos = env.step(actions)
    time.sleep(0.1)  # For visualization


class DefenderVisualizer:
    def __init__(self, env):
        pygame.init()
        self.env = env
        self.window = pygame.display.set_mode((500, 500))
        self.running = True

    def render(self):
        self.window.fill((255, 255, 255))
        for agent, pos in self.env.pos.items():
            pygame.draw.circle(self.window, (0, 0, 255), (pos[0] * 50, pos[1] * 50), 10)  # Defender
        for attacker, pos in self.env.attackers.items():
            pygame.draw.circle(self.window, (255, 0, 0), (pos[0] * 50, pos[1] * 50), 10)  # Attacker
        pygame.display.flip()

    def run(self):
        obs = self.env.reset()
        while self.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False

            actions = {agent: model.predict(obs[agent], deterministic=True)[0] for agent in self.env.agents}
            obs, rewards, dones, infos = self.env.step(actions)
            self.render()
            pygame.time.delay(100)

        pygame.quit()

visualizer = DefenderVisualizer(env)
visualizer.run()