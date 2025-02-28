import os
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from tower_defence_gym import DefenderEnv
# Convert PettingZoo env to Gym-compatible
from pettingzoo.utils.wrappers import BaseParallelWrapper

CHECKPOINT_DIR = "saves/"
MODEL_NAME = "sample_sb_model.zip"  # Change this if you need to load different model
MODEL_PATH = os.path.join(CHECKPOINT_DIR, MODEL_NAME)

gym_env = BaseParallelWrapper(DefenderEnv())

# Train using PPO
model = PPO("MlpPolicy", gym_env, verbose=1)
model.learn(total_timesteps=100_000)

# Save and test
model.save("defender_policy")




