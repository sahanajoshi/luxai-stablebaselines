import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

from gymnasium.envs.registration import register

register(
    id='LuxAI-v0',
    entry_point='stable_baselinesagent:LuxAIEnv',
)

env = gym.make('LuxAI-v0')

observation = env.reset()

model = PPO("MultiInputPolicy", env, verbose=1, )

model.learn(total_timesteps=10)
