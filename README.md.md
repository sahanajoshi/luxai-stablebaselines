# LUX AI season 1 interface with stable-baselines PPO

## Objective
Environments in kaggle challenges are generally supported by [kaggle-environment](https://github.com/Kaggle/kaggle-environments), but the kaggle APIs cannot be easily interfaced with stable-baselines3's Policy Gradient algorithms. This project creates a gym wrapper environment which can be interfaced with stable-baselines3's PPO algorithm. The wrapper environment internally initializes and maintains the LUX AI game.

## Files
- stable_baselinesagent.py - contains the wrapper env description
- ppo.py - initializes the game with the environment defined in stable_baselinesagent.py and trains an agent with PPO environment.
