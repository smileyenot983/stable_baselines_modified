import gym
import pybulletgym
import numpy as np
import torch as th
import argparse
# from stable_baselines3 import PPO
# from stable_baselines3.ppo import ppo as PPO
from stable_baselines3.ppo.ppo import PPO
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='Parameter noise with stable baselines')
parser.add_argument('--env-name', type=str, default='Pendulum-v0')
parser.add_argument('--n-train-timesteps', type=int, default=1000000)
parser.add_argument('--n-seeds', type=int, default=5)

args = parser.parse_args()
'''Observations on stable-baselines:

    1. discrete action space - categorical disrtribution
       continuous action space - DiagGaussian distribution

    2. there are 2 neural networks: -feature extractor MLp
                                    -output nn.Linear

        -feature extractor consists of 2 networks: policy and value with architecture: 
            Linear(28,64)->Tanh()->Linear(64,64)->Tanh()
            
        -output = nn.Linear(64,n_actions)

        I am not adding noise to parameters of output layer, that's why i should only modify extractor Mlp

'''





def train_test(seed,env_name,total_timesteps,parameter_noise):
    env = gym.make(env_name)

    env.seed(seed)
    th.manual_seed(seed)

    # training
    model = PPO('MlpPolicy', env, verbose=0)
    model.learn(total_timesteps=total_timesteps,parameter_noise=parameter_noise)

    test_rewards = 0
    # testing
    obs = env.reset()
    for i in range(1000):
        action, _state = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        test_rewards += reward
        # env.render()
        if done:
            break

    return test_rewards

n_seeds = 5
# env_name = "Pendulum-v0"
# train_steps = 1000000

test_noisy = []
test_unnoisy = []
for i in range(n_seeds):
    noisy_reward = train_test(i,args.env_name,total_timesteps=args.n_train_timesteps,parameter_noise=True)
    unnoisy_reward = train_test(i,args.env_name,total_timesteps=args.n_train_timesteps,parameter_noise=False)
    test_noisy.append(noisy_reward)
    test_unnoisy.append(unnoisy_reward)

print(test_noisy)
print(test_unnoisy)

