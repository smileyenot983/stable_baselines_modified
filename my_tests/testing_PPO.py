import gym
import pybulletgym
import numpy as np
import torch as th
import argparse
# from stable_baselines3 import PPO
# from stable_baselines3.ppo import ppo as PPO
from stable_baselines3.ppo.ppo import PPO
import matplotlib.pyplot as plt


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


    # testing

    rewards_batch = []
    #making a post-evaluation
    for i in range(5):
        episode_rewards = 0
        obs = env.reset()
        for i in range(1000):
            action, _state = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            episode_rewards += reward
            # env.render()
            if done:
                break

        rewards_batch.append(episode_rewards)

    return rewards_batch

n_seeds = 5
env_name = "Pendulum-v0"
n_train_timesteps = 100000


test_noisy = []
test_unnoisy = []

#lists for average values of post evaluation after training
test_noisy_avg = []
test_unnoisy_avg = []

for i in range(n_seeds):
    noisy_reward = train_test(i,env_name,total_timesteps=n_train_timesteps,parameter_noise=True)
    unnoisy_reward = train_test(i,env_name,total_timesteps=n_train_timesteps,parameter_noise=False)
    test_noisy.append(noisy_reward)
    test_unnoisy.append(unnoisy_reward)

    test_noisy_avg.append(np.mean(noisy_reward))
    test_unnoisy_avg.append(np.mean(unnoisy_reward))


print('_______All results______')
print(test_noisy)
print(test_unnoisy)
print('_______Average results______')
print(test_noisy_avg)
print(test_unnoisy_avg)


#returns for pendulum:
# res_noisy = np.array([[-890.9602376261371, -1503.393886162202, -939.6161591689901, -1071.3334939731649, -1296.3587453755847], [-395.46389471923436, -1184.4693136807243, -417.51426307699023, -1192.4416277773282, -1121.9499253627928], [-1333.5232891879607, -1492.4545003967446, -1501.787835282463, -1346.2855398607057, -941.5154877644054], [-1515.638757274032, -1656.9181801652217, -1491.5183771506586, -1314.0380108964787, -948.8419345920822], [-886.0569501365346, -1243.5643372694324, -1103.2281846118376, -1500.435345394972, -1384.5753994117347]])
# res_unnoisy = np.array([[-1194.5171675632712, -1238.8782307105473, -1651.4555439277835, -1510.096286182288, -1644.2634651546098], [-1491.6256642730316, -1298.9438700757278, -1464.5447889761506, -1306.580586379228, -1215.5445062185427], [-1632.8190363130345, -1647.3783482440176, -1657.9494764537376, -1633.3224496352734, -1656.2139690726049], [-1581.9792511905457, -1656.9181801652217, -1575.2019220568593, -1571.934675452756, -1653.9713720644013], [-1555.5705896312502, -1243.5643372694324, -1103.2281846118376, -1500.435345394972, -1384.5753994117347]])

