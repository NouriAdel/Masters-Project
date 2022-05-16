import os
import time
import random
import numpy as np
import matplotlib.pyplot as plt
import pybullet_envs
import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
from gym import wrappers
from torch.autograd import Variable
from collections import deque

from TD3 import ReplayBuffer, Actor, Critic, TD3
#from gym_pybullet_drones.envs.single_agent_rl.TakeoffAviary import TakeoffAviary
from gym_pybullet_drones.utils.Logger import Logger
from gym_pybullet_drones.utils.utils import sync, str2bool
from NewAviary import NewAviary


def evaluate_policy(policy, eval_episodes=50):
  avg_reward = 0.
  #env.INIT_XYZS = [-0.1,-0.1,0.1]
  #desired = [[0.1,-0.1,0.1], [0.1,0.1,0.1], [-0.1,0.1,0.1], [-0.1,-0.1,0.1]]
  #i = 0
  j = 0
  #r = eval_episodes/4
  #env.desired_pos = desired[0]
  #desired = [0.1,-0.1,0.1]
  env.INIT_XYZS[0] = [-0.1,-0.1,0.4]
  env.desired_pos = [-0.1,-0.1,0.4]
  for _ in range(eval_episodes):
    #env.desired_pos = env._generate_desired_pos()[0]
    #env.INIT_XYZS = env._generate_init_pos()
    obs = env.reset()
    
    '''
    if j==0:
       #env.INIT_XYZS[0] = [-0.1,-0.1,0.1]
       env.desired_pos = [0.1,-0.1,0.4]
    elif j==15:
       #env.INIT_XYZS[0] = [0.1,-0.1,0.1]
       env.desired_pos = [0.1,0.1,0.4]
    elif j==30:
       #env.INIT_XYZS[0] = [0.1,0.1,0.1]
       env.desired_pos = [-0.1,0.1,0.4]
    elif j==45:
       #env.INIT_XYZS[0] = [-0.1,0.1,0.1]
       env.desired_pos = [-0.1,-0.1,0.4]
    #else: 
       #do nothing
    '''
    done = False
    #i = i + 1
    j = j + 1
    while not done:
      action = policy.select_action(np.array(obs))
      obs, reward, done, _ = env.step(action)
      avg_reward += reward
  avg_reward /= eval_episodes
  print ("---------------------------------------")
  print ("Average Reward over the Evaluation Step: %f" % (avg_reward))
  print ("---------------------------------------")
  return avg_reward

#env_name = "HalfCheetahBulletEnv-v0"
env_name = "NewAviary"
seed = 0

file_name = "%s_%s_%s" % ("TD3", env_name, str(seed))
print ("---------------------------------------")
print ("Settings: %s" % (file_name))
print ("---------------------------------------")

eval_episodes = 1
save_env_vid = True
#env = gym.make(env_name)
env = NewAviary(freq = 240, gui=True, record=False, box_bound = [0.6, 0.6, 0.6], hover = False)
max_episode_steps = env._max_episode_steps

logger = Logger(logging_freq_hz=int(env.SIM_FREQ/env.AGGR_PHY_STEPS),
                num_drones=1
                )


'''if save_env_vid:
  env = wrappers.Monitor(env, monitor_dir, force = True)
  env.reset()'''
start = time.time()
env.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
max_action = float(env.action_space.high[0])
policy = TD3(state_dim, action_dim, max_action)
policy.load(file_name, './pytorch_models/')
_ = evaluate_policy(policy, eval_episodes=eval_episodes)

"""for i in range(8*int(env.SIM_FREQ/env.AGGR_PHY_STEPS)): # Up to 6''
        action, _states = model.predict(obs,
                                        deterministic=True # OPTIONAL 'deterministic=False'
                                        )
        obs, reward, done, info = env.step(action)
        env.render()
        if OBS==ObservationType.KIN:
            logger.log(drone=0,
                       timestamp=i/env.SIM_FREQ,
                       state= np.hstack([obs[0:3], np.zeros(4), obs[3:15],  np.resize(action, (4))]),
                       control=np.zeros(12)
                       )"""
#env.desired_pos = env._generate_desired_pos()[0]
#env.INIT_XYZS = env._generate_init_pos()

env.INIT_XYZS[0] = [-0.1,-0.1,0.4]
desired = [[0.1,-0.1,0.4], [0.1,0.1,0.4], [-0.1,0.1,0.4], [-0.1,-0.1,0.4]]
j = 0
#env.desired_pos = [0.1,-0.1,0.4]
#env.desired_pos = [0.1,0.1,0.4]
obs = env.reset()

for i in range(50*env.SIM_FREQ):
    action = policy.select_action(np.array(obs))
    obs, reward, done, info = env.step(action)
    #env.desired_pos = [-0.1,-0.1,0.4]
    if i % (30*env.SIM_FREQ/4) == 0:
       env.desired_pos = desired[j]
       j = j + 1

    logger.log(drone=0,
               timestamp=i/env.SIM_FREQ,
               state=np.hstack([obs[0:3], np.zeros(4), obs[3:15],  np.resize(action, (4))]),
               control=np.zeros(12)
               )
    if i%env.SIM_FREQ == 0:
        env.render()
        print(done)
    sync(i, start, env.TIMESTEP)
    #if done:
        #obs = env.reset()

env.close()
#logger.save_as_csv("sa")
logger.plot()
