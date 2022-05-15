#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug  8 12:15:55 2021

@author: mokhtar
"""

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
from gym_pybullet_drones.envs.single_agent_rl.TakeoffAviary import TakeoffAviary
from gym_pybullet_drones.utils.Logger import Logger
from gym_pybullet_drones.utils.utils import sync, str2bool
#from DroneAviary import DroneAviary
#from NewHoverAviary import NewHoverAviary
from NewAviary import NewAviary

"""**Inference**"""
def mkdir(base, name):
    path = os.path.join(base, name)
    if not os.path.exists(path):
        os.makedirs(path)
    return path
  
def evaluate_policy(policy, eval_episodes=50):
  avg_reward = 0.
  for _ in range(eval_episodes):
    #env.randDes()
    #env.randInit() 
    #env.INIT_XYZS=[0,0,0.1125]
    #env.desired_pos=[0,0,0.38] 
    obs = env.reset()
    done = False
    while not done:
      action = policy.select_action(np.array(obs))
      obs, reward, done, _ = env.step(action)
      avg_reward += reward
  avg_reward /= eval_episodes
  print ("---------------------------------------")
  print ("Average Reward over the Evaluation Step: %f" % (avg_reward))
  print ("---------------------------------------")
  return avg_reward

#### Show (and record a video of) the model's performance ##
#env = DroneAviary(gui=True,
                    #record=False
                    #)
env = NewAviary(gui=True,
                    record=False,
                    freq=50
                    )

logger = Logger(logging_freq_hz=int(env.SIM_FREQ/env.AGGR_PHY_STEPS),
                num_drones=1
                )

seed = 0
#env_name = "drone-aviary-v0"
env_name = "NewAviary"
file_name = "%s_%s_%s" % ("TD3", env_name, str(0))
#file_name = "%s_%s_%s" % ("TD3", env_name, str(seed))
print ("---------------------------------------")
print ("Settings: %s" % (file_name))
print ("---------------------------------------")

#bound = np.array([0.5,-0.5,0.5,-0.5,1,0])
##bound = np.array([1,-1,1,-1,2,0])
#bound = bound * 2
##env.updateBound(bound)
#env.updateDes(np.array([0,0,0.3]))
##env.randDes()
##env.randInit()
#print(env.INIT_XYZS)
#print(env.desired_pos)
obs = env.reset()
start = time.time()
env.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)
#%%


state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
max_action = float(env.action_space.high[0])

policy = TD3(state_dim, action_dim, max_action)
policy.load(file_name, directory='./pytorch_models/')
evaluate_policy(policy,eval_episodes=(10))
env.close()


for i in range(10*env.SIM_FREQ):
    action = policy.select_action(np.array(obs))
    obs, reward, done, info = env.step(action)
    logger.log(drone=0,
               timestamp=i/env.SIM_FREQ,
               state=np.hstack([obs[0:3], np.zeros(4), obs[3:15],  np.resize(action, (4))]),
               control=np.zeros(12)
               )
    if i%env.SIM_FREQ == 0:
        env.render()
        print(done)
    sync(i, start, env.TIMESTEP)
    if done:
        obs = env.reset()
env.close()
logger.plot()


"""
env_name = "Walker2DBulletEnv-v0"


file_name = "%s_%s_%s" % ("TD3", env_name, str(seed))
print ("---------------------------------------")
print ("Settings: %s" % (file_name))
print ("---------------------------------------")

eval_episodes = 10
save_env_vid = True
env = gym.make(env_name)
max_episode_steps = env._max_episode_steps
work_dir = mkdir('exp', 'brs')
monitor_dir = mkdir(work_dir, 'monitor')
if save_env_vid:
  env = wrappers.Monitor(env, monitor_dir, force = True)
  env.reset()

_ = evaluate_policy(policy, eval_episodes=eval_episodes)
"""
