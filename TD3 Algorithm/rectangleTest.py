#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  2 11:29:01 2021

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

from td3 import ReplayBuffer, Actor, Critic, TD3
from gym_pybullet_drones.envs.single_agent_rl.TakeoffAviary import TakeoffAviary
from gym_pybullet_drones.utils.Logger import Logger
from gym_pybullet_drones.utils.utils import sync, str2bool
from DroneAviary import DroneAviary

def creatRec(base,points,inc):
    rec = base
    for i in range(points):
        base[0,0] = base[0,0] + inc
        rec = np.append(rec,base,axis=0)
    for i in range(points):
        base[0,1] = base[0,1] + inc
        rec = np.append(rec,base,axis=0)
    for i in range(points):
        base[0,0] = base[0,0] - inc
        rec = np.append(rec,base,axis=0)
    for i in range(points):
        base[0,1] = base[0,1] - inc
        rec = np.append(rec,base,axis=0)
    return rec[1:]

  

env = DroneAviary(gui=True,
                    record=False
                    )

seed = 0
env_name = "drone-aviary-v0"
#file_name = "%s_%s_%s" % ("TD3", env_name, str(0))
file_name = "%s_%s_%s" % ("TD3", env_name, str(seed))
print ("---------------------------------------")
print ("Settings: %s" % (file_name))
print ("---------------------------------------")

bound = np.array([0.5,-0.5,0.5,-0.5,1,0])
#bound = bound * 2
env.updateBound(bound)
env.randDes()
env.randInit()
obs = env.reset()
start = time.time()
env.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
max_action = float(env.action_space.high[0])



policy = TD3(state_dim, action_dim, max_action)
policy.load(file_name, './pytorch_models/')

base = np.array([[-0.2,-0.2,0.5]])
points = 20
inc = 0.2
Trajectory = creatRec(base, points, inc)
env.setInit(base)

obs = env.reset()
done = False
for i in range(Trajectory.shape[0]):
    des = Trajectory[i]
    env.updateDes(np.array([0.2,0.2,0.5]))
    action = policy.select_action(np.array(obs))
    obs, reward, done, _ = env.step(action)