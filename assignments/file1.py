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
print(os.getcwd())
os.chdir('/content/drive/MyDrive/gym-pybullet-drones/')
print(os.getcwd())
import gym_pybullet_drones
from gym_pybullet_drones import *
os.chdir(/content/drive/MyDrive/gym-pybullet-drones/gym_pybullet_drones/envs)
import BaseAviary
os.chdir(/content/drive/MyDrive/gym-pybullet-drones/gym_pybullet_drones/envs/single_agent_rl)
import BaseSingleAgentAviary
os.chdir(/content/drive/MyDrive/gym-pybullet-drones/gym_pybullet_drones/envs/multi_agent_rl)
import BaseMultiagentAviary
import NewLeaderAviary
os.chdir(/content/drive/MyDrive/gym-pybullet-drones/experiments/learning)
import multiagent2
import test_multiagent2

python3 multiagent2.py --env newleader --num_drones 2 --act rpm

