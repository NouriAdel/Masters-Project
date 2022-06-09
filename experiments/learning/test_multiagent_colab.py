"""Test script for multiagent problems.

This scripts runs the best model found by one of the executions of `multiagent.py`

Example
-------
To run the script, type in a terminal:

    $ python test_multiagent.py --exp ./results/save-<env>-<num_drones>-<algo>-<obs>-<act>-<time_date>

"""
import os
import time
import argparse
from datetime import datetime
import pdb
import math
import numpy as np
import pybullet as p
import pickle
import matplotlib.pyplot as plt
import gym
from gym import error, spaces, utils
from gym.utils import seeding
from gym.spaces import Box, Dict
import torch
import torch.nn as nn
import ray
from ray import tune
from ray.tune import register_env
from ray.rllib.agents import ppo
from ray.rllib.agents.ppo import PPOTrainer, PPOTFPolicy
from ray.rllib.examples.policy.random_policy import RandomPolicy
from ray.rllib.utils.test_utils import check_learning_achieved
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.agents.callbacks import DefaultCallbacks
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork
from ray.rllib.models import ModelCatalog
from ray.rllib.policy.sample_batch import SampleBatch

from gym_pybullet_drones.envs.BaseAviary import DroneModel, Physics
from gym_pybullet_drones.envs.multi_agent_rl.FlockAviary import FlockAviary
from gym_pybullet_drones.envs.multi_agent_rl.LeaderFollowerAviary import LeaderFollowerAviary
from gym_pybullet_drones.envs.multi_agent_rl.MeetupAviary import MeetupAviary
from gym_pybullet_drones.envs.single_agent_rl.BaseSingleAgentAviary import ActionType, ObservationType
from gym_pybullet_drones.utils.Logger import Logger
from gym_pybullet_drones.utils.utils import sync

#from ray.rllib.agents import ddpg ##Nouran
#from ray.rllib.agents.ddpg.ddpg_tf_policy import DDPGTFPolicy  ##Nouran
#from ray.rllib.agents.ddpg.td3 import TD3Trainer  ##Nouran

#import td3_model
#import neural_network_AC_model
#from td3_model import Agent
from collections import deque
import td3
from td3 import ReplayBuffer, Actor, Critic, TD3

import shared_constants

OWN_OBS_VEC_SIZE = None # Modified at runtime
ACTION_VEC_SIZE = None # Modified at runtime

def evaluate_policy(policy, eval_episodes=10):
  num_agents = 2  
  avg_reward = np.zeros(num_agents)
  for _ in range(eval_episodes):
    obs = env.reset()
    #obs_ = []
    #for i in range (len(obs)):
      #obs_.append(obs[i])
    done = False
    #actions = np.zeros(num_agents)
    while not done:
      action = policy.select_action([np.array(obs[0]),np.array(obs[1])])  
      #action = policy.select_action(np.array(obs_))
      #print("action", np.array(action))
      #action = np.array(action)
      #actions = []
      #actions.append(action[0])
      #actions.append(action[1])
      #actions = np.array(actions).flatten()
      #print("actions",actions)
      actions = {0: np.array(action[0]), 1:np.array(action[1])}
      obs, reward, done, _ = env.step(actions)
      #avg_reward += reward
      avg_reward[0] += reward[0]
      avg_reward[1] += reward[1]
  #avg_reward /= eval_episodes
  avg_reward_ = np.mean(avg_reward)
  print ("---------------------------------------")
  print ("Average Reward over the Evaluation Step: %f" % (avg_reward_))
  print ("---------------------------------------")
  return avg_reward_

'''
def evaluate_policy(policy, eval_episodes=50):
  num_agents = 2
  avg_reward = np.zeros(num_agents)
  for _ in range(eval_episodes):
    obs = env.reset()
    #env.desired_pos = env._generate_desired_pos()[0]
    #env.INIT_XYZS = env._generate_init_pos()
    done = False
    actions = np.zeros(num_agents)
    while not done:
      action = policy.select_action(np.array(obs))
      obs, reward, done, _ = env.step(action)
      avg_reward += reward
  avg_reward /= eval_episodes
  print ("---------------------------------------")
  print ("Average Reward over the Evaluation Step: %f" % (avg_reward))
  print ("---------------------------------------")
  return avg_reward
'''

############################################################
if __name__ == "__main__":

    #### Define and parse (optional) arguments for the script ##
    parser = argparse.ArgumentParser(description='Multi-agent reinforcement learning experiments script')
    parser.add_argument('--exp',    type=str,       help='The experiment folder written as ./results/save-<env>-<num_drones>-<algo>-<obs>-<act>-<time_date>', metavar='')
    ARGS = parser.parse_args()

    #### Parameters to recreate the environment ################
    
    NUM_DRONES = 2
    OBS = ObservationType.KIN
    ACT = ActionType.RPM

    #### Constants, and errors #################################
    if OBS == ObservationType.KIN:
        OWN_OBS_VEC_SIZE = 12
    elif OBS == ObservationType.RGB:
        print("[ERROR] ObservationType.RGB for multi-agent systems not yet implemented")
        exit()
    else:
        print("[ERROR] unknown ObservationType")
        exit()
    if ACT in [ActionType.ONE_D_RPM, ActionType.ONE_D_DYN, ActionType.ONE_D_PID]:
        ACTION_VEC_SIZE = 1
    elif ACT in [ActionType.RPM, ActionType.DYN, ActionType.VEL]:
        ACTION_VEC_SIZE = 4
    elif ACT == ActionType.PID:
        ACTION_VEC_SIZE = 3
    else:
        print("[ERROR] unknown ActionType")
        exit()

 
    #### Unused env to extract the act and obs spaces ##########
    
    env = LeaderFollowerAviary(num_drones=NUM_DRONES,
                               aggregate_phy_steps=shared_constants.AGGR_PHY_STEPS,
                               obs=OBS,
                               act=ACT,
                               gui=True,
                               record=False
                               )

    observer_space = env.observation_space[0]
    action_space = env.action_space[0]
    num_agents = NUM_DRONES
    action_size = ACTION_VEC_SIZE
    state_size = OWN_OBS_VEC_SIZE
    state_dim = env.observation_space[0].shape[0]
    action_dim = env.action_space[0].shape[0]
    max_action = float(env.action_space[0].high[0])
    #### Set up the trainer's config ###########################
    
    #### Set up the model parameters of the trainer's config ###
    
    #### Set up the multiagent params of the trainer's config ##
    
    env_name = "LeaderFollowerAviary"
    seed = 0

    file_name = "%s_%s_%s" % ("TD3", env_name, str(seed))
    print ("---------------------------------------")
    print ("Settings: %s" % (file_name))
    print ("---------------------------------------")

    
    #### Restore agent #########################################

    policy = TD3(state_dim, action_dim, max_action)
    policy.load(file_name, './pytorch_models/')

    eval_episodes = 10
    _ = evaluate_policy(policy, eval_episodes=eval_episodes)

    #### Extract and print policies ############################

    #### Create test environment ###############################

    
    #### Show, record a video, and log the model's performance #
    obs = env.reset()
    logger = Logger(logging_freq_hz=int(env.SIM_FREQ/env.AGGR_PHY_STEPS),
                    num_drones=NUM_DRONES
                    )
    if ACT in [ActionType.ONE_D_RPM, ActionType.ONE_D_DYN, ActionType.ONE_D_PID]:
        actions = {i: np.array([0]) for i in range(NUM_DRONES)}
    elif ACT in [ActionType.RPM, ActionType.DYN, ActionType.VEL]:
        actions = {i: np.array([0, 0, 0, 0]) for i in range(NUM_DRONES)}
    elif ACT==ActionType.PID:
         actions = {i: np.array([0, 0, 0]) for i in range(NUM_DRONES)}
    else:
        print("[ERROR] unknown ActionType")
        exit()
    start = time.time()
    env.new_goal = [0,0,0.5]
    for i in range(6*int(env.SIM_FREQ/env.AGGR_PHY_STEPS)): # Up to 6''
        #### Deploy the policies ###################################
        #actions = np.zeros(num_agents)
        #for n in range (num_agents):
            #actions[n] = agents[n].select_action(np.array(obs[n]))
        actions = policy.select_action([np.array(obs[0]),np.array(obs[1])])
        actions = {0: actions[0], 1: actions[1]}
        #print ("action", action)
        #action = {0: np.array([1]), 1: np.array([0])}
        obs, reward, done, info = env.step(actions)
        #print ("obs", obs)
        #test_env.render()
        if OBS==ObservationType.KIN: 
            for j in range(NUM_DRONES):
                logger.log(drone=j,
                           timestamp=i/env.SIM_FREQ,
                           state= np.hstack([obs[j][0:3], np.zeros(4), obs[j][3:15], np.resize(actions[j], (4))]),
                           control=np.zeros(12)
                           )
        sync(np.floor(i*env.AGGR_PHY_STEPS), start, env.TIMESTEP)
        # if done["__all__"]: obs = test_env.reset() # OPTIONAL EPISODE HALT
    env.close()
    logger.save_as_csv("ma") # Optional CSV save
    logger.plot()


