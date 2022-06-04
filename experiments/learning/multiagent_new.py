import os
import time
import argparse
from datetime import datetime
import subprocess
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
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork
import ray
from ray import tune
from ray.tune.logger import DEFAULT_LOGGERS
from ray.tune import register_env
from ray.rllib.agents import ppo
from ray.rllib.agents.ppo import PPOTrainer, PPOTFPolicy
from ray.rllib.examples.policy.random_policy import RandomPolicy
from ray.rllib.utils.test_utils import check_learning_achieved
from ray.rllib.agents.callbacks import DefaultCallbacks
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models import ModelCatalog
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.env.multi_agent_env import ENV_STATE

from gym_pybullet_drones.envs.BaseAviary import DroneModel, Physics
from gym_pybullet_drones.envs.multi_agent_rl.FlockAviary import FlockAviary
from gym_pybullet_drones.envs.multi_agent_rl.LeaderFollowerAviary import LeaderFollowerAviary
from gym_pybullet_drones.envs.multi_agent_rl.MeetupAviary import MeetupAviary
from gym_pybullet_drones.envs.single_agent_rl.BaseSingleAgentAviary import ActionType, ObservationType
from gym_pybullet_drones.utils.Logger import Logger

import td3_model
import neural_network_AC_model
from td3_model import Agent
from collections import deque

import shared_constants

OWN_OBS_VEC_SIZE = None # Modified at runtime
ACTION_VEC_SIZE = None # Modified at runtime

############################################################

def td3(n_episodes=700, max_t=500, consecutive_episodes = 100, 
        learn_every = 10, n_experiences = 20, noise = 2.0, noise_grad = 0.999,
        num_agents = 2, print_every=5):
    """Deep Deterministic Policy Gradients DDPG  from Pendulum exercise was employed: 
       Params:
            n_episodes (int): maximum number of training episodes
            max_t (int): maximum number of timesteps per episode
            consecutive_episodes (int): number of consecutive episodes
            learn_every (int): update the network every n timesteps
            n_experiences (int): update the netword n times
            target_avg_score (int): target avg score of the range of n consecutive_episodes
            noise = (float): noise added to actions
            noise_grad = noise adjestment per episode
            num_agents = number of agents
            print_every (int): print the training every n timesteps
    """
    total_rewards = [] 
    rewards_windows = deque(maxlen=consecutive_episodes) 
    save = True
    for i_episode in range(1, n_episodes+1):
        states = env.reset()
        agents_rewards = np.zeros(num_agents)
        noise_level= noise
        noise_gradient = noise_grad
        
        for t in range(max_t):
            actions = []
            for i in range(len(states)):
                actions.append(agents[i].act(states[i], noise_reduction= noise_level))
            actions = {0: np.array(actions[0]), 1:np.array(actions[1])}
            noise_level*=noise_grad
            noise_level = max(noise_level, 0)
            next_states, rewards, dones, _ = env.step(actions)

            for i in range(num_agents):
                agents[i].step(states[i], actions[i], rewards[i], next_states[i], dones[i],
                               t, learn_every, n_experiences)

            states = next_states
            print("########### before #########")
            print(rewards)
            data = rewards.items()
            data = list(data)
            print(rewards[0])
            data = [data[0][1], data[1][1]]
            rewards = np.array(data)
            print("########### After #########")
            print(rewards)

            agents_rewards += rewards

        rewards_windows.append(np.max(agents_rewards))
        total_rewards.append(agents_rewards)
        average_reward = np.mean(rewards_windows)
        

        if i_episode % print_every == 0:
            print('\n\rEpisode {}\tAverage Reward: {:.4f}\tReward: {:.4f}\n'.format(i_episode, 
                                                                              average_reward, np.max(agents_rewards)))
 
    if save:
        print('\nEnvironment solved in {:d} episodes!\tAverage Reward: {:.5f}'.format(i_episode, average_reward))
        for i in range(num_agents):
                torch.save(agents[i].actor_local.state_dict(),  'td3_actor{}.pth'.format(i))
                torch.save(agents[i].critic_local.state_dict(), 'td3_critic{}.pth'.format(i))
        save = False   
    return total_rewards



if __name__ == "__main__":

    #### Define and parse (optional) arguments for the script ##
    parser = argparse.ArgumentParser(description='Multi-agent reinforcement learning experiments script')
    parser.add_argument('--num_drones',  default=2,                 type=int,                                                                 help='Number of drones (default: 2)', metavar='')
    parser.add_argument('--env',         default='leaderfollower',  type=str,             choices=['leaderfollower', 'flock', 'meetup'],      help='Task (default: leaderfollower)', metavar='')
    parser.add_argument('--obs',         default='kin',             type=ObservationType,                                                     help='Observation space (default: kin)', metavar='')
    parser.add_argument('--act',         default='one_d_rpm',       type=ActionType,                                                          help='Action space (default: one_d_rpm)', metavar='')
    parser.add_argument('--algo',        default='td3',              type=str,             choices=['td3'],                                     help='MARL approach (default: cc)', metavar='')
    parser.add_argument('--workers',     default=0,                 type=int,                                                                 help='Number of RLlib workers (default: 0)', metavar='')        
    ARGS = parser.parse_args()

    #### Save directory ########################################
    filename = os.path.dirname(os.path.abspath(__file__))+'/results/save-'+ARGS.env+'-'+str(ARGS.num_drones)+'-'+ARGS.algo+'-'+ARGS.obs.value+'-'+ARGS.act.value+'-'+datetime.now().strftime("%m.%d.%Y_%H.%M.%S")
    if not os.path.exists(filename):
        os.makedirs(filename+'/')

    #### Print out current git commit hash #####################
    git_commit = subprocess.check_output(["git", "describe", "--tags"]).strip()
    with open(filename+'/git_commit.txt', 'w+') as f:
        f.write(str(git_commit))

    #### Constants, and errors #################################
    if ARGS.obs==ObservationType.KIN:
        OWN_OBS_VEC_SIZE = 12
    elif ARGS.obs==ObservationType.RGB:
        print("[ERROR] ObservationType.RGB for multi-agent systems not yet implemented")
        exit()
    else:
        print("[ERROR] unknown ObservationType")
        exit()
    if ARGS.act in [ActionType.ONE_D_RPM, ActionType.ONE_D_DYN, ActionType.ONE_D_PID]:
        ACTION_VEC_SIZE = 1
    elif ARGS.act in [ActionType.RPM, ActionType.DYN, ActionType.VEL]:
        ACTION_VEC_SIZE = 4
    elif ARGS.act == ActionType.PID:
        ACTION_VEC_SIZE = 3
    else:
        print("[ERROR] unknown ActionType")
        exit()

    #### Uncomment to debug slurm scripts ######################
    # exit()
    
    if ARGS.env == 'flock':
        env = FlockAviary(num_drones=ARGS.num_drones,
                          aggregate_phy_steps=shared_constants.AGGR_PHY_STEPS,
                          obs=ARGS.obs,
                          act=ARGS.act
                          )
    elif ARGS.env == 'leaderfollower':
        env = LeaderFollowerAviary(num_drones=ARGS.num_drones,
                                   aggregate_phy_steps=shared_constants.AGGR_PHY_STEPS,
                                   obs=ARGS.obs,
                                   act=ARGS.act
                                   )
    elif ARGS.env == 'meetup':
        env = MeetupAviary(num_drones=ARGS.num_drones,
                           aggregate_phy_steps=shared_constants.AGGR_PHY_STEPS,
                           obs=ARGS.obs,
                           act=ARGS.act
                           )
    else:
        print("[ERROR] environment not yet implemented")
        exit()

    # number of agents
    num_agents = ARGS.num_drones
    print("Number of agents:" , num_agents)
   
    # size of each action
    action_size = ACTION_VEC_SIZE

    # examine the state space
    states = env._getDroneStateVector(0)
    state_size = OWN_OBS_VEC_SIZE
    print("There are {} agents. Each observes a state with length: {}" .format(num_agents, state_size))
    print("The state for the first agent looks like:" , states)

    #Agent1 = Agent(state_size, action_size, random_seed = 1)

    agents = []

    for i in range(num_agents):
        agents.append(Agent(state_size, action_size, random_seed = 1, agent_size = 1, max_action = 1))

    # Evaluate Policy
    '''
    for i in range(5):
         obs = env.reset()
         rew = np.zeros(num_agents)
         done = False
         while not done:
                #actions = Agent.select_action(np.array(obs))
                #actions = np.random.randn(num_agents, action_size)
                #actions = np.clip(actions, -1, 1)
                obs_ = []
                for n in range (num_agents):
                    obs_.append(obs[n])
                #print("obs_",obs_)
                actions = Agent1.select_action(np.array(obs_))
                print("actions1",actions)
                actions = {0: np.array(actions[0]), 1: np.array(actions[1])}
                obs, rewards, done, _ = env.step(actions)
                rew[0] += rewards[0]
                rew[1] += rewards[1]
         print("Total reward (averaged over agents) this episode: {}" .format(np.mean(rew)))
    
    '''

    for i in range(5):
         obs = env.reset()
         rew = np.zeros(num_agents)
         done = False
         actions = np.zeros(num_agents)
         while not done:
                for n in range (num_agents):
                    actions[n] = agents[n].select_action(np.array(obs[n]))
                #print("actions1",actions)
                actions = {0: np.array(actions[0]), 1: np.array(actions[1])}
                obs, rewards, done, _ = env.step(actions)
                rew[0] += rewards[0]
                rew[1] += rewards[1]
         print("Total reward (averaged over agents) this episode: {}" .format(np.mean(rew)))


    rewards_ = [] # list containing rewards
    rewards_ = td3()

    fig = plt.figure(figsize=(16,5))
    ax = fig.add_subplot(121)
    plt.plot(np.arange(1, len(rewards_)+1), np.max(rewards_,axis=1))
    #plt.axhline(y=target_avg_score, color='navy', linestyle='-')
    plt.title('Reward vs Episode')
    plt.ylabel('Reward')
    plt.xlabel('Episode #')
    plt.show()

    env.close()
