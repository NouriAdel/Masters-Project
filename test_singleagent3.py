"""Test script for single agent problems.

This scripts runs the best model found by one of the executions of `singleagent.py`

Example
-------
To run the script, type in a terminal:

    $ python test_singleagent.py --exp ./results/save-<env>-<algo>-<obs>-<act>-<time_date>

"""
import os
import time
from datetime import datetime
import argparse
import re
import numpy as np
import gym
import torch
#from stable_baselines3.common.env_checker import check_env
#from stable_baselines3 import A2C
#from stable_baselines3 import PPO
#from stable_baselines3 import SAC
#from stable_baselines3 import TD3
#from stable_baselines3 import DDPG
#from stable_baselines3.common.policies import ActorCriticPolicy as a2cppoMlpPolicy
#from stable_baselines3.common.policies import ActorCriticCnnPolicy as a2cppoCnnPolicy
#from stable_baselines3.sac.policies import SACPolicy as sacMlpPolicy
#from stable_baselines3.sac import CnnPolicy as sacCnnPolicy
#from stable_baselines3.td3 import MlpPolicy as td3ddpgMlpPolicy
#from stable_baselines3.td3 import CnnPolicy as td3ddpgCnnPolicy
#from stable_baselines3.common.evaluation import evaluate_policy
from TD3 import Actor, Critic, ReplayBuffer, TD3
from gym_pybullet_drones.utils.utils import sync
from gym_pybullet_drones.utils.Logger import Logger
from gym_pybullet_drones.envs.single_agent_rl.TakeoffAviary import TakeoffAviary
from gym_pybullet_drones.envs.single_agent_rl.HoverAviary import HoverAviary
#from gym_pybullet_drones.envs.single_agent_rl.FlyThruGateAviary import FlyThruGateAviary
#from gym_pybullet_drones.envs.single_agent_rl.TuneAviary import TuneAviary
from gym_pybullet_drones.envs.single_agent_rl.BaseSingleAgentAviary import ActionType, ObservationType
from NewAviary import NewAviary

#import shared_constants

if __name__ == "__main__":

    #### Define and parse (optional) arguments for the script ##
    parser = argparse.ArgumentParser(description='Single agent reinforcement learning example script using TakeoffAviary')
    parser.add_argument('--exp',                           type=str,            help='The experiment folder written as ./results/save-<env>-<algo>-<obs>-<act>-<time_date>', metavar='')
    ARGS = parser.parse_args()

    #### Load the model from file ##############################

    """if os.path.isfile(ARGS.exp+'/success_model.zip'):
        path = ARGS.exp+'/success_model.zip'
    elif os.path.isfile(ARGS.exp+'/best_model.zip'):
        path = ARGS.exp+'/best_model.zip'
    else:
        print("[ERROR]: no model under the specified path", ARGS.exp)"""
    '''print(ARGS.exp)
    if os.path.isfile('./results/save-hover-td3-kin-one_d_rpm-09.08.2021_23.08.03/success_model.zip'):
        path = './results/save-hover-td3-kin-one_d_rpm-09.08.2021_23.08.03/success_model.zip'
    elif os.path.isfile('./results/save-hover-td3-kin-one_d_rpm-09.08.2021_23.08.03/best_model.zip'):
        path = './results/save-hover-td3-kin-one_d_rpm-09.08.2021_23.08.03/best_model.zip'
    else:
        print("[ERROR]: no model under the specified path", ARGS.exp)

    model = TD3.load(path)'''
        
    #### Parameters to recreate the environment ################
    env_name = "NewAviary"
    OBS = ObservationType.KIN 
    ACT = ActionType.ONE_D_RPM
    seed = 0
    file_name = "%s_%s_%s" % ("TD3", env_name, str(seed))
    
    #### Evaluate the model ####################################
    '''eval_env = NewAviary(obs=OBS, act=ACT)
    mean_reward, std_reward = evaluate_policy(model,
                                              eval_env,
                                              n_eval_episodes=10
                                              )
    print("\n\n\nMean reward ", mean_reward, " +- ", std_reward, "\n\n")'''

    #### Show, record a video, and log the model's performance #
    test_env = NewAviary(gui=True, record=False, obs=OBS, act=ACT)
    test_env.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    state_dim = test_env.observation_space.shape[0]
    action_dim = test_env.action_space.shape[0]
    max_action = float(test_env.action_space.high[0])
    policy = TD3(state_dim, action_dim, max_action)
    model = policy.load(file_name, directory="./pytorch_models")

    #test_env.goal = [0,0,1]
    logger = Logger(logging_freq_hz=int(test_env.SIM_FREQ/test_env.AGGR_PHY_STEPS),
                    num_drones=1
                    )
    obs = test_env.reset()
    start = time.time()
    for i in range(6*int(test_env.SIM_FREQ/test_env.AGGR_PHY_STEPS)): # Up to 6''
        action, _states = model.predict(obs,
                                        deterministic=True # OPTIONAL 'deterministic=False'
                                        )
        obs, reward, done, info = test_env.step(action)
        test_env.render()
        
        logger.log(drone=0,
                   timestamp=i/test_env.SIM_FREQ,
                   state= np.hstack([obs[0:3], np.zeros(4), obs[3:15],  np.resize(action, (4))]),
                   control=np.zeros(12)
                   )
        sync(np.floor(i*test_env.AGGR_PHY_STEPS), start, test_env.TIMESTEP)
        # if done: obs = test_env.reset() # OPTIONAL EPISODE HALT
    test_env.close()
    logger.save_as_csv("sa") # Optional CSV save
    logger.plot()

    # with np.load(ARGS.exp+'/evaluations.npz') as data:
    #     print(data.files)
    #     print(data['timesteps'])
    #     print(data['results'])
    #     print(data['ep_lengths'])
