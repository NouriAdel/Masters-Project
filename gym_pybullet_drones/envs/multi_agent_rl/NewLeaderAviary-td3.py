import math
import numpy as np
from gym import spaces
from ray.rllib.env.multi_agent_env import MultiAgentEnv, ENV_STATE

from gym_pybullet_drones.envs.BaseAviary import DroneModel, Physics, BaseAviary

from gym_pybullet_drones.envs.BaseAviary import DroneModel, Physics, BaseAviary
from gym_pybullet_drones.envs.single_agent_rl.BaseSingleAgentAviary import ActionType, ObservationType
from gym_pybullet_drones.envs.multi_agent_rl.BaseMultiagentAviary import BaseMultiagentAviary

from statistics import mean
import os
import math

class NewLeaderAviary(BaseMultiagentAviary):
    """Multi-agent RL problem: leader-follower."""

    ################################################################################

    def __init__(self,
                 drone_model: DroneModel=DroneModel.CF2X,
                 num_drones: int=2,
                 neighbourhood_radius: float=np.inf,
                 initial_xyzs=None,
                 initial_rpys=None,
                 physics: Physics=Physics.PYB,
                 freq: int=240,
                 aggregate_phy_steps: int=1,
                 gui=False,
                 record=False, 
                 obs: ObservationType=ObservationType.KIN,
                 act: ActionType=ActionType.RPM,
		 box_bound=[1,1,1],
		 desired_pos=[0,0,0.5],
		 p1=[10,10,20],
		 p2=[0,0,0],
		 p3=0.8,
		 p4=0
		 ):
        """Initialization of a multi-agent RL environment.

        Using the generic multi-agent RL superclass.

        Parameters
        ----------
        drone_model : DroneModel, optional
            The desired drone type (detailed in an .urdf file in folder `assets`).
        num_drones : int, optional
            The desired number of drones in the aviary.
        neighbourhood_radius : float, optional
            Radius used to compute the drones' adjacency matrix, in meters.
        initial_xyzs: ndarray | None, optional
            (NUM_DRONES, 3)-shaped array containing the initial XYZ position of the drones.
        initial_rpys: ndarray | None, optional
            (NUM_DRONES, 3)-shaped array containing the initial orientations of the drones (in radians).
        physics : Physics, optional
            The desired implementation of PyBullet physics/custom dynamics.
        freq : int, optional
            The frequency (Hz) at which the physics engine steps.
        aggregate_phy_steps : int, optional
            The number of physics steps within one call to `BaseAviary.step()`.
        gui : bool, optional
            Whether to use PyBullet's GUI.
        record : bool, optional
            Whether to save a video of the simulation in folder `files/videos/`.
        obs : ObservationType, optional
            The type of observation space (kinematic information or vision)
        act : ActionType, optional
            The type of action space (1 or 3D; RPMS, thurst and torques, or waypoint with PID control)

        """
        super().__init__(drone_model=drone_model,
                         num_drones=num_drones,
                         neighbourhood_radius=neighbourhood_radius,
                         initial_xyzs=initial_xyzs,
                         initial_rpys=initial_rpys,
                         physics=physics,
                         freq=freq,
                         aggregate_phy_steps=aggregate_phy_steps,
                         gui=gui,
                         record=record, 
                         obs=obs,
                         act=act
                         )

        #self.register_variables(self.model.variables())
        #print("hi")
        self.box_bound = box_bound
        self.desired_pos = desired_pos
        self.p1 = p1
        self.p2 = p2
        self.p3 = p3
        self.p4 = p4
        self.max_x = (self.box_bound[0]/2) - 0.1
        self.min_x = -(self.box_bound[0]/2) + 0.1
        self.max_y = (self.box_bound[1]/2) - 0.1
        self.min_y = -(self.box_bound[1]/2) + 0.1
        self.max_z = (self.box_bound[2]/2) - 0.1
        self.min_z = 0
        self.current_clipped_action_ = np.zeros((self.NUM_DRONES, 4))
        self.last_clipped_action_ = np.zeros((self.NUM_DRONES, 4))
        self.max_rpy = 8
        self.min_rpy = -8
        #self.hover = hover
        self.EPISODE_LEN_SEC = 10
        #self._max_episode_steps = 500 ##
        self.max_reward_ = np.zeros(self.NUM_DRONES)
        self.mean_rewards_ = []
        self.max_mean_reward = 0
        self.rewards_ = []
        self.time_ = []
        self.time_max_rew = 0
        self.rewards_splitted = []
        for i in range(self.NUM_DRONES):
            self.rewards_splitted.append([])
            #time_.append([])

    ################################################################################
    
    def _computeReward(self):
        """Computes the current reward value(s).

        Returns
        -------
        dict[int, float]
            The reward value for each drone.

        """
        rewards = {}
        states = np.array([self._getDroneStateVector(i) for i in range(self.NUM_DRONES)])
        '''
        rewards[0] = -1 * np.linalg.norm(np.array([0, 0, 0.5]) - states[0, 0:3])**2
        # rewards[1] = -1 * np.linalg.norm(np.array([states[1, 0], states[1, 1], 0.5]) - states[1, 0:3])**2 # DEBUG WITH INDEPENDENT REWARD 
        for i in range(1, self.NUM_DRONES):
            rewards[i] = -(1/self.NUM_DRONES) * np.linalg.norm(np.array([states[i, 0], states[i, 1], states[0, 2]]) - states[i, 0:3])**2
        '''
        '''	
        pos_error = self._get_pos_error(0,True,True)
        acc_error = self._get_acc_error(0)
        angle_error = self._get_angle_error(0)	
        rewards[0] = np.maximum(0, ((np.tanh(1-pos_error-angle_error))-acc_error))
        for i in range(1, self.NUM_DRONES):
            pos_error = self._get_pos_error(i,True,False)
            acc_error = self._get_acc_error(i)
            angle_error = self._get_angle_error(i)
            rewards[i] = np.maximum(0, ((np.tanh(1-pos_error-angle_error))-acc_error))
        '''
        '''
        pos_error = self._get_pos_error(0,True,True)
        acc_error = self._get_acc_error(0)
        angle_error = self._get_angle_error(0)	
        rewards[0] = 2 * math.exp(-0.25 * ((pos_error - angle_error) ** 2) / 18)
        for i in range(1, self.NUM_DRONES):
            pos_error = self._get_pos_error(i,True,False)
            acc_error = self._get_acc_error(i)
            angle_error = self._get_angle_error(i)
            rewards[i] = 2 * math.exp(-0.25 * ((pos_error - angle_error) ** 2) / 18)
        '''

        r1 = self._get_pos_rew(0,True)
        r2 = self._get_acc_rew(0)
        r3 = self._get_angle_rew(0)
        state = self._getDroneStateVector(0)
        if (state[0:3] == [0,0,0.5]).all():
            r4 = 1000
        else:
            r4 = 0	
        rewards[0] = 3*r1 + r2 + 2*r3 + r4
        for i in range(1, self.NUM_DRONES):
            r1 = self._get_pos_rew(i,False)
            r2 = self._get_acc_rew(i)
            r3 = self._get_angle_rew(i)
            state = self._getDroneStateVector(i)
            if state[2] == 0.5:
                r4 = 1000
            else:
                r4 = 0
            rewards[i] = 3*r1 + r2 + 2*r3 + r4
        return rewards

    ################################################################################
    
    def _computeDone(self):
        """Computes the current done value(s).

        Returns
        -------
        dict[int | "__all__", bool]
            Dictionary with the done value of each drone and 
            one additional boolean value for key "__all__".

        """
        #bool_val = True if self.step_counter/self.SIM_FREQ > self.EPISODE_LEN_SEC else False

        bool_val = False	
        for i in range(self.NUM_DRONES):
            if self._check_pos_out(i):
                bool_val = True
                print("pos out", i)
                break
            elif self._check_angles_out(i):
                bool_val = True
                print("angle out", i)
                break
        if self.step_counter/self.SIM_FREQ > self.EPISODE_LEN_SEC:
            bool_val = True
	
        if bool_val:
            rew = self._computeReward()
            print("Current Time Step: ", self.step_counter)
            print("\nCurrent Reward: ", rew)
            '''
            self.rewards_.append(rew)
            self.time_.append(self.step_counter)
            #mean_ = mean(rew)
            s = 0
            for a in range (self.NUM_DRONES):
                s += rew[a]
            mean_ = float(s/self.NUM_DRONES)
            #mean_ = float(sum(rew)/len(rew))
            self.mean_rewards_.append(mean_)
            if mean_ > self.max_mean_reward:
                self.max_mean_reward = mean_
                self.time_max_rew = self.step_counter
            for c in range (self.NUM_DRONES):
                self.rewards_splitted[c].append(rew[c])
                if rew[c] > self.max_reward_[c]:
                    self.max_reward_[c] = rew[c]
            '''
        '''
        path_ = os.path.join(os.path.dirname(os.path.abspath(__file__)),"result.txt")
        with open(path_,'w') as f:
            pass
        with open(path_, 'w') as output:
            output.write("Results:\n\n")
            output.write("Max Mean Reward Across All Episodes:\n")
            output.write(str(self.max_mean_reward))
            output.write("\n\nTime of Max Mean Reward:\n")
            output.write(str(self.time_max_rew))
            output.write("\n\nMax Reward For Each Agent:\n")
            output.write(str(self.max_reward_))
            output.write("\n\nMean Reward Of Each Episode:\n")
            output.write(str(self.mean_rewards_))
            output.write("\n\nRewards Of All Episodes:\n")
            output.write(str(self.rewards_))
            output.write("\n\nRewards Of Each Agent Across All Episodes:\n")
            for a in range (self.NUM_DRONES):
                output.write(str(self.rewards_splitted[a])+"\n")
            #output.write(str(self.rewards_splitted))
            output.write("\nThe Time Steps In Each Episode:\n")
            output.write(str(self.time_))
            output.write("\n\nTraining Info:\n\n")
            output.write("\n\nName of Environment:\n")
            output.write("NewLeaderAviary")
            output.write("\n\nType Of Environment:\n")
            output.write("Cooperative -> Leader Follower")
            output.write("\n\nAlgorithm:\n")
            output.write("TD3 -> Custom Critic")
            output.write("\n\nNumber Of Agents:\n")
            output.write(str(self.NUM_DRONES))
            output.write("\n\nObservation Type:\n")
            output.write("Kinematic information (pose, linear and angular velocities)")
            output.write("\n\nAction Type:\n")
            output.write("RPM")
            output.write("\n\nExistance Of Obstacles:\n")
            output.write("No")
            output.write("\n\nFrequency:\n")
            output.write("240")
            output.write("\n\nSimulation Frequency:\n")
            output.write(str(self.SIM_FREQ))	
        '''
        done = {i: bool_val for i in range(self.NUM_DRONES)}
        done["__all__"] = bool_val # True if True in done.values() else False
        return done

    ################################################################################
    
    def _computeInfo(self):
        """Computes the current info dict(s).

        Unused.

        Returns
        -------
        dict[int, dict[]]
            Dictionary of empty dictionaries.

        """
        return {i: {} for i in range(self.NUM_DRONES)}

    ################################################################################

    def _clipAndNormalizeState(self,
                               state
                               ):
        """Normalizes a drone's state to the [-1,1] range.

        Parameters
        ----------
        state : ndarray
            (20,)-shaped array of floats containing the non-normalized state of a single drone.

        Returns
        -------
        ndarray
            (20,)-shaped array of floats containing the normalized state of a single drone.

        """
        MAX_LIN_VEL_XY = 3 
        MAX_LIN_VEL_Z = 1

        MAX_XY = MAX_LIN_VEL_XY*self.EPISODE_LEN_SEC
        MAX_Z = MAX_LIN_VEL_Z*self.EPISODE_LEN_SEC

        MAX_PITCH_ROLL = np.pi # Full range

        clipped_pos_xy = np.clip(state[0:2], -MAX_XY, MAX_XY)
        clipped_pos_z = np.clip(state[2], 0, MAX_Z)
        clipped_rp = np.clip(state[7:9], -MAX_PITCH_ROLL, MAX_PITCH_ROLL)
        clipped_vel_xy = np.clip(state[10:12], -MAX_LIN_VEL_XY, MAX_LIN_VEL_XY)
        clipped_vel_z = np.clip(state[12], -MAX_LIN_VEL_Z, MAX_LIN_VEL_Z)

        if self.GUI:
            self._clipAndNormalizeStateWarning(state,
                                               clipped_pos_xy,
                                               clipped_pos_z,
                                               clipped_rp,
                                               clipped_vel_xy,
                                               clipped_vel_z
                                               )

        normalized_pos_xy = clipped_pos_xy / MAX_XY
        normalized_pos_z = clipped_pos_z / MAX_Z
        normalized_rp = clipped_rp / MAX_PITCH_ROLL
        normalized_y = state[9] / np.pi # No reason to clip
        normalized_vel_xy = clipped_vel_xy / MAX_LIN_VEL_XY
        normalized_vel_z = clipped_vel_z / MAX_LIN_VEL_XY
        normalized_ang_vel = state[13:16]/np.linalg.norm(state[13:16]) if np.linalg.norm(state[13:16]) != 0 else state[13:16]

        norm_and_clipped = np.hstack([normalized_pos_xy,
                                      normalized_pos_z,
                                      state[3:7],
                                      normalized_rp,
                                      normalized_y,
                                      normalized_vel_xy,
                                      normalized_vel_z,
                                      normalized_ang_vel,
                                      state[16:20]
                                      ]).reshape(20,)

        return norm_and_clipped
    
    ################################################################################
    
    def _clipAndNormalizeStateWarning(self,
                                      state,
                                      clipped_pos_xy,
                                      clipped_pos_z,
                                      clipped_rp,
                                      clipped_vel_xy,
                                      clipped_vel_z,
                                      ):
        """Debugging printouts associated to `_clipAndNormalizeState`.

        Print a warning if values in a state vector is out of the clipping range.
        
        """
        if not(clipped_pos_xy == np.array(state[0:2])).all():
            print("[WARNING] it", self.step_counter, "in LeaderFollowerAviary._clipAndNormalizeState(), clipped xy position [{:.2f} {:.2f}]".format(state[0], state[1]))
        if not(clipped_pos_z == np.array(state[2])).all():
            print("[WARNING] it", self.step_counter, "in LeaderFollowerAviary._clipAndNormalizeState(), clipped z position [{:.2f}]".format(state[2]))
        if not(clipped_rp == np.array(state[7:9])).all():
            print("[WARNING] it", self.step_counter, "in LeaderFollowerAviary._clipAndNormalizeState(), clipped roll/pitch [{:.2f} {:.2f}]".format(state[7], state[8]))
        if not(clipped_vel_xy == np.array(state[10:12])).all():
            print("[WARNING] it", self.step_counter, "in LeaderFollowerAviary._clipAndNormalizeState(), clipped xy velocity [{:.2f} {:.2f}]".format(state[10], state[11]))
        if not(clipped_vel_z == np.array(state[12])).all():
            print("[WARNING] it", self.step_counter, "in LeaderFollowerAviary._clipAndNormalizeState(), clipped z velocity [{:.2f}]".format(state[12]))


    ################################################################################

    def _get_pos_rew(self,nth,l):
        """Calculate the position reward

        to be added to the total reward

        nth is the drone number

        l defines if the drone is a leader or follower

	"""

        state = self._getDroneStateVector(nth)
        if l:
            #desired_pos_ = self.desired_pos
            desired_pos_ = [0,0,0.5]
        else:
            state_l = self._getDroneStateVector(0)
            #newX = state_l[0] + 0.1
            #newY = state_l[1]
            #newZ = state_l[2]
            desired_pos_ = [state[0], state[1], state_l[2]]
 
        error = desired_pos_ - state[0:3]
        rew = np.sum((1/math.sqrt(2*math.pi*0.04)) * np.exp(-(np.power(error,2))/(2*0.04)))

        return rew

    ################################################################################

    def _get_acc_rew(self,nth):
        """Calculate the acceleration reward

        to be used in the reward function

        nth is the drone number

        """

        state = self._getDroneStateVector(nth)
        self.current_clipped_action_[nth] = state[16:20]
        normalized_current_action_ = self.current_clipped_action_[nth] / self.MAX_RPM
        normalized_last_action_ = self.last_clipped_action_[nth] / self.MAX_RPM
        #error = np.sum(self.p3 * np.square(normalized_current_action_ - normalized_last_action_))
        error = normalized_current_action_ - normalized_last_action_
        rew = np.sum((1/math.sqrt(2*math.pi*0.04)) * np.exp(-(np.power(error,2))/(2*0.04)))
        self.last_clipped_action_[nth] = self.current_clipped_action_[nth]

        return rew

    ################################################################################

    def _get_angle_rew(self,nth):
        """Calculate the angle reward

        to be used in the reward function

        nth is the drone number

        """

        state = self._getDroneStateVector(nth)
        #error = np.sum(self.p2 * np.square(state[7:10]))
        error = state[7:10]
        rew = np.sum((1/math.sqrt(2*math.pi*0.04)) * np.exp(-(np.power(error,2))/(2*0.04)))

        return rew

    ################################################################################

    def _get_pos_error(self,nth,r,l):
        """Calculate the position error

        to be added to the states

        nth is the drone number

	"""

        state = self._getDroneStateVector(nth)
        if l:
            desired_pos_ = self.desired_pos
        else:
            state_l = self._getDroneStateVector(0)
            desired_pos_ = [state[0], state[1], state_l[2]]
        if r:
            error = np.sum(self.p1 * np.square(desired_pos_ - state[0:3]))
        else:
            error = desired_pos_ - state[0:3]

        return error
  
    ################################################################################

    def _get_acc_error(self,nth):
        """Calculate the acceleration error

        to be used in the reward function

        nth is the drone number

        """

        state = self._getDroneStateVector(nth)
        self.current_clipped_action_[nth] = state[16:20]
        normalized_current_action_ = self.current_clipped_action_[nth] / self.MAX_RPM
        normalized_last_action_ = self.last_clipped_action_[nth] / self.MAX_RPM
        error = np.sum(self.p3 * np.square(normalized_current_action_ - normalized_last_action_))
        self.last_clipped_action_[nth] = self.current_clipped_action_[nth]

        return error

    ################################################################################

    def _get_angle_error(self,nth):
        """Calculate the angle error

        to be used in the reward function

        nth is the drone number

        """

        state = self._getDroneStateVector(nth)
        error = np.sum(self.p2 * np.square(state[7:10]))

        return error
  
    ################################################################################
    
    def _generate_init_pos(self):
        """Generate random initial position for the drone
        
        returns an array containing x,y,z
        
        """
        """if self.hover:
            random_init_zs = np.zeros(self.NUM_DRONES)
            for i in range (self.NUM_DRONES):
              random_init_zs[i] = random.uniform(0, 1)  ## Added by Nouran
            return np.vstack([np.array([0 for x in range(self.NUM_DRONES)]), \
                              np.array([0 for y in range(self.NUM_DRONES)]), \
                              np.array([random_init_zs[z] for z in range(self.NUM_DRONES)])]).transpose().reshape(self.NUM_DRONES, 3)  ## Added by Nouran
        else:
            random_init_xs = np.zeros(self.NUM_DRONES)   ## Added by Nouran
            random_init_ys = np.zeros(self.NUM_DRONES)   ## Added by Nouran
            random_init_zs = np.zeros(self.NUM_DRONES)   ## Added by Nouran
            for i in range (self.NUM_DRONES):
              random_init_xs[i] = random.uniform(-0.5, 0.5) ## Added by Nouran
              random_init_ys[i] = random.uniform(-0.5, 0.5)  ## Added by Nouran
              random_init_zs[i] = random.uniform(1, 2)  ## Added by Nouran

            return np.vstack([np.array([random_init_xs[x] for x in range(self.NUM_DRONES)]), \
                              np.array([random_init_ys[y] for y in range(self.NUM_DRONES)]), \
                              np.array([random_init_zs[z] for z in range(self.NUM_DRONES)])]).transpose().reshape(self.NUM_DRONES, 3)  ## Added by Nouran
        """
        random_init_xs = np.zeros(self.NUM_DRONES)   ## Added by Nouran
        random_init_ys = np.zeros(self.NUM_DRONES)   ## Added by Nouran
        random_init_zs = np.zeros(self.NUM_DRONES)   ## Added by Nouran
        for i in range (self.NUM_DRONES):
            random_init_xs[i] = random.uniform(-0.2, 0.2) ## Added by Nouran
            random_init_ys[i] = random.uniform(-0.2, 0.2)  ## Added by Nouran
            random_init_zs[i] = random.uniform(0, 0.5)  ## Added by Nouran

        return np.vstack([np.array([random_init_xs[x] for x in range(self.NUM_DRONES)]), \
                          np.array([random_init_ys[y] for y in range(self.NUM_DRONES)]), \
                          np.array([random_init_zs[z] for z in range(self.NUM_DRONES)])]).transpose().reshape(self.NUM_DRONES, 3)  ## Added by Nouran

    ################################################################################

    def _generate_desired_pos(self): ##hereee
        """Generate random target position for the drone
        
        returns an array containing x,y,z
        
        """
        if not self.hover:
            random_goal_xs = np.zeros(self.NUM_DRONES)   ## Added by Nouran
            random_goal_ys = np.zeros(self.NUM_DRONES)   ## Added by Nouran
            random_goal_zs = np.zeros(self.NUM_DRONES)   ## Added by Nouran
            for i in range (self.NUM_DRONES):
                random_goal_xs[i] = random.uniform(-0.2, 0.2) ## Added by Nouran
                random_goal_ys[i] = random.uniform(-0.2, 0.2)  ## Added by Nouran
                random_goal_zs[i] = random.uniform(0, 0.5)  ## Added by Nouran

            return np.vstack([np.array([random_goal_xs[x] for x in range(self.NUM_DRONES)]), \
                              np.array([random_goal_ys[y] for y in range(self.NUM_DRONES)]), \
                              np.array([random_goal_zs[z] for z in range(self.NUM_DRONES)])]).transpose().reshape(self.NUM_DRONES, 3)  ## Added by Nouran
  
    ################################################################################
    
    def _check_angles_out(self, nth):
        """Check if the angles exceeded their limits 
        
        they should be in the range [-pi/2, pi/2]
        
        nth is the drone number
        
        returns bool
        
        """
        state = self._getDroneStateVector(nth)
        #print("state in check angles", state)
        roll = state[7]
        pitch = state[8]
        yaw = state[9]
        
        max = np.pi/2
        min = -np.pi/2
        
        if (roll > max or roll < min) or (yaw > max or yaw < min) or (pitch > max or pitch < min):
            return True
        else:
            return False
        
    ################################################################################

    def _check_pos_out(self, nth):
        """Check if x,y,z exceeded their limits
        
        they should be in the box_bound
        
        nth is the drone number
        
        returns bool
        
        """
        state = self._getDroneStateVector(nth)
        #print("state in check pos", state)
        x = state[0]
        y = state[1]
        z = state[2]
        """
        x_bound = self.box_bound[0]
        y_bound = self.box_bound[1]
        z_bound = self.box_bound[2]
        
        max_x = x_bound/2 - 0.1
        min_x = -x_bound/2 + 0.1
        max_y = y_bound/2 - 0.1
        min_y = -y_bound/2 + 0.1
        max_z = z_bound/2 - 0.1
        min_z = -z_bound/2 + 0.1
        """
        if (x > self.max_x or x < self.min_x) or (y > self.max_y or y < self.min_y) or (z > self.max_z or z < self.min_z):
            return True
        else:
            return False
        
    ################################################################################

    def _normalizedActionToRPM(self,
                               action
                               ):
        """De-normalizes the [-1, 1] range to the [0, MAX_RPM] range.

        Parameters
        ----------
        action : ndarray
            (4)-shaped array of ints containing an input in the [-1, 1] range.

        Returns
        -------
        ndarray
            (4)-shaped array of ints containing RPMs for the 4 motors in the [0, MAX_RPM] range.

        """
        Min = (60*100)/(2*np.pi)  ## Added by Nouran
        Max = (60*750)/(2*np.pi)  ## Added by Nouran

        if np.any(np.abs(action)) > 1:
            print("\n[ERROR] it", self.step_counter, "in BaseAviary._normalizedActionToRPM(), out-of-bound action")
        #return np.where(action <= 0, (action+1)*self.HOVER_RPM, action*self.MAX_RPM) # Non-linear mapping: -1 -> 0, 0 -> HOVER_RPM, 1 -> MAX_RPM
        else:
            return (((Max-Min)*(action+1))/2)+Min  ## Added by Nouran

    ################################################################################

    def _computeObs(self):
        """Returns the current observation of the environment.

        Returns
        -------
        dict[int, ndarray]
            A Dict with NUM_DRONES entries indexed by Id in integer format,
            each a Box() os shape (H,W,4) or (12,) depending on the observation type.

        """
        if self.OBS_TYPE == ObservationType.RGB:
            if self.step_counter%self.IMG_CAPTURE_FREQ == 0: 
                for i in range(self.NUM_DRONES):
                    self.rgb[i], self.dep[i], self.seg[i] = self._getDroneImages(i,
                                                                                 segmentation=False
                                                                                 )
                    #### Printing observation to PNG frames example ############
                    if self.RECORD:
                        self._exportImage(img_type=ImageType.RGB,
                                          img_input=self.rgb[i],
                                          path=self.ONBOARD_IMG_PATH+"drone_"+str(i),
                                          frame_num=int(self.step_counter/self.IMG_CAPTURE_FREQ)
                                          )
            return {i: self.rgb[i] for i in range(self.NUM_DRONES)}
        elif self.OBS_TYPE == ObservationType.KIN: 
            ############################################################
            #### OBS OF SIZE 20 (WITH QUATERNION AND RPMS)
            # return {   i   : self._clipAndNormalizeState(self._getDroneStateVector(i)) for i in range(self.NUM_DRONES) }
            ############################################################
            #### OBS SPACE OF SIZE 12 ## 15 modified by Nouran
            obs_15 = np.zeros((self.NUM_DRONES,15))
            for i in range(self.NUM_DRONES):
                obs = self._clipAndNormalizeState(self._getDroneStateVector(i))
                if i == 0:
                    pos_error = self._get_pos_error(0,False,True)
                else:
                    pos_error = self._get_pos_error(i,False,False)
                #pos_error = obs[0:3] - self.desired_pos[0:3] ### hereee
                obs_15[i, :] = np.hstack([obs[0:3], obs[7:10], obs[10:13], obs[13:16], pos_error[0:3]]).reshape(15,)
            return {i: obs_15[i, :] for i in range(self.NUM_DRONES)}
            ############################################################
        else:
            print("[ERROR] in BaseMultiagentAviary._computeObs()")

            
    ################################################################################
    
    def _observationSpace(self):
        """Returns the observation space of the environment.

        Returns
        -------
        dict[int, ndarray]
            A Dict with NUM_DRONES entries indexed by Id in integer format,
            each a Box() os shape (H,W,4) or (12,) depending on the observation type.

        """
        if self.OBS_TYPE == ObservationType.RGB:
            return spaces.Dict({i: spaces.Box(low=0,
                                              high=255,
                                              shape=(self.IMG_RES[1], self.IMG_RES[0], 4), dtype=np.uint8
                                              ) for i in range(self.NUM_DRONES)})
        elif self.OBS_TYPE == ObservationType.KIN:
            ############################################################
            #### OBS OF SIZE 20 (WITH QUATERNION AND RPMS)
            #### Observation vector ### X        Y        Z       Q1   Q2   Q3   Q4   R       P       Y       VX       VY       VZ       WX       WY       WZ       P0            P1            P2            P3
            # obs_lower_bound = np.array([-1,      -1,      0,      -1,  -1,  -1,  -1,  -1,     -1,     -1,     -1,      -1,      -1,      -1,      -1,      -1,      -1,           -1,           -1,           -1])
            # obs_upper_bound = np.array([1,       1,       1,      1,   1,   1,   1,   1,      1,      1,      1,       1,       1,       1,       1,       1,       1,            1,            1,            1])          
            # return spaces.Box( low=obs_lower_bound, high=obs_upper_bound, dtype=np.float32 )
            ############################################################
            #### OBS SPACE OF SIZE 12
            return spaces.Dict({i: spaces.Box(low=np.array([-np.inf,-np.inf,0, -np.inf,-np.inf,-np.inf, -np.inf,-np.inf,-np.inf, -np.inf,-np.inf,-np.inf, -np.inf,-np.inf,-np.inf]),
                                              high=np.array([np.inf,np.inf,np.inf, np.inf,np.inf,np.inf, np.inf,np.inf,np.inf, np.inf,np.inf,np.inf, np.inf,np.inf,np.inf]),
                                              dtype=np.float32
                                              ) for i in range(self.NUM_DRONES)})  ## Modified by Nouran 15 instead of 12
            ############################################################
        else:
            print("[ERROR] in BaseMultiagentAviary._observationSpace()")

    
    ################################################################################
    '''
    def reset(self):
        """Resets the environment.

        Returns
        -------
        ndarray | dict[..]
            The initial observation, check the specific implementation of `_computeObs()`
            in each subclass for its format.

        """
        p.resetSimulation(physicsClientId=self.CLIENT)
        #### Housekeeping ##########################################
        self._housekeeping()
        #### Update and store the drones kinematic information #####
        self._updateAndStoreKinematicInformation()
        #### Start video recording #################################
        self._startVideoRecording()
        #### Return the initial observation ########################
        #self.INIT_XYZS = self._generate_init_pos()
        return self._computeObs()
	'''
    ################################################################################
  
    def _preprocessAction(self,
                          action
                          ):
        """Pre-processes the action passed to `.step()` into motors' RPMs.

        Parameter `action` is processed differenly for each of the different
        action types: the input to n-th drone, `action[n]` can be of length
        1, 3, or 4, and represent RPMs, desired thrust and torques, or the next
        target position to reach using PID control.

        Parameter `action` is processed differenly for each of the different
        action types: `action` can be of length 1, 3, or 4 and represent 
        RPMs, desired thrust and torques, the next target position to reach 
        using PID control, a desired velocity vector, etc.

        Parameters
        ----------
        action : dict[str, ndarray]
            The input action for each drone, to be translated into RPMs.

        Returns
        -------
        ndarray
            (NUM_DRONES, 4)-shaped array of ints containing to clipped RPMs
            commanded to the 4 motors of each drone.

        """
        rpm = np.zeros((self.NUM_DRONES,4))
        for k, v in action.items():
            if self.ACT_TYPE == ActionType.RPM: 
                rpm[int(k),:] = np.array(self.HOVER_RPM * (1+0.05*v))
            elif self.ACT_TYPE == ActionType.DYN: 
                rpm[int(k),:] = nnlsRPM(thrust=(self.GRAVITY*(v[0]+1)),
                                        x_torque=(0.05*self.MAX_XY_TORQUE*v[1]),
                                        y_torque=(0.05*self.MAX_XY_TORQUE*v[2]),
                                        z_torque=(0.05*self.MAX_Z_TORQUE*v[3]),
                                        counter=self.step_counter,
                                        max_thrust=self.MAX_THRUST,
                                        max_xy_torque=self.MAX_XY_TORQUE,
                                        max_z_torque=self.MAX_Z_TORQUE,
                                        a=self.A,
                                        inv_a=self.INV_A,
                                        b_coeff=self.B_COEFF,
                                        gui=self.GUI
                                        )
            elif self.ACT_TYPE == ActionType.PID: 
                state = self._getDroneStateVector(int(k))
                rpm_k, _, _ = self.ctrl[int(k)].computeControl(control_timestep=self.AGGR_PHY_STEPS*self.TIMESTEP, 
                                                        cur_pos=state[0:3],
                                                        cur_quat=state[3:7],
                                                        cur_vel=state[10:13],
                                                        cur_ang_vel=state[13:16],
                                                        target_pos=state[0:3]+0.1*v
                                                        )
                rpm[int(k),:] = rpm_k
            elif self.ACT_TYPE == ActionType.VEL:
                state = self._getDroneStateVector(int(k))
                if np.linalg.norm(v[0:3]) != 0:
                    v_unit_vector = v[0:3] / np.linalg.norm(v[0:3])
                else:
                    v_unit_vector = np.zeros(3)
                temp, _, _ = self.ctrl[int(k)].computeControl(control_timestep=self.AGGR_PHY_STEPS*self.TIMESTEP, 
                                                        cur_pos=state[0:3],
                                                        cur_quat=state[3:7],
                                                        cur_vel=state[10:13],
                                                        cur_ang_vel=state[13:16],
                                                        target_pos=state[0:3], # same as the current position
                                                        target_rpy=np.array([0,0,state[9]]), # keep current yaw
                                                        target_vel=self.SPEED_LIMIT * np.abs(v[3]) * v_unit_vector # target the desired velocity vector
                                                        )
                rpm[int(k),:] = temp
            elif self.ACT_TYPE == ActionType.ONE_D_RPM: 
                rpm[int(k),:] = np.repeat(self.HOVER_RPM * (1+0.05*v), 4)
            elif self.ACT_TYPE == ActionType.ONE_D_DYN: 
                rpm[int(k),:] = nnlsRPM(thrust=(self.GRAVITY*(1+0.05*v[0])),
                                        x_torque=0,
                                        y_torque=0,
                                        z_torque=0,
                                        counter=self.step_counter,
                                        max_thrust=self.MAX_THRUST,
                                        max_xy_torque=self.MAX_XY_TORQUE,
                                        max_z_torque=self.MAX_Z_TORQUE,
                                        a=self.A,
                                        inv_a=self.INV_A,
                                        b_coeff=self.B_COEFF,
                                        gui=self.GUI
                                        )
            elif self.ACT_TYPE == ActionType.ONE_D_PID:
                state = self._getDroneStateVector(int(k))
                rpm, _, _ = self.ctrl[k].computeControl(control_timestep=self.AGGR_PHY_STEPS*self.TIMESTEP, 
                                                        cur_pos=state[0:3],
                                                        cur_quat=state[3:7],
                                                        cur_vel=state[10:13],
                                                        cur_ang_vel=state[13:16],
                                                        target_pos=state[0:3]+0.1*np.array([0,0,v[0]])
                                                        )
                rpm[int(k),:] = rpm
            else:
                print("[ERROR] in BaseMultiagentAviary._preprocessAction()")
                exit()
        return rpm

    ################################################################################   

