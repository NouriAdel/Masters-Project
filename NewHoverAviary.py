import numpy as np
from gym import spaces

from gym_pybullet_drones.envs.BaseAviary import DroneModel, Physics, BaseAviary
from gym_pybullet_drones.envs.single_agent_rl.BaseSingleAgentAviary import ActionType, ObservationType, BaseSingleAgentAviary

# generate random integer values
#from numpy.random import seed
#from numpy.random import randint
# seed random number generator
#seed(1)
import random

class NewHoverAviary(BaseSingleAgentAviary):
    """Single agent RL problem: hover at position."""

    ################################################################################
    
    def __init__(self,
                 drone_model: DroneModel=DroneModel.CF2X,
                 initial_xyzs=None,
                 initial_rpys=None,
                 physics: Physics=Physics.PYB,
                 freq: int=240,
                 aggregate_phy_steps: int=1,
                 gui=False,
                 record=False, 
                 obs: ObservationType=ObservationType.KIN,
                 act: ActionType=ActionType.RPM
                 ):
        """Initialization of a single agent RL environment.

        Using the generic single agent RL superclass.

        Parameters
        ----------
        drone_model : DroneModel, optional
            The desired drone type (detailed in an .urdf file in folder `assets`).
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
        
        #self.initial_xyzs = generate_init_pos()
        #self.initial_rpys = generate_init_orient()
        #random_xs = randint(-3, 3, 2 * self.NUM_DRONES) ## Added by Nouran
        #random_ys = randint(-3, 3, 2 * self.NUM_DRONES)  ## Added by Nouran
        #random_zs = randint(0, 3, 2 * self.NUM_DRONES)  ## Added by Nouran
        #self.goal = generate_goal()
        self.goal_pos = np.zeros((self.NUM_DRONES, 3))
        self.goal_pos[0] = [0.25,0.25,0.25]
        self._max_episode_steps = 500  ## Added by Nouran
        self.EPISODE_LEN_SEC = 10  ## Added by Nouran
        #self.last_action = np.zeros((self.NUM_DRONES, 4))
        #self.current_action = np.zeros((self.NUM_DRONES, 4))
        self.last_action_ = np.zeros(4)   ## Added by Nouran
        self.current_action = np.zeros(4)  ## Added by Nouran
        self.normalized_last_action_ = np.zeros(4)   ## Added by Nouran
        self.normalized_current_action = np.zeros(4)  ## Added by Nouran

    ################################################################################
    
    def _computeReward(self):
        """Computes the current reward value.

        Returns
        -------
        float
            The reward.

        """
        state = self._getDroneStateVector(0)
        self.current_action = state[16:20]
        #print("last action", state[16:20])
        #print("last clipped action", env.last_clipped_action)
        #print("init pos", env.initial_xyzs)
        #print("goal", env.goal)

        #return -1 * np.linalg.norm(np.array([0, 0, 1])-state[0:3])**2
        ##p1 = [3,3,3]  ## Added by Nouran
        ##p2 = [20,20,10]  ## Added by Nouran
        p1 = [10,10,20]
        p2 = [0,0,0]
        ##p3 = 0.8  ## Added by Nouran
        p3 = 0
        self.normalized_current_action = self.current_action/self.MAX_RPM  ## Added by Nouran
        self.normalized_last_action_ = self.last_action/self.MAX_RPM  ## Added by Nouran
     #   return np.tanh(1 - p1 * ((state[20:23])**2) - p2 * ((state[7:10])**2)) - p3 * (self.last_action - self.last_last_action)**2   ## Added by Nouran
        res = np.tanh(1 - p1 * ((state[20:23])**2) - p2 * ((state[7:10])**2))    ## Added by Nouran
        res2 = (res[0] + res[1] + res[2]) - np.sum(p3 * (np.subtract(self.normalized_current_action, self.normalized_last_action_))**2)
        #print("current", self.current_action)
        #print("last", self.last_action_)
        #print("state", state[16:20])
        self.last_action_ = self.current_action
        if res2 < 0:
          return 0
        else:
          return res2

    ################################################################################
    
    def _computeDone(self):
        """Computes the current done value.

        Returns
        -------
        bool
            Whether the current episode is done.

        """
        state = self._getDroneStateVector(0)
        if state[0]>0.5 or state[0]<-0.5 or state[1]>0.5 or state[1]<-0.5 or state[2]>1 or state[2]<0 or state[7]>np.pi/2 or state[7]<-np.pi/2 or state[8]>np.pi/2 or state[8]<-np.pi/2 or state[9]>np.pi/2 or state[9]<-np.pi/2:
            return True
        elif self.step_counter/self.SIM_FREQ > self.EPISODE_LEN_SEC:
            return True
        else:
            return False

    ################################################################################
    
    def _computeInfo(self):
        """Computes the current info dict(s).

        Unused.

        Returns
        -------
        dict[str, int]
            Dummy value.

        """
        return {"answer": 42} #### Calculated by the Deep Thought supercomputer in 7.5M years

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
                                      state[16:20],
                                      state[20:23]
                                      ]).reshape(23,) ## Added by Nouran

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
            print("[WARNING] it", self.step_counter, "in HoverAviary._clipAndNormalizeState(), clipped xy position [{:.2f} {:.2f}]".format(state[0], state[1]))
        if not(clipped_pos_z == np.array(state[2])).all():
            print("[WARNING] it", self.step_counter, "in HoverAviary._clipAndNormalizeState(), clipped z position [{:.2f}]".format(state[2]))
        if not(clipped_rp == np.array(state[7:9])).all():
            print("[WARNING] it", self.step_counter, "in HoverAviary._clipAndNormalizeState(), clipped roll/pitch [{:.2f} {:.2f}]".format(state[7], state[8]))
        if not(clipped_vel_xy == np.array(state[10:12])).all():
            print("[WARNING] it", self.step_counter, "in HoverAviary._clipAndNormalizeState(), clipped xy velocity [{:.2f} {:.2f}]".format(state[10], state[11]))
        if not(clipped_vel_z == np.array(state[12])).all():
            print("[WARNING] it", self.step_counter, "in HoverAviary._clipAndNormalizeState(), clipped z velocity [{:.2f}]".format(state[12]))


################################################################################

    def _observationSpace(self):
        """Returns the observation space of the environment.

        Returns
        -------
        ndarray
            A Box() of shape (H,W,4) or (12,) depending on the observation type.

        """
        if self.OBS_TYPE == ObservationType.RGB:
            return spaces.Box(low=0,
                              high=255,
                              shape=(self.IMG_RES[1], self.IMG_RES[0], 4),
                              dtype=np.uint8
                              )
        elif self.OBS_TYPE == ObservationType.KIN:
            ############################################################
            #### OBS OF SIZE 20 (WITH QUATERNION AND RPMS)
            #### Observation vector ### X        Y        Z       Q1   Q2   Q3   Q4   R       P       Y       VX       VY       VZ       WX       WY       WZ       P0            P1            P2            P3
            # obs_lower_bound = np.array([-1,      -1,      0,      -1,  -1,  -1,  -1,  -1,     -1,     -1,     -1,      -1,      -1,      -1,      -1,      -1,      -1,           -1,           -1,           -1])
            # obs_upper_bound = np.array([1,       1,       1,      1,   1,   1,   1,   1,      1,      1,      1,       1,       1,       1,       1,       1,       1,            1,            1,            1])          
            # return spaces.Box( low=obs_lower_bound, high=obs_upper_bound, dtype=np.float32 )
            ############################################################
            #### OBS SPACE OF SIZE 12
            return spaces.Box(low=np.array([-1,-1,0, -1,-1,-1, -1,-1,-1, -1,-1,-1, -np.inf,-np.inf,-np.inf]),
                              high=np.array([1,1,1, 1,1,1, 1,1,1, 1,1,1, np.inf,np.inf,np.inf]),
                              dtype=np.float32
                              )  ## Modified by Nouran 15 instead of 12
            ############################################################
        else:
            print("[ERROR] in BaseSingleAgentAviary._observationSpace()")
    
    ################################################################################

    def _computeObs(self):
        """Returns the current observation of the environment.

        Returns
        -------
        ndarray
            A Box() of shape (H,W,4) or (12,) depending on the observation type.

        """
        if self.OBS_TYPE == ObservationType.RGB:
            #print("here2")
            if self.step_counter%self.IMG_CAPTURE_FREQ == 0: 
                self.rgb[0], self.dep[0], self.seg[0] = self._getDroneImages(0,
                                                                             segmentation=False
                                                                             )
                #### Printing observation to PNG frames example ############
                if self.RECORD:
                    self._exportImage(img_type=ImageType.RGB,
                                      img_input=self.rgb[0],
                                      path=self.ONBOARD_IMG_PATH,
                                      frame_num=int(self.step_counter/self.IMG_CAPTURE_FREQ)
                                      )
            return self.rgb[0]
        elif self.OBS_TYPE == ObservationType.KIN: 
            obs = self._clipAndNormalizeState(self._getDroneStateVector(0))
            #print("here")
            ############################################################
            #### OBS OF SIZE 20 (WITH QUATERNION AND RPMS)
            # return obs
            ############################################################
            #### OBS SPACE OF SIZE 12
            return np.hstack([obs[0:3], obs[7:10], obs[10:13], obs[13:16], obs[20:23]]).reshape(15,)   ## Modified by Nouran
            ############################################################
        else:
            print("[ERROR] in BaseSingleAgentAviary._computeObs()")
    
    ################################################################################

    def _getDroneStateVector(self,
                             nth_drone
                             ):
        """Returns the state vector of the n-th drone.

        Parameters
        ----------
        nth_drone : int
            The ordinal number/position of the desired drone in list self.DRONE_IDS.

        Returns
        -------
        ndarray 
            (20,)-shaped array of floats containing the state vector of the n-th drone.
            Check the only line in this method and `_updateAndStoreKinematicInformation()`
            to understand its format.

        """
        state = np.hstack([self.pos[nth_drone, :], self.quat[nth_drone, :], self.rpy[nth_drone, :],
                           self.vel[nth_drone, :], self.ang_v[nth_drone, :], self.last_clipped_action[nth_drone, :], np.subtract(self.goal_pos[nth_drone, :], self.pos[nth_drone, :])])   ## Modified by Nouran
        #print("hi")
        return state.reshape(23,)

    ################################################################################

    def generate_goal(self):

        random_goal_xs = np.zeros(self.NUM_DRONES)   ## Added by Nouran
        random_goal_ys = np.zeros(self.NUM_DRONES)   ## Added by Nouran
        random_goal_zs = np.zeros(self.NUM_DRONES)   ## Added by Nouran
        for i in range (self.NUM_DRONES):
          random_goal_xs[i] = random.uniform(-0.5, 0.5) ## Added by Nouran
          random_goal_ys[i] = random.uniform(-0.5, 0.5)  ## Added by Nouran
          random_goal_zs[i] = random.uniform(1, 2)  ## Added by Nouran

        return np.vstack([np.array([random_goal_xs[x] for x in range(self.NUM_DRONES)]), \
                               np.array([random_goal_ys[y] for y in range(self.NUM_DRONES)]), \
                               np.array([random_goal_zs[z] for z in range(self.NUM_DRONES)])]).transpose().reshape(self.NUM_DRONES, 3)  ## Added by Nouran


    ################################################################################

    def generate_init_pos(self):

        random_init_xs = np.zeros(self.NUM_DRONES)   ## Added by Nouran
        random_init_ys = np.zeros(self.NUM_DRONES)   ## Added by Nouran
        random_init_zs = np.zeros(self.NUM_DRONES)   ## Added by Nouran
        for i in range (self.NUM_DRONES):
          random_init_xs[i] = random.uniform(-0.5, 0.5) ## Added by Nouran
          random_init_ys[i] = random.uniform(-0.5, 0.5)  ## Added by Nouran
          random_init_zs[i] = random.uniform(1, 2)  ## Added by Nouran
        #for j in range (self.NUM_DRONES):
          #random_zs[j] = random.uniform(1, 2)  ## Added by Nouran

        return np.vstack([np.array([random_init_xs[x] for x in range(self.NUM_DRONES)]), \
                                       np.array([random_init_ys[y] for y in range(self.NUM_DRONES)]), \
                                       np.array([random_init_zs[z] for z in range(self.NUM_DRONES)])]).transpose().reshape(self.NUM_DRONES, 3)  ## Added by Nouran


    ################################################################################

    def generate_init_orient(self):

        random_rpys = np.zeros(3*self.NUM_DRONES)   ## Added by Nouran
        for k in range (3*self.NUM_DRONES):
          random_rpys[k] = random.uniform(-8*(np.pi/180), 8*(np.pi/180))  ## Added by Nouran

        return np.vstack([np.array([random_rpys[r] for r in range(self.NUM_DRONES)]), \
                                       np.array([random_rpys[p+self.NUM_DRONES] for p in range(self.NUM_DRONES)]), \
                                       np.array([random_rpys[yw+2*self.NUM_DRONES] for yw in range(self.NUM_DRONES)])]).transpose().reshape(self.NUM_DRONES, 3)  ## Added by Nouran

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


