import os
import numpy as np
import pybullet as p
from gymnasium import spaces
from collections import deque

from gym_pybullet_drones.envs.BaseAviary import BaseAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics, ActionType, ObservationType, ImageType
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl



class BaseRLAviary(BaseAviary):
    """Base single and multi-agent environment class for reinforcement learning."""
    
    ################################################################################

    def __init__(self,
                 drone_model: DroneModel=DroneModel.CF2X,
                 num_drones: int=1,
                 neighbourhood_radius: float=np.inf,
                 initial_xyzs=None,
                 initial_rpys=None,
                 physics: Physics=Physics.PYB,
                 pyb_freq: int = 240,
                 ctrl_freq: int = 240,
                 gui=False,
                 record=False,
                 obs: ObservationType=ObservationType.KIN,
                 act: ActionType=ActionType.RPM
                 ):
        """Initialization of a generic single and multi-agent RL environment.

        Attributes `vision_attributes` and `dynamics_attributes` are selected
        based on the choice of `obs` and `act`; `obstacles` is set to True 
        and overridden with landmarks for vision applications; 
        `user_debug_gui` is set to False for performance.

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
        pyb_freq : int, optional
            The frequency at which PyBullet steps (a multiple of ctrl_freq).
        ctrl_freq : int, optional
            The frequency at which the environment steps.
        gui : bool, optional
            Whether to use PyBullet's GUI.
        record : bool, optional
            Whether to save a video of the simulation.
        obs : ObservationType, optional
            The type of observation space (kinematic information or vision)
        act : ActionType, optional
            The type of action space (1 or 3D; RPMS, thurst and torques, waypoint or velocity with PID control; etc.)

        """

        #### Create a buffer for the last .5 sec of actions ########
        self.ACTION_BUFFER_SIZE = int(ctrl_freq//2)
        self.action_buffer = deque(maxlen=self.ACTION_BUFFER_SIZE)
        ####
        vision_attributes = True if obs == ObservationType.RGB else False
        self.OBS_TYPE = obs
        self.ACT_TYPE = act
        #### Create integrated controllers #########################
        if act in [ActionType.PID, ActionType.VEL, ActionType.ONE_D_PID]:
            os.environ['KMP_DUPLICATE_LIB_OK']='True'
            if drone_model in [DroneModel.CF2X, DroneModel.CF2P]:
                self.ctrl = [DSLPIDControl(drone_model=DroneModel.CF2X) for i in range(num_drones)]
            else:
                print("[ERROR] in BaseRLAviary.__init()__, no controller is available for the specified drone_model")
        super().__init__(drone_model=drone_model,
                         num_drones=num_drones,
                         neighbourhood_radius=neighbourhood_radius,
                         initial_xyzs=initial_xyzs,
                         initial_rpys=initial_rpys,
                         physics=physics,
                         pyb_freq=pyb_freq,
                         ctrl_freq=ctrl_freq,
                         gui=gui,
                         record=record, 
                         obstacles=True, # Add obstacles for RGB observations and/or FlyThruGate
                         user_debug_gui=False, # Remove of RPM sliders from all single agent learning aviaries
                         vision_attributes=vision_attributes,
                         )
        #### Set a limit on the maximum target speed ###############
        if act == ActionType.VEL:
            self.SPEED_LIMIT = 0.03 * self.MAX_SPEED_KMH * (1000/3600)

        if act == ActionType.DISCRETE_2D_COMPLEX:
            self.strength_levels = {
                'strong': 0.05,
                'med': 0.025,
                'weak': 0.0125,
                'very_weak': 0.00625,
                'hover': 0
            }

            self.action_mappings = {
                0: ('hover', 'strong'),
                1: ('hover', 'med'),
                2: ('hover', 'weak'),
                3: ('med', 'strong'),
                4: ('weak', 'med'),
                5: ('very_weak', 'weak'),
                6: ('strong', 'strong'),
                7: ('med', 'med'),
                8: ('weak', 'weak'),
                9: ('hover', 'hover'),
                10: ('strong', 'med'),
                11: ('med', 'weak'),
                12: ('weak', 'very_weak'),
                13: ('strong', 'hover'),
                14: ('med', 'hover'),
                15: ('weak', 'hover'),
                16: ('strong', 'med'),  #Starting from here, they are negative!
                17: ('med', 'weak'),
                18: ('weak', 'very_weak'),
                19: ('strong', 'strong'),
                20: ('med', 'med'),
                21: ('weak', 'weak'),
                22: ('med', 'strong'),
                23: ('weak', 'med'),
                24: ('very_weak', 'weak')

}

    ################################################################################

    def _addObstacles(self):
        """Add obstacles to the environment.

        Only if the observation is of type RGB, 4 landmarks are added.
        Overrides BaseAviary's method.

        """
        if self.OBS_TYPE == ObservationType.RGB:
            p.loadURDF("block.urdf",
                       [1, 0, .1],
                       p.getQuaternionFromEuler([0, 0, 0]),
                       physicsClientId=self.CLIENT
                       )
            p.loadURDF("cube_small.urdf",
                       [0, 1, .1],
                       p.getQuaternionFromEuler([0, 0, 0]),
                       physicsClientId=self.CLIENT
                       )
            p.loadURDF("duck_vhacd.urdf",
                       [-1, 0, .1],
                       p.getQuaternionFromEuler([0, 0, 0]),
                       physicsClientId=self.CLIENT
                       )
            p.loadURDF("teddy_vhacd.urdf",
                       [0, -1, .1],
                       p.getQuaternionFromEuler([0, 0, 0]),
                       physicsClientId=self.CLIENT
                       )
        else:
            pass

    ################################################################################

    def _actionSpace(self):
        """Returns the action space of the environment.

        Returns
        -------
        spaces.Box
            A Box of size NUM_DRONES x 4, 3, or 1, depending on the action type.

        """
        if self.ACT_TYPE in [ActionType.RPM, ActionType.VEL]:
            size = 4
        elif self.ACT_TYPE==ActionType.PID:
            size = 3
        elif self.ACT_TYPE in [ActionType.ONE_D_RPM, ActionType.ONE_D_PID]:
            size = 1
        # ALTERED
        elif self.ACT_TYPE==ActionType.TWO_D_RPM:
            size = 2
        elif self.ACT_TYPE==ActionType.DISCRETE_2D:   #STRONG LEFT, WEAK LEFT, UP, HOVER, DOWN, WEAK RIGHT, STRONG RIGHT
            size = 7
        elif self.ACT_TYPE==ActionType.DISCRETE_3D:   #STRONG LEFT, WEAK LEFT, UP, HOVER, DOWN, WEAK RIGHT, STRONG RIGHT
            size = 11
        elif self.ACT_TYPE==ActionType.DISCRETE_2D_COMPLEX:   #STRONG LEFT, MED LEFT, WEAK LEFT, STRONG UP, MED UP, WEAK UP, HOVER,WEAK DOWN, MED DOWN, STRONG DOWN, WEAK RIGHT, MED RIGHT, STRONG RIGHT, STRONG UP-LEFT, MED UP-LEFT, WEAK UP-LEFT, STRONG UP-RIGHT, MED UP-RIGHT, WEAK UP-RIGHT, STRONG DOWN-LEFT, MED DOWN-LEFT, WEAK DOWN-LEFT, STRONG DOWN-RIGHT, MED DOWN-RIGHT, WEAK DOWN-RIGHT
            size = 25
        else:
            print("[ERROR] in BaseRLAviary._actionSpace()")
            exit()
        if self.ACT_TYPE==ActionType.DISCRETE_2D or self.ACT_TYPE==ActionType.DISCRETE_3D or self.ACT_TYPE==ActionType.DISCRETE_2D_COMPLEX:
            ##
            act_lower_bound = np.array([0*np.ones(size) for i in range(self.NUM_DRONES)])
            act_upper_bound = np.array([+1*np.ones(size) for i in range(self.NUM_DRONES)])
            return spaces.Box(low=act_lower_bound, high=act_upper_bound, dtype=np.float32)

        #TODO actions stay in the range [-1,1] ?
        act_lower_bound = np.array([-1*np.ones(size) for i in range(self.NUM_DRONES)])
        act_upper_bound = np.array([+1*np.ones(size) for i in range(self.NUM_DRONES)])
        #
        for i in range(self.ACTION_BUFFER_SIZE):
            self.action_buffer.append(np.zeros((self.NUM_DRONES,size)))
        #
        return spaces.Box(low=act_lower_bound, high=act_upper_bound, dtype=np.float32)

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
        action : ndarray
            The input action for each drone, to be translated into RPMs.

        Returns
        -------
        ndarray
            (NUM_DRONES, 4)-shaped array of ints containing to clipped RPMs
            commanded to the 4 motors of each drone.

        """
        self.action_buffer.append(action)
        rpm = np.zeros((self.NUM_DRONES,4))

        if self.ACT_TYPE == ActionType.DISCRETE_2D or self.ACT_TYPE == ActionType.DISCRETE_3D or self.ACT_TYPE == ActionType.DISCRETE_2D_COMPLEX:
            action_to_take = np.argmax(action)
            rpm = self._getRPMs(action_to_take)
   
        else:
            for k in range(action.shape[0]):
                target = action[k, :]
                if self.ACT_TYPE == ActionType.RPM:
                    rpm[k,:] = np.array(self.HOVER_RPM * (1+0.05*target))
                elif self.ACT_TYPE == ActionType.PID:
                    state = self._getDroneStateVector(k)
                    next_pos = self._calculateNextStep(
                        current_position=state[0:3],
                        destination=target,
                        step_size=1,
                        )
                    rpm_k, _, _ = self.ctrl[k].computeControl(control_timestep=self.CTRL_TIMESTEP,
                                                            cur_pos=state[0:3],
                                                            cur_quat=state[3:7],
                                                            cur_vel=state[10:13],
                                                            cur_ang_vel=state[13:16],
                                                            target_pos=next_pos
                                                            )
                    rpm[k,:] = rpm_k
                elif self.ACT_TYPE == ActionType.VEL:
                    state = self._getDroneStateVector(k)
                    if np.linalg.norm(target[0:3]) != 0:
                        v_unit_vector = target[0:3] / np.linalg.norm(target[0:3])
                    else:
                        v_unit_vector = np.zeros(3)
                    temp, _, _ = self.ctrl[k].computeControl(control_timestep=self.CTRL_TIMESTEP,
                                                            cur_pos=state[0:3],
                                                            cur_quat=state[3:7],
                                                            cur_vel=state[10:13],
                                                            cur_ang_vel=state[13:16],
                                                            target_pos=state[0:3], # same as the current position
                                                            target_rpy=np.array([0,0,state[9]]), # keep current yaw
                                                            target_vel=self.SPEED_LIMIT * np.abs(target[3]) * v_unit_vector # target the desired velocity vector
                                                            )
                    rpm[k,:] = temp
                elif self.ACT_TYPE == ActionType.ONE_D_RPM:
                    rpm[k,:] = np.repeat(self.HOVER_RPM * (1+0.05*target), 4)
                elif self.ACT_TYPE == ActionType.ONE_D_PID:
                    state = self._getDroneStateVector(k)
                    res, _, _ = self.ctrl[k].computeControl(control_timestep=self.CTRL_TIMESTEP,
                                                            cur_pos=state[0:3],
                                                            cur_quat=state[3:7],
                                                            cur_vel=state[10:13],
                                                            cur_ang_vel=state[13:16],
                                                            target_pos=state[0:3]+0.1*np.array([0,0,target[0]])
                                                            )
                    rpm[k,:] = res
                elif self.ACT_TYPE == ActionType.TWO_D_RPM:
                    #if the urdf file is correct and the axis are: x -> forward, y -> left, z -> up
                    #then the following is:
                    rpm[k,0] = self.HOVER_RPM * (1+0.05*target[1]) #x=0.028; y=-0.028 so it is the front right motor
                    rpm[k,1] = self.HOVER_RPM * (1+0.05*target[1]) #x=-0.028; y=-0.028 so it is the back right motor
                    rpm[k,2] = self.HOVER_RPM * (1+0.05*target[0]) #x=-0.028; y=0.028 so it is the back left motor
                    rpm[k,3] = self.HOVER_RPM * (1+0.05*target[0]) #x=0.028; y=0.028 so it is the front left motor

                else:
                    print("[ERROR] in BaseRLAviary._preprocessAction()")
                    exit()
        return rpm

    ################################################################################
    def _getRPMs(self, action_to_take):
        rpm = np.zeros((self.NUM_DRONES,4))
        if self.ACT_TYPE == ActionType.DISCRETE_2D:
            strong = 0.05
            weak = 0.025
            if action_to_take == 0: #STRONG LEFT
                rpm[0,:2] = np.repeat(self.HOVER_RPM * (1+strong), 2)
                rpm[0,2:] = np.repeat(self.HOVER_RPM, 2)
            elif action_to_take == 1: #WEAK LEFT
                rpm[0,:2] = np.repeat(self.HOVER_RPM * (1+weak), 2)
                rpm[0,2:] = np.repeat(self.HOVER_RPM, 2)
            elif action_to_take == 2: #UP
                rpm[0,:] = np.repeat(self.HOVER_RPM * (1+strong), 4)
            elif action_to_take == 3: #HOVER
                rpm[0,:] = np.repeat(self.HOVER_RPM, 4)
            elif action_to_take == 4: #DOWN
                rpm[0,:] = np.repeat(self.HOVER_RPM * (1-strong), 4)
            elif action_to_take == 5: #WEAK RIGHT
                rpm[0,:2] = np.repeat(self.HOVER_RPM, 2)
                rpm[0,2:] = np.repeat(self.HOVER_RPM * (1+weak), 2)
            elif action_to_take == 6: #STRONG RIGHT
                rpm[0,:2] = np.repeat(self.HOVER_RPM, 2)
                rpm[0,2:] = np.repeat(self.HOVER_RPM * (1+strong), 2)
        
        elif self.ACT_TYPE == ActionType.DISCRETE_3D:
            raise NotImplementedError("DISCRETE_3D not implemented yet")
        
        elif self.ACT_TYPE == ActionType.DISCRETE_2D_COMPLEX:
            right, left = self.action_mappings[action_to_take]
            if action_to_take < 16:
                rpm[0,2:] = np.repeat(self.HOVER_RPM * (1+self.strength_levels[right]), 2)
                rpm[0,:2] = np.repeat(self.HOVER_RPM * (1+self.strength_levels[left]), 2)
            else:
                rpm[0,2:] = np.repeat(self.HOVER_RPM * (1-self.strength_levels[right]), 2)
                rpm[0,:2] = np.repeat(self.HOVER_RPM * (1-self.strength_levels[left]), 2)
                
        return rpm

    ################################################################################

    def _observationSpace(self):
        """Returns the observation space of the environment.

        Returns
        -------
        ndarray
            A Box() of shape (NUM_DRONES,H,W,4) or (NUM_DRONES,12) depending on the observation type.

        """
        if self.OBS_TYPE == ObservationType.RGB:
            return spaces.Box(low=0,
                              high=255,
                              shape=(self.NUM_DRONES, self.IMG_RES[1], self.IMG_RES[0], 4), dtype=np.uint8)
        elif self.OBS_TYPE == ObservationType.KIN:
            ############################################################
            #### OBS SPACE OF SIZE 12
            #### Observation vector ### X        Y        Z       Q1   Q2   Q3   Q4   R       P       Y       VX       VY       VZ       WX       WY       WZ
            lo = -np.inf
            hi = np.inf
            obs_lower_bound = np.array([[lo,lo,0, lo,lo,lo,lo,lo,lo,lo,lo,lo] for i in range(self.NUM_DRONES)])
            obs_upper_bound = np.array([[hi,hi,hi,hi,hi,hi,hi,hi,hi,hi,hi,hi] for i in range(self.NUM_DRONES)])
            
            return spaces.Box(low=obs_lower_bound, high=obs_upper_bound, dtype=np.float32)
            ############################################################
        elif self.OBS_TYPE == ObservationType.POS:
            ############################################################
            lo = -np.inf
            hi = np.inf
            #### OBS SPACE OF SIZE 3
            obs_lower_bound = np.array([[lo,lo,0] for i in range(self.NUM_DRONES)])
            obs_upper_bound = np.array([[hi,hi,hi] for i in range(self.NUM_DRONES)])
            return spaces.Box(low=obs_lower_bound, high=obs_upper_bound, dtype=np.float32)
            ############################################################
        elif self.OBS_TYPE == ObservationType.POS_RPY:
            ############################################################
            lo = -np.inf
            hi = np.inf
            #### OBS SPACE OF SIZE 6
            obs_lower_bound = np.array([[lo,lo,0,lo,lo,lo] for i in range(self.NUM_DRONES)])
            obs_upper_bound = np.array([[hi,hi,hi,hi,hi,hi] for i in range(self.NUM_DRONES)])
            return spaces.Box(low=obs_lower_bound, high=obs_upper_bound, dtype=np.float32)
            ############################################################
        else:
            print("[ERROR] in BaseRLAviary._observationSpace()")
    
    ################################################################################

    def _computeObs(self):
        """Returns the current observation of the environment.

        Returns
        -------
        ndarray
            A Box() of shape (NUM_DRONES,H,W,4) or (NUM_DRONES,12) depending on the observation type.

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
            return np.array([self.rgb[i] for i in range(self.NUM_DRONES)]).astype('float32')
        elif self.OBS_TYPE == ObservationType.KIN:
            ############################################################
            #### OBS SPACE OF SIZE 12
            obs_12 = np.zeros((self.NUM_DRONES,12))
            for i in range(self.NUM_DRONES):
                #obs = self._clipAndNormalizeState(self._getDroneStateVector(i))
                obs = self._getDroneStateVector(i)
                obs_12[i, :] = np.hstack([obs[0:3], obs[7:10], obs[10:13], obs[13:16]]).reshape(12,)
            ret = np.array([obs_12[i, :] for i in range(self.NUM_DRONES)]).astype('float32')

            return ret
            ############################################################
        elif self.OBS_TYPE == ObservationType.POS:
            ############################################################
            #### OBS SPACE OF SIZE 3
            obs_3 = np.zeros((self.NUM_DRONES,3))
            for i in range(self.NUM_DRONES):
                obs = self._getDroneStateVector(i)
                obs_3[i, :] = obs[0:3].reshape(3,)
            ret = np.array([obs_3[i, :] for i in range(self.NUM_DRONES)]).astype('float32')
            return ret
            ############################################################
        elif self.OBS_TYPE == ObservationType.POS_RPY:
            ############################################################
            #### OBS SPACE OF SIZE 6
            obs_6 = np.zeros((self.NUM_DRONES,6))
            for i in range(self.NUM_DRONES):
                #obs = self._clipAndNormalizeState(self._getDroneStateVector(i))
                obs = self._getDroneStateVector(i)
                obs_6[i, :] = np.hstack([obs[0:3], obs[7:10]]).reshape(6,)
            ret = np.array([obs_6[i, :] for i in range(self.NUM_DRONES)]).astype('float32')    
            return ret    
        else:
            print("[ERROR] in BaseRLAviary._computeObs()")
