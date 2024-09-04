import numpy as np

from gym_pybullet_drones.envs.BaseRLAviary import BaseRLAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics, ActionType, ObservationType

class HoverAviary(BaseRLAviary):
    """Single agent RL problem: hover at position."""

    ################################################################################
    
    def __init__(self,
                 drone_model: DroneModel=DroneModel.CF2X,
                 initial_xyzs=None,
                 initial_rpys=None,
                 physics: Physics=Physics.PYB,
                 pyb_freq: int = 240,
                 ctrl_freq: int = 30,
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
            The type of action space (1 or 3D; RPMS, thurst and torques, or waypoint with PID control)

        """
        #TODO I CHANGED TIHS FROM 1,1,1 TO: 0,0,1
        self.TARGET_POS = np.array([0,1,1])
        #TODO I CHANGED TIHS FROM 8 TO 16
        self.EPISODE_LEN_SEC = 8
        super().__init__(drone_model=drone_model,
                         num_drones=1,
                         initial_xyzs=initial_xyzs,
                         initial_rpys=initial_rpys,
                         physics=physics,
                         pyb_freq=pyb_freq,
                         ctrl_freq=ctrl_freq,
                         gui=gui,
                         record=record,
                         obs=obs,
                         act=act
                         )

    ################################################################################
    
    def _computeReward(self):

        state = self._getDroneStateVector(0)

        #Penalize position       
        position_error = np.linalg.norm(self.TARGET_POS[:3]-state[:3]) 
        # Penalize high pitch and roll
        pitch_penalty = abs(state[7])**0.5 if abs(state[7]) > 0.2 else 0
        roll_penalty = abs(state[8])**0.5 if abs(state[8]) > 0.2 else 0
        
        # Penalize low altitude
        altitude_penalty = 20 if state[2] < 0.11 else 0
        
        # Aggregate rewards with penalties
        ret = 2 - (position_error + pitch_penalty + roll_penalty + altitude_penalty)

        # Ensure reward is non-negative
        ret = max(ret, 0)

        return ret

    ################################################################################
    
    def _computeTerminated(self):
        """Computes the current done value.

        Returns
        -------
        bool
            Whether the current episode is done.

        """
        state = self._getDroneStateVector(0)
        if np.linalg.norm(self.TARGET_POS[:3]-state[:3]) < .0001:
            return True
        else:
            return False
        
    ################################################################################
    
    def _computeTruncated(self):
        """Computes the current truncated value.

        Returns
        -------
        bool
            Whether the current episode timed out.

        """
        state = self._getDroneStateVector(0)
        if ((state[0] > 4.0 or state[0] < -1.0) or (state[1] > 4.0 or state[1] < -1.0) or state[2] > 4.0 # Truncate when the drone is too far away
             or abs(state[7]) > .4 or abs(state[8]) > .4 # Truncate when the drone is too tilted
        ):
            ret = 2 - (np.linalg.norm(self.TARGET_POS[:3]-state[:3])) - (abs(state[7])*5 if abs(state[7])>0.2 else 0) - (abs(state[8])*5 if abs(state[8])>0.2 else 0)
            return True
        if self.step_counter/self.PYB_FREQ > self.EPISODE_LEN_SEC:
            ret = 2 - (np.linalg.norm(self.TARGET_POS[:3]-state[:3])) - (abs(state[7])**0.5 if abs(state[7])>0.2 else 0) - (abs(state[8])**0.5 if abs(state[8])>0.2 else 0)
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