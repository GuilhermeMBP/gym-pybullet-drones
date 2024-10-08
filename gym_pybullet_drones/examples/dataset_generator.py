"""Script to generate dataset files.

Class HoverAviary is used as learning envs for the SAC algorithm.

"""
import os
import time
from datetime import datetime
import argparse
import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.evaluation import evaluate_policy

from gym_pybullet_drones.utils.Logger import Logger
from gym_pybullet_drones.envs.HoverAviary import HoverAviary
from gym_pybullet_drones.envs.MultiHoverAviary import MultiHoverAviary
from gym_pybullet_drones.utils.utils import sync, str2bool
from gym_pybullet_drones.utils.enums import ObservationType, ActionType

DEFAULT_GUI = False
DEFAULT_RECORD_VIDEO = False
DEFAULT_OUTPUT_FOLDER = 'results'
DEFAULT_COLAB = False

DEFAULT_OBS = ObservationType('kin') # 'kin' or 'rgb'
DEFAULT_ACT = ActionType('discrete_2d_complex') # 'rpm' or 'one_d_rpm' or two_d_rpm or discrete_2d or discrete_2d_complex
DEFAULT_STEP = 0.1
DEFAULT_AXIS = 'y'

def run(output_folder=DEFAULT_OUTPUT_FOLDER, gui=DEFAULT_GUI, plot=True, colab=DEFAULT_COLAB, record_video=DEFAULT_RECORD_VIDEO, local=True, step=DEFAULT_STEP, x=0, y=0, z=0.1125, step_axis=DEFAULT_AXIS):

    filename = os.path.join(output_folder, '/app/VFRN4UAVs/discrete/models/save-440reward')
    if not os.path.exists(filename):
        os.makedirs(filename+'/')

    #### Print training progression ############################
    best = 0
    #with np.load(filename+'/evaluations.npz') as data:
    #    for j in range(data['timesteps'].shape[0]):
    #        print(str(data['timesteps'][j])+","+str(data['results'][j][0]))
    #        if data['results'][j][0] > best:
    #            best = data['results'][j][0]

    print("Best reward:", best)

    ############################################################
    ############################################################
    ############################################################
    ############################################################
    ############################################################

    # if os.path.isfile(filename+'/final_model.zip'):
    #     path = filename+'/final_model.zip'
    if os.path.isfile(filename+'/best_model.zip'):
        path = filename+'/best_model.zip'
    else:
        print("[ERROR]: no model un5der the specified path", filename)
    model = SAC.load(path)

    if z < 0.1125:
        print("Z coordinate is less than 0.1125, setting it to 0.1125")
        z = 0.1125
    
    #### Show (and record a video of) the model's performance ##

    test_env = HoverAviary(gui=gui,
                            obs=DEFAULT_OBS,
                            act=DEFAULT_ACT,
                            record=record_video, 
                            initial_xyzs=np.array([[x, y, z]]))
    test_env_nogui = HoverAviary(obs=DEFAULT_OBS, act=DEFAULT_ACT)
    logger = Logger(logging_freq_hz=int(test_env.CTRL_FREQ),
                num_drones=1,
                output_folder=output_folder,
                colab=colab
                )

    mean_reward, std_reward = evaluate_policy(model,
                                              test_env_nogui,
                                              n_eval_episodes=10
                                              )
    print("\n\n\nMean reward ", mean_reward, " +- ", std_reward, "\n\n")

    index_axis = 1
    if step_axis == 'x':
        index_axis = 0
    elif step_axis == 'y':
        index_axis = 1
    elif step_axis == 'z':
        index_axis = 2
    else:
        print("Invalid step_axis, using y as default")

    obs, info = test_env.reset(seed=42, options={})
    start = time.time()
    #variable to be used in the register of the dataset
    j = 1
    dir_name = '/app/VFRN4UAVs/discrete/datasets/dataset_'+datetime.now().strftime("%d.%m.%Y")+"/"+datetime.now().strftime("%H.%M.%S")
    for i in range((test_env.EPISODE_LEN_SEC)*test_env.CTRL_FREQ):
        action, _states = model.predict(obs,
                                        deterministic=True
                                        )
        obs, reward, terminated, truncated, info = test_env.step(action)
        obs2 = obs.squeeze()
        act2 = action.squeeze()
        print("Obs:", obs, "\tAction", action, "\tReward:", reward, "\tTerminated:", terminated, "\tTruncated:", truncated)

        if DEFAULT_ACT != ActionType.DISCRETE_2D and DEFAULT_ACT != ActionType.DISCRETE_3D and DEFAULT_ACT != ActionType.DISCRETE_2D_COMPLEX and DEFAULT_OBS == ObservationType.KIN:
            logger.log(drone=0,
                timestamp=i/test_env.CTRL_FREQ,
                state=np.hstack([obs2[0:3],
                                    np.zeros(4),
                                    obs2[3:15],
                                    act2[1],
                                    act2[1],
                                    act2[0],
                                    act2[0]
                                    ]),
                control=np.zeros(12)
                )
        elif DEFAULT_OBS == ObservationType.POS_RPY or DEFAULT_OBS == ObservationType.KIN:
            if DEFAULT_ACT == ActionType.DISCRETE_2D or DEFAULT_ACT == ActionType.DISCRETE_3D or DEFAULT_ACT == ActionType.DISCRETE_2D_COMPLEX:
                action_to_take = np.argmax(act2)
                rpm = test_env._getRPMs(action_to_take)
                logger.log(drone=0,
                    timestamp=i/test_env.CTRL_FREQ,
                    state=np.hstack([obs2[0:3],
                                        np.zeros(4),
                                        obs2[3:6],
                                        np.zeros(5),
                                        reward,
                                        rpm[0,:]
                                        ]),
                    control=np.zeros(12)
                    )
                #if step is positive and the coordinate in [step_axis] is greater than (starting coordinate in step_axis) + step, (if step = default (0.1) and starting coordinate in [step_axis] = 0 then 0.1, 0.2 0.3 etc) 
                # or if step is negative and ... is less than ... then
                # register the obs and action
                if (step > 0 and obs2[index_axis] > (j * step) + test_env.INIT_XYZS[0][index_axis]) or (step < 0 and obs2[index_axis] < (j * step) + test_env.INIT_XYZS[0][index_axis]):
                    if not os.path.exists(dir_name):
                        os.makedirs(dir_name)
                    with open(dir_name+'/obs_act_'+step_axis+'_equals_'+ str("%.2f" % (obs2[index_axis])).zfill(4)+'_start_'+str(y)+'_'+str(z)+'.csv', 'w') as file:
                        for obs_mom in obs2:
                            file.write(f'{obs_mom}\n')
                        file.write(f'{action_to_take}\n')
                    j += 1or discrete_3d 

        test_env.render()
        print(terminated)
        sync(i, start, test_env.CTRL_TIMESTEP)
        if terminated:
            obs = test_env.reset(seed=42, options={})
    test_env.close()

    if plot and DEFAULT_OBS == ObservationType.KIN:
        logger.plot()
    input("Press Enter to continue...")

if __name__ == '__main__':
    #### Define and parse arguments for the script ##
    parser = argparse.ArgumentParser(description='dataset generator')
    parser.add_argument('--gui',                default=DEFAULT_GUI,           type=str2bool,      help='Whether to use PyBullet GUI (default: True)', metavar='')
    parser.add_argument('--record_video',       default=DEFAULT_RECORD_VIDEO,  type=str2bool,      help='Whether to record a video (default: False)', metavar='')
    parser.add_argument('--output_folder',      default=DEFAULT_OUTPUT_FOLDER, type=str,           help='Folder where to save logs (default: "results")', metavar='')
    parser.add_argument('--colab',              default=DEFAULT_COLAB,         type=bool,          help='Whether example is being run by a notebook (default: "False")', metavar='')
    parser.add_argument('--x',                  default=0,                     type=float,         help='X coordinate for the starting point of the drone', metavar='')
    parser.add_argument('--y',                  default=0,                     type=float,         help='Y coordinate for the starting point of the drone', metavar='')
    parser.add_argument('--z',                  default=0.1125,                type=float,         help='Z coordinate for the starting point of the drone', metavar='')
    parser.add_argument('--step',               default=DEFAULT_STEP,          type=float,         help='What is the step in the [step_axis] axis to record for the dataset', metavar='')
    parser.add_argument('--step_axis',          default=DEFAULT_AXIS,          type=str,           help='What axis to use for the step (x, y, z)', metavar='')
    ARGS = parser.parse_args()

    run(**vars(ARGS))
