"""Script demonstrating the use of `gym_pybullet_drones`'s Gymnasium interface.

Classes HoverAviary and MultiHoverAviary are used as learning envs for the PPO algorithm.

Example
-------
In a terminal, run as:

    $ python learn.py --multiagent false
    $ python learn.py --multiagent true

Notes
-----
This is a minimal working example integrating `gym-pybullet-drones` with 
reinforcement learning library `stable-baselines3`.

"""
import os
import time
from datetime import datetime
import argparse
import gymnasium as gym
import numpy as np
import torch
import wandb
import optuna

from stable_baselines3 import SAC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor

from gym_pybullet_drones.utils.Logger import Logger
from gym_pybullet_drones.envs.HoverAviary import HoverAviary
from gym_pybullet_drones.envs.MultiHoverAviary import MultiHoverAviary
from gym_pybullet_drones.utils.utils import sync, str2bool
from gym_pybullet_drones.utils.enums import ObservationType, ActionType
from wandb.integration.sb3 import WandbCallback

DEFAULT_GUI = True
DEFAULT_RECORD_VIDEO = False
DEFAULT_OUTPUT_FOLDER = 'results'
DEFAULT_COLAB = False

DEFAULT_OBS = ObservationType('kin') # 'kin' or 'rgb'
DEFAULT_ACT = ActionType('discrete_2d') # 'rpm' or 'pid' or 'vel' or 'one_d_rpm' or 'one_d_pid'
DEFAULT_AGENTS = 2
DEFAULT_MA = False

def run(multiagent=DEFAULT_MA, 
        output_folder=DEFAULT_OUTPUT_FOLDER, 
        gui=DEFAULT_GUI, 
        plot=True, 
        colab=DEFAULT_COLAB, 
        record_video=DEFAULT_RECORD_VIDEO, 
        local=True, 
        learning_rate=0.0003, 
        gamma=0.99, 
        batch_size=128, 
        net_arch=(64, 64), 
        ent_coef=0.01, 
        target_entropy=-1.0,
        num_epochs=300,
        total_steps=1000,
        num_envs=4
        ):

    # Initialize W&B
    wandb.login()
    run = wandb.init(
        project="gym-pybullet-drones", 
        entity="greis",
        config={
            "learning_rate": learning_rate,
            "gamma": gamma,
            "batch_size": batch_size,
            "buffer_size": 500000,
            "tau": 0.005,
            "train_freq": 64,
            "gradient_steps": 64,
            "n_eval_episodes": 10,
            "eval_freq": 5000,
            "epochs": num_epochs,
            "net_arch": net_arch,
            "ent_coef": ent_coef,
            "target_entropy": target_entropy
        },
    )

    # Explicitly log the learning rate
    wandb.log({"learning_rate": learning_rate})

    filename = os.path.join(output_folder, 'save-'+datetime.now().strftime("%m.%d.%Y_%H.%M.%S"))
    if not os.path.exists(filename):
        os.makedirs(filename+'/')

    if not multiagent:
        train_env = make_vec_env(HoverAviary,
                                 env_kwargs=dict(obs=DEFAULT_OBS, act=DEFAULT_ACT),
                                 n_envs=num_envs,
                                 seed=0
                                 )
        eval_env = HoverAviary(obs=DEFAULT_OBS, act=DEFAULT_ACT)
    else:
        train_env = make_vec_env(MultiHoverAviary,
                                 env_kwargs=dict(num_drones=DEFAULT_AGENTS, obs=DEFAULT_OBS, act=DEFAULT_ACT),
                                 n_envs=num_envs,
                                 seed=0
                                 )
        eval_env = MultiHoverAviary(num_drones=DEFAULT_AGENTS, obs=DEFAULT_OBS, act=DEFAULT_ACT)

    eval_env = Monitor(eval_env)

    #### Check the environment's spaces ########################
    print('[INFO] Action space:', train_env.action_space)
    print('[INFO] Observation space:', train_env.observation_space)

    #### Train the model #######################################
    policy_kwargs = dict(activation_fn=torch.nn.ReLU,
                         net_arch=dict(pi=net_arch, vf=net_arch, qf=[128, 128]))
    
    model = SAC('MlpPolicy',
                train_env,
                learning_rate=learning_rate,
                gamma=gamma,
                device="cuda",
                # tensorboard_log=filename+'/tb/',
                policy_kwargs=policy_kwargs,
                verbose=1)
   
    # Set up optimizer and learning rate scheduler
    optimizer = torch.optim.Adam(model.policy.parameters(), lr=learning_rate)
                       
    #### Target cumulative rewards (problem-dependent) ##########
    if DEFAULT_ACT == ActionType.ONE_D_RPM:
        target_reward = 474.15 if not multiagent else 949.5
    else:
        target_reward = 467. if not multiagent else 920.
    callback_on_best = StopTrainingOnRewardThreshold(reward_threshold=target_reward,
                                                     verbose=1)
    eval_callback = EvalCallback(eval_env,
                                 callback_on_new_best=callback_on_best,
                                 verbose=1,
                                 best_model_save_path=filename+'/',
                                 log_path=filename+'/',
                                 eval_freq=int(1000),
                                 deterministic=True,
                                 render=False)
    
    # Use WandbCallback for logging to wandb
    wandb_callback = WandbCallback(
        gradient_save_freq=100,
        model_save_path=filename+'/',
        verbose=2
    )

    # Create a list of callbacks
    callbacks = [eval_callback, wandb_callback]

    #### Simulating Training Run with W&B Logging ####
    for epoch in range(num_epochs):
        # Train the model
        model.learn(total_timesteps=total_steps, callback=callbacks, log_interval=100)

        # Log training metrics
        train_mean_reward, train_std_reward = evaluate_policy(model, train_env, n_eval_episodes=10)
        eval_mean_reward, eval_std_reward = evaluate_policy(model, eval_env, n_eval_episodes=10)

        # Safe division to avoid ZeroDivisionError
        if epoch > 0:
            acc = 1 - 2**-epoch - np.random.random() / epoch
            loss = 2**-epoch + np.random.random() / epoch
        else:
            acc = 0
            loss = 1
        
        wandb.log({
            "epoch": epoch, 
            "train_mean_reward": train_mean_reward, 
            "train_std_reward": train_std_reward, 
            "accuracy": acc, 
            "loss": loss, 
            "eval_mean_reward": eval_mean_reward, 
            "eval_std_reward": eval_std_reward
        })

    #### Save the model ########################################
    model.save(filename+'/final_model.zip')
    print(filename)

    #### Print training progression ############################
    with np.load(filename+'/evaluations.npz') as data:
        for j in range(data['timesteps'].shape[0]):
            print(str(data['timesteps'][j])+","+str(data['results'][j][0]))

    ############################################################
    ############################################################
    ############################################################
    ############################################################
    ############################################################

#Desligar
    if local:
    #  input("Press Enter to continue...")
       print("Done!")


    if os.path.isfile(filename+'/final_model.zip'):
         path = filename+'/final_model.zip'
    if os.path.isfile(filename+'/best_model.zip'):
        path = filename+'/best_model.zip'
    else:
        print("[ERROR]: no model under the specified path", filename)
#Desligar
    model = SAC.load(path)

    #### Show (and record a video of) the model's performance ##
    if not multiagent:
        test_env = HoverAviary(gui=gui,
                               obs=DEFAULT_OBS,
                               act=DEFAULT_ACT,
                               record=record_video)
        test_env_nogui = HoverAviary(obs=DEFAULT_OBS, act=DEFAULT_ACT)
    else:
        test_env = MultiHoverAviary(gui=gui,
                                        num_drones=DEFAULT_AGENTS,
                                        obs=DEFAULT_OBS,
                                        act=DEFAULT_ACT,
                                        record=record_video)
        test_env_nogui = MultiHoverAviary(num_drones=DEFAULT_AGENTS, obs=DEFAULT_OBS, act=DEFAULT_ACT)
    logger = Logger(logging_freq_hz=int(test_env.CTRL_FREQ),
                num_drones=DEFAULT_AGENTS if multiagent else 1,
                output_folder=output_folder,
                colab=colab
                )

    mean_reward, std_reward = evaluate_policy(model,
                                              test_env_nogui,
                                              n_eval_episodes=10
                                              )
    print("\n\n\nMean reward ", mean_reward, " +- ", std_reward, "\n\n")

    obs, info = test_env.reset(seed=42, options={})
    start = time.time()
    for i in range((test_env.EPISODE_LEN_SEC+2)*test_env.CTRL_FREQ):
        action, _states = model.predict(obs,
                                        deterministic=True
                                        )
        obs, reward, terminated, truncated, info = test_env.step(action)
        obs2 = obs.squeeze()
        act2 = action.squeeze()
        print("Obs:", obs, "\tAction", action, "\tReward:", reward, "\tTerminated:", terminated, "\tTruncated:", truncated)
        if DEFAULT_OBS == ObservationType.KIN:
            if not multiagent:
                logger.log(drone=0,
                    timestamp=i/test_env.CTRL_FREQ,
                    state=np.hstack([obs2[0:3],
                                        np.zeros(4),
                                        obs2[3:15],
                                        act2
                                        ]),
                    control=np.zeros(12)
                    )
            else:
                for d in range(DEFAULT_AGENTS):
                    logger.log(drone=d,
                        timestamp=i/test_env.CTRL_FREQ,
                        state=np.hstack([obs2[d][0:3],
                                            np.zeros(4),
                                            obs2[d][3:15],
                                            act2[d]
                                            ]),
                        control=np.zeros(12)
                        )
        test_env.render()
        print(terminated)
        sync(i, start, test_env.CTRL_TIMESTEP)
        if terminated:
            obs = test_env.reset(seed=42, options={})
    test_env.close()

#Desligar
    #if plot and DEFAULT_OBS == ObservationType.KIN:
    #    logger.plot()

    return mean_reward

def objective(trial):
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-1, log=True)
    batch_size = trial.suggest_categorical('batch_size', [32, 64, 128, 512])
    gamma = trial.suggest_float('gamma', 0.9, 0.9999)
    #net_arch = trial.suggest_categorical('net_arch', [(32, 32), (64, 64), (128, 128), (256, 256)])
    net_arch_str = trial.suggest_categorical('net_arch', ['(32, 32)', '(64, 64)', '(128, 128)', '(256, 256)'])
    net_arch = tuple(map(int, net_arch_str.strip('()').split(', ')))
    ent_coef = trial.suggest_float('ent_coef', 1e-5, 0.1, log=True)
    target_entropy = trial.suggest_float('target_entropy', -5.0, 0.0)

    
    # Call the run function with these hyperparameters
    mean_reward = run(
        multiagent=DEFAULT_MA,
        output_folder=DEFAULT_OUTPUT_FOLDER,
        gui=DEFAULT_GUI,
        plot=False,
        colab=DEFAULT_COLAB,
        record_video=DEFAULT_RECORD_VIDEO,
        local=True,
        learning_rate=learning_rate,
        gamma=gamma,
        batch_size=batch_size,
        net_arch=net_arch,
        ent_coef=ent_coef,
        target_entropy=target_entropy,
        num_envs=4 
    )
    
    return mean_reward

if __name__ == '__main__':
    #### Define and parse (optional) arguments for the script ##
    parser = argparse.ArgumentParser(description='Single agent reinforcement learning example script')
    parser.add_argument('--multiagent',         default=DEFAULT_MA,            type=str2bool,      help='Whether to use example LeaderFollower instead of Hover (default: False)', metavar='')
    parser.add_argument('--gui',                default=DEFAULT_GUI,           type=str2bool,      help='Whether to use PyBullet GUI (default: True)', metavar='')
    parser.add_argument('--record_video',       default=DEFAULT_RECORD_VIDEO,  type=str2bool,      help='Whether to record a video (default: False)', metavar='')
    parser.add_argument('--output_folder',      default=DEFAULT_OUTPUT_FOLDER, type=str,           help='Folder where to save logs (default: "results")', metavar='')
    parser.add_argument('--colab',              default=DEFAULT_COLAB,         type=str2bool,      help='Whether example is being run by a notebook (default: "False")', metavar='')
    ARGS = parser.parse_args()

    # Run Optuna optimization
    study = optuna.create_study(storage='sqlite:///my_study.db', study_name="drones", direction='maximize', load_if_exists=True)
    study.optimize(objective, n_trials=50)

    print("Best hyperparameters: ", study.best_params)
    print("Best reward: ", study.best_value)

    run(**vars(ARGS))
