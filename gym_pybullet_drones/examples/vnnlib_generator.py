import os
import argparse

DEFAULT_OBS = 'kin'
DEFAULT_ACT = 'discrete_2d'
DEFAULT_OUT_CONDITION = 'robustness'

def run(obs=DEFAULT_OBS, act=DEFAULT_ACT, dataset_folder=None, perturb=0, output_condition=DEFAULT_OUT_CONDITION):
    if not dataset_folder:
        raise ValueError('dataset_folder must be provided')
    
    # make a constant based on the observation space
    obs_space = 0
    if obs == 'kin':
        obs_space = 12

    act_space = 0
    if act == 'discrete_2d':
        act_space = 7
    elif act == 'discrete_2d_complex':
        act_space = 25

    # Create the folder to store the vnnlib files
    vnnlibs_dir = os.path.join(dataset_folder, (f'vnnlibs_{perturb}_{output_condition}'))
    os.makedirs(vnnlibs_dir, exist_ok=True)

    # Create a csv file that contains the names of each vnnlib file
    with open(os.path.join(vnnlibs_dir, 'vnnlib_files.csv'), 'w') as csv:

        # Iterate over each item in the dataset folder
        for item in sorted(os.listdir(dataset_folder)):
            item_path = os.path.join(dataset_folder, item)
            
            # Check if it is a directory
            if os.path.isdir(item_path):
                continue

            # Split the item into base name and extension
            base_name, _ = os.path.splitext(item)
            
            # Process the file
            with open(item_path, 'r') as f:
                # If the output condition is quadrants we need to define the correct actions
                if output_condition == 'quadrants':
                    # Check which quadrant the drone is in so we can define the correct actions
                    # There are situations that the l-ball is in between quadrants so we need to check if the drone is in the middle of the quadrants and in that cases we set looser constraints
                    left = False
                    right = False
                    correct_actions = []
                    # Read the lines corresponding to the drone's coordinates in the observation space
                    for i in range(3):
                        # Read the line
                        try:
                            line = f.readline().strip()
                            line_float = float(line)
                        except ValueError:
                            raise ValueError('The file in dataset is malformed somehow (NOT FLOATS)')
                        correct_actions.append(9) # HOVER
                        # Check which quadrant the drone is in
                        if i == 1 and line_float + perturb >= 1:    # y + perturb >= 1
                            correct_actions.append(13,14,15) # RIGHT actions
                            left = True
                        if i == 1 and line_float - perturb < 1:  # y - perturb < 1
                            correct_actions.append(0,1,2) # LEFT actions
                            right = True
                        if i == 2 and line_float + perturb >= 1:  # z + perturb >= 1
                            correct_actions.append(19,20,21)    # DOWN actions
                            if left:
                                correct_actions.append(22,23,24)    # RIGHT DOWN actions
                            if right:
                                correct_actions.append(16,17,18)    # LEFT DOWN actions
                        if i == 2 and line_float - perturb < 1:   # z - perturb < 1
                            correct_actions.append(6,7,8)    # UP actions
                            if left:
                                correct_actions.append(3,4,5)       # LEFT UP actions
                            if right:
                                correct_actions.append(10,11,12)    # RIGHT UP actions
                    
                    # Go back to the start of the file
                    f.seek(0)

                # Create the vnnlib file
                vnnlib_name = base_name + '_epsilon_'+ str(perturb) +'.vnnlib'
                with open(os.path.join(vnnlibs_dir, vnnlib_name), 'a') as vnnlib:
                    # Check if obs_space is 0
                    if obs_space == 0:
                        raise NotImplementedError('Not implemented')
                    
                    # Check if act_space is 0
                    if act_space == 0:
                        raise NotImplementedError('Not implemented')

                    # Write the name of the vnnlib file in the csv file
                    csv.write(vnnlib_name+'\n')
                    
                    # Write the header of the vnnlib file
                    vnnlib.write(';; --- INPUT VARIABLES ---\n')
                    for i in range(obs_space):
                        vnnlib.write(f'(declare-const X_{i} Real)\n')
                    
                    vnnlib.write('\n;; --- OUTPUT VARIABLES ---\n')
                    for i in range(act_space):
                        vnnlib.write(f'(declare-const Y_{i} Real)\n')
                    
                    vnnlib.write('\n;; --- INPUT CONSTRAINTS ---\n')
                    
                    # Read the lines corresponding to the observation space 
                    for i in range(obs_space):
                        # Read the line
                        try:
                            line = f.readline().strip()
                            line_float = float(line)
                        except ValueError:
                            raise ValueError('The file in dataset is malformed somehow (NOT FLOATS)')
                        # Create 2 lines in the vnnlib file with the perturbation
                        # There is situations in 1D and 2D that some observations are always 0 so if it is the case we shoud not perturb them
                        if line_float == 0:
                            vnnlib.write(f'(assert (>= X_{i} 0.0))\n')
                            vnnlib.write(f'(assert (<= X_{i} 0.0))\n')
                        else:
                            vnnlib.write(f'(assert (>= X_{i} {line_float - perturb}))\n')
                            vnnlib.write(f'(assert (<= X_{i} {line_float + perturb}))\n')
                    
                    vnnlib.write('\n;; --- OUTPUT CONSTRAINTS ---\n')
                    vnnlib.write('(assert (or\n     (and ')
                    if output_condition == 'robustness':
                        # Read the line that corresponds to the action choosen by the model
                        try:
                            line = f.readline().strip()
                            line_int = int(line)
                        except ValueError:
                            raise ValueError('The file in dataset is malformed somehow (ACTION NOT INT)')
                        
                        # Create the constraints for the action
                        for i in range(act_space):
                            if i == line_int:
                                continue
                            vnnlib.write(f'(>= Y_{line_int} Y_{i}) ')
                    elif output_condition == 'not_strong_right':
                        if act == 'discrete_2d':
                            strong_right = 6
                            for i in range(act_space):
                                if i == strong_right:
                                    continue
                                vnnlib.write(f'(>= Y_{strong_right} Y_{i}) ')
                        elif act == 'discrete_2d_complex':
                            strong_right = 13
                            for i in range(act_space):
                                if i == strong_right:
                                    continue
                                vnnlib.write(f'(>= Y_{strong_right} Y_{i}) ')
                    elif output_condition == 'quadrants':
                        # In this case we are looking for the accpetable actions that are in the right down quadrant
                        # this being: all the ones corresponding to the left and up actions -> 0,1,2,3,4,5,6,7,8,9
                        # Since in the formal verification we give to the verifier the output condition that we don't desire so if it finds a solution it is a counterexample
                        # the prob of this actions (0,1,2,3,4,5,6,7,8,9) needs to be less than the prob of the actions that will give a counter example...
                        for i in range(act_space):
                            if i in correct_actions:
                                continue
                            for j in correct_actions:
                                vnnlib.write(f'(>= Y_{i} Y_{j}) ')
                            # Close "and" and open a new "and" if it is not the last action
                            if i != act_space - 1:
                                vnnlib.write(')\n     (and ')
                    else:
                        raise NotImplementedError('Not implemented or invalid output_condition')
                            
                    vnnlib.write(')\n))\n;; --- END OF FILE ---')

if __name__ == '__main__':
    #### Define and parse (optional) arguments for the script ##
    parser = argparse.ArgumentParser(description='Single agent reinforcement learning example script')
    parser.add_argument('--obs',         default=DEFAULT_OBS,            type=str,      help='what is the observation space of the dataset(kin)', metavar='')
    parser.add_argument('--act',                default=DEFAULT_ACT,           type=str,      help='what is the action space of the dataset(rpm,one_d_rpm, two_d_rpm, discrete_2d, discrete_2d_complex)', metavar='')
    parser.add_argument('--dataset_folder', type=str, required=True, help='The folder that contains the dataset recorded', metavar='')
    parser.add_argument('--perturb', type=float, required=True, help='What is the value to perturb the inputs', metavar='')
    parser.add_argument('--output_condition', default=DEFAULT_OUT_CONDITION, type=str, help='What is the output condition (robustness, not_strong_right, right_down_quadrant, quadrants)', metavar='')
    ARGS = parser.parse_args()

    run(**vars(ARGS))