import os
import argparse
import matplotlib.pyplot as plt
import numpy as np

DEFAULT_OBS = 'kin'
DEFAULT_ACT = 'discrete_2d_complex'
DEFAULT_OUT_CONDITION = 'robustness'
DEFAULT_VIS = False
DEFAULT_VIS_ACTIONS = False

def run(obs=DEFAULT_OBS, act=DEFAULT_ACT, dataset_folder=None, perturb=0, output_condition=DEFAULT_OUT_CONDITION, visualize_dataset=DEFAULT_VIS, visualize_actions=DEFAULT_VIS_ACTIONS):

    # make a constant based on the observation space
    obs_space = 0
    if obs == 'kin':
        obs_space = 12

    act_space = 0
    if act == 'discrete_2d':
        act_space = 7
    elif act == 'discrete_2d_complex':
        act_space = 25

    all_y_values = []
    all_z_values = []
    actions = []

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
                
                action = 0

                # If the output condition is quadrants we need to define the correct actions
                if output_condition == 'quadrants':
                    # Check which quadrant the drone is in so we can define the correct actions
                    # There are situations that the l-ball is in between quadrants so we need to check if the drone is in the middle of the quadrants and in that cases we set looser constraints
                    left = right = up = down = False    #Quadrant occupied by the drone
                    correct_actions = [9]
                    # Read the lines corresponding to the drone's coordinates in the observation space
                    for i in range(3):
                        # Read the line
                        try:
                            line = f.readline().strip()
                            line_float = float(line)
                        except ValueError:
                            raise ValueError('The file in dataset is malformed somehow (NOT FLOATS)')

                        if i == 0:
                            continue
                        # Check which quadrant the drone is in
                        elif i == 1:
                            if line_float + perturb >= 1:    # y + perturb >= 1
                                correct_actions.extend([14,15]) # LEFT actions
                                right = True
                            if line_float - perturb < 1:  # y - perturb < 1
                                correct_actions.extend([1,2]) # RIGHT actions
                                left = True
                            if right and not left:
                                correct_actions.append(13) # STRONG LEFT
                            if left and not right:
                                correct_actions.append(0) # STRONG RIGHT
                        elif i == 2:
                            if line_float + perturb >= 1:  # z + perturb >= 1
                                correct_actions.extend([20,21])    # DOWN actions
                                up = True
                                if left:
                                    correct_actions.extend([23,24])    # RIGHT DOWN actions
                                if right:
                                    correct_actions.extend([17,18])    # LEFT DOWN actions
                            if line_float - perturb < 1:   # z - perturb < 1
                                correct_actions.extend([7,8])    # UP actions
                                down = True
                                if left:
                                    correct_actions.extend([4,5])       # RIGHT UP actions
                                if right:
                                    correct_actions.extend([11,12])    # LEFT UP actions
                            if up and not down:
                                correct_actions.append(19) # STRONG DOWN
                                if left and not right:
                                    correct_actions.append(22) # STRONG RIGHT DOWN
                                if right and not left:
                                    correct_actions.append(16) # STRONG LEFT DOWN
                            if down and not up:
                                correct_actions.append(6) # STRONG UP
                                if left and not right:
                                    correct_actions.append(3) # STRONG RIGHT UP
                                if right and not left:
                                    correct_actions.append(10) # STRONG LEFT UP
                        
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
                        # Register the values of y and z to plot the data and to use in the "too_tilted" output condition
                        if i == 1:
                            all_y_values.append(line_float)
                        elif i == 2:
                            all_z_values.append(line_float)
                        # Create 2 lines in the vnnlib file with the perturbation
                        # If the output condition is too_tilted we need to put the too_tilted constraints
                        if i == 3 and output_condition == 'too_tilted':
                            # Create the constraints for the too_tilted (roll)
                            # if the y value is greater than 1 the drone is in the right side of the target point
                            # and it is supposed to be tilted to the left so the roll should be positive (checked in the simulator)
                            if all_y_values[-1] > 1:
                                vnnlib.write(f'(assert (>= X_{i} {0.2 - 0.01}))\n')
                                vnnlib.write(f'(assert (<= X_{i} {0.2 + 0.01}))\n')
                                # Append the actions that are correct (all except the left actions)
                                correct_actions = [0,1,2,3,4,5,6,7,8,9,19,20,21,22,23,24]
                            else:
                                vnnlib.write(f'(assert (>= X_{i} {-0.2 - 0.01}))\n')
                                vnnlib.write(f'(assert (<= X_{i} {-0.2 + 0.01}))\n')
                                # Append the actions that are correct (all except the right actions)
                                correct_actions = [6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21]
                        # There is situations in 1D and 2D that some observations are always 0 so if it is the case we shoud not perturb them
                        elif line_float == 0:
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
                            action = int(line)
                        except ValueError:
                            raise ValueError('The file in dataset is malformed somehow (ACTION NOT INT)')
                        
                        # Create the constraints for the action
                        for i in range(act_space):
                            if i == action:
                                continue
                            vnnlib.write(f'(>= Y_{action} Y_{i}) ')
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
                    elif output_condition == 'quadrants' or output_condition == 'too_tilted':

                        first = True
                        # Since in the formal verification we give to the verifier the output condition that we don't desire so if it finds a solution it is a counterexample
                        # the prob of the correct actions needs to be less than the prob of the actions that will give a counter example...
                        for i in range(act_space):
                            if i in correct_actions:
                                continue
                            if first:
                                first = False
                            else:
                                # Close "and" and open a new "and" if it is not the first action to be written
                                vnnlib.write(')\n     (and ')
                            for j in correct_actions:
                                vnnlib.write(f'(>= Y_{i} Y_{j}) ')
                        
                    else:
                        raise NotImplementedError('Not implemented or invalid output_condition')
                            
                    vnnlib.write(')\n))\n;; --- END OF FILE ---')
                
                # Read the line that corresponds to the action choosen by the model
                if output_condition != 'robustness': # This is checked because in that we have already read the line!
                    try:
                        line = f.readline().strip()
                        action = int(line)
                    except ValueError:
                        raise ValueError('The file in dataset is malformed somehow (ACTION NOT INT)')
                actions.append(action)
    if visualize_dataset:
        if visualize_actions:
            plot_data(all_y_values, all_z_values, actions)
        else:
            plot_data(all_y_values, all_z_values)

# Action directions and intensities table
action_table = [
    {"direction": "Direita", "intensity": "Forte"}, #0
    {"direction": "Direita", "intensity": "Média"}, #1
    {"direction": "Direita", "intensity": "Fraca"}, #2
    {"direction": "Direita-Cima", "intensity": "Forte"}, #3
    {"direction": "Direita-Cima", "intensity": "Média"}, #4
    {"direction": "Direita-Cima", "intensity": "Fraca"}, #5
    {"direction": "Cima", "intensity": "Forte"}, #6
    {"direction": "Cima", "intensity": "Média"}, #7
    {"direction": "Cima", "intensity": "Fraca"}, #8
    {"direction": "Voo estacionário", "intensity": "Nula"}, #9
    {"direction": "Esquerda-Cima", "intensity": "Forte"}, #10
    {"direction": "Esquerda-Cima", "intensity": "Média"}, #11
    {"direction": "Esquerda-Cima", "intensity": "Fraca"}, #12
    {"direction": "Esquerda", "intensity": "Forte"}, #13
    {"direction": "Esquerda", "intensity": "Média"}, #14
    {"direction": "Esquerda", "intensity": "Fraca"}, #15
    {"direction": "Esquerda-Baixo", "intensity": "Forte"}, #16
    {"direction": "Esquerda-Baixo", "intensity": "Média"}, #17
    {"direction": "Esquerda-Baixo", "intensity": "Fraca"}, #18
    {"direction": "Baixo", "intensity": "Forte"}, #19
    {"direction": "Baixo", "intensity": "Média"}, #20
    {"direction": "Baixo", "intensity": "Fraca"}, #21
    {"direction": "Direita-Baixo", "intensity": "Forte"}, #22
    {"direction": "Direita-Baixo", "intensity": "Média"}, #23
    {"direction": "Direita-Baixo", "intensity": "Fraca"}  #24
]

# Directions corresponding to the action Y indices
direction_vectors = {
    "Direita": (1, 0),
    "Direita-Cima": (1, 1),
    "Cima": (0, 1),
    "Esquerda-Cima": (-1, 1),
    "Esquerda": (-1, 0),
    "Esquerda-Baixo": (-1, -1),
    "Baixo": (0, -1),
    "Direita-Baixo": (1, -1),
    "Voo estacionário": (0, 0)
}

# Intensity scale factors
intensity_scale = {
    "Forte": 0.2,
    "Média": 0.1,
    "Fraca": 0.05,
    "Nula": 0.0
}

intensity_colors = {
    "Forte": "red",
    "Média": "yellow",
    "Fraca": "green",
    "Nula": "black"
}

# Function to plot the data
def plot_data(all_x_values, all_y_values, actions=None):
    plt.figure(figsize=(8, 8))

    #Draw the target point
    plt.scatter(1.0, 1.0, color='red', linewidths=5, label='Objetivo')
    
    # Draw horizontal line at y=1
    plt.axhline(y=1, color='gray', linestyle='--', linewidth=1, label='y=1')
    
    # Draw vertical line at z=1
    plt.axvline(x=1, color='gray', linestyle='--', linewidth=1, label='z=1')
    
    for x_values, y_values, act in zip(all_x_values, all_y_values, actions):
        plt.scatter(x_values, y_values, color='blue')
        if actions is not None:
            # Get the direction and intensity for the action
            action = action_table[act]
            direction = action["direction"]
            intensity = action["intensity"]

            # Get the vector and scale it according to the intensity
            vector = direction_vectors[direction]
            scale = intensity_scale[intensity]
            color = intensity_colors[intensity]

            # Plot the arrow
            if scale != 0:
                plt.quiver(x_values, y_values, vector[0], vector[1], angles='xy', scale_units='xy', scale=1/scale , color=color, width=0.002)

    plt.xlim(0, 2)
    plt.ylim(0, 2)
    plt.xlabel('y')
    plt.ylabel('z')
    if actions is not None:
        plt.title('Pontos do dataset com as ações tomadas')
    else:
        plt.title('Pontos do dataset')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    #### Define and parse (optional) arguments for the script ##
    parser = argparse.ArgumentParser(description='vnnlib_generator.py: Generates vnnlib files from a dataset')
    parser.add_argument('--obs',         default=DEFAULT_OBS,            type=str,      help='what is the observation space of the dataset(kin)', metavar='')
    parser.add_argument('--act',                default=DEFAULT_ACT,           type=str,      help='what is the action space of the dataset(rpm,one_d_rpm, two_d_rpm, discrete_2d, discrete_2d_complex)', metavar='')
    parser.add_argument('--dataset_folder', type=str, required=True, help='The folder that contains the dataset recorded', metavar='')
    parser.add_argument('--perturb', type=float, required=True, help='What is the value to perturb the inputs', metavar='')
    parser.add_argument('--output_condition', default=DEFAULT_OUT_CONDITION, type=str, help='What is the output condition (robustness, not_strong_right, quadrants, too_tilted)', metavar='')
    parser.add_argument('--visualize_dataset', default=DEFAULT_VIS, type=bool, help='if is desired to plot the points in the dataset', metavar='')
    parser.add_argument('--visualize_actions', default=DEFAULT_VIS_ACTIONS, type=bool, help='if is desired to plot the actions of the dataset', metavar='')
    ARGS = parser.parse_args()

    run(**vars(ARGS))
