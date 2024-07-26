import os
import argparse

DEFAULT_OBS = 'kin'
DEFAULT_ACT = 'discrete_2d'

def run(obs=DEFAULT_OBS, act=DEFAULT_ACT, dataset_folder=None, perturb=0):
    if not dataset_folder:
        raise ValueError('dataset_folder must be provided')
    
    # make a constant based on the observation space
    obs_space = 0
    if obs == 'kin':
        obs_space = 12

    act_space = 0
    if act == 'discrete_2d':
        act_space = 7

    # Open the dataset folder and for each file create a vnnlib file
    for file in os.listdir(dataset_folder):
        #Open the file
        with open(os.path.join(dataset_folder, file), 'r') as f:
            # Create the vnnlib file
            vnnlibs_dir = os.path.join(dataset_folder, 'vnnlibs')
            os.makedirs(vnnlibs_dir, exist_ok=True)
            with open(os.path.join(vnnlibs_dir, file + '.vnnlib'), 'a') as vnnlib:

                #Check if obs_space is 0
                if obs_space == 0:
                    raise NotImplementedError('Not implemented')
                
                #Check if act_space is 0
                if act_space == 0:
                    raise NotImplementedError('Not implemented')
                
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
                    # There is situations in 1D and 2D that some observations are always 0 so if it is the case we don't need to perturb it
                    #TODO: check if this if is correct!
                    if line_float == 0:
                        vnnlib.write(f'(assert (>= X_{i} 0.0))\n')
                        vnnlib.write(f'(assert (<= X_{i} 0.0))\n')
                    else:
                        #TODO check if this really works in all cases
                        vnnlib.write(f'(assert (>= X_{i} {line_float - perturb})\n')
                        vnnlib.write(f'(assert (<= X_{i} {line_float + perturb})\n')
                
                vnnlib.write('\n;; --- OUTPUT CONSTRAINTS ---\n')
                vnnlib.write('(assert (or\n     (and ')
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
                        
                vnnlib.write(')\n))\n;; --- END OF FILE ---')

if __name__ == '__main__':
    #### Define and parse (optional) arguments for the script ##
    parser = argparse.ArgumentParser(description='Single agent reinforcement learning example script')
    parser.add_argument('--obs',         default=DEFAULT_OBS,            type=str,      help='what is the observation space of the dataset (rpm,one_d_rpm, two_d_rpm, discrete_2d, discrete_3d)', metavar='')
    parser.add_argument('--act',                default=DEFAULT_ACT,           type=str,      help='what is the action space of the dataset(default: True)', metavar='')
    parser.add_argument('--dataset_folder', type=str, required=True, help='The folder that contains the dataset recorded', metavar='')
    parser.add_argument('--perturb', type=float, required=True, help='What is the value to perturb the inputs', metavar='')
    ARGS = parser.parse_args()

    run(**vars(ARGS))