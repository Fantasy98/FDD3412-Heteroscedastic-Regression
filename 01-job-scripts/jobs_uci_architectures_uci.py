import os 
import subprocess
from itertools import product

# Path to the restart file and executed commands file
# Define a function to check if a command was already executed
def use_command(command):
    restart_file = 'restart-hyper-param-uci.dat'
    # Load previously executed commands from restart_file if it exists
    if os.path.exists(restart_file):
        with open(restart_file, 'r') as f:
            executed_commands = set(f.read().splitlines())
    else:
        f= open(restart_file,'w')
        f.close()
        executed_commands = set()
    
    # Check if the command is already executed
    if command in executed_commands:
        print(f"Skipping already executed command: {command}")
        return False  # Command was found, so skip it
    # Command was not executed, so execute it and record it
    print(f"Executing command: {command}")
    # Uncomment the following line to actually execute
    # os.system(command)
    # Record command as executed by appending it to the file
    with open(restart_file, 'a') as f:
        f.write(command + '\n')
    return True  # Command was new, so execute it

def run_program(base_cmd, args):
    cmd_line = base_cmd + " " + args

    if_run = use_command(cmd_line)
    if if_run:
        # subprocess.run(cmd_line,shell=True)
        print(f'[SYS] {cmd_line}',flush=True)
    else:
        print(f'[SYS] CASE EXIST!',flush=True)

    pass 



UCI_DATASETS = [
    'boston-housing', 'concrete', 'energy', 'kin8nm','naval-propulsion-plant',
    'power-plant', 'wine-quality-red', 'yacht'
]

# FIXED 
#----------------------------
seeds = [6,11]
dataset='wine-quality-red'          # REASON: Dataset with the highest noisy levels 
heads = ['natural', 'meanvar']
#----------------------------

# HYPER-PARAM
#----------------------------
widths=[25,50,75]                   # Origin=50
depths=[1,2,3]                      # Origin=1
activations=['gelu','selu','silu'] # Origin gelu
#----------------------------

icount = 0
for seed, width, depth,activation in product(seeds,widths,depths,activations):
    base_cmd = f"python run_uci_crispr_regression.py --seed {seed} " + \
                f"--dataset {dataset} --config configs/uci.yaml " +\
                f"--width {width} --depth {depth} --activation {activation}"
    # Proposed Laplace approximation
    for head in heads:
        # print(base_cmd, f'--likelihood heteroscedastic --method map --head {head}')
        run_program(base_cmd, f'--likelihood heteroscedastic --method map --head {head}')
        # print(base_cmd, f'--likelihood heteroscedastic --method marglik --head {head}')
        print(base_cmd, f'--likelihood heteroscedastic --method marglik --head {head}')
        icount +=1 
print(f'[SUMMARY] NCASE={icount}')

