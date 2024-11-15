"""
Fixing the missing VI method and Re train the Naive NLL 
@yuningw
"""
import os 
import subprocess
from itertools import product

# Path to the restart file and executed commands file
# Define a function to check if a command was already executed
def use_command(command):
    restart_file = 'restart-uci-fix1.dat'
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
        subprocess.run(cmd_line,shell=True)
        print(f'[SYS] {cmd_line}',flush=True)
    else:
        print(f'[SYS] CASE EXIST!',flush=True)

    pass 



UCI_DATASETS = [
    'boston-housing', 'concrete', 'energy', 'kin8nm','naval-propulsion-plant',
    'power-plant', 'wine-quality-red', 'yacht'
]

# ORIGINAL:
seeds = [seed for seed in range(1, 21)]
# YW: Reduce the n_seed to use 
# seeds = [seed for seed in range(1, 21, 5)]
print(f"[JOB] SEED LIST:{seeds}")
for seed, dataset in product(seeds, UCI_DATASETS):
    base_cmd = f'python run_uci_crispr_regression.py --seed {seed} --dataset {dataset} --config configs/uci.yaml'

    mcdo_vi_params = '--n_epochs 1000 --lr 0.001 --lr_min 0.001 --optimizer Adam'
    # print(base_cmd, f'--likelihood heteroscedastic --method mcdropout {mcdo_vi_params}')
    # run_program(base_cmd, f'--likelihood heteroscedastic --method mcdropout {mcdo_vi_params}')
    
    # print(base_cmd, f'--likelihood heteroscedastic --method vi {mcdo_vi_params}')
    run_program(base_cmd, f'--likelihood heteroscedastic --method vi {mcdo_vi_params}')


# dataset_ = ['boston-housing','naval-propulsion-plant',]
# heads = ['natural', 'meanvar']
# for seed, dataset in product(seeds, UCI_DATASETS):
#     # Proposed Laplace approximation
#     for head in heads:
#         # print(base_cmd, f'--likelihood heteroscedastic --method map --head {head}')
#         run_program(base_cmd, f'--likelihood heteroscedastic --method map --head {head}')
#         # print(base_cmd, f'--likelihood heteroscedastic --method marglik --head {head}')
#         run_program(base_cmd, f'--likelihood heteroscedastic --method marglik --head {head}')
    

