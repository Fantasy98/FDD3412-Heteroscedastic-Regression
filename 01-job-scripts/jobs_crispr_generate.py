import os 
import subprocess
from itertools import product

# Path to the restart file and executed commands file
# Define a function to check if a command was already executed
def use_command(command):
    restart_file = 'restart-crispr.dat'
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
    # print(f"Executing command: {command}")
    # Uncomment the following line to actually execute
    os.system(command)

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



CRISPR_DATASETS = [
    'flow-cytometry-HEK293', 
    'survival-screen-A375', 
    'survival-screen-HEK293'
]
#YW: Reduce the seed number to make the training lighter
#NOTE: n_case * n_data * n_seed = 11 * 3 * 3 = 99! Assume at most 1hr per case, then it cost 4 days! 
seeds = [seed for seed in range(1,11,4)]
heads = ['natural', 'meanvar']
for seed, dataset in product(seeds, CRISPR_DATASETS):
    base_cmd = f'python run_uci_crispr_regression.py --seed {seed} --dataset {dataset} --config configs/crispr.yaml'
    # homoscedastic
    run_program(base_cmd, '--likelihood homoscedastic --method map')

    run_program(base_cmd, '--likelihood homoscedastic --method marglik')

    # heteroscedastic
    run_program(base_cmd,f'--likelihood heteroscedastic --method faithful --head gaussian')
    
    run_program(base_cmd, f'--likelihood heteroscedastic --method betanll --head gaussian --beta 0.0')

    run_program(base_cmd, f'--likelihood heteroscedastic --method betanll --head gaussian --beta 0.5')
    
    run_program(base_cmd, f'--likelihood heteroscedastic --method betanll --head gaussian --beta 1.0')
    
    # Bayes (VI)
    vi_params = '--n_epochs 500 --lr 0.001 --lr_min 0.001 --optimizer Adam'
    run_program(base_cmd, f'--likelihood heteroscedastic --method vi {vi_params}')

    # Proposed Laplace approximation
    for head in heads:
        run_program(base_cmd, f'--likelihood heteroscedastic --method map --head {head}')
        run_program(base_cmd, f'--likelihood heteroscedastic --method marglik --head {head}')
        

