from itertools import product
import subprocess 
import os 

# Path to the restart file and executed commands file
# Define a function to check if a command was already executed
def use_command(command):
    restart_file = 'restart-image.dat'
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



# seeds = [117, 189, 509, 832, 711]

# YW: Reduce the number of seeds to try 
seeds = [189,711]
heads = ['natural', 'meanvar']
# MNIST and FMNIST
datasets = ['mnist','fmnist','cifar10']
het_flags = ['--het_noise label', '--het_noise rotation', '--het_noise neither']
for seed, dataset, hf in product(seeds, datasets, het_flags):
    base_cmd = f'python run_image_regression.py --seed {seed} --config configs/{dataset}.yaml {hf}'
    # homoscedastic
    # print(base_cmd, '--likelihood homoscedastic --method map')
    run_program(base_cmd, '--likelihood homoscedastic --method map')
    # print(base_cmd, '--likelihood homoscedastic --method marglik')
    run_program(base_cmd, '--likelihood homoscedastic --method marglik')

    # heteroscedastic
    # print(base_cmd, f'--likelihood heteroscedastic --method faithful --head gaussian')
    run_program(base_cmd, f'--likelihood heteroscedastic --method faithful --head gaussian')
    # print(base_cmd, f'--likelihood heteroscedastic --method betanll --head gaussian --beta 0.0')
    run_program(base_cmd, f'--likelihood heteroscedastic --method betanll --head gaussian --beta 0.0')
    # print(base_cmd, f'--likelihood heteroscedastic --method betanll --head gaussian --beta 0.5')
    run_program(base_cmd, f'--likelihood heteroscedastic --method betanll --head gaussian --beta 0.5')
    # print(base_cmd, f'--likelihood heteroscedastic --method betanll --head gaussian --beta 1.0')
    run_program(base_cmd, f'--likelihood heteroscedastic --method betanll --head gaussian --beta 1.0')
    
    # Bayes (MCDO, VI)
    mcdo_params = '--n_epochs 50 --lr 0.01 --lr_min 0.01 --optimizer Adam'
    # print(base_cmd, f'--likelihood heteroscedastic --method mcdropout {mcdo_params}')
    run_program(base_cmd, f'--likelihood heteroscedastic --method mcdropout {mcdo_params}')
    vi_lr = 1e-2 if 'fmnist' in dataset else 1e-3
    vi_lr_min = vi_lr / 100
    vi_params = f'--lr {vi_lr} --lr_min {vi_lr_min} --optimizer Adam --vi-posterior-rho-init -3.0'
    # print(base_cmd, f'--likelihood heteroscedastic --method vi {vi_params}')
    run_program(base_cmd, f'--likelihood heteroscedastic --method vi {vi_params}')
    # Proposed Laplace approximation
    for head in heads:
        # print(base_cmd, f'--likelihood heteroscedastic --method map --head {head}')
        run_program(base_cmd, f'--likelihood heteroscedastic --method map --head {head}')
        # print(base_cmd, f'--likelihood heteroscedastic --method marglik --head {head}')
        run_program(base_cmd, f'--likelihood heteroscedastic --method marglik --head {head}')

