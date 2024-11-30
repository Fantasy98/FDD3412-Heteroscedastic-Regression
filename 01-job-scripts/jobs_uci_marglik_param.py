import os 
import subprocess
from itertools import product

# Path to the restart file and executed commands file
# Define a function to check if a command was already executed
def use_command(command):
    restart_file = 'restart-marglik-param-uci.dat'
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
marglik_frequencys =[25,50,100]             # Default=50
n_hyperstepses  = [25,50,100]               # Default=50
approxs = ['full','kron','diag','kernel']   # Default=full
#----------------------------

icount = 0
for seed,marglik_frequency,n_hypersteps,approx in product(seeds,
                                                                marglik_frequencys,
                                                                n_hyperstepses,
                                                                approxs):
    base_cmd = f"python run_uci_crispr_regression.py --seed {seed} " + \
                f"--dataset {dataset} --config configs/uci.yaml " +\
                f"--marglik_frequency {marglik_frequency} " +\
                f"--n_hypersteps {n_hypersteps} " +\
                f"--approx {approx}"
    
    # Proposed Laplace approximation
    for head in heads:
        # print(base_cmd, f'--likelihood heteroscedastic --method map --head {head}')
        run_program(base_cmd, f'--likelihood heteroscedastic --method map --head {head}')
        # print(base_cmd, f'--likelihood heteroscedastic --method marglik --head {head}')
        print(base_cmd, f'--likelihood heteroscedastic --method marglik --head {head}')
        icount +=1 
print(f'[SUMMARY] NCASE={icount}')

