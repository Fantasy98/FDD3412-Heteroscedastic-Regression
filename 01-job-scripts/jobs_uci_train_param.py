import os 
import subprocess
from itertools import product

# Path to the restart file and executed commands file
# Define a function to check if a command was already executed
def use_command(command):
    restart_file = 'restart-train-param-uci.dat'
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
seeds = [6]
dataset='wine-quality-red'          # REASON: Dataset with the highest noisy levels 
heads = ['natural', 'meanvar']
#----------------------------

# HYPER-PARAM
#----------------------------
lrs          = [1e-3,1e-2,1e-1]         # Default=1e-2
lr_mins      = [1e-6,1e-5,1e-4]         # Default=1e-5
lr_hyps      = [1e-3,1e-2,1e-1]         # Default=1e-2
lr_hyp_mins  = [1e-4,1e-3,1e-2]         # Default=1e-3
lr_hyps      = [1e-3,1e-2,1e-1]         # Default=1e-2
lr_hyp_mins  = [1e-4,1e-3,1e-2]         # Default=1e-3
batch_sizes  = [128,256,512]            # Default=256
n_epochss    = [5000]         # Default=5000
#----------------------------

icount = 0
for seed,lr,lr_min,lr_hyp,lr_hyp_min,batch_size,n_epochs in product(seeds,
                                                                lrs,
                                                                lr_mins,
                                                                lr_hyps,
                                                                lr_hyp_mins,
                                                                batch_sizes,
                                                                n_epochss,
                                                                ):
    base_cmd = f"python run_uci_crispr_regression.py --seed {seed} " + \
                f"--dataset {dataset} --config configs/uci.yaml " +\
                f"--lr {lr} --lr_min {lr_min}" +\
                f"--lr_hyp {lr_hyp} --lr_hyp_min {lr_hyp_min}" +\
                f"--batch_size {batch_size} --n_epochs {n_epochs}"
    
    # Proposed Laplace approximation
    for head in heads:
        # print(base_cmd, f'--likelihood heteroscedastic --method map --head {head}')
        run_program(base_cmd, f'--likelihood heteroscedastic --method map --head {head}')
        # print(base_cmd, f'--likelihood heteroscedastic --method marglik --head {head}')
        print(base_cmd, f'--likelihood heteroscedastic --method marglik --head {head}')
        icount +=1 
print(f'[SUMMARY] NCASE={icount}')

