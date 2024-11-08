#!/bin/env bash
#SBATCH -A NAISS2024-5-129
#SBATCH --job-name=UCI-TRAIN-PARAM
#SBATCH -p alvis
#SBATCH --nodes=1 
#SBATCH --gpus-per-node=A40:1
#SBATCH -t 7-00:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=yuningw@kth.se
#SBATCH --output logs/UCI-TRAIN-PARAM-log.out
#SBATCH --error  logs/UCI-TRAIN-PARAM-err.error


ENV_NAME="/mimer/NOBACKUP/groups/kthmech/yuningw/bnn"
ml purge

module load virtualenv/20.23.1-GCCcore-12.3.0
source ${ENV_NAME}/bin/activate
echo "INFO: All module has been loaded"

# ${ENV_NAME}/bin/python run_image_regression.py 
python jobs_uci_train_param.py 