ml purge
ml PyTorch/1.11.0-foss-2021a-CUDA-11.3.1
ml SciPy-bundle/2021.05-foss-2021a
ml matplotlib/3.4.2-foss-2021a
ml h5py/3.2.1-foss-2021a
ml tqdm/4.61.2-GCCcore-10.3.0
ml scikit-learn/0.24.2-foss-2021a

echo "INFO: All module has been loaded"

srun -A NAISS2023-3-13 -n 1 --gpus-per-node=A40:1 -t 1:00:00 --pty bash

echo "INFO: End interactive job"