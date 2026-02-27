#!/bin/sh
### General options
### –- specify queue --
#SBATCH -C a100
### -- set the job Name --
#SBATCH -J train_two_layers
### -- set the job array --
#SBATCH --array=1-36
### -- ask for number of cores (default: 1) --
#SBATCH -n 1
### -- set walltime limit: j-h:m:s
#SBATCH --time 1:0:0
### -- Specify the output and error file. %A_%a is the job-id --
### -- -o and -e mean append, -oo and -eo mean overwrite --
#SBATCH -o /beegfs/lfontain/unsupervised-pcn/outputs/logs/experiment_%A_%a.out
#SBATCH -e /beegfs/lfontain/unsupervised-pcn/outputs/logs/experiment_%A_%a.err
# -- end of Slurm options --


# Load modules
module load build/conda/4.10
module load compiler/cuda/12.3

conda activate torch_env

F_ca3=(0.06 0.1 0.2 0.3 0.4 0.5)
n=${#F_ca3[@]}
n_index=$(( (SLURM_ARRAY_TASK_ID - 1) % $n ))
f_ca3=${F_ca3[$n_index]}

F_dg=(0.01 0.1 0.2 0.3 0.4 0.5)
f_dg_index=$(( (SLURM_ARRAY_TASK_ID - 1) / $n ))
f_dg=${F_dg[$f_dg_index]}

srun python /beegfs/lfontain/static-hpc/src/hpc/experiment.py --f_dg=$f_dg --f_ca3=$f_ca3