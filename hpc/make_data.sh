#!/bin/sh
### General options
### -- set the job Name --
#SBATCH -J mak_data
### -- ask for number of cores (default: 1) --
#SBATCH -n 1
### -- set walltime limit: j-h:m:s
#SBATCH --time 1:0:0
### -- Specify the output and error file. %J is the job-id --
### -- -o and -e mean append, -oo and -eo mean overwrite --
#SBATCH -o /beegfs/lfontain/static-hpc/outputs/logs/make_data_%J.out
#SBATCH -e /beegfs/lfontain/static-hpc/outputs/logs/make_data_%J.err
# -- end of Slurm options --


# Load modules
module load build/conda/4.10
module load compiler/cuda/12.3

conda activate torch_env

srun python /beegfs/lfontain/static-hpc/src/hpc/make_data.py