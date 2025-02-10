#!/bin/bash

#SBATCH -A cis240551-gpu   # Allocation name 
#SBATCH --nodes=1         # Total # of nodes (must be 1 for serial job)
#SBATCH --ntasks=1        # Total # of MPI tasks (should be 1 for serial job)
#SBATCH --time=2:40:00    # Total run time limit (hh:mm:ss)
#SBATCH -J v_01           # Job name
#SBATCH -o pytorch-snn-subtest.o%j      # Name of stdout output file
#SBATCH -p gpu
#SBATCH --gpus-per-node=1

module purge
module load modtree/gpu
module load anaconda/2021.05-py38
conda activate mypackages
ml modtree/gpu
ml learning
ml ml-toolkit-gpu/pytorch
echo

# Here we are going to run the PyTorch super_resolution example from the PyTorch examples
# GitHub Repository: https://github.com/pytorch/examples/tree/master/super_resolution

# Run PyTorch Training
echo "Training Start:"
date

python train_resnet.py

echo

# You're done!
echo "Ending script..."
date
