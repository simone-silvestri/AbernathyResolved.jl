#!/bin/bash

#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH -p pi_raffaele 
#SBATCH --time=120:00:00
#SBATCH --mem=50GB

julia --project --check-bounds=no prototype_omip_simulation.jl
