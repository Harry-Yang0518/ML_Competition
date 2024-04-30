#!/bin/bash

#SBATCH --job-name=ml_competition
#SBATCH --output=aug_resnet18_output_%j.out
#SBATCH --error=aug_resnet18_error_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=80GB
#SBATCH --time=48:00:00
#SBATCH --gres=gpu
#SBATCH --partition=a100_1,a100_2,v100,rtx8000

# job info

# Singularity path
ext3_path=/scratch/$USER/overlay-25GB-500K.ext3
sif_path=/scratch/hy2611/cuda11.4.2-cudnn8.2.4-devel-ubuntu20.04.3.sif

# start running
singularity exec --nv \
--overlay ${ext3_path}:ro \
${sif_path} /bin/bash -c "
source /ext3/env.sh
cd /scratch/hy2611/ML_Competition/
python -u main.py 
"
# python main_wb.py --dataset cifar10 -a baseline_resnet32 --imbalance_type exp --imbalance_rate 0.01 --bn_type bn --lr 0.01 --seed 2021 --epochs 200 --loss ce 

