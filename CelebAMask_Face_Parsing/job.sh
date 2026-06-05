#!/bin/bash
#SBATCH --partition=M2
#SBATCH --qos=q_a_norm
#SBATCH --nodelist=TC2N05
#SBATCH --gres=gpu:1
#SBATCH --mem=20G
#SBATCH --job-name=experiment_lr0.0007_epoch80_bs8_wd1e-4_leakyrelu_cbam_plateau_upsample_labelsmoothing_newnormalize_dilatedconv
#SBATCH --output=output_%x_%j.out
#SBATCH --error=error_%x_%j.err
#SBATCH --time=04:00:00

module load anaconda
eval "$(conda shell.bash hook)"
conda activate Face_Parsing

# CBAM Attention MobileNet_UNet_CBAM
python main.py --lr 0.0007 --epochs 80 --batch_size 8 --weight_decay 1e-4
