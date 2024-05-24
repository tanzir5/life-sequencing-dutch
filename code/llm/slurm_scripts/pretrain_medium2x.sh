#!/bin/bash
#
#SBATCH --job-name=pretrain
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --time=100:00:00
#SBATCH --mem=100G
#SBATCH -p comp_env
#SBATCH --gres=gpu:1
#SBATCH -e /gpfs/ostor/ossc9424/homedir/Tanzir/LifeToVec_Nov/projects/dutch_real/logs/pretrain_medium2x_stderr.txt
#SBATCH -o /gpfs/ostor/ossc9424/homedir/Tanzir/LifeToVec_Nov/projects/dutch_real/logs/pretrain_medium2x_stdout.txt


echo "job started"


date
time python pretrain.py projects/dutch_real/pretrain_cfg_medium2x.json

echo "job ended successfully"