#!/bin/bash
#
#SBATCH --job-name=pretrain
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=36
#SBATCH --mem=180G
#SBATCH -p comp_env
#SBATCH -e /gpfs/ostor/ossc9424/homedir/Tanzir/LifeToVec_Nov/projects/dutch_real/logs/%x.%j.err
#SBATCH -o /gpfs/ostor/ossc9424/homedir/Tanzir/LifeToVec_Nov/projects/dutch_real/logs/%x.%j.out

HOMEDIR="/gpfs/ostor/ossc9424/homedir/"
VENV="ossc_new/"

echo "job started"


module purge 
module load 2022 
module load Python/3.10.4-GCCCore-11.3.0
module load PyTorch/1.12.0-foss-CUDA-11.7.0
module load SciPy-bundle/2022.05-foss-2022a
module load matplotlib/3.5.2-foss-2022a
source "$HOMEDIR"/"$VENV"/bin/activate

echo "job started" 
export CUDA_VISIBLE_DEVICES=0

cd "$HOMEDIR"/Tanzir/LifeToVec_Nov/

date
time python src/new_code/pretrain.py projects/dutch_real/pretrain_cfg.json

echo "job ended successfully"