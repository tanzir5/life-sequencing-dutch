#!/bin/bash
#
#SBATCH --job-name=net_summary
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --time=24:00:00
#SBATCH --mem=0
#SBATCH -p comp_env
#SBATCH -e /gpfs/ostor/ossc9424/homedir/logs/net_summary_stdout.txt
#SBATCH -o /gpfs/ostor/ossc9424/homedir/logs/net_summary_stdout.txt

module purge 
module load 2022 
module load Python/3.10.4-GCCcore-11.3.0
module load SciPy-bundle/2022.05-foss-2022a

echo "job started" 
cd /gpfs/ostor/ossc9424/homedir/Dakota_network/

date
time python get_network_summary_statistics.py

echo "job ended successfully" 