#!/bin/bash
# This is how the venv on snellius (OSSC and regular) should be activated

declare ENV_NAME="ossc_env_may2"

module purge 
module load 2022
module load Python/3.10.4-GCCcore-11.3.0
source ${ENV_NAME}/bin/activate