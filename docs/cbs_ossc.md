
# Documentation for working on the CBS RA and on the OSSC


## Disk space

This is information from Tom.

- It is cheaper on the OSSC than on the RA 
- On RA, the current limit is 100 GB. 
- On the OSSC, the default is a couple of TBs. Above some threshold, it costs SBUs, but this cost is lower than the cost on the RA. 

## Running slurm scripts on the OSSC

- For running evaluation scripts (`generate_llm_report.sh` and `generate_network_report.sh`), 240G memory works and 120G memory is too little. 


## Permissions on the OSSC

It goes through [access control lists](https://servicedesk.surf.nl/wiki/pages/viewpage.action?pageId=30660238). 
This currently creates constraints on the permissions of directories/files. If user A creates directory `mydir/`, user B in the same project may not automatically have permissions to write/execute code on in `mydir/`. As far as we understand, the user that created `mydir/` needs to use the `setfacl` and `getfacl` to give other project users those permissions. 


## Filesystem

Available partitions and accounting on Snellius https://servicedesk.surf.nl/wiki/display/WIKI/Snellius+partitions+and+accounting. For the OSSC, one can only get full nodes.


## Working on GPU nodes


Note that slurm does not accept the two common GPU specifications
```bash
#SBATCH --gres=gpu:4
#SBATCH --gpus:4
```

Instead, for GPU, Ben suggests to always use 
```bash
#SBATCH --exclusive 
```

But I'm not sure how this impacts the ability to run multiple jobs in parallel, and how this works on CPU-only nodes.



### Using a single GPU

```bash
#SBATCH xyx


module load abc 


export CUDA_VISIBLE_DEVICES=0

python script.py

```


```python
# script.py
import pytorch_lightning as pl 

trainer = pl.trainer(strategy="ddp")

```


### Using multiple GPUs (1 node with 4 GPUs)

```bash
#SBATCH xyx

module load abc 


export CUDA_VISIBLE_DEVICES=0,1,2,3

srun --mpi=pmi2 python script.py 

```

Currently, the `nccl` communication backend for torch lightning does not work; instead, we need to use `mpi`. The difference is that `nccl` allows GPUs to communicate directly with each other, while `mpi` needs to go throug the CPU. I think this requires pickling the data, which may slow down the code substantially. But we have not done any systematic comparison of timing.

One can use the `mpi` backend in a DDP strategy in lightning as follows:


```python
# script.py
from pytorch_lightning.strategies import DDPStrategy
import pytorch_lightning as pl 

ddp = DDPStrategy(process_group_backend="mpi")
trainer = pl.trainer(strategy=ddp)

```


## Open questions 

- [ ] I noticed a couple of times that the GPU node crashes when doing `scancel`. It then goes into `down` state and does not recover fast / maybe not even automatically. Maybe this has to do with the virtualization of the OSSC. 