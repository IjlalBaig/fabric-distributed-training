# ml-ddp-example
A toy example to play with Pytorch DDP basics 

## Setup: 
```bash
conda create -n <env_name> python=3.10

conda activate <env_name>

pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 --extra-index-url https://download.pytorch.org/whl/cu113
```

## Launch Training:
```bash
# simple run
python train.py

# distributed run
torchrun --nnodes=1 --nproc_per_node=<num_procs> train.py
```