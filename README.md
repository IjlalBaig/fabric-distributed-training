# ml-ddp-example
A toy example to play with Pytorch DDP basics 

```bash
# simple run
python train.py

# distributed run
torchrun --nnodes=1 --nproc_per_node=<num_procs> train.py
```