# Fabric Distributed Training
A toy example to test distributed training configuration using `lightning fabric` and `Amazon Sagemaker`.

## Installation 
```bash
conda create -n <env_name> python=3.8

conda activate <env_name>

pip install -r requirements.txt
```

## Training
#### Setup Pytorch Estimator
Set `PyTorch` estimator configuration. Ensure that the `distribution` argument 
is set according to the desired training strategy.
```python 
# train.py [ddp]
distribution = {"pytorchddp":  {"enabled": True}}
```
or
```python
# train.py [fsdp]
distribution = {"torch_distributed":  {"enabled": True}}
```
#### Setup Fabric Instance
Set `strategy` and `num_nodes` arguments in `Fabric` instance.

```python
# train_with_lightning_fabric.py [ddp]
from lightning.fabric.strategies import DDPStrategy
import lightning as L

ddp_strategy = DDPStrategy()
fabric = L.Fabric(strategy=ddp_strategy, num_nodes=2, devices="auto")
```
or
```python
# train_with_lightning_fabric.py [fsdp]
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy
from lightning.fabric.strategies import FSDPStrategy
import lightning as L
import functools

my_auto_wrap_policy = functools.partial(
  size_based_auto_wrap_policy, 
  min_num_params=20000
)
fsdp_strategy = FSDPStrategy(auto_wrap_policy=my_auto_wrap_policy)
fabric = L.Fabric(strategy=fsdp_strategy, num_nodes=2, devices="auto")
```

#### Launch Training Job
```bash
python train.py
``` 
