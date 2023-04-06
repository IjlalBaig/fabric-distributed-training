import os
import argparse

import numpy as np

import torch
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel
from torch.distributed import init_process_group, destroy_process_group

import torchvision.transforms as transforms

from model import EncoderDecoder

device = 'cuda'

# init argparse
parser = argparse.ArgumentParser(
    description='Pytorch DDP',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)

parser.add_argument(
    '--lr',
    type=float, 
    help='learning rate',
    required=False,
    default=0.001,
)

args = parser.parse_args()

# check if ddp run
ddp = int(os.environ.get('RANK', -1)) != -1

if ddp:
    # ddp setup
    init_process_group(backend='nccl')
    global_rank = int(os.environ['RANK'])               # global rank
    local_rank = int(os.environ['LOCAL_RANK'])          # local rank
    device = f'cuda:{local_rank}'
    master_process = global_rank == 0                   # this process will perform logging etc.
    seed_offset = global_rank
else:
    # single process, single gpu run
    master_process = True
    seed_offset = 0

# init data transform
transform = transforms.Compose([
    transforms.Resize(254), 
    transforms.Normalize([0.5], [0.5]),
])

# loading model to device
model = EncoderDecoder()
model.to(device)

# init loss function and optimizer
criterion = torch.nn.MSELoss()

learning_rate = args.lr
optimizer = optim.SGD(model.parameters(), lr=learning_rate)

# wrap model in DDP
if ddp:
    model = DistributedDataParallel(model, device_ids=[local_rank])

# start training
epochs = 100
for e in range(epochs):
    # random input tensor
    x = transform(torch.randn(
        1, 3, 284, 284,
        requires_grad=True, 
        device=device,
    ))

    # forward pass
    y = model(x)

    # backward pass
    loss = criterion(y, x)
    loss.backward()
    optimizer.step()

    # log loss if master_process
    if master_process:
        print(f"Epoch: {e+1} | Loss: {loss:.4f}")

# ddp cleanup
if ddp:
    destroy_process_group()