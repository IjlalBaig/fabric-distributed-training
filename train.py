import os
import argparse

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel
from torch.distributed import init_process_group, destroy_process_group

from model import Generator, Discriminator

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
    default=0.00001,
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

# init params
z_dim = 64
batch_size=6
epochs = 200

# loading model to device
gen = Generator(z_dim).to(device)
disc = Discriminator().to(device)

# init loss function and optimizer
criterion = nn.BCEWithLogitsLoss()

learning_rate = args.lr
gen_opt = optim.Adam(gen.parameters(), lr=learning_rate)
disc_opt = optim.Adam(disc.parameters(), lr=learning_rate)

# wrap model in DDP
if ddp:
    gen = DistributedDataParallel(gen, device_ids=[local_rank])
    gen = nn.SyncBatchNorm.convert_sync_batchnorm(gen)

    disc = DistributedDataParallel(disc, device_ids=[local_rank])
    disc = nn.SyncBatchNorm.convert_sync_batchnorm(disc)

# training utilities
def get_noise(n_samples, z_dim, device='cpu'):
    return torch.randn(n_samples, z_dim, device=device)

def get_disc_loss(gen, disc, criterion, real, num_images, z_dim, device):
    # fake images
    noise_vec = get_noise(num_images, z_dim=z_dim, device=device)
    out_gen = gen(noise_vec).detach()
    out_dfake = disc(out_gen)
    loss_dfake = criterion(out_dfake, torch.zeros(num_images, 1, device=device))

    # real images
    out_dreal = disc(real)
    loss_dreal = criterion(out_dreal, torch.ones(num_images, 1, device=device))

    # total loss
    disc_loss = (loss_dfake + loss_dreal) / 2

    return disc_loss

def get_gen_loss(gen, disc, criterion, num_images, z_dim, device):
    noise_vec = get_noise(num_images, z_dim=z_dim, device=device)
    out_gen = gen(noise_vec)
    out_disc = disc(out_gen)

    gen_loss = criterion(out_disc, torch.ones(num_images, 1, device=device))
    return gen_loss

# start training
for e in range(epochs):
    # placeholder data: using random tensor as real data 
    real = torch.randn(batch_size, 784).to(device)

    # discriminator update
    disc_opt.zero_grad()
    disc_loss = get_disc_loss(gen, disc, criterion, real, batch_size, z_dim, device)
    disc_loss.backward(retain_graph=True)
    disc_opt.step()

    # generator update
    gen_opt.zero_grad()
    gen_loss = get_gen_loss(gen, disc, criterion, batch_size, z_dim, device)
    gen_loss.backward(retain_graph=True)
    gen_opt.step()

    if master_process:
        print(
            f"Epoch: {e+1:0=3d}"
            f" | Generator Loss: {gen_loss.item():.4f}"
            f" | Discriminator Loss: {disc_loss.item():.4f}"
        )

# ddp cleanup
if ddp:
    destroy_process_group()