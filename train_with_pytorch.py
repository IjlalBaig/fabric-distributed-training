from model import WGAN_GP, Discriminator, Generator
import torch
import lightning as L

from torchvision.datasets import CIFAR10
import torchvision.transforms as T

from lightning.fabric.strategies import FSDPStrategy
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import os

import functools


class RandnDataset(Dataset):
    def __init__(self, num_samples=10000):
        super().__init__()
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        image = torch.randn(3, 32, 32)
        noise = torch.randn(100, 1, 1)
        return image, noise


def discriminator_step(gen, disc, optimizer_disc, real_images, noise):
    optimizer_disc.zero_grad()

    # Discriminate real images
    d_real_out = disc(real_images)
    d_real_tgt = torch.ones(real_images.shape[0], device=real_images.device)
    d_real_loss = F.binary_cross_entropy_with_logits(d_real_out, d_real_tgt)

    # Discriminate fake images
    fake_images = gen(noise)
    d_fake_out = disc(fake_images.detach())
    d_fake_tgt = torch.zeros(fake_images.shape[0], device=fake_images.device)
    d_fake_loss = F.binary_cross_entropy_with_logits(d_fake_out, d_fake_tgt)

    d_loss = (d_fake_loss + d_real_loss) / 2
    # Backward and optimize
    d_loss.backward()
    optimizer_disc.step()
    return d_loss


def generator_step(gen, disc, optimizer_gen, noise):
    optimizer_gen.zero_grad()

    # Discriminate fake images
    fake_images = gen(noise)
    d_fake_out = disc(fake_images)
    d_fake_tgt = torch.ones(real_images.shape[0], device=real_images.device)
    g_loss = F.binary_cross_entropy_with_logits(d_fake_out, d_fake_tgt)

    g_loss.backward()
    optimizer_gen.step()
    return g_loss


if __name__ == "__main__":
    # Set seed
    from lightning.fabric import seed_everything
    seed_everything(999)

    # Define hyper-parameters
    BATCH_SIZE = 8
    NUM_WORKERS = 8
    NUM_EPOCHS = 1000
    LR = 1e-4
    CHANNEL_MULTIPLIER = 2
    ADDITIONAL_CONVS = 20
    DEVICE = "cuda:7"

    # Prepare data
    train_set = RandnDataset(10000)
    train_loader = DataLoader(dataset=train_set, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)

    # Setup modules and optimizers
    gen = Generator(CHANNEL_MULTIPLIER, ADDITIONAL_CONVS).to(DEVICE)
    disc = Discriminator(CHANNEL_MULTIPLIER, ADDITIONAL_CONVS).to(DEVICE)

    optimizer_gen = torch.optim.Adam(gen.parameters(), lr=LR)
    optimizer_disc = torch.optim.Adam(disc.parameters(), lr=LR)

    # Training loop
    for epoch in range(NUM_EPOCHS):
        for i, batch in enumerate(train_loader):
            real_images, noise = batch
            real_images = real_images.to(DEVICE)
            noise = noise.to(DEVICE)
            d_loss = discriminator_step(gen, disc, optimizer_disc, real_images, noise)
            g_loss = generator_step(gen, disc, optimizer_gen, noise)
            if i % 5 == 0:
                print(
                    f"Epoch: {epoch + 1:0=3d}"
                    f" | STEP: {i:0=4d}"
                    f" | Generator Loss: {g_loss.item():.4f}"
                    f" | Discriminator Loss: {d_loss.item():.4f}"
                )