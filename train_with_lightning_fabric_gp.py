from model import Discriminator, Generator
import torch
import lightning as L
import warnings
import functools

# Set seed
from lightning.fabric import seed_everything

seed_everything(999)

from lightning.fabric.strategies import FSDPStrategy, DDPStrategy
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import os

# Optional: cluster environment setup
# env = None
# process_group_backend = None
# try:
#     print(f"world_size: ({int(os.environ['WORLD_SIZE'])}), rank: ({int(os.environ['RANK'])})")
#
#     from lightning.fabric.plugins.environments import LightningEnvironment
#     env = LightningEnvironment()
#     env.world_size = lambda: int(os.environ["WORLD_SIZE"])
#     env.global_rank = lambda: int(os.environ["RANK"])
#     process_group_backend = "nccl"
# except ModuleNotFoundError as e:
#     warnings.warn(e.msg)


# Define Dataset
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


def critic_step(gen, disc, optimizer_disc, real_images, noise, fabric):
    optimizer_disc.zero_grad()

    # Discriminate real images
    d_real_out = disc(real_images)
    d_real_loss = -torch.mean(d_real_out)

    # Discriminate fake images
    fake_images = gen(noise)
    d_fake_out = disc(fake_images.detach())
    d_fake_loss = torch.mean(d_fake_out)

    gp = gradient_penalty(real_images, fake_images, disc)

    d_loss = d_fake_loss + d_real_loss + gp * 10
    # Backward and optimize
    fabric.backward(d_loss)
    optimizer_disc.step()
    return d_loss


def generator_step(gen, disc, optimizer_gen, noise, fabric):
    optimizer_gen.zero_grad()

    # Discriminate fake images
    fake_images = gen(noise)
    d_fake_out = disc(fake_images)
    # d_fake_tgt = torch.ones(real_images.shape[0], device=real_images.device)
    # g_loss = F.binary_cross_entropy_with_logits(d_fake_out, d_fake_tgt)
    g_loss = - torch.mean(d_fake_out)

    fabric.backward(g_loss)
    optimizer_gen.step()
    return g_loss


def gradient_penalty(real_images, fake_images, critic):
    # Random weight term for interpolation between real and fake data
    alpha = torch.randn((real_images.size(0), 1, 1, 1), device=real_images.device)
    # Get random interpolation between real and fake data
    interp_images = (alpha * real_images + ((1 - alpha) * fake_images.data)).requires_grad_(True)

    critique = critic(interp_images)
    grad_outputs = torch.ones(critique.size(), device=real_images.device, requires_grad=False)

    # Get gradient w.r.t. interp_images
    gradients = torch.autograd.grad(
        outputs=critique,
        inputs=interp_images,
        grad_outputs=grad_outputs,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = torch.mean((gradients.norm(2, dim=1) - 1) ** 2)
    return gradient_penalty



if __name__ == "__main__":

    # Define hyper-parameters
    BATCH_SIZE = 8
    NUM_WORKERS = 8
    NUM_EPOCHS = 1000
    LR = 1e-4
    CHANNEL_MULTIPLIER = 2
    ADDITIONAL_CONVS = 1

    # Setup fabric instance
    my_auto_wrap_policy = functools.partial(
        size_based_auto_wrap_policy, min_num_params=20000
    )
    fsdp_strategy = FSDPStrategy(
        auto_wrap_policy=my_auto_wrap_policy,
        # cluster_environment=env,
        # process_group_backend=process_group_backend,
    )
    ddp_strategy = DDPStrategy(
        # cluster_environment=env,
        # process_group_backend=process_group_backend,
    )
    #     strategy = "auto"

    fabric = L.Fabric(strategy=ddp_strategy, num_nodes=1, devices="auto")
    fabric.launch()

    # Prepare data
    train_set = RandnDataset(10000)
    train_loader = DataLoader(dataset=train_set, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)
    train_loader = fabric.setup_dataloaders(train_loader)

    # Setup modules and optimizers
    gen = Generator(CHANNEL_MULTIPLIER, ADDITIONAL_CONVS)
    disc = Discriminator(CHANNEL_MULTIPLIER, ADDITIONAL_CONVS)

    gen = fabric.setup_module(gen)
    disc = fabric.setup_module(disc)

    optimizer_gen = torch.optim.Adam(gen.parameters(), lr=LR)
    optimizer_disc = torch.optim.Adam(disc.parameters(), lr=LR)

    optimizer_gen, optimizer_disc = fabric.setup_optimizers(optimizer_gen, optimizer_disc)

    # Training loop
    for epoch in range(NUM_EPOCHS):
        for i, batch in enumerate(train_loader):
            real_images, noise = batch
            d_loss = critic_step(gen, disc, optimizer_disc, real_images, noise, fabric)
            g_loss = generator_step(gen, disc, optimizer_gen, noise, fabric)
            if i % 5 == 0:
                fabric.print(
                    f"Epoch: {epoch + 1:0=3d}"
                    f" | STEP: {i:0=4d}"
                    f" | Generator Loss: {g_loss.item():.4f}"
                    f" | Discriminator Loss: {d_loss.item():.4f}"
                )