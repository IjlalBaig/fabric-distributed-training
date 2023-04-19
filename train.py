from model import WGAN_GP
import torch
from lightning.fabric.strategies import FSDPStrategy, DDPStrategy
from torch.utils.data import DataLoader
torch.manual_seed(0)
import lightning as L
from torchvision.datasets import CIFAR10
import torchvision.transforms as T
import traceback
if __name__ == "__main__":

    BATCH_SIZE = 64
    NUM_WORKERS = 8
    NUM_EPOCHS = 1000
    NUM_CRITIC_STEPS = 1

    # Prepare Distributed Training Strategy
    strategy = FSDPStrategy()
    # strategy = DDPStrategy(find_unused_parameters=True)
    # strategy = "auto"
    fabric = L.Fabric(strategy=strategy, devices=[0])
    fabric.launch()

    train_set = CIFAR10(root='./data', train=True, download=True, transform=lambda x: T.ToTensor()(x))
    train_loader = DataLoader(dataset=train_set, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)
    train_loader = fabric.setup_dataloaders(train_loader)

    # model
    model = WGAN_GP(fabric, lr=1e-5, noise_dim=100, channel_multiplier=1, additional_convs=100)

    # Training loop
    for epoch in range(NUM_EPOCHS):
        fabric.print("Epoch:", epoch)

        losses_gen = dict(gen_loss=0.)
        for i, batch in enumerate(train_loader):
            real_images, _ = batch
            n = real_images.size(0)
            noise = torch.randn(n, 100, 1, 1, device=real_images.device)

            losses_critic = model.critic_training_step(real_images, noise, fabric)
            if (i + 1) % NUM_CRITIC_STEPS == 0:
                losses_gen = model.generator_training_step(noise, fabric)

            if i % 100 == 0:
                fabric.print(f"losses: gen ({losses_gen['gen_loss']:.2f}), critic ({losses_critic['critic_loss']:.2f})")