import lightning

from model_ptl import Generator, Discriminator
import torch
from lightning.fabric.strategies import FSDPStrategy
from randn_dataset import RandnDataset
from torch.utils.data import DataLoader
torch.manual_seed(0)
import lightning as L
import torch.nn.functional as F

if __name__ == "__main__":

    batch_size = 8
    num_workers = 8
    num_train_samples = 10000
    num_val_samples = 1000

    strategy = FSDPStrategy()
    # strategy = DDPStrategy(find_unused_parameters=True)
    # data
    fabric = L.Fabric(strategy=strategy, devices=[0])
    fabric.launch()

    train_set = RandnDataset(num_train_samples)
    train_loader = DataLoader(dataset=train_set, batch_size=batch_size, num_workers=num_workers)
    train_loader = fabric.setup_dataloaders(train_loader)

    # model
    z_dim = 64
    hidden_dim = 128
    im_dim = 784
    lr = 1e-5

    gen = Generator(z_dim, hidden_dim, im_dim)
    gen = fabric.setup_module(gen)

    disc = Discriminator(im_dim, hidden_dim)
    disc = fabric.setup_module(disc)

    opt_g = torch.optim.Adam(gen.parameters(), lr=lr)
    opt_g = fabric.setup_optimizers(opt_g)

    opt_d = torch.optim.Adam(disc.parameters(), lr=lr)
    opt_d = fabric.setup_optimizers(opt_d)

    # Training loop
    for epoch in range(2):
        fabric.print("Epoch:", epoch)
        for i, batch in enumerate(train_loader):
            real = batch["image"]

            # sample noise
            n, l = real.shape
            z = torch.randn(n, z_dim, device=real.device)

            # train generator
            opt_g.zero_grad()
            fake = gen(z)
            d_fake = disc(fake)

            labels = torch.ones(n, 1, device=real.device)
            g_loss = F.binary_cross_entropy_with_logits(d_fake, labels)

            # backward and optimize
            fabric.backward(g_loss)
            opt_g.step()

            # train discriminator
            opt_d.zero_grad()
            d_real = disc(real)
            label_real = torch.ones(n, 1, device=real.device)
            loss_d_real = F.binary_cross_entropy_with_logits(d_real, label_real)

            fake = gen(z)
            d_fake = disc(fake.detach())
            label_fake = torch.zeros(n, 1, device=real.device)
            loss_d_fake = F.binary_cross_entropy_with_logits(d_fake, label_fake)

            d_loss = (loss_d_real + loss_d_fake) / 2
            if i % 1000 == 0:
                fabric.print("g_loss", float(g_loss))
                fabric.print("d_loss", float(d_loss))


