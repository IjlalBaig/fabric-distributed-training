import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L


class Generator(nn.Module):
    def __init__(self, z_dim=10, hidden_dim=128, im_dim=784):
        # todo: use conv2d instead
        super(Generator, self).__init__()
        self.encoder = nn.Sequential(
            self._generator_block(z_dim, hidden_dim),
            self._generator_block(hidden_dim, hidden_dim * 2),
            self._generator_block(hidden_dim * 2, hidden_dim * 4),
            nn.Linear(hidden_dim * 4, im_dim),
            nn.Sigmoid(),
        )

    def _generator_block(self, input_dim, output_dim):
        return nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, z):
        return self.encoder(z)


class Discriminator(nn.Module):
    def __init__(self, im_dim=784, hidden_dim=128):
        super(Discriminator, self).__init__()

        self.decoder = nn.Sequential(
            self._discriminator_block(im_dim, hidden_dim * 4),
            self._discriminator_block(hidden_dim * 4, hidden_dim),
            # self._discriminator_block(hidden_dim * 2, hidden_dim),
            nn.Linear(hidden_dim, 1),
        )

    def _discriminator_block(self, input_dim, output_dim):
        return nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )

    def forward(self, x):
        x = self.decoder(x)
        return x


class GANModel(L.LightningModule):
    def __init__(self, z_dim=64, hidden_dim=128, im_dim=784, lr=1e-5):
        super().__init__()
        self.save_hyperparameters()
        self.automatic_optimization = False

        self.gen = Generator(z_dim, hidden_dim, im_dim)
        self.disc = Discriminator(im_dim, hidden_dim)

    def forward(self, z):
        return self.gen(z)

    def configure_optimizers(self):
        opt_g = torch.optim.Adam(self.gen.parameters(), lr=self.hparams.lr)
        opt_d = torch.optim.Adam(self.disc.parameters(), lr=self.hparams.lr)
        return [opt_g, opt_d], []

    def training_step(self, batch, batch_idx):
        real = batch["image"]
        opt_g, opt_d, = self.optimizers()

        # sample noise
        n, l = real.shape

        z = torch.randn(n, self.hparams.z_dim, device=self.device)

        # train generator
        self.toggle_optimizer(opt_g)
        fake = self.gen(z)
        d_fake = self.disc(fake)

        labels = torch.ones(n, 1, device=self.device)
        g_loss = F.binary_cross_entropy_with_logits(d_fake, labels)
        self.log("g_loss", g_loss, prog_bar=True)

        self.manual_backward(g_loss)
        opt_g.step()
        opt_g.zero_grad()
        self.untoggle_optimizer(opt_g)

        # train discriminator
        self.toggle_optimizer(opt_d)
        d_real = self.disc(real)
        label_real = torch.ones(n, 1, device=self.device)
        loss_d_real = F.binary_cross_entropy_with_logits(d_real, label_real)

        fake = self.gen(z)
        d_fake = self.disc(fake.detach())
        label_fake = torch.zeros(n, 1, device=self.device)
        loss_d_fake = F.binary_cross_entropy_with_logits(d_fake, label_fake)

        d_loss = (loss_d_real + loss_d_fake) / 2
        self.log("d_loss", d_loss, prog_bar=True)
        self.manual_backward(d_loss)
        opt_d.step()
        opt_d.zero_grad()
        self.untoggle_optimizer(opt_d)





