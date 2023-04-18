import torch
import torch.nn as nn
import lightning as L


class Generator(torch.nn.Module):
    def __init__(self, channel_multiplier=1):
        super().__init__()

        self.main = nn.Sequential(
            nn.ConvTranspose2d(100, 512 * channel_multiplier, 4, 1, 0, bias=False),
            nn.BatchNorm2d(512 * channel_multiplier),
            nn.ReLU(True),

            nn.ConvTranspose2d(512 * channel_multiplier, 256 * channel_multiplier, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256 * channel_multiplier),
            nn.ReLU(True),

            nn.ConvTranspose2d(256 * channel_multiplier, 128 * channel_multiplier, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128 * channel_multiplier),
            nn.ReLU(True),

            nn.ConvTranspose2d(128 * channel_multiplier, 3, 4, 2, 1),
            nn.Tanh()
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        out = self.main(input)
        return out


class Discriminator(nn.Module):
    def __init__(self, channel_multiplier=1):
        super(Discriminator, self).__init__()

        self.main = nn.Sequential(
            nn.Conv2d(3, 64 * channel_multiplier, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(64 * channel_multiplier, 128 * channel_multiplier, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(128 * channel_multiplier, affine=True),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(128 * channel_multiplier, 256 * channel_multiplier, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(256 * channel_multiplier, affine=True),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(256 * channel_multiplier, 1, 4, 1, 0),
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        out = self.main(input)
        out = torch.flatten(out)
        return out


class WGAN_GP(nn.Module):
    def __init__(self, fabric: L.Fabric, lr: float = 1e-5, noise_dim: int = 100, channel_multiplier: int = 1):
        super().__init__()
        self.fabric = fabric
        self.lr = lr
        self.noise_dim = noise_dim
        self.channel_multiplier = channel_multiplier

        self.gen, self.critic = self.prepare_modules()
        self.optimizer_gen, self.optimizer_critic = self.prepare_optimizers()

    def prepare_modules(self):
        gen = Generator(self.channel_multiplier)
        gen = self.fabric.setup_module(gen)

        critic = Discriminator(self.channel_multiplier)
        critic = self.fabric.setup_module(critic)
        return gen, critic

    def prepare_optimizers(self):
        optimizer_g = torch.optim.Adam(self.gen.parameters(), lr=self.lr)
        optimizer_g = self.fabric.setup_optimizers(optimizer_g)

        optimizer_c = torch.optim.Adam(self.critic.parameters(), lr=self.lr)
        optimizer_c = self.fabric.setup_optimizers(optimizer_c)
        return optimizer_g, optimizer_c

    def gradient_penalty(self, real_images, fake_images):
        # Random weight term for interpolation between real and fake data
        alpha = torch.randn((real_images.size(0), 1, 1, 1), device=real_images.device)
        # Get random interpolation between real and fake data
        interp_images = (alpha * real_images + ((1 - alpha) * fake_images.data)).requires_grad_(True)

        critique = self.critic(interp_images)
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

    def critic_training_step(self, real_images: torch.Tensor, noise: torch.Tensor, fabric: L.Fabric):
        self.optimizer_critic.zero_grad()

        # Critique real images
        real_critique = self.critic(real_images)
        err_critic_real = -torch.mean(real_critique)

        # Critique fake images
        fake_images = self.gen(noise)
        fake_critique = self.critic(fake_images.detach())
        err_critic_fake = torch.mean(fake_critique)

        # Gradient penalty
        # gp = self.gradient_penalty(real_images, fake_images)

        # Combine losses
        loss = err_critic_real + err_critic_fake # + gp * 10

        # Backward and optimize
        fabric.backward(loss)
        self.optimizer_critic.step()

        return dict(critic_loss=loss)

    def generator_training_step(self, noise: torch.Tensor, fabric: L.Fabric):
        self.optimizer_gen.zero_grad()

        # Critique fake images
        fake_images = self.gen(noise)
        fake_critique = self.critic(fake_images.detach())
        loss = - torch.mean(fake_critique)

        # Backward and optimize
        fabric.backward(loss)
        self.optimizer_gen.step()

        return dict(gen_loss=loss)

    def forward(self, noise: torch.Tensor):
        return self.gen(noise)


