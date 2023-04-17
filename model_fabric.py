from model_ptl import Generator, Discriminator

import torch
import torch.nn.functional as F
import torch.nn as nn


class GANFabricModel(nn.Module):
    def __init__(self, z_dim=64, hidden_dim=128, im_dim=784, lr=1e-5):
        super().__init__()
        self.z_dim = z_dim
        self.hidden_dim = hidden_dim
        self.im_dim = im_dim
        self.lr =lr

        self.gen = Generator(z_dim, hidden_dim, im_dim)
        self.disc = Discriminator(im_dim, hidden_dim)

    def forward_training(self):
        pass

    def forward(self):
        pass
