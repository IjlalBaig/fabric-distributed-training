import torch
import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, z_dim=10, hidden_dim=128, im_dim=784):
        super(Generator, self).__init__()

        self.encoder = nn.Sequential(
            self._generator_block(z_dim, hidden_dim),
            self._generator_block(hidden_dim, hidden_dim*2),
            self._generator_block(hidden_dim*2, hidden_dim*4),
            nn.Linear(hidden_dim*4, im_dim),
            nn.Sigmoid(),
        )
    
    def _generator_block(self, input_dim, output_dim):
        return nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, x):
        x = self.encoder(x)
        return x

class Discriminator(nn.Module):
    def __init__(self, im_dim=784, hidden_dim=128):
        super(Discriminator, self).__init__()

        self.decoder = nn.Sequential(
            self._discriminator_block(im_dim, hidden_dim*4),
            self._discriminator_block(hidden_dim*4, hidden_dim*2),
            self._discriminator_block(hidden_dim*2, hidden_dim),
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