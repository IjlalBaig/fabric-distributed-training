import torch
from pytorch_lightning.strategies import FSDPStrategy, DDPStrategy

from model_ptl import GANModel


import lightning as L
from lightning.pytorch.strategies import FSDPStrategy, DDPStrategy
from randn_dataset import RandnDataset
from torch.utils.data import DataLoader
torch.manual_seed(0)

if __name__ == "__main__":

    batch_size = 8
    num_workers = 8
    num_train_samples = 10000
    num_val_samples = 1000

    strategy = FSDPStrategy(use_orig_params=True, cpu_offload=True)
    # strategy = DDPStrategy(find_unused_parameters=True)
    # data
    train_set = RandnDataset(num_train_samples)
    train_loader = DataLoader(dataset=train_set, batch_size=batch_size, num_workers=num_workers)

    # model
    model = GANModel(z_dim=64, hidden_dim=128, im_dim=784, lr=1e-5)

    # training

    trainer = L.Trainer(max_epochs=1000, num_nodes=1, devices=[0],  strategy=strategy)
    trainer.fit(model, train_loader)