import torch
from pytorch_lightning.strategies import FSDPStrategy
from model_ptl import GANModel

import pytorch_lightning as pl
from randn_dataset import RandnDataset
from torch.utils.data import DataLoader
torch.manual_seed(0)

if __name__ == "__main__":
    batch_size = 8
    num_workers = 8
    num_train_samples = 10000
    num_val_samples = 1000

    # data
    train_set = RandnDataset(num_train_samples)
    train_loader = DataLoader(dataset=train_set, batch_size=batch_size, num_workers=num_workers)

    # model
    model = GANModel(z_dim=64, hidden_dim=128, im_dim=784, lr=1e-5)

    # training
    trainer = pl.Trainer(num_nodes=1, devices=[1, 2, 3], max_epochs=1000, )
    trainer.fit(model, train_loader)