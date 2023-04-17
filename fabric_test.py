
import lightning as L
from torch import nn, optim, utils
import torchvision

def main():
    # create models
    encoder = nn.Sequential(nn.Linear(28 * 28, 64), nn.ReLU(), nn.Linear(64, 3))
    decoder = nn.Sequential(nn.Linear(3, 64), nn.ReLU(), nn.Linear(64, 28 * 28))



    # setup data
    dataset = torchvision.datasets.MNIST(".", download=True, transform=torchvision.transforms.ToTensor())
    train_loader = utils.data.DataLoader(dataset, batch_size=64)

    # setup Fabric
    fabric = L.Fabric(strategy="fsdp")
    fabric.launch()
    encoder = fabric.setup_module(encoder)
    decoder = fabric.setup_module(decoder)
    params = list(encoder.parameters()) + list(decoder.parameters())
    optimizer = optim.Adam(params, lr=1e-3)

    optimizer = fabric.setup_optimizers(optimizer)
    # encoder, optimizer = fabric.setup(encoder, optimizer)
    # decoder = fabric.setup(decoder)
    train_loader = fabric.setup_dataloaders(train_loader)

    # train the model
    for epoch in range(2):
        fabric.print("Epoch:", epoch)
        for i, batch in enumerate(train_loader):
            # get the inputs; data is a list of [inputs, labels]
            x, y = batch
            x = x.view(x.size(0), -1)

            optimizer.zero_grad()

            # forward + loss
            z = encoder(x)
            x_hat = decoder(z)
            loss = nn.functional.mse_loss(x_hat, x)

            # backward + optimize
            fabric.backward(loss)
            optimizer.step()

            if i % 100 == 0:
                fabric.print("train_loss", float(loss))

                fabric.log("train_loss", loss)

if __name__ == "__main__":
    main()