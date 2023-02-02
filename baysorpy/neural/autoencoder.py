import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl


class Encoder(nn.Module):
    def __init__(self, input_size, l1_size, hidden_size):
        super().__init__()
        self.l1 = nn.Sequential(nn.Linear(input_size, l1_size), nn.ReLU(), nn.Linear(l1_size, hidden_size))

    def forward(self, x):
        return self.l1(x)


class Decoder(nn.Module):
    def __init__(self, output_size, l1_size, hidden_size):
        super().__init__()
        self.l1 = nn.Sequential(nn.Linear(hidden_size, l1_size), nn.ReLU(), nn.Linear(l1_size, output_size))

    def forward(self, x):
        return self.l1(x)


class AutoEncoder(pl.LightningModule):
    def __init__(
        self,
        input_size: int,
        l1_size: int,
        hidden_size: int,
        encoder_class: object = Encoder,
        decoder_class: object = Decoder
    ):
        super().__init__()
        self.save_hyperparameters()

        self.encoder = encoder_class(input_size, l1_size, hidden_size)
        self.decoder = decoder_class(input_size, l1_size, hidden_size)
        # Example input array needed for visualizing the graph of the network
        self.example_input_array = torch.zeros(input_size)

    def forward(self, x):
        """The forward function takes in an image and returns the reconstructed image."""
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat

    def _get_reconstruction_loss(self, batch):
        x, = batch
        x_hat = self.forward(x)
        loss = F.mse_loss(x_hat, x)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)

        # TODO: uncomment
        # # Using a scheduler is optional but can be helpful.
        # # The scheduler reduces the LR if the validation performance hasn't improved for the last N epochs
        # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.2, patience=20, min_lr=5e-5)
        # return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val_loss"}

        return optimizer


    def training_step(self, batch, batch_idx):
        loss = self._get_reconstruction_loss(batch)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        self.log("val_loss", self._get_reconstruction_loss(batch))

#     def test_step(self, batch, batch_idx):
#         self.log("test_loss", self._get_reconstruction_loss(batch))