import torch.nn as nn
from models.my_layer import View, Conv2d, Deconv2d, Linear


class CAE(nn.Module):
    def __init__(self, args):
        super(CAE, self).__init__()
        self.recon_loss = nn.MSELoss()
        encode_dim = args.encode_dim

        self.encoder = nn.Sequential(
            Conv2d(1, 32, 4, stride=2, padding=1, activation="relu"),
            Conv2d(32, 32, 4, stride=2, padding=1, activation="relu"),
            Conv2d(32, 64, 4, stride=2, padding=1, activation="relu"),
            Conv2d(64, 64, 4, stride=2, padding=1, activation="relu"),
            View((-1, 64 * 4 * 4)),
            Linear(64 * 4 * 4, 256, activation="relu"),
            nn.Linear(256, encode_dim),
        )
        self.decoder = nn.Sequential(
            Linear(encode_dim, 256, activation="relu"),
            Linear(256, 64 * 4 * 4, activation="relu"),
            View((-1, 64, 4, 4)),
            Deconv2d(64, 64, 4, stride=2, padding=1, activation="relu"),
            Deconv2d(64, 32, 4, stride=2, padding=1, activation="relu"),
            Deconv2d(32, 32, 4, stride=2, padding=1, activation="relu"),
            Deconv2d(32, 1, 4, stride=2, padding=1, activation="sigmoid"),
        )

    def forward(self, x):
        z = self.encoder(x)
        x_recon = self.decoder(z)

        return {"x_recon": x_recon, "z": z}

    def loss(self, outputs, x):
        x_recon = outputs["x_recon"]

        return {"loss": self.recon_loss(x, x_recon)}
