import torch
import torch.nn as nn
from torch.nn import functional as F
from models.my_layer import View, Conv2d, Deconv2d, Linear


class BetaVAE(nn.Module):
    def __init__(self, args):
        super(BetaVAE, self).__init__()
        self.beta = args.beta
        self.encode_dim = args.encode_dim

        self.encoder = nn.Sequential(
            Conv2d(1, 32, 4, stride=2, padding=1, activation="relu"),
            Conv2d(32, 32, 4, stride=2, padding=1, activation="relu"),
            Conv2d(32, 64, 4, stride=2, padding=1, activation="relu"),
            Conv2d(64, 64, 4, stride=2, padding=1, activation="relu"),
            View((-1, 64 * 4 * 4)),
            Linear(64 * 4 * 4, 256, activation="relu"),
            nn.Linear(256, self.encode_dim * 2),
        )
        self.decoder = nn.Sequential(
            Linear(self.encode_dim, 256, activation="relu"),
            Linear(256, 64 * 4 * 4, activation="relu"),
            View((-1, 64, 4, 4)),
            Deconv2d(64, 64, 4, stride=2, padding=1, activation="relu"),
            Deconv2d(64, 32, 4, stride=2, padding=1, activation="relu"),
            Deconv2d(32, 32, 4, stride=2, padding=1, activation="relu"),
            Deconv2d(32, 1, 4, stride=2, padding=1, activation="sigmoid"),
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)

        return mu + eps * std

    def forward(self, x):
        mu_logvar = self.encoder(x)
        mu = mu_logvar[:, : self.encode_dim]
        logvar = mu_logvar[:, self.encode_dim :]

        z = self.reparameterize(mu, logvar)

        x_recon = self.decoder(z)

        return {"x_recon": x_recon, "z": z, "mu": mu, "logvar": logvar}

    def loss(self, outputs, x):
        x_recon = outputs["x_recon"]
        mu = outputs["mu"]
        logvar = outputs["logvar"]

        bce = F.binary_cross_entropy(x_recon, x, reduction="sum")
        kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        return {
            "loss": (bce + self.beta * kld) / x.size(0),
            "recon_loss": bce / x.size(0),
            "kld_loss": kld / x.size(0),
        }
