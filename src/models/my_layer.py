import torch.nn as nn
from base.layer import Layer


class View(nn.Module):
    def __init__(self, size):
        super(View, self).__init__()
        self.size = size

    def forward(self, tensor):
        return tensor.view(self.size)


class Conv2d(Layer):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        activation=None,
    ):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            activation,
        )


class Deconv2d(Layer):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        activation=None,
    ):
        super().__init__(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding),
            activation,
        )


class Linear(Layer):
    def __init__(
        self, in_features, out_features, activation=None,
    ):
        super().__init__(
            nn.Linear(in_features, out_features), activation,
        )
