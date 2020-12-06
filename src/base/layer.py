import torch.nn as nn


class Layer(nn.Module):
    def __init__(
        self, layer, activation=None,
    ):
        super(Layer, self).__init__()
        assert activation is not None, "activation is None"

        self.layer = layer
        self.activation_name = activation

        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "sigmoid":
            self.activation = nn.Sigmoid()

    def forward(self, x):
        x = self.layer(x)
        x = self.activation(x)

        return x
