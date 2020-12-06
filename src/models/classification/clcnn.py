import torch.nn as nn
from models.my_layer import Conv2d, View
from base.classification import Classification


class CLCNN(Classification):
    def __init__(self, args):
        super().__init__(args)

        self.features = nn.Sequential(
            Conv2d(self.encode_dim, 512, kernel_size=(1, 3), activation="relu"),
            nn.MaxPool2d((1, 3), stride=3),
            Conv2d(512, 512, kernel_size=(1, 3), activation="relu"),
            nn.MaxPool2d((1, 3), stride=3),
            Conv2d(512, 512, kernel_size=(1, 3), activation="relu"),
            Conv2d(512, 512, kernel_size=(1, 3), activation="relu"),
        )

        if args.dataset == "livedoor":
            feature_dim = 512 * 4
        elif args.dataset == "newspaper":
            feature_dim = 512 * 9

        self.classifier = nn.Sequential(
            View((-1, feature_dim)), nn.Linear(feature_dim, self.num_class)
        )

    def forward(self, x):
        x = self.da(x)
        x = self.features(x)
        x = self.classifier(x)

        return x
