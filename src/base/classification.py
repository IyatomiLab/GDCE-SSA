import torch
import torch.nn as nn
from models.ssa import SemanticSubcharacterAugmentation


class Classification(nn.Module):
    def __init__(self, args):
        super(Classification, self).__init__()

        if args.da == "wt":
            self.da = nn.Dropout(args.wildcard_ratio, inplace=True)
        elif args.da == "ssa":
            self.da = SemanticSubcharacterAugmentation(args=args)
        else:
            self.da = lambda x: x

        self.criterion = nn.CrossEntropyLoss()
        self.encode_dim = args.encode_dim
        self.num_class = args.num_class

    def forward(self, x):
        pass

    def accuracy(self, output, labels):
        predicts = torch.argmax(output, dim=1)
        return (predicts == labels).sum().float() / predicts.size(0)

    def loss(self, output, labels):
        return self.criterion(output, labels)
