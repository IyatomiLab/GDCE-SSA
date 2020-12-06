import torch
import torch.nn as nn


class SemanticSubcharacterAugmentation(nn.Module):
    def __init__(self, args):
        super(SemanticSubcharacterAugmentation, self).__init__()
        self.encode_dim = args.encode_dim
        self.char_len = args.char_len
        self.gamma = abs(args.gamma)

    def forward(self, x):
        if not self.training:
            return x

        batch_size = x.size(0)
        mask = (
            torch.eye(self.encode_dim)[
                torch.randint(0, self.encode_dim, (batch_size * self.char_len,))
            ]
            .view(batch_size, self.char_len, self.encode_dim)
            .unsqueeze(2)
            .transpose(3, 1)
            .to(x.device)
        )
        perturbation = torch.empty_like(x).uniform_(-self.gamma, self.gamma)
        perturbation = (perturbation * mask).to(x.dtype)

        return x + perturbation
