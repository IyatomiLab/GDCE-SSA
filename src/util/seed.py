import torch
import random
import numpy as np


def worker_init_fn(worker_id):
    random.seed(worker_id + 42)
    np.random.seed(worker_id + 42)


def seed():
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    random.seed(42)
    np.random.seed(42)
