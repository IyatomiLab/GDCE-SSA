import argparse


def get_args():
    parser = argparse.ArgumentParser("GDCE_SSA")

    # dataset
    parser.add_argument(
        "--dataset", default="livedoor", choices=["newspaper", "livedoor"], type=str
    )
    parser.add_argument("--char_len", default=80, type=int)
    parser.add_argument("--num_class", type=int)

    # data loader
    parser.add_argument("--train_batch", default=256, type=int)
    parser.add_argument("--val_batch", default=256, type=int)
    parser.add_argument("--test_batch", default=256, type=int)
    parser.add_argument("--num_workers", default=8, type=int)

    # model
    parser.add_argument(
        "--character_encoder",
        default="BetaVAE",
        choices=["BetaVAE", "CAE"],
        type=str,
    )
    parser.add_argument(
        "--classification",
        default="CLCNN",
        choices=["CLCNN"],
        type=str,
    )

    # hparam
    parser.add_argument("--beta", default=8.0, type=float)
    parser.add_argument("--encode_dim", default=10, type=int)

    # optimizer
    parser.add_argument("--lr", default=1e-4, type=float)
    parser.add_argument("--weight_decay", default=1e-4, type=float)

    # data augmentation
    parser.add_argument("--da", default="", choices=["wt", "ssa"], type=str)

    # wildcard training
    parser.add_argument("--wildcard_ratio", default=0.1, type=float)

    # SSA
    parser.add_argument("--gamma", default=2.0, type=float)

    # early stopping
    parser.add_argument("--patience", default=70, type=int)
    parser.add_argument("--verbose", default=True, type=bool)

    # tensorboard logger
    parser.add_argument("--character_encoder_version", default=0, type=int)
    parser.add_argument("--classification_version", default=0, type=int)

    # trainer
    parser.add_argument("--num_epoch", default=3000, type=int)
    parser.add_argument("--gpu", default=[0], type=list)

    args = parser.parse_args()
    return args
