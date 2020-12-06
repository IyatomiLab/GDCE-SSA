import torch
from models import ARCHS
from util.args import get_args
from util.path import get_path
from util.seed import seed
from pathlib import Path
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger


def main():
    seed()

    args = get_args()
    path = get_path(args)

    model = ARCHS["classification"](args=args, path=path)

    logger = TensorBoardLogger(
        save_dir=path["save_dir"],
        name=path["classification"]["name"],
        version=args.classification_version,
    )
    early_stop = EarlyStopping(
        monitor="validation/loss",
        patience=args.patience,
        verbose=args.verbose,
        mode="min",
    )

    checkpoint = ModelCheckpoint(
        filepath=f"{logger.log_dir}/",
        save_weights_only=True,
        verbose=False,
        monitor="validation/loss",
        mode="min",
    )

    trainer = Trainer(
        logger=logger,
        max_epochs=args.num_epoch,
        checkpoint_callback=checkpoint,
        early_stop_callback=early_stop,
        gpus=args.gpu,
    )
    trainer.fit(model)

    # test
    checkpoint_path = list(Path(logger.log_dir).glob("*.ckpt"))[0]
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    model.freeze()
    if args.dataset == "newspaper":
        model.newspaper_test()
    if args.dataset == "livedoor":
        trainer.test(model)


if __name__ == "__main__":
    main()
