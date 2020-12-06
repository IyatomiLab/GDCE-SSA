import torch
import pytorch_lightning as pl
from . import Classification
from models.character_encoder import CharacterEncoder
from util.seed import worker_init_fn
from util.dataset import DATASET
from tqdm import tqdm
from pathlib import Path
from torch.utils.data import DataLoader


class Model(pl.LightningModule):
    def __init__(self, args, path):
        super(Model, self).__init__()

        if args.dataset == "newspaper":
            args.num_class = 4
        elif args.dataset == "livedoor":
            args.num_class = 9

        self.hparams = args

        self.path = path

        if args.character_encoder == "BetaVAE":
            self.character_encoder = CharacterEncoder[args.character_encoder](args)
            checkpoint_path = list(
                (
                    path["save_dir"]
                    / Path(path["character_encoder"]["name"])
                    / f"version_{args.character_encoder_version}"
                ).glob("*.ckpt")
            )[0]

            checkpoint = torch.load(checkpoint_path)
            model_checkpoint = {
                key.replace("model.", ""): value
                for key, value in checkpoint["state_dict"].items()
            }
            self.character_encoder.load_state_dict(model_checkpoint)
            self.character_encoder.eval()

        self.model = Classification[args.classification](self.hparams)

    def forward(self, x):
        if self.hparams.character_encoder == "BetaVAE":
            with torch.no_grad():
                x = torch.stack(
                    [self.character_encoder(char_img)["z"].detach() for char_img in x]
                )
                x = x.unsqueeze(2).transpose(3, 1)

        output = self.model(x)

        return output

    def training_step(self, batch, batch_nb):
        data, target = batch
        output = self.forward(data)

        train_loss = self.model.criterion(output, target)
        train_acc = self.model.accuracy(output, target)

        return {
            "loss": train_loss,
            "progress_bar": {"train/acc": train_acc},
            "log": {"train/acc": train_acc, "train/loss": train_loss},
        }

    def validation_step(self, batch, batch_nb):
        data, target = batch
        output = self.forward(data)

        val_loss = self.model.criterion(output, target)
        val_acc = self.model.accuracy(output, target)

        return {"val_loss": val_loss, "val_acc": val_acc}

    def validation_epoch_end(self, outputs):
        avg_val_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        avg_val_acc = torch.stack([x["val_acc"] for x in outputs]).mean()

        return {
            "progress_bar": {"validation/acc": avg_val_acc},
            "log": {"validation/acc": avg_val_acc, "validation/loss": avg_val_loss},
        }

    def test_step(self, batch, batch_nb):
        data, target = batch
        output = self.forward(data)

        test_acc = self.model.accuracy(output, target)

        return {"test_acc": test_acc}

    def test_epoch_end(self, outputs):
        avg_test_acc = torch.stack([x["test_acc"] for x in outputs]).mean()
        self.logger.experiment.add_text(
            tag="classification",
            text_string=f"test accuracy: {avg_test_acc}",
            global_step=0,
        )

        return {
            "progress_bar": {"test/acc": avg_test_acc},
            "log": {"test/acc": avg_test_acc},
        }

    def newspaper_test(self):
        predicts = torch.tensor(
            [
                self.newspaper_test_predict(subseqs)
                for subseqs, _ in tqdm(self.test_dataset)
            ]
        )
        targets = torch.tensor([target for _, target in self.test_dataset])
        accuracy = (predicts == targets).sum().float() / targets.size(0)

        self.logger.experiment.add_text(
            tag="classification",
            text_string=f"test accuracy: {accuracy}",
            global_step=0,
        )

        self.logger.experiment.close()

    def newspaper_test_predict(self, subseqs):
        subseq_length = len(subseqs)
        subseq_batches = [
            subseqs[b : b + self.hparams.test_batch]
            for b in range(0, subseq_length, self.hparams.test_batch)
        ]
        output = torch.stack(
            [
                self.forward(torch.stack(subseq).to("cuda"))
                .to("cpu")
                .detach()
                .mean(dim=0)
                for subseq in subseq_batches
            ]
        ).sum(dim=0)
        predict = torch.argmax(output)

        return predict

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
        return optimizer

    def prepare_data(self):
        dataset_param = {
            "args": self.hparams,
            "path": self.path,
        }
        train_dataset = DATASET[self.hparams.dataset](**dataset_param)
        val_dataset = DATASET[self.hparams.dataset](test=True, **dataset_param)
        test_dataset = DATASET[self.hparams.dataset](
            test=True, slide=True, **dataset_param
        )
        self.test_dataset = test_dataset

        dataloader_param = {
            "num_workers": self.hparams.num_workers,
            "pin_memory": True,
            "worker_init_fn": worker_init_fn,
        }
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.hparams.train_batch,
            shuffle=True,
            **dataloader_param,
        )
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=self.hparams.val_batch,
            shuffle=False,
            **dataloader_param,
        )

        self.test_loader = DataLoader(
            test_dataset,
            batch_size=self.hparams.test_batch,
            shuffle=False,
            **dataloader_param,
        )

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.val_loader

    def test_dataloader(self):
        return self.test_loader
