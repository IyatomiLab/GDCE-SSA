import torch
import pickle
import pytorch_lightning as pl
from . import CharacterEncoder
from util.seed import worker_init_fn
from util.dataset import DATASET
from torch.utils.data import DataLoader


class Model(pl.LightningModule):
    def __init__(self, args, path):
        super(Model, self).__init__()
        self.hparams = args

        self.path = path
        self.char2embedding_path = path["char2embedding"]
        self.model = CharacterEncoder[args.character_encoder](args)

    def forward(self, x):
        outputs = self.model(x)

        return outputs

    def training_step(self, batch, batch_nb):
        data, _ = batch
        outputs = self.forward(data)

        train_loss = self.model.loss(outputs, data)
        log = {f"train/{name}": loss for name, loss in train_loss.items()}

        return {"loss": train_loss["loss"], "log": log}

    def test_step(self, batch, batch_nb):
        data, labels = batch
        outputs = self.forward(data)
        test_loss = self.model.loss(outputs, data)["loss"]

        if self.hparams.character_encoder != "CAE":
            outputs["std"] = torch.exp(0.5 * outputs["logvar"])[0]
            outputs["mu"] = outputs["mu"][0]

        outputs["char_img"] = data
        outputs["label"] = labels[0]
        outputs["test_loss"] = test_loss
        outputs["z"] = outputs["z"][0]

        return outputs

    def test_epoch_end(self, outputs):
        avg_test_loss = torch.stack([x["test_loss"] for x in outputs]).mean()

        if self.hparams.character_encoder == "CAE":
            char2embedding = {x["label"]: x["z"].cpu().numpy() for x in outputs}
            with open(self.char2embedding_path, "wb") as f:
                pickle.dump(char2embedding, f)
        else:
            std = torch.stack([x["std"] for x in outputs]).mean()
            mu = torch.stack([x["mu"] for x in outputs]).mean()
            self.logger.experiment.add_text(
                tag="test/mean_std",
                text_string=f"mean: {mu}\nstd: {std}",
                global_step=0,
            )
            for embedding_idx in range(self.hparams.encode_dim):
                embedding_std = torch.stack([x["std"][embedding_idx] for x in outputs])
                embedding_mu = torch.stack([x["mu"][embedding_idx] for x in outputs])

                self.logger.experiment.add_histogram(
                    "embedding_std", embedding_std, embedding_idx
                )
                self.logger.experiment.add_histogram(
                    "embedding_mu", embedding_mu, embedding_idx
                )

        for embedding_idx in range(self.hparams.encode_dim):
            embedding = torch.stack([x["z"][embedding_idx] for x in outputs])

            self.logger.experiment.add_histogram("embedding", embedding, embedding_idx)

        mat = []
        metadata = []
        label_img = []

        for output in outputs:
            mat.append(output["z"])
            metadata.append(output["label"])
            label_img.append(output["char_img"])

        mat = torch.stack(mat, dim=0)
        label_img = torch.stack(label_img, dim=1)[0]

        self.logger.experiment.add_embedding(
            mat=mat, metadata=metadata, label_img=label_img
        )

        return {"progress_bar": {"test/loss": avg_test_loss}}

    def configure_optimizers(self):
        return torch.optim.Adam(
            self.model.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )

    def prepare_data(self):
        train_dataset = DATASET["ja_chars"](self.path)
        test_dataset = DATASET["ja_chars"](self.path)

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
        self.test_loader = DataLoader(
            test_dataset, batch_size=1, shuffle=False, **dataloader_param
        )

    def train_dataloader(self):
        return self.train_loader

    def test_dataloader(self):
        return self.test_loader
