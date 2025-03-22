from pathlib import Path
from typing import Callable

import torch
from torch import types
from torch.utils.data import Dataset, DataLoader
from typing_extensions import Optional

import torchcv_tmpl
from dataset import MNISTDataset
from model import MnistModel
from torchcv_tmpl.trainer import BaseTrainer
from torchcv_tmpl.utils.plotter import Plotter


class MnistCsvTrainer(BaseTrainer):

    def __init__(self,
                 csv_train_path: str | Path,
                 csv_validate_path: str | Path,
                 max_epoch: int,
                 batch_size: int,
                 learning_rate: float = 0.001,
                 save_folder_path: Optional[Path | str] = None,
                 validate_period: Optional[int] = None,
                 load_checkpoint_path: Optional[Path] = None,
                 ):
        self.csv_train_path = Path(csv_train_path)
        self.csv_validate_path = Path(csv_validate_path)
        self.learning_rate = learning_rate
        self.device = torchcv_tmpl.utils.hardware.auto_choice_device()
        super().__init__(
            max_epoch=max_epoch,
            batch_size=batch_size,
            save_folder_path=save_folder_path,
            save_best_for_metric='accuracy',
            have_validate=True,
            validate_period=validate_period,
            load_checkpoint_path=load_checkpoint_path,
        )

    def build_model(self) -> torch.nn.Module:
        return MnistModel(
            in_channels=784,
            hidden_channels=[2500, 2000, 1500, 1000, 500, 10],
            activation_layer=torch.nn.ReLU,
            bias=True,
            dropout=0.3
        )

    def build_validate_dataset(self) -> Dataset:
        return MNISTDataset(self.csv_validate_path, False)

    def build_train_dataset(self) -> Dataset:
        return MNISTDataset(self.csv_train_path, True)

    def build_train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.build_train_dataset(),
            batch_size=self.batch_size,
            shuffle=True,
        )

    def build_validate_dataloader(self) -> DataLoader:
        return DataLoader(
            self.build_validate_dataset(),
            batch_size=self.batch_size,
        )

    def build_loss_func(self) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
        return torch.nn.CrossEntropyLoss()

    def build_optimizer(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

    def build_scheduler(self) -> Optional[torch.optim.lr_scheduler.LRScheduler]:
        return None

    def _setup_train(self):
        self.model.to(self.device)

    def preprocessing_batch(self, batch: dict | list) -> dict | list:
        return [n.to(self.device) for n in batch]

    def _train_step(self, batch: dict | list) -> dict[str, types.Number]:
        X, Y = batch
        X = X.view(X.size(0), -1)

        self.optimizer.zero_grad()

        Y_pred = self.model(X)
        loss = self.loss_func(Y_pred, Y)

        loss.backward()
        self.optimizer.step()

        return dict(loss=loss.item())

    def _validate_step(self, batch: dict | list) -> dict[str, types.Number]:
        X, Y = batch

        with torch.inference_mode():
            X = X.view(X.size(0), -1)

            Y_pred: torch.Tensor = self.model(X)

            _, Y_pred = torch.max(Y_pred, 1)
            acc = 100 * (Y_pred == Y).sum() / Y_pred.shape[0]
        return dict(accuracy=acc.item())

    def plot_history(self, history_metric: dict):
        plotter = Plotter(
            history_metric,
            self.save_folder_path / 'figure',
            self.log,
        )
        plotter.epoch_series('train_epoch', [('train', 'loss')], 'train')
        plotter.epoch_series('validate_epoch', [('validate', 'accuracy')], 'validata')
