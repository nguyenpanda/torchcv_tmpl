import copy
import json
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import torch
from torch import types
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from typing_extensions import Optional, Callable

from ..utils.log import Logger
from ..utils.path import auto_naming
from ..utils.typing import HistoryDict


class BaseTrainer(ABC):

    def __init__(self,
                 max_epoch: int,
                 batch_size: int,
                 save_folder_path: Optional[Path | str] = None,
                 save_period: Optional[int] = None,
                 save_best_for_metric: str = None,
                 save_for_highest: bool = True,
                 have_validate: bool = False,
                 validate_period: Optional[int] = None,
                 load_checkpoint_path: Optional[Path] = None,
                 logger: Optional[Logger | str] = None,
                 ):
        self.max_epoch: int = int(max_epoch)
        self.cur_epoch: int = 0
        self.batch_size: int = int(batch_size)

        self.save_folder_path: Path = Path(save_folder_path) if save_folder_path else Path.cwd()
        self.save_folder_path = self.save_folder_path / auto_naming(save_folder_path, 'history_', '_')
        self.save_period: int = 1 if save_period is None else int(save_period)
        self.save_best_for_metric: str = str(save_best_for_metric)
        self.save_for_highest: bool = bool(save_for_highest)

        self.model = self.build_model()
        self.loss_func = self.build_loss_func()
        self.optimizer = self.build_optimizer()
        self.scheduler = self.build_scheduler()
        self.train_dataloader = self.build_train_dataloader()

        self.have_validate = bool(have_validate)
        self.validate_period = 1 if validate_period is None else int(validate_period)
        self.validate_dataloader = self.build_validate_dataloader() if self.have_validate else None

        if load_checkpoint_path:
            self._load_checkpoint(load_checkpoint_path)

        self.log: Logger = Logger(logger or 'model', self.save_folder_path, 'a') \
            if isinstance(logger, (str, type(None))) \
            else logger

    @abstractmethod
    def build_model(self) -> torch.nn.Module:
        pass

    @abstractmethod
    def build_validate_dataset(self) -> Dataset:
        pass

    @abstractmethod
    def build_train_dataset(self) -> Dataset:
        pass

    @abstractmethod
    def build_train_dataloader(self) -> DataLoader:
        pass

    @abstractmethod
    def build_validate_dataloader(self) -> DataLoader:
        pass

    @abstractmethod
    def build_loss_func(self) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
        pass

    @abstractmethod
    def build_optimizer(self) -> torch.optim.Optimizer:
        pass

    @abstractmethod
    def build_scheduler(self) -> Optional[torch.optim.lr_scheduler.LRScheduler]:
        pass

    @abstractmethod
    def preprocessing_batch(self, batch: dict | list) -> dict | list:
        pass

    @abstractmethod
    def _train_step(self, batch: dict | list) -> dict[str, types.Number]:
        pass

    @abstractmethod
    def _validate_step(self, batch: dict | list) -> dict[str, types.Number]:
        pass

    @abstractmethod
    def plot_history(self, history_metric: dict):
        pass

    def _setup_train(self):
        pass

    def _post_train(self):
        pass

    def _train(self, history: HistoryDict, best_validate: dict):
        for epoch in range(self.cur_epoch, self.max_epoch):
            self.cur_epoch = epoch

            # Validating
            if self.have_validate and epoch % self.validate_period == 0:
                validate_metric: dict = self.validate()
                history['validate_epoch'].append(epoch)
                history['validate'].append(validate_metric)

                metric = self.save_best_for_metric
                if best_validate['epoch'] == -1 or (
                        best_validate['value'] <= validate_metric[metric]
                        if self.save_for_highest
                        else best_validate['value'] >= validate_metric[metric]
                ):
                    best_validate['epoch'] = epoch
                    best_validate['value'] = validate_metric[metric]
                    best_validate['metrics'] = copy.deepcopy(validate_metric)
                    self._save_checkpoint(epoch, name='best.pt', log_mess=f'\t=> Saving the best model at epoch {epoch}')

                log_mess = f'\t=> The best model counting utils is at epoch {best_validate["epoch"]}, which have |'
                for k, v in best_validate['metrics'].items():
                    log_mess += f' {k.upper()} = {v} |'
                self.log(log_mess)
                self.log(100 * '=')

            # Training
            self.log(100 * '=')
            self.log('Training progress')
            train_metric = dict()
            progress_bar = tqdm(self.train_dataloader, total=len(self.train_dataloader), position=0, leave=True)

            self.model.train()
            for batch in progress_bar:
                batch_metrics: dict = self._train_step(self.preprocessing_batch(batch))

                for k, num_v in batch_metrics.items():
                    if k not in train_metric:
                        train_metric[k] = [num_v]
                    else:
                        train_metric[k].append(num_v)

            for k, list_v in train_metric.items():
                train_metric[k] = np.mean(list_v)

            history['train_epoch'].append(epoch)
            history['train'].append(train_metric)

            # Updating scheduler
            if self.scheduler is not None:
                self.scheduler.step()
                self.log(f'The next learning rate is {self.scheduler.get_last_lr()[0]}')

            # Logging out after each epoch
            log_msg = f'\t=> Total training loss: |'
            for k, v in train_metric.items():
                log_msg += f' {k} = {np.mean(v)} |'
            self.log(log_msg)

            # Save checkpoint
            self.plot_history(history)
            self.save_dict_to_json(history, self.save_folder_path / 'history.json')
            self.save_dict_to_json(best_validate, self.save_folder_path / 'best_validate.json')
            self._save_checkpoint(epoch + 1, name=f'last.pt')
            if epoch % self.save_period == 0:
                self._save_checkpoint(
                    epoch + 1,
                    name=f'checkpoint.pt',
                    log_mess=f'Saving model at epoch {epoch}!'
                )

    def train(self) -> tuple[dict, dict]:
        self._setup_train()

        history: HistoryDict = HistoryDict(train_epoch=[], train=[], validate_epoch=[], validate=[])
        best_validate: dict = dict(epoch=-1, value=None, metrics={})
        try:
            self._train(history, best_validate)
        except KeyboardInterrupt:
            self.log('Interrupt training process!', level='warning')

        self._post_train()

        self.log('Finish training!!!', level='info')
        return history, best_validate

    def validate(self) -> dict:
        self.log(100 * '=')
        self.log('Validating progress')
        validate_metric = dict()
        progress_bar = tqdm(
            self.validate_dataloader,
            total=len(self.validate_dataloader),
            position=0, leave=True
        )

        self.model.eval()
        with torch.inference_mode():
            for batch in progress_bar:
                batch_metrics = self._validate_step(self.preprocessing_batch(batch))
                progress_bar.set_postfix(batch_metrics)

                for k, num_v in batch_metrics.items():
                    if k not in validate_metric:
                        validate_metric[k] = [num_v]
                    else:
                        validate_metric[k].append(num_v)

        for k, list_v in validate_metric.items():
            validate_metric[k] = np.mean(list_v)

        log_msg = '\t=> Validation result |'
        for k, v in validate_metric.items():
            log_msg += f' {k} = {v} |'
        self.log(log_msg)

        return validate_metric

    def _save_checkpoint(self, epoch: int, name: str, log_mess: Optional[str] = None):
        checkpoint = dict(
            epoch=int(epoch),
            model_state_dict=self.model.state_dict(),
            optimizer_state_dict=self.optimizer.state_dict(),
            scheduler_state_dict=self.scheduler.state_dict() if self.scheduler else None,
        )
        torch.save(checkpoint, self.save_folder_path / Path(name).with_suffix('.pt'))
        if log_mess:
            self.log(str(log_mess), level='info')

    def _load_checkpoint(self, path: str | Path):
        checkpoint = torch.load(Path(path), map_location="cpu")
        self.cur_epoch = int(checkpoint["epoch"])
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.log(f'Loading model from path={path.absolute()}', level='info')

    @staticmethod
    def save_dict_to_json(history_dict: HistoryDict, path: Path | str):
        with open(path, 'w') as f:
            json.dump(history_dict, f, indent=4)

    @staticmethod
    def load_history_dict(path: Path | str) -> HistoryDict:
        with open(path, 'w') as f:
            history_dict = json.load(f)
        return history_dict
