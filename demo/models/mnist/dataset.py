from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torchvision import transforms

from torchcv_tmpl.data import BaseDataset
from transform import RandomHomographyTransform


class MNISTDataset(BaseDataset):

    def __init__(self,
                 csv_file: str | Path,
                 is_train: bool,
                 ):
        super().__init__()
        self.df: pd.DataFrame = pd.read_csv(csv_file)
        self.is_train = is_train
        self.transform = self._build_argumentation()

        # Uncomment the follow lines for faster testing
        # p = 0.001
        # n = int(max(len(self.df) * p, 0))
        # self.df = self.df.head(n)

    def _build_argumentation(self):
        if self.is_train:
            return transforms.Compose([
                RandomHomographyTransform(max_warp=7),  # Apply random perspective warp
                transforms.ToTensor(),  # Convert to tensor and scale to [0,1]
                transforms.Normalize((0.5,), (0.5,))  # Normalize to [-1,1]
            ])
        else:
            return transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,))
            ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        label = torch.tensor(self.df.iloc[idx, 0], dtype=torch.long)
        image = self.df.iloc[idx, 1:].values.reshape(28, 28).astype(np.uint8)

        if self.transform:
            image = self.transform(image)

        return image, label


if __name__ == '__main__':
    DEMO = Path(__file__).parents[2]
    MNIST_CSV = DEMO / 'dataset' / 'MNIST_CSV'

    TRAIN_CSV = MNIST_CSV / 'mnist_train.csv'
    TEST_CSV = MNIST_CSV / 'mnist_test.csv'

    a = MNISTDataset(TRAIN_CSV, True)
    x, y = a[10]
    print(x)
    print(y)

    print(x.shape)
    print(y.shape)
