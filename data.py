import os

import torch
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import MNIST
from torchvision import transforms as tr
from tonic.datasets import NMNIST
from tonic import transforms as ttr
# from tonic import DiskCachedDataset
from tonic import MemoryCachedDataset
from tonic.collation import PadTensors
import pytorch_lightning as pl


class MNISTDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str = os.environ.get('TMPDIR', './') + "data/", batch_size: int = 32):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.transform = tr.Compose(
            [
                tr.ToTensor(),
                tr.Normalize((0.1307,), (0.3081,)), 
                torch.flatten
            ],
        )

    def setup(self, stage: str):
        self.mnist_test = MNIST(self.data_dir, train=False, transform=self.transform, download=True)
        mnist_full = MNIST(self.data_dir, train=True, transform=self.transform, download=True)
        self.mnist_train, self.mnist_val = random_split(mnist_full, [55000, 5000])

    def train_dataloader(self):
        return DataLoader(
            self.mnist_train,
            batch_size=self.batch_size,
            num_workers=8,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.mnist_val,
            batch_size=self.batch_size,
            num_workers=8,
            pin_memory=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.mnist_test,
            batch_size=self.batch_size,
            num_workers=8,
            pin_memory=True,
        )


class NMNISTDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str = os.environ.get('TMPDIR', './') + "cache/", batch_size: int = 32):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.transform = ttr.Compose(
            [
                ttr.Denoise(filter_time=10000),
                ttr.ToFrame(sensor_size=NMNIST.sensor_size,
                            time_window=5000),
                torch.from_numpy,
                torch.nn.Flatten(start_dim=1),
                lambda x: x.bool().int(),
            ],
        )

    def setup(self, stage: str):
        # self.nmnist_test = DiskCachedDataset(NMNIST(self.data_dir, train=False, transform=self.transform), cache_path=self.data_dir+'nmnist/test/')
        self.nmnist_test = MemoryCachedDataset(NMNIST(self.data_dir, train=False, transform=self.transform))
        # nmnist_test = DiskCachedDataset(NMNIST(self.data_dir, train=True, transform=self.transform), cache_path=self.data_dir+'nmnist/train/')
        nmnist_full = MemoryCachedDataset(NMNIST(self.data_dir, train=True, transform=self.transform))
        self.nmnist_train, self.nmnist_val = random_split(nmnist_full, [.9, .1])

    def train_dataloader(self):
        return DataLoader(
            self.nmnist_train,
            batch_size=self.batch_size,
            collate_fn=PadTensors(batch_first=False),
            shuffle=True,
            num_workers=2,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.nmnist_val,
            batch_size=self.batch_size,
            collate_fn=PadTensors(batch_first=False),
            num_workers=2,
            pin_memory=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.nmnist_test,
            batch_size=self.batch_size,
            collate_fn=PadTensors(batch_first=False),
            num_workers=4,
            pin_memory=True,
        )