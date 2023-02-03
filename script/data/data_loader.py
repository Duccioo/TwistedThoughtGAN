import torchvision.datasets as dset
from torch.utils.data import DataLoader
import torch
import os

import torchvision.transforms as transforms

transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
        transforms.RandomHorizontalFlip(0.5),
        transforms.Grayscale(),
    ]
)


class Data_L:
    def __init__(
        self,
        data_folder,
        percent_train=0.8,
        percent_test=0.1,
        transform=transform,
        batch_size=0,
    ):
        self.transform = transform
        self.batch_size = batch_size
        self.data_folder = data_folder

        if os.path.exists(data_folder) == False:
            os.mkdir(data_folder)

        self.dataset = dset.ImageFolder(root=data_folder, transform=self.transform)

        # Dividiamo i dati in train, validation e test
        self.train_size = int(percent_train * len(self.dataset))
        self.test_size = int(percent_test * len(self.dataset))
        self.val_size = len(self.dataset) - self.train_size - self.test_size

        (
            self.train_dataset,
            self.val_dataset,
            self.test_dataset,
        ) = torch.utils.data.random_split(
            self.dataset, [self.train_size, self.val_size, self.test_size]
        )

    def get_train_set(self, shuffle=True, batch_size=0):
        if batch_size == 0:
            batch_size = self.batch_size

        return DataLoader(self.train_dataset, batch_size=batch_size, shuffle=shuffle)

    def get_test_set(self, shuffle=True, batch_size=0):
        if batch_size == 0:
            batch_size = self.batch_size
        return DataLoader(self.test_dataset, batch_size=batch_size, shuffle=shuffle)

    def get_val_set(self, shuffle=True, batch_size=0):
        if batch_size == 0:
            batch_size = self.batch_size
        return DataLoader(self.val_dataset, batch_size=batch_size, shuffle=shuffle)

    def set_batch_size(self, batch_size):
        self.batch_size = batch_size
