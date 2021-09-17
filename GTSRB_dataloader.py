"""
@author:yebin
@file:GTSRB_dataloader.py
@IDE:Pycharm
@time:2020/12/18 下午1:46
@function:加载经过resize到２５６＊２５６的GTSRB数据
@example:
@tip:
"""

from torch.utils.data import DataLoader, Dataset
from skimage import io
from torchvision import transforms
import os
import torch


class yebin_data_Train(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.images = os.listdir(self.root)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image_index = self.images[index]
        img_path = os.path.join(self.root, image_index)
        image = io.imread(img_path)
        label = img_path.split('/')[-1].split('_')[0]
        label = torch.tensor(int(label))

        if self.transform:
            image = self.transform(image)

        return image, label


class yebin_data_Test(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.images = os.listdir(self.root)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image_index = self.images[index]
        img_path = os.path.join(self.root, image_index)
        image = io.imread(img_path)
        label = img_path.split('/')[-1].split('_')[0]
        label = torch.tensor(int(label))

        if self.transform:
            image = self.transform(image)

        return image, label


def get_GTSRB_train_dataloader(root, mean, std, batch_size, num_workers, shuffle):
    transform_train = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        # transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    train_gtsrb = yebin_data_Train(root=root, transform=transform_train)
    #kwargs = {"num_workers": 4, "pin_memory": True}
    train_gtsrb_loader = DataLoader(train_gtsrb, shuffle=shuffle, num_workers=num_workers,batch_size=batch_size, pin_memory=True)

    return train_gtsrb_loader

def get_GTSRB_test_dataloader(root, mean, std, batch_size, num_workers, shuffle):
    transform_test = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    test_gtsrb = yebin_data_Test(root=root, transform=transform_test)
    #kwargs = {"num_workers": 4, "pin_memory": True}
    test_gtsrb_loader = DataLoader(test_gtsrb, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size,pin_memory=True)

    return test_gtsrb,test_gtsrb_loader
