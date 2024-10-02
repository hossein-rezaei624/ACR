import os
from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data.dataset import Dataset

from backbone.ResNet18 import resnet18
from datasets.transforms.denormalization import DeNormalize
from datasets.utils.continual_dataset import (ContinualDataset,
                                              store_masked_loaders)
from utils.conf import base_path


class CUB200(Dataset):
    """
    Overrides dataset to change the getitem function.
    """
    IMG_SIZE = 32
    N_CLASSES = 200
    MEAN, STD = (0.4856, 0.4994, 0.4324), (0.2272, 0.2226, 0.2613)

    def __init__(self, root: str, train: bool = True, transform: transforms = None,
                 target_transform: transforms = None, download: bool =False) -> None:
        self.not_aug_transform = transforms.Compose([transforms.Resize(32), transforms.ToTensor()])
        self.root = root
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.download = download

        if download:
            if os.path.isdir(root) and len(os.listdir(root)) > 0:
                print('Download not needed, files already on disk.')
            else:
                from onedrivedownloader import download
                ln = '<iframe src="https://onedrive.live.com/embed?cid=D3924A2D106E0039&resid=D3924A2D106E0039%21110&authkey=AIEfi5nlRyY1yaE" width="98" height="120" frameborder="0" scrolling="no"></iframe>'
                print('Downloading dataset')
                download(ln, filename=os.path.join(root, 'cub_200_2011.zip'), unzip=True, unzip_path=root, clean=True)

        data_file = np.load(os.path.join(root, 'train_data.npz' if self.train else 'test_data.npz'), allow_pickle=True)

        self.data = data_file['data']
        self.targets = torch.from_numpy(data_file['targets']).long()


    def __len__(self):
      return len(self.data)
    
    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]

        # to return a PIL Image
        img = Image.fromarray(img)
        original_img = img.copy()


        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        if hasattr(self, 'logits'):
            return img, target, original_img, self.logits[index]

        return img, target



class MyCUB200(CUB200):
    """Base CUB200 dataset."""

    def __init__(self, root: str, train: bool = True, transform: transforms = None, 
                 target_transform: transforms = None, download: bool = False) -> None:
        super(MyCUB200, self).__init__(
          root, train, transform, target_transform, download)

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]

        # to return a PIL Image
        img = Image.fromarray(img)
        original_img = img.copy()

        not_aug_img = self.not_aug_transform(original_img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        if hasattr(self, 'logits'):
            return img, target, not_aug_img, self.logits[index]

        return img, target, not_aug_img, index


class SequentialCUB200(ContinualDataset):

    NAME = 'seq-cub200'
    SETTING = 'class-il'
    N_CLASSES_PER_TASK = 20
    N_TASKS = 10
    MEAN, STD = (0.4856, 0.4994, 0.4324), (0.2272, 0.2226, 0.2613)
    TRANSFORM = transforms.Compose([
        transforms.Resize(32),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD)])


    def get_examples_number(self):
        train_dataset = MyCUB200(base_path() + 'CUB200', train=True,
                                  download=True)
        return len(train_dataset.data)

  
    def get_data_loaders(self):
        transform = self.TRANSFORM

        test_transform = transforms.Compose(
            [transforms.Resize(32), transforms.ToTensor(), self.get_normalization_transform()])

        train_dataset = MyCUB200(base_path() + 'CUB200', train=True,
                                 download=True, transform=transform)

        train_dataset.not_aug_transform = test_transform  # store normalized images in the buffer
        if self.args.validation:
            train_dataset, test_dataset = get_train_val(train_dataset,
                                                        test_transform, self.NAME)
        else:
            test_dataset = CUB200(base_path() + 'CUB200', train=False,
                                download=True, transform=test_transform)

        train, test = store_masked_loaders(train_dataset, test_dataset, self)
        return train, test

  

    @staticmethod
    def get_backbone():
      return resnet18(SequentialCUB200.N_CLASSES_PER_TASK * SequentialCUB200.N_TASKS)

    @staticmethod
    def get_loss():
      return F.cross_entropy
  

    def get_transform(self):
        transform = transforms.Compose(  # weaken random crop to reproduce results
            [transforms.RandomCrop(32, padding=1), transforms.RandomHorizontalFlip()])
        return transform

  
    @staticmethod
    def get_normalization_transform():
      transform = transforms.Normalize(SequentialCUB200.MEAN, SequentialCUB200.STD)
      return transform

    @staticmethod
    def get_denormalization_transform():
      transform = DeNormalize(SequentialCUB200.MEAN, SequentialCUB200.STD)
      return transform

    @staticmethod
    def get_scheduler(model, args):
      return None
  
    @staticmethod
    def get_batch_size():
      return 32

    @staticmethod
    def get_epochs():
      return 50

    @staticmethod
    def get_minibatch_size():
      return 32
      
