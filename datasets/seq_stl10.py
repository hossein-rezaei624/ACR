# Copyright 2022-present, Lorenzo Bonicelli, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Tuple
import numpy as np
import torch.nn.functional as F
import torch.optim
import torchvision.transforms as transforms
from backbone.ResNet18 import resnet18
from PIL import Image
from torchvision.datasets import STL10

from datasets.transforms.denormalization import DeNormalize
from datasets.utils.continual_dataset import (ContinualDataset,
                                              store_masked_loaders)
from datasets.utils.validation import get_train_val
from utils.conf import base_path_dataset as base_path


class TSTL10(STL10):
    """Workaround to avoid printing the already downloaded messages."""
    def __init__(self, root, split = 'train', folds=None, transform=None,
                 target_transform=None, download=False) -> None:
        self.not_aug_transform = transforms.Compose([transforms.Resize(32), transforms.ToTensor()])
        self.root = root
        super(TSTL10, self).__init__(root, split, folds, transform, target_transform, download=download)
        self.targets = self.labels

    def __getitem__(self, index):
        """
        Gets the requested element from the dataset.
        :param index: index of the element to be returned
        :returns: tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], int(self.targets[index])

        # to return a PIL Image
        ##img = Image.fromarray(img, mode='RGB')
        img = Image.fromarray(np.transpose(img, (1, 2, 0)))
        original_img = img.copy()

        not_aug_img = self.not_aug_transform(original_img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        if hasattr(self, 'logits'):
            return img, target, not_aug_img, self.logits[index]

        return img, target


class MySTL10(STL10):
    """
    Overrides the CIFAR100 dataset to change the getitem function.
    """
    def __init__(self, root, split = 'train', folds=None, transform=None,
                 target_transform=None, download=False) -> None:
        self.not_aug_transform = transforms.Compose([transforms.Resize(32), transforms.ToTensor()])
        self.root = root
        super(MySTL10, self).__init__(root, split, folds, transform, target_transform, download=download)
        self.targets = self.labels

    def __getitem__(self, index: int) -> Tuple[Image.Image, int, Image.Image]:
        """
        Gets the requested element from the dataset.
        :param index: index of the element to be returned
        :returns: tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], int(self.targets[index])

        # to return a PIL Image
        ##img = Image.fromarray(img, mode='RGB')
        img = Image.fromarray(np.transpose(img, (1, 2, 0)))
        original_img = img.copy()

        not_aug_img = self.not_aug_transform(original_img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        if hasattr(self, 'logits'):
            return img, target, not_aug_img, self.logits[index]

        return img, target, not_aug_img, index


class SequentialSTL10(ContinualDataset):

    NAME = 'seq-stl10'
    SETTING = 'class-il'
    N_CLASSES_PER_TASK = 2
    N_TASKS = 5
    TRANSFORM = transforms.Compose(
            [transforms.Resize(32),
             transforms.RandomCrop(32, padding=4),
             transforms.RandomHorizontalFlip(),
             transforms.ToTensor(),
             transforms.Normalize((0.4467, 0.4398, 0.4066),
                                  (0.2326, 0.2300, 0.2343))])

    def get_examples_number(self):
        train_dataset = MySTL10(base_path() + 'STL10', split = 'train',
                                  download=True)
        return len(train_dataset.data)

    def get_data_loaders(self):
        transform = self.TRANSFORM

        test_transform = transforms.Compose(
            [transforms.Resize(32), transforms.ToTensor(), self.get_normalization_transform()])

        train_dataset = MySTL10(base_path() + 'STL10', split = 'train',
                                  download=True, transform=transform)
        train_dataset.not_aug_transform = test_transform  # store normalized images in the buffer
        if self.args.validation:
            train_dataset, test_dataset = get_train_val(train_dataset,
                                                    test_transform, self.NAME)
        else:
            test_dataset = TSTL10(base_path() + 'STL10', split = 'test',
                                   download=True, transform=test_transform)

        train, test = store_masked_loaders(train_dataset, test_dataset, self)

        return train, test

    @staticmethod
    def get_transform():
        transform = transforms.Compose(
            [transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip()])
        return transform

    @staticmethod
    def get_backbone():
        return resnet18(SequentialSTL10.N_CLASSES_PER_TASK
                        * SequentialSTL10.N_TASKS)

    @staticmethod
    def get_loss():
        return F.cross_entropy

    @staticmethod
    def get_normalization_transform():
        transform = transforms.Normalize((0.4467, 0.4398, 0.4066),
                                  (0.2326, 0.2300, 0.2343))
        return transform

    @staticmethod
    def get_denormalization_transform():
        transform = DeNormalize((0.4467, 0.4398, 0.4066),
                                  (0.2326, 0.2300, 0.2343))
        return transform

    @staticmethod
    def get_epochs():
        return 50

    @staticmethod
    def get_batch_size():
        return 32

    @staticmethod
    def get_minibatch_size():
        return SequentialSTL10.get_batch_size()

    @staticmethod
    def get_scheduler(model, args):
        return None
