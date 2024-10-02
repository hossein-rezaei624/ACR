# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from contextlib import suppress
import os
import sys
from typing import Any, Dict

import numpy as np

from utils import create_if_not_exists
from utils.conf import base_path
from utils.metrics import *

useless_args = ['dataset', 'tensorboard', 'validation', 'model',
                'csv_log', 'notes', 'load_best_args']


def print_mean_accuracy(mean_acc: np.ndarray, task_number: int,
                        setting: str) -> None:
    """
    Prints the mean accuracy on stderr.
    :param mean_acc: mean accuracy value
    :param task_number: task index
    :param setting: the setting of the benchmark
    """
    if setting == 'domain-il':
        mean_acc, _, mean_acc_ood, __ = mean_acc
        print('\nAccuracy for {} task(s): {} %'.format(
            task_number, round(mean_acc, 2)), file=sys.stderr)
        print('\nOOD Accuracy for {} task(s): {} %'.format(
            task_number, round(mean_acc_ood, 2)), file=sys.stderr)
    else:
        mean_acc_class_il, mean_acc_task_il, mean_acc_class_il_ood, mean_acc_task_il_ood = mean_acc
        print('\nAccuracy for {} task(s): \t [Class-IL]: {} %'
              ' \t [Task-IL]: {} %\n'.format(task_number, round(
                  mean_acc_class_il, 2), round(mean_acc_task_il, 2)), file=sys.stderr)
        print('\nOOD Accuracy for {} task(s): \t [Class-IL]: {} %'
              ' \t [Task-IL]: {} %\n'.format(task_number, round(
                  mean_acc_class_il_ood, 2), round(mean_acc_task_il_ood, 2)), file=sys.stderr)


class Logger:
    def __init__(self, setting_str: str, dataset_str: str,
                 model_str: str) -> None:
        self.accs = []
        self.accs_ood = []
        self.fullaccs = []
        self.fullaccs_ood = []
        if setting_str == 'class-il':
            self.accs_mask_classes = []
            self.accs_mask_classes_ood = []
            self.fullaccs_mask_classes = []
            self.fullaccs_mask_classes_ood = []
        self.setting = setting_str
        self.dataset = dataset_str
        self.model = model_str
        self.fwt = None
        self.fwt_ood = None
        self.fwt_mask_classes = None
        self.fwt_mask_classes_ood = None
        self.bwt = None
        self.bwt_ood = None
        self.bwt_mask_classes = None
        self.bwt_mask_classes_ood = None
        self.forgetting = None
        self.forgetting_ood = None
        self.forgetting_mask_classes = None
        self.forgetting_mask_classes_ood = None

    def dump(self):
        dic = {
            'accs': self.accs,
            'accs_ood': self.accs_ood,
            'fullaccs': self.fullaccs,
            'fullaccs_ood': self.fullaccs_ood,
            'fwt': self.fwt,
            'fwt_ood': self.fwt_ood,
            'bwt': self.bwt,
            'bwt_ood': self.bwt_ood,
            'forgetting': self.forgetting,
            'forgetting_ood': self.forgetting_ood,
            'fwt_mask_classes': self.fwt_mask_classes,
            'fwt_mask_classes_ood': self.fwt_mask_classes_ood,
            'bwt_mask_classes': self.bwt_mask_classes,
            'bwt_mask_classes_ood': self.bwt_mask_classes_ood,
            'forgetting_mask_classes': self.forgetting_mask_classes,
            'forgetting_mask_classes_ood': self.forgetting_mask_classes_ood,
        }
        if self.setting == 'class-il':
            dic['accs_mask_classes'] = self.accs_mask_classes
            dic['accs_mask_classes_ood'] = self.accs_mask_classes_ood
            dic['fullaccs_mask_classes'] = self.fullaccs_mask_classes
            dic['fullaccs_mask_classes_ood'] = self.fullaccs_mask_classes_ood

        return dic

    def load(self, dic):
        self.accs = dic['accs']
        self.accs_ood = dic['accs_ood']
        self.fullaccs = dic['fullaccs']
        self.fullaccs_ood = dic['fullaccs_ood']
        self.fwt = dic['fwt']
        self.fwt_ood = dic['fwt_ood']
        self.bwt = dic['bwt']
        self.bwt_ood = dic['bwt_ood']
        self.forgetting = dic['forgetting']
        self.forgetting_ood = dic['forgetting_ood']
        self.fwt_mask_classes = dic['fwt_mask_classes']
        self.fwt_mask_classes_ood = dic['fwt_mask_classes_ood']
        self.bwt_mask_classes = dic['bwt_mask_classes']
        self.bwt_mask_classes_ood = dic['bwt_mask_classes_ood']
        self.forgetting_mask_classes = dic['forgetting_mask_classes']
        self.forgetting_mask_classes_ood = dic['forgetting_mask_classes_ood']
        if self.setting == 'class-il':
            self.accs_mask_classes = dic['accs_mask_classes']
            self.accs_mask_classes_ood = dic['accs_mask_classes_ood']
            self.fullaccs_mask_classes = dic['fullaccs_mask_classes']
            self.fullaccs_mask_classes_ood = dic['fullaccs_mask_classes_ood']

    def rewind(self, num):
        self.accs = self.accs[:-num]
        self.accs_ood = self.accs_ood[:-num]
        self.fullaccs = self.fullaccs[:-num]
        self.fullaccs_ood = self.fullaccs_ood[:-num]
        with suppress(BaseException):
            self.fwt = self.fwt[:-num]
            self.fwt_ood = self.fwt_ood[:-num]
            self.bwt = self.bwt[:-num]
            self.bwt_ood = self.bwt_ood[:-num]
            self.forgetting = self.forgetting[:-num]
            self.forgetting_ood = self.forgetting_ood[:-num]
            self.fwt_mask_classes = self.fwt_mask_classes[:-num]
            self.fwt_mask_classes_ood = self.fwt_mask_classes_ood[:-num]
            self.bwt_mask_classes = self.bwt_mask_classes[:-num]
            self.bwt_mask_classes_ood = self.bwt_mask_classes_ood[:-num]
            self.forgetting_mask_classes = self.forgetting_mask_classes[:-num]
            self.forgetting_mask_classes_ood = self.forgetting_mask_classes_ood[:-num]

        if self.setting == 'class-il':
            self.accs_mask_classes = self.accs_mask_classes[:-num]
            self.accs_mask_classes_ood = self.accs_mask_classes_ood[:-num]
            self.fullaccs_mask_classes = self.fullaccs_mask_classes[:-num]
            self.fullaccs_mask_classes_ood = self.fullaccs_mask_classes_ood[:-num]

    def add_fwt(self, results, accs, results_mask_classes, accs_mask_classes, results_ood, accs_ood, results_mask_classes_ood, accs_mask_classes_ood):
        self.fwt = forward_transfer(results, accs)
        self.fwt_ood = forward_transfer(results_ood, accs_ood)
        if self.setting == 'class-il':
            self.fwt_mask_classes = forward_transfer(results_mask_classes, accs_mask_classes)
            self.fwt_mask_classes_ood = forward_transfer(results_mask_classes_ood, accs_mask_classes_ood)

    def add_bwt(self, results, results_mask_classes, results_ood, results_mask_classes_ood):
        self.bwt = backward_transfer(results)
        self.bwt_ood = backward_transfer(results_ood)
        self.bwt_mask_classes = backward_transfer(results_mask_classes)
        self.bwt_mask_classes_ood = backward_transfer(results_mask_classes_ood)

    def add_forgetting(self, results, results_mask_classes, results_ood, results_mask_classes_ood):
        self.forgetting = forgetting(results)
        self.forgetting_ood = forgetting(results_ood)
        self.forgetting_mask_classes = forgetting(results_mask_classes)
        self.forgetting_mask_classes_ood = forgetting(results_mask_classes_ood)

    def log(self, mean_acc: np.ndarray, mean_acc_ood: np.ndarray) -> None:
        """
        Logs a mean accuracy value.
        :param mean_acc: mean accuracy value
        """
        if self.setting == 'general-continual':
            self.accs.append(mean_acc)
            self.accs_ood.append(mean_acc_ood)
        elif self.setting == 'domain-il':
            mean_acc, _ = mean_acc
            mean_acc_ood, _ = mean_acc_ood
            self.accs.append(mean_acc)
            self.accs_ood.append(mean_acc_ood)
        else:
            mean_acc_class_il, mean_acc_task_il = mean_acc
            mean_acc_class_il_ood, mean_acc_task_il_ood = mean_acc_ood
            self.accs.append(mean_acc_class_il)
            self.accs_ood.append(mean_acc_class_il_ood)
            self.accs_mask_classes.append(mean_acc_task_il)
            self.accs_mask_classes_ood.append(mean_acc_task_il_ood)

    def log_fullacc(self, accs, accs_ood):
        if self.setting == 'class-il':
            acc_class_il, acc_task_il = accs
            acc_class_il_ood, acc_task_il_ood = accs_ood
            self.fullaccs.append(acc_class_il)
            self.fullaccs_ood.append(acc_class_il_ood)
            self.fullaccs_mask_classes.append(acc_task_il)
            self.fullaccs_mask_classes_ood.append(acc_task_il_ood)

    def write(self, args: Dict[str, Any]) -> None:
        """
        writes out the logged value along with its arguments.
        :param args: the namespace of the current experiment
        """
        wrargs = args.copy()

        for i, acc in enumerate(self.accs):
            wrargs['accmean_task' + str(i + 1)] = acc
        for i, acc_ood in enumerate(self.accs_ood):
            wrargs['accmean_task_ood' + str(i + 1)] = acc_ood

        for i, fa in enumerate(self.fullaccs):
            for j, acc in enumerate(fa):
                wrargs['accuracy_' + str(j + 1) + '_task' + str(i + 1)] = acc
        for i, fa_ood in enumerate(self.fullaccs_ood):
            for j, acc_ood in enumerate(fa_ood):
                wrargs['accuracy_' + str(j + 1) + '_task_ood' + str(i + 1)] = acc_ood

        wrargs['forward_transfer'] = self.fwt
        wrargs['forward_transfer_ood'] = self.fwt_ood
        wrargs['backward_transfer'] = self.bwt
        wrargs['backward_transfer_ood'] = self.bwt_ood
        wrargs['forgetting'] = self.forgetting
        wrargs['forgetting_ood'] = self.forgetting_ood

        target_folder = base_path() + "results/"

        create_if_not_exists(target_folder + self.setting)
        create_if_not_exists(target_folder + self.setting +
                             "/" + self.dataset)
        create_if_not_exists(target_folder + self.setting +
                             "/" + self.dataset + "/" + self.model)

        path = target_folder + self.setting + "/" + self.dataset\
            + "/" + self.model + "/logs.pyd"
        with open(path, 'a') as f:
            f.write(str(wrargs) + '\n')

        if self.setting == 'class-il':
            create_if_not_exists(os.path.join(*[target_folder, "task-il/", self.dataset]))
            create_if_not_exists(target_folder + "task-il/"
                                 + self.dataset + "/" + self.model)

            for i, acc in enumerate(self.accs_mask_classes):
                wrargs['accmean_task' + str(i + 1)] = acc
            for i, acc_ood in enumerate(self.accs_mask_classes_ood):
                wrargs['accmean_task_ood' + str(i + 1)] = acc_ood

            for i, fa in enumerate(self.fullaccs_mask_classes):
                for j, acc in enumerate(fa):
                    wrargs['accuracy_' + str(j + 1) + '_task' + str(i + 1)] = acc
            for i, fa_ood in enumerate(self.fullaccs_mask_classes_ood):
                for j, acc_ood in enumerate(fa_ood):
                    wrargs['accuracy_' + str(j + 1) + '_task_ood' + str(i + 1)] = acc_ood

            wrargs['forward_transfer'] = self.fwt_mask_classes
            wrargs['forward_transfer_ood'] = self.fwt_mask_classes_ood
            wrargs['backward_transfer'] = self.bwt_mask_classes
            wrargs['backward_transfer_ood'] = self.bwt_mask_classes_ood
            wrargs['forgetting'] = self.forgetting_mask_classes
            wrargs['forgetting_ood'] = self.forgetting_mask_classes_ood

            path = target_folder + "task-il" + "/" + self.dataset + "/"\
                + self.model + "/logs.pyd"
            with open(path, 'a') as f:
                f.write(str(wrargs) + '\n')
