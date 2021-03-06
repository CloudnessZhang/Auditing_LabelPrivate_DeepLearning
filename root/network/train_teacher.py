#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import logging
import os
import random
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from network import fixmatch

from args.args_Setting import PateTeacherConfig,PateCommonConfig


def partition_dataset_indices(dataset_len, n_teachers, teacher_id, seed=None):
    #数据集互斥分割
    random.seed(seed)

    teacher_data_size = dataset_len // n_teachers
    indices = list(range(dataset_len))
    # 打乱数据集D
    random.shuffle(indices)

    # 获取数据集D—_teacher_id
    result = indices[
        teacher_id * teacher_data_size : (teacher_id + 1) * teacher_data_size
    ]

    logging.info(
        f"Teacher {teacher_id} processing {len(result)} samples. "
        f"First index: {indices[0]}, last index: {indices[-1]}. "
        f"Range: [{teacher_id * teacher_data_size}:{(teacher_id + 1) * teacher_data_size}]"
    )

    return result


def _vote_one_teacher(
        # 单个教师对输入的x对应标签投票
    model: nn.Module,
    student_dataset: Dataset,
    config_teacher: PateTeacherConfig,
    n_classes: int,
    device: Any,
):
    student_data_loader = DataLoader(
        student_dataset,
        batch_size=config_teacher.learning.batch_size,
    )

    r = torch.zeros(0, n_classes).to(device)

    with torch.no_grad():
        for data, _ in student_data_loader:
            data = data.to(device)
            output = model(data)
            binary_vote = torch.isclose(
                output, output.max(dim=1, keepdim=True).values #大于阈值的方可被统计为vote
            ).double()

            r = torch.cat((r, binary_vote), 0)

    return r



class Tearcher():
    def __init__(self, trainset, testset, num_classes=10, config_common=PateCommonConfig,config_teacher=PateTeacherConfig):
        # (self, trainset, testset, num_classes=10, setting=ALIBI_Settings): alibi传入参数
        self.trainset = trainset
        self.testset = testset
        self.num_classes = num_classes
        self.config_common = config_common
        self.config_teacher = config_teacher
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.n_tearchers = config_common.n_teachers



        for teacher_id in range(self.n_tearchers):
            fixmatch_config = config_teacher.fixmatch

            #生成数据集

        # if config_common.dataset == "cifar10":
        #     datasets = get_cifar10(
        #         root=config_common.dataset_dir,
        #         student_dataset_max_size=config_common.student_dataset_max_size,
        #         student_seed=config_common.seed + 100
        #         # we want different seeds for splitting data between teachers and for picking student subset
        #     )
        #     # labeled为数据集D，unlabel为数据集D-：一个样本包括软增强unlabel和强增强unlabel两个
        #     labeled_dataset, unlabeled_dataset = datasets["labeled"], datasets["unlabeled"]
        #     # 均为labeled
        #     test_dataset, student_dataset = datasets["test"], datasets["student"]
        #
        #     n_classes = 10
        # elif config_common.dataset == "cifar100":
        #     datasets = get_cifar100(
        #         root=config_common.dataset_dir,
        #         student_dataset_max_size=config_common.student_dataset_max_size,
        #         student_seed=config_common.seed + 100
        #         # we want different seeds for splitting data between teachers and for picking student subset
        #     )
        #     labeled_dataset, unlabeled_dataset = datasets["labeled"], datasets["unlabeled"]
        #     test_dataset, student_dataset = datasets["test"], datasets["student"]
        #
        #     n_classes = 100
        # else:
        #     raise ValueError(f"Unexpected dataset: {config_common.dataset}")
        #
        # if config_common.canary_dataset is not None:
        #     logging.info(
        #         f"Injecting canaries. Seed: {config_common.canary_dataset.seed}, N:{config_common.canary_dataset.N}"
        #     )
        #     orig_label_sum = sum(labeled_dataset.targets)
        #     fill_canaries(
        #         dataset=labeled_dataset,
        #         num_classes=len(labeled_dataset.classes),
        #         N=config_common.canary_dataset.N,
        #         seed=config_common.canary_dataset.seed,
        #     )
        #     canary_label_sum = sum(labeled_dataset.targets)
        #     logging.info(
        #         f"Canaries injected. Label sum before: {orig_label_sum}, after: {canary_label_sum}"
        #     )
        #
        # labeled_indices = partition_dataset_indices(
        #     dataset_len=len(labeled_dataset),
        #     n_teachers=config_common.n_teachers,
        #     teacher_id=teacher_id,
        #     seed=config_common.seed,
        # )
        #
        # labeled_dataset.data = labeled_dataset.data[labeled_indices]
        # labeled_dataset.targets = np.array(labeled_dataset.targets)[labeled_indices]
        #
        # logging.info(f"Training teacher {teacher_id} with {len(labeled_dataset)} samples")
        #
        # checkpoint_path = os.path.join(config_common.model_dir, f"teacher_{teacher_id}.ckp")
        # summary_writer = SummaryWriter(log_dir=config_common.tensorboard_log_dir)
        #
        # logging.info(
        #     f"Launching training. Tensorboard dir: {summary_writer.log_dir}. Checkpoint path: {checkpoint_path}"
        # )
        #
        # model, acc, loss = fixmatch.train(
        #     labeled_dataset=labeled_dataset,
        #     unlabeled_dataset=unlabeled_dataset,
        #     test_dataset=test_dataset,
        #     fixmatch_config=fixmatch_config,
        #     learning_config=config_teacher.learning,
        #     device=device,
        #     n_classes=n_classes,
        #     writer=summary_writer,
        #     writer_tag="teacher",
        #     checkpoint_path=checkpoint_path,
        # )
        #
        # logging.info(f"Finished training. Reported accuracy: {acc}")
        #
        # summary_writer.add_scalar("All teachers final accuracy", acc, teacher_id)
        #
        # r = _vote_one_teacher(
        #     model=model,
        #     student_dataset=student_dataset,
        #     config_teacher=config_teacher,
        #     n_classes=n_classes,
        #     device=device,
        # )
        # votes_path = os.path.join(config_common.model_dir, f"votes{teacher_id}.pt")
        #
        # torch.save(r, votes_path)
        # logging.info(f"Finished voting. Votes shape: {r.shape}. Path: {votes_path}")


