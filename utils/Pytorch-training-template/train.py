#!/usr/bin/env python
# _*_ coding: utf-8 _*_
# @Time : 2021/6/5 下午4:24
# @Author : PH
# @Version：V 0.1
# @File : train.py
# @desc :
import time

import torch
import torch.nn as nn
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
from torchvision.models.resnet import resnet101
from torchvision.transforms import Compose, ToTensor, RandomHorizontalFlip, RandomRotation
from torch.utils.tensorboard import SummaryWriter
from template import TemplateModel


class ModelT(TemplateModel):
    def __init__(self, model_list, optimizer_list, loss_fn, train_loader, test_loader, writer):
        super(ModelT, self).__init__()
        # 必须设定
        # 模型架构
        # 将模型和优化器以list保存，方便对整个模型的多个部分设定对应的优化器
        self.model_list = model_list  # 模型的list
        self.optimizer_list = optimizer_list  # 优化器的list
        self.criterion = loss_fn

        # 数据集
        self.train_loader = train_loader
        self.test_loader = test_loader

        # 下面的可以不设定
        # tensorboard
        self.writer = writer  # 推荐设定
        # 训练时print的间隔
        self.log_per_step = 100  # 推荐按数据集大小设定


def get_loader(batch_size, transforms):
    train_set = CIFAR10(root="/home/ph/Desktop/Tutorials/Pytorch/data",
                        train=True, transform=transforms, download=True)
    test_set = CIFAR10(root="/home/ph/Desktop/Tutorials/Pytorch/data",
                       train=False, transform=transforms, download=True)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              num_workers=6, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True,
                             num_workers=6, pin_memory=True)
    return train_loader, test_loader


def main(continue_model=None, clean_log=False, write_params=False):
    epochs = 20
    batch_size = 100
    init_lr = 3e-4
    transforms = Compose([RandomRotation(degrees=60),
                          RandomHorizontalFlip(p=0.5),
                          ToTensor()])
    train_loader, test_loader = get_loader(batch_size, transforms)
    model = resnet101(pretrained=False)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=init_lr)
    loss_fn = nn.CrossEntropyLoss()
    writer = SummaryWriter()

    model_t = ModelT([model], [optimizer], loss_fn, train_loader, test_loader, writer)
    model_t.check_init(clean_log=clean_log)
    model_t.get_model_info(fake_inp=torch.randn(1, 3, 64, 64))
    if continue_model is not None:
        model_t.load_state(continue_model)

    start = time.time()
    for _ in range(epochs):
        model_t.train_loop(write_params)
        model_t.eval(save_per_epochs=5)
    print(f"DONE! best {model_t.key_metric}:{model_t.best_metric[model_t.key_metric]}")
    print(f"Final_lr:{optimizer.param_groups[0]['lr']}")
    print(f"Total Time:{time.time() - start}")


if __name__ == '__main__':
    main(clean_log=True, write_params=False)
