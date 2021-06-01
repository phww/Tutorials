#!/usr/bin/env python
# _*_ coding: utf-8 _*_
# @Time : 2021/5/14 下午4:59
# @Author : PH
# @Version：V 1.0
# @File : template.py
# @desc :
import torch
from torchsummary import summary
import os
import os.path as osp


class TemplateModel:
    def __init__(self):
        # 必须设定
        # 模型架构
        # 将模型和优化器以list保存，方便对整个模型的多个部分设定对应的优化器
        self.model_list = None  # 模型的list
        self.optimizer_list = None  # 优化器的list
        self.criterion = None
        # 数据集
        self.train_loader = None
        self.test_loader = None

        # 下面的可以不设定
        # tensorboard
        self.writer = None  # 推荐设定
        # 训练状态
        self.global_step = 0
        self.epoch = 0
        self.best_metric = {}
        self.key_metric = None
        # 运行设备
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # check_point 目录
        self.ckpt_dir = "./check_point"
        # 训练时print的间隔
        self.log_per_step = 5  # 推荐按数据集大小设定

    def check_init(self):
        # 检测摸板的初始状态，可以在这加上很多在训练之前的操作
        # 我喜欢加一个清空cuda的cache，保证训练时显存尽量不浪费
        assert self.model_list
        assert self.optimizer_list
        assert self.criterion
        assert self.metric
        assert self.train_loader
        assert self.test_loader
        assert self.device
        assert self.ckpt_dir
        assert self.log_per_step
        torch.cuda.empty_cache()
        for model in self.model_list:
            model.to(self.device)
        if not osp.exists(self.ckpt_dir):
            os.mkdir(self.ckpt_dir)

    def load_state(self, fname, optim=True):
        # 读取保存的模型到模板之中。如果要继续训练的模型optim=True
        # 使用最佳模型做推断optim=False
        state = torch.load(fname)
        for idx, model in enumerate(self.model_list):
            if isinstance(model, torch.nn.DataParallel):  # 多卡训练
                model.module.load_state_dict(state[f'model{idx}'])
            else:  # 非多卡训练
                model.load_state_dict(state[f'model{idx}'])
            # 恢复一些状态参数
            if optim and f'optimizer_list{idx}' in state:
                self.optimizer_list[idx].load_state_dict(state[f'optimizer_list{idx}'])
        self.global_step = state['global_step']
        self.epoch = state['epoch']
        self.best_metric = state['best_metric']
        self.key_metric = state['key_metric']
        print('load model from {}'.format(fname))

    def save_state(self, fname, optim=True):
        # 保存模型，其中最佳模型不用保存优化器中的梯度。
        # 而训练过程中保存的其他模型需要保存优化器中的梯度以便继续训练
        state = {}
        for idx, model in enumerate(self.model_list):
            if isinstance(model, torch.nn.DataParallel):
                state[f'model{idx}'] = model.module.state_dict()
            else:
                state[f'model{idx}'] = model.state_dict()
            # 训练过程中的模型除了保存模型的参数外，还要保存当前训练的状态：optim中的参数
            if optim:
                state[f'optimizer_list{idx}'] = self.optimizer_list[idx].state_dict()
        state['global_step'] = self.global_step
        state['epoch'] = self.epoch
        state['best_metric'] = self.best_metric
        state['key_metric'] = self.key_metric
        torch.save(state, fname)
        print('save model at {}'.format(fname))

    def train_loop(self):
        print("*" * 15, f"epoch:{self.epoch + 1}", "*" * 15)
        # 训练一个epoch
        for model in self.model_list:
            model.train()
        self.epoch += 1
        running_loss = 0.0
        for step, batch in enumerate(self.train_loader):
            self.global_step += 1
            batch_loss = self.train_loss_per_batch(batch)
            # 多个优化器需要按逆序更新每一个优化器
            for optimizer in reversed(self.optimizer_list):
                optimizer.zero_grad()

            batch_loss.backward()

            for optimizer in reversed(self.optimizer_list):
                optimizer.step()
            running_loss += batch_loss.item()

            # 记录损失除了训练刚开始时是用此时的loss外，其他都是用一批loss的平均loss
            if self.global_step == 1:
                if self.writer is not None:
                    self.writer.add_scalar('loss', batch_loss.item(), self.global_step)
                print(f"loss:{batch_loss.item() : .5f}\t"
                      f"cur:[{step * self.train_loader.batch_size}]\[{len(self.train_loader.dataset)}]")

            # 记录每一批loss的平均loss
            elif step % self.log_per_step == 0 and step != 0:
                avg_loss = running_loss / (self.log_per_step * len(batch))
                if self.writer is not None:
                    self.writer.add_scalar('loss', avg_loss, self.global_step)
                print(f"loss:{avg_loss : .5f}\t"
                      f"cur:[{step * self.train_loader.batch_size}]\[{len(self.train_loader.dataset)}]")

                # 每训练一定的批次就记录此时的参数和梯度
                if self.writer is not None:
                    for model in self.model_list:
                        for tag, value in model.named_parameters():
                            tag = tag.replace('.', '/')
                            self.writer.add_histogram('weights/' + tag, value.data.cpu().numpy())
                            if value.grad is not None:  # 在FineTurn时有些参数被冻结了，没有梯度。也就不用记录了
                                self.writer.add_histogram('grads/' + tag, value.grad.data.cpu().numpy())
                running_loss = 0.0

    def train_loss_per_batch(self, batch):
        """
        计算数据集的一个batch的loss，这个部分是可能要按需求修改的部分
        # Pytorch 中的loss函数中一般规定x和y都是float，而有些loss函数规定y要为long（比如经常用到的CrossEntropyLoss）
        # 如果官网：https://pytorch.org/docs/stable/nn.html#loss-functions 对y的数据类型有要求请做相应的修改
        # 这里除了CrossEntropyLoss将y的数据类型设为long外， 其他都默认x和y的数据类型为float

        """
        x, y = batch
        x = x.to(self.device, dtype=torch.float)

        # 标签y的数据类型
        y_dtype = torch.float
        if isinstance(self.criterion, torch.nn.CrossEntropyLoss):
            y_dtype = torch.long

        # 保证标签y至少是个列向量，即shape "B, 1"
        if y.dim == 1:
            y = y.unsqueeze(dim=1).to(self.device, dtype=y_dtype)
        else:
            y = y.to(self.device, dtype=y_dtype)

        # 若模型的输入不是一个tensor，按需求改
        pred = x
        for model in self.model_list:
            pred = model(pred)
        loss = self.criterion(pred, y)
        return loss

    def eval(self, save_per_epochs=1):
        # 在整个测试集上做评估，使用分批次的metric的平均值表示训练集整体的metric
        for model in self.model_list:
            model.eval()
        scores = {}
        # 分批计算metric
        cnt = 0
        for step, batch in enumerate(self.test_loader):
            if step == 0:
                scores = self.eval_scores_per_batch(batch)
            # 累加所有批次的metric。这里有个问题：
            # 准确率可以使用分批的准确率之和除以分批数量得到，并且与用全部数据集计算准确率是等价的
            # 但是有的metric使用一部分批次的计算出来的结果可能与使用全部数据集计算出来的结果不同
            else:
                batch_scores = self.eval_scores_per_batch(batch)
                for key in scores.keys():
                    scores[key] += batch_scores[key]
            cnt += 1
        # 最后的metric要取所有批次的平均
        for key in scores.keys():
            scores[key] /= cnt

        # Tensorboard记录
        if self.writer is not None:
            for key in scores.keys():
                self.writer.add_scalar(f"{key}", scores[key], self.epoch)

        # 根据scores[self.key_metric]来判定是否保存最佳模型.
        # self.key_metric需要在metric函数中初始化，分类任务常用self.key_metric = "acc"
        for key in self.best_metric.keys():
            # 更新所有metric的最佳结果到self.best_metric字典中
            if scores[key] >= self.best_metric[key]:
                self.best_metric[key] = scores[key]
                # 保存最佳模型
                if key == self.key_metric:
                    self.save_state(osp.join(self.ckpt_dir, f'best.pth'), False)
        if self.epoch % save_per_epochs == 0:  # 每次save_per_epochs次评估保存当前模型
            self.save_state(osp.join(self.ckpt_dir, f'epoch{self.epoch}.pth'))
        print('epoch:{}\tbleu {:.5f}'.format(self.epoch, scores["bleu"]))
        return scores["bleu"]

    # 以下eval_scores_per_batch()和metric()，有时要按需求修改
    def eval_scores_per_batch(self, batch):
        x, y = batch
        x = x.to(self.device)
        y = y.to(self.device)

        pred = x
        for model in self.model_list():
            pred = model(pred)
        scores_pre_batch = self.metric(pred, y)
        return scores_pre_batch

    def metric(self, pred, y):
        """
        不同任务的性能指标太难统一了，这里只是实现了多分类任务求准确率的方法。其他任务请按需求继承
        这个类的时候再重载这个metric函数，注意返回数据类型为字典,且一定要有准确率acc这个指标，因为
        acc用于保存训练过程中的最优模型.这个模板使用分批计算metric再求全部批次的平均值的策略得到整
        体的metric。不会将全部的预测和ground truth保存在preds和ys中然后在cpu上进行预测。因为如
        果测试集或验证集太大（>50000）可能CPU内存装不下，训练会报错.但是有的metric可能不能使用分
        批得到的metric求平均来表示整体的metric,按需求改吧
        Args:
            pred: torch.tensor
                测试集或验证集的一个批次的预测
            y: torch.tensor
                测试集或验证集的一个批次的ground truth

        Returns:
            scores：dict
                各种性能指标的字典，务必要有scores["acc"]

        """
        # 初始化self.key_metric
        self.key_metric = "acc"

        scores = {}
        correct = (torch.argmax(pred, dim=1) == y).type(torch.float).sum().item()
        scores[self.key_metric] = correct / self.test_loader.batch_size
        return scores

    # 有时要按需求修改
    def inference(self, x):
        x = x.to(self.device)
        for model in self.model_list:
            x = model(x)
        return x

    def get_model_info(self, fake_inp):
        # 输出模型信息和在Tensorboard中绘制模型计算图
        if self.writer is not None:
            for model in self.model_list:
                self.writer.add_graph(model, fake_inp.to(self.device))
        # summary对transformer有BUG
        # print(summary(self.model, batch_size=32, input_size=fake_inp.shape[1:], device=self.device))

    def num_parameters(self):
        num = 0
        for model in self.model_list:
            num += sum([p.data.nelement() for p in model.parameters()])
        return num
