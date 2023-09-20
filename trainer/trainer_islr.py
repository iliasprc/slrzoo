import json
import logging
import os
import time

import torch
from torch import nn

from base.base_trainer import BaseTrainer
from trainer.training_utils import AverageMeter, ProgressMeter, Summary
from utils import check_dir, save_checkpoint_islr
from utils.metrics import accuracy
from utils.utils import MetricTracker, get_logger

logger = logging.getLogger('train_islr.train')


class Trainer_ISLR(BaseTrainer):
    """
    Trainer class
    """

    def __init__(self, args, model, optimizer, config, checkpoint_dir, logger, data_loader,
                 valid_data_loader=None, test_data_loader=None, lr_scheduler=None, len_epoch=None, writer=None,
                 id2w=None):
        super().__init__(model, optimizer, config)

        self.args = args
        if (args.cuda):
            use_cuda = torch.cuda.is_available()
            self.device = torch.device("cuda" if use_cuda else "cpu")
        else:
            self.device = torch.device("cpu")
        self.start_epoch = 1
        self.train_data_loader = data_loader

        self.len_epoch = len(self.train_data_loader)
        self.epochs = args.epochs
        self.valid_data_loader = valid_data_loader
        self.test_data_loader = test_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.do_test = self.test_data_loader is not None
        self.lr_scheduler = lr_scheduler
        self.log_step = args.log_interval
        self.model = model
        self.optimizer = optimizer
        self.logger = logger
        self.mnt_best = 0.0
        self.loss = nn.CrossEntropyLoss()
        self.checkpoint_dir = checkpoint_dir

        if writer != None:
            self.writer = writer
        self.metric_ftns = ['loss', 'acc1']
        self.train_metrics = MetricTracker(*[m for m in self.metric_ftns], writer=self.writer, mode='train')
        self.metric_ftns = ['loss', 'acc1']
        self.valid_metrics = MetricTracker(*[m for m in self.metric_ftns], writer=self.writer, mode='valid')
        self.logger = get_logger('train_islr')

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        Args:
            epoch (int): current training epoch.
        """

        self.model.train()

        self.train_metrics.reset()

        batch_time = AverageMeter('Time', ':6.3f')
        data_time = AverageMeter('Data', ':6.3f')
        losses = AverageMeter('Loss', ':.4e')
        top1 = AverageMeter('Acc@1', ':6.2f')
        top5 = AverageMeter('Acc@5', ':6.2f')
        progress = ProgressMeter(
            len(self.train_data_loader),
            [batch_time, data_time, losses, top1, top5],
            prefix="    Epoch: [{}]".format(epoch))

        end = time.time()

        for batch_idx, (data, target) in enumerate(self.train_data_loader):

            data = data.to(self.device)

            target = target.to(self.device)

            logits = self.model(data)  # , target)
            loss = self.loss(logits, target)
            acc1, acc5 = accuracy(logits, target, topk=(1, 5))
            losses.update(loss.item(), logits.size(0))
            top1.update(acc1[0], logits.size(0))
            top5.update(acc5[0], logits.size(0))

            loss.backward()

            self.optimizer.step()  # Now we can do an optimizer step
            self.optimizer.zero_grad()  # Reset gradients tensors
            batch_time.update(time.time() - end)
            end = time.time()

            if batch_idx % self.args.log_interval == 0:
                self.logger.info(progress.display(batch_idx + 1))
        self.logger.info(progress.display_summary(len(self.train_data_loader)))
        return [str(epoch), losses.avg,
                top1.avg]
        # writer_step = (epoch - 1) * self.len_epoch + batch_idx
        # self.train_metrics.update_all_metrics(
        #     {
        #         'ctc_loss': loss_ctc.item(), 'wer': 100.0 * temp_wer,
        #     }, writer_step=writer_step)
        #
        #     self._progress(batch_idx, epoch, metrics=self.train_metrics, mode='train')
        #
        # self._progress(batch_idx, epoch, metrics=self.train_metrics, mode='train', self.logger.info_summary=True)

    def _valid_epoch(self, epoch, mode, loader):
        """
        Validate after training an epoch
        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        self.model.eval()

        self.valid_metrics.reset()
        batch_time = AverageMeter('Time', ':6.3f', Summary.NONE)
        losses = AverageMeter('Loss', ':.4e', Summary.NONE)
        top1 = AverageMeter('Acc@1', ':6.2f', Summary.AVERAGE)
        top5 = AverageMeter('Acc@5', ':6.2f', Summary.AVERAGE)
        progress = ProgressMeter(
            len(loader),
            [batch_time, losses, top1, top5],
            prefix="Val Epoch: [{}]".format(epoch))

        end = time.time()
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(loader):
                data = data.to(self.device)

                target = target.long().to(self.device)

                logits = self.model(data)
                loss = self.loss(logits, target)
                acc1, acc5 = accuracy(logits, target, topk=(1, 5))
                losses.update(loss.item(), data.size(0))
                top1.update(acc1[0], data.size(0))
                top5.update(acc5[0], data.size(0))
                batch_time.update(time.time() - end)
                end = time.time()

                writer_step = (epoch - 1) * len(loader) + batch_idx

        self.logger.info(progress.display_summary(len(loader)))

        return [str(epoch), losses.avg,
                top1.avg]

    def train(self):
        for epoch in range(self.start_epoch, self.epochs):
            self._train_epoch(epoch)

            self.logger.info(f"{'!' * 10}   validation  {'!' * 10}")
            _, avg_loss, val_acc1 = self._valid_epoch(epoch, 'validation', self.valid_data_loader)
            check_dir(self.checkpoint_dir)
            self.checkpointer(epoch, avg_loss)
            # TODO
            self.lr_scheduler.step(float(avg_loss))
            if self.do_test:
                self.logger.info(f"{'!' * 10}  test  {'!' * 10}")
                self._valid_epoch(epoch, 'test', self.test_data_loader)

    def test(self):
        pass

    def checkpointer(self, epoch, val_acc):
        name = 'last'
        is_best = val_acc > self.mnt_best
        if (is_best):
            self.mnt_best = val_acc
            name = 'best'

            self.logger.info("Best acc1 {} so far ".format(self.mnt_best))
        with open(os.path.join(self.checkpoint_dir, 'training_arguments.txt'), 'w') as f:
            json.dump(self.args.__dict__, f, indent=2)
        save_checkpoint_islr(model=self.model, optimizer=self.optimizer, epoch=epoch, acc1=val_acc,
                             checkpoint=self.checkpoint_dir, name=name,
                             save_seperate_layers=True, is_best=is_best)

    def _progress(self, batch_idx, epoch, metrics, mode='', print_summary=False):
        metrics_string = metrics.calc_all_metrics()
        if (batch_idx % self.log_step == 0):

            if metrics_string == None:
                self.logger.info(" No metrics")
            else:
                self.logger.info(
                    '{} Epoch: [{:2d}/{:2d}]\t Video [{:5d}/{:5d}]\t {}'.format(
                        mode, epoch, self.epochs, batch_idx, self.len_epoch, metrics_string))
        elif self.logger.info_summary:
            self.logger.info(
                '{} summary  Epoch: [{}/{}]\t {}'.format(
                    mode, epoch, self.epochs, metrics_string))
