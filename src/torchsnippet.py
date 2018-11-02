import torch
import time
import numpy as np
import os, shutil
from utils import AverageMeter


class NN(object):
    """
    This is a prototype for NN wrapper in Pytorch.
    Please follow this coding style carefully.
    Args:
        train_loader (torch.dataset.DataLoader) :
            pytorch DataLoader for training dataset
        val_loader (torch.dataset.DataLoader) :
            pytorch DataLodaer for validation dataset
        opt (torch.optim) :
            optimizer
        criterion:
            Loss function.
        initial_lr (float):
            Initial learning rate. TODO implement using lr_find()
        checkpoint_save (str):
            Directory to save check point.
        model_save (str):
            Directory to save model.
        model:
            Pytorch Model.
        param_diagonstic (bool):
            check parameters, will be print. TODO record parameters.
        shape_diagonstic (bool):
            print shape for some important variables
        result_diagonstic (bool):
            print result for some variables.
    """

    def __init__(self, train_loader=None, val_loader=None, epochs=None,
                 opt=None, criterion=None, initial_lr=None, checkpoint_save=None,
                 model_save=None, param_diagonstic=None, shape_diagonstic=None,
                 result_diagonstic=None, lr_adjust=None, metrics=None,
                 model=None):
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.epochs = epochs
        self.criterion = criterion
        self.model = model
        self.checkpoint_save = checkpoint_save
        self.initial_lr = initial_lr
        self.model_save = model_save
        self.train_current_batch_data = {}
        self.valid_current_batch_data = {}
        self.param_diagonsitc = param_diagonstic
        self.shape_diagonstic = shape_diagonstic
        self.result_diagonstic = result_diagonstic
        self.metrics=metrics
        self.optimizer = opt
        self.lr_adjust = lr_adjust

        if not isinstance(self.optimizer, torch.optim.Optimizer):
            raise TypeError('should be an torch.optim.Optimizer type, instead of {}'.format(type(self.optimizer)))

        if not isinstance(lr_adjust, torch.optim.lr_scheduler._LRScheduler) and lr_adjust is not None:
            raise TypeError('should be inheritant learning rate scheudler.')
        global best_val_acc, best_test_acc

    def train(self):
        # if learning_rate_reschedule is a subclass of torch.optim.lr_scheduler, do as usual lr adjust function.
        if self.lr_adjust is not None:
            self.lr_adjust.step()
        best_val_acc = 0
        for epoch in range(self.epochs):
            train_losses, train_acc = self.train_epoch(data_loader=self.train_loader,
                                                       model=self.model,
                                                       criterion=self.criterion,
                                                       optimizer=self.optimizer)
            val_losses, val_acc = self.validate_epoch(data_loader=self.val_loader,
                                                      model=self.model,
                                                      criterion=self.criterion)

            is_best = val_acc.avg > best_val_acc
            print('>>>>>>>>>>>>>>>>>>>>>>')
            print(
                'Epoch: {} train loss: {}, train acc: {}, valid loss: {}, valid acc: {}'.format(epoch, train_losses.avg,
                                                                                                train_acc.avg,
                                                                                                val_losses.avg,
                                                                                                val_acc.avg))
            print('>>>>>>>>>>>>>>>>>>>>>>')
            self.save_checkpoint({'epoch': epoch + 1,
                                  'state_dict': self.model.state_dict(),
                                  'best_val_acc': best_val_acc,
                                  'optimizer': self.optimizer.state_dict(), }, is_best)

    def eval(self, data_loader, model, criterion):
        return None

    def train_epoch(self, data_loader, model, criterion, optimizer, print_freq=100):
        """
        Train function for every epoch. Standard for supervised learning.
        Args:
            data_loader (torch.utils.dataset.Dataloader): Dataloader for training.
            model :
            criterion (torch.nn.Module): Loss function.
            optimizer (torch.opt) : Optimizer.
            print_freq(int): number of step to print results. The first round always print.
        """
        losses = AverageMeter()
        scores = AverageMeter()
        model.train()
        time_now = time.time()

        for batch_idx, (data, target) in enumerate(data_loader):
            if torch.cuda.is_available():
                data = data.cuda()
                target = target.cuda()

            output = model(data)
            loss, pred = criterion(output, target)

            losses.update(loss.item(), 1)

            scores = self.metrics(pred, target)

            scores.update(scores, 1)
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()

            time_end = time.time() - time_now
            if batch_idx % print_freq == 0:
                print('Training Round: {}, Time: {}'.format(batch_idx, np.round(time_end, 2)))
                print('Training Loss: val:{} avg:{} Acc: val:{} avg:{}'.format(losses.val, losses.avg,
                                                                               scores.val, scores.avg))
        return losses, scores

    def validate_epoch(self, data_loader, model, criterion, print_freq=10000):
        """
        Validation function for every epoch.
        Args:
            data_loader (torch.utils.dataset.Dataloader): Dataloader for training.
            model :
            criterion (torch.nn.Module): Loss function.
            print_freq(int): number of step to print results. The first round always print.
        """
        model.eval()
        losses = AverageMeter()
        scores = AverageMeter()

        with torch.no_grad():
            time_now = time.time()
            for batch_idx, (data, target) in enumerate(data_loader):
                if torch.cuda.is_available():
                    data = data.cuda()
                    target = target.cuda()

                output = model(data)
                loss, pred = criterion(output, target)

                losses.update(loss.item(), data.size(0))

                score = self.metrics(pred, target)
                scores.update(score, data.size(0))
                time_end = time.time() - time_now
                if batch_idx % print_freq == 0:
                    print('Validation Round: {}, Time: {}'.format(batch_idx, np.round(time_end, 2)))
                    print('Validation Loss: val:{} avg:{} Acc: val:{} avg:{}'.format(losses.val, losses.avg,
                                                                                     scores.val, scores.avg))
        return losses, scores

    def adjust_learing_rate(self, opt):
        lr = self.initial_lr - 0.0000  # reduce 10 percent every 50 epoch
        for param_group in opt.param_groups:
            param_group['lr'] = lr

    def save_checkpoint(self, state, is_best, ckpt_filename='checkpoint.path.tar',
                        model_filename='model_best.path.tar'):
        """
        save the best states.
        :param state:
        :param is_best: if the designated benchmark is the best in this epoch.
        :param ckpt_filename: the file path to save checkpoint, will be create if not exist.
        """
        if not os.path.exists(self.checkpoint_save):
            os.mkdir(self.checkpoint_save)
        ckpt_savefile = os.path.join(self.checkpoint_save, ckpt_filename)
        torch.save(state, ckpt_savefile)
        if is_best:
            model_savefile = os.path.join(self.checkpoint_save, model_filename)
            shutil.copyfile(ckpt_savefile, model_savefile)
