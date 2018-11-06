from utils import AverageMeter
import torch
import time
import numpy as np
import os, shutil


class NN(object):
    """
    This is a prototype for NN wrapper in Pytorch.
    Please follow this coding style carefully.
    Args:
        model:
            Pytorch Model.
        train_loader (torch.dataset.DataLoader) :
            pytorch DataLoader for training dataset
        val_loader (torch.dataset.DataLoader) :
            pytorch DataLodaer for validation dataset
        epochs:

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
        dataset:

        model:
            Pytorch Model.
        param_diagonstic (bool):
            check parameters, will be print. TODO record parameters.
        if_checkpoint_save (bool):
            save checkpoint if True
        print_result_epoch (bool):
            true if results some steps at every epochs are print.
        metrics :
            Evaluation metrics.
    """

    def __init__(self, model=None, train_loader=None, val_loader=None,
                 test_loader=None, epochs=None,
                 opt=None, criterion=None, initial_lr=None, checkpoint_save=None,
                 model_save=None, dataset=None, param_diagonstic=None,
                 shape_diagonstic=None, if_checkpoint_save=True,
                 result_diagonstic=None, lr_adjust=None, penalty=None,
                 print_result_epoch=False, metrics=None,
                 target_reshape=None, **kwargs):
        self.test_loader = test_loader
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.epochs = epochs
        self.criterion = criterion
        self.model = model
        self.checkpoint_save = checkpoint_save
        self.initial_lr = initial_lr
        self.model_save = model_save
        self.dataset = dataset
        self.train_current_batch_data = {}
        self.valid_current_batch_data = {}
        self.param_diagonsitc = param_diagonstic
        self.shape_diagonstic = shape_diagonstic
        self.result_diagonstic = result_diagonstic
        self.optimizer = opt
        self.lr_adjust = lr_adjust
        self.if_checkpoint_save = if_checkpoint_save
        self.print_result_epoch = print_result_epoch
        self.penalty = penalty
        self.target_reshape = target_reshape
        self.metrics = metrics

        if not isinstance(self.optimizer, torch.optim.Optimizer):
            raise TypeError('should be an torch.optim.Optimizer type, instead of {}'.format(type(self.optimizer)))

        if not isinstance(lr_adjust, torch.optim.lr_scheduler._LRScheduler) and lr_adjust is not None:
            raise TypeError('should be inheritant learning rate scheudler.')
        global best_val_acc, best_test_acc

    def train(self):
        print('Start training process.')
        if self.lr_adjust is not None:
            self.lr_adjust.step()
        best_val_acc = 0
        for epoch in range(self.epochs):
            train_losses, train_acc = self.train_epoch(data_loader=self.train_loader,
                                                       criterion=self.criterion,
                                                       optimizer=self.optimizer)
            val_losses, val_acc = self.validate_epoch(data_loader=self.val_loader,
                                                      criterion=self.criterion)
            if self.if_checkpoint_save:
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
        print('Training process end.')

    def train_epoch(self, data_loader, criterion,
                    optimizer, print_freq=100):
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
        percent_acc = AverageMeter()
        self.model.train()
        time_now = time.time()

        for batch_idx, (data, target) in enumerate(data_loader):
            target = target.float()
            if self.target_reshape is not None:
                target = self.target_reshape(target)
            if torch.cuda.is_available():
                data = data.cuda()
                target = target.cuda()

            output = self.model(data)
            loss = criterion(output, target)
            if self.penalty is not None:
                penalty_val = self.loss_penalty()
                loss += penalty_val

            losses.update(loss.item(), 1)

            acc = self.metrics(output, target)

            percent_acc.update(acc, 1)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            time_end = time.time() - time_now
            if batch_idx % print_freq == 0 and self.print_result_epoch:
                print('Training Round: {}, Time: {}'.format(batch_idx,
                                                            np.round(time_end, 2)))
                print('Training Loss: val:{} avg:{} Acc: val:{} avg:{}'.format(losses.val,
                                                                               losses.avg,
                                                                               percent_acc.val, percent_acc.avg))
        return losses, percent_acc

    def validate_epoch(self, data_loader, criterion,
                       print_freq=10000):
        """
        Validation function for every epoch.
        Args:
            data_loader (torch.utils.dataset.Dataloader): Dataloader for training.
            model :
            criterion (torch.nn.Module): Loss function.
            print_freq(int): number of step to print results. The first round always print.
        """
        self.model.eval()
        losses = AverageMeter()
        percent_acc = AverageMeter()

        with torch.no_grad():
            time_now = time.time()
            for batch_idx, (data, target) in enumerate(data_loader):
                if self.target_reshape is not None:
                    target = self.target_reshape(target)
                target = target.float()
                if torch.cuda.is_available():
                    data = data.cuda()
                    target = target.cuda()

                output = self.model(data)
                # implement loss
                loss = criterion(output, target)
                if self.penalty is not None:
                    penalty_val = self.loss_penalty()
                    loss += penalty_val

                losses.update(loss.item(), data.size(0))

                acc = self.metrics(output, target)
                percent_acc.update(acc, data.size(0))
                time_end = time.time() - time_now
                if batch_idx % print_freq == 0 and self.print_result_epoch:
                    print('Validation Round: {}, Time: {}'.format(batch_idx, np.round(time_end, 2)))
                    print('Validation Loss: val:{} avg:{} Acc: val:{} avg:{}'.format(losses.val, losses.avg,
                                                                                     percent_acc.val, percent_acc.avg))
        return losses, percent_acc

    def adjust_learing_rate(self, opt):
        lr = self.initial_lr - 0.0000  # reduce 10 percent every 50 epoch
        for param_group in opt.param_groups:
            param_group['lr'] = lr

    def save_checkpoint(self, state, is_best):
        """
        save the best states.
        :param state:
        :param is_best: if the designated benchmark is the best in this epoch.
        :param ckpt_filename: the file path to save checkpoint, will be create if not exist.
        """
        #if not os.path.exists(self.checkpoint_save):
        #    os.mkdir(self.checkpoint_save)
        torch.save(state, self.checkpoint_save)
        if is_best:
            shutil.copyfile(self.checkpoint_save, self.model_save)

    def save_model(self):
        return None

    def resume_model(self, resume_file_path):
        if not os.path.exists(resume_file_path):
            raise ValueError('Resume file does not exist')
        else:
            print('=> loading checkpoint {}'.format(resume_file_path))
            checkpoint = torch.load(resume_file_path)
            start_epoch = checkpoint['epoch']
            self.best_val_acc = checkpoint['best_val_acc']
            self.model.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            print('=> loaded checkpoint {} of epoch {}'.format(resume_file_path,
                checkpoint['epoch']))

    def evaluate(self, weights,
                       print_freq=10000):
        """
        Validation function for every epoch.
        Args:
            data_loader (torch.utils.dataset.Dataloader): Dataloader for testing.
            print_freq(int): number of step to print results. The first round always print.
        """
        print('Start evaluating process.')
        if weights is not None:
            print('Loading weights from {}'.format(weights))
            self.resume_model(weights)
            print('Weights loaded.')
        self.model.eval()
        percent_acc = AverageMeter()

        with torch.no_grad():
            time_now = time.time()
            for batch_idx, (data, target) in enumerate(self.test_loader):
                if torch.cuda.is_available():
                    data = data.cuda()
                    target = target.cuda()

                output = self.model(data)

                acc = self.metrics(output, target)
                percent_acc.update(acc, data.size(0))
                time_end = time.time() - time_now
        print('Test AUC: val:{} avg:{}'.format(percent_acc.val, percent_acc.avg))
        if not weights:
            print('Test evaluation is end!')
        print('Test evaluation is end!')

    def loss_penalty(self):
        if self.penalty['type'] == 'l2':
            l2_penalty = 0
            for param in self.model.parameters():
                l2_penalty = (0.5 / self.penalty['reg']) * l2_penalty
            return l2_penalty
        else:
            raise ValueError('Currently only l2 penalty are supported')