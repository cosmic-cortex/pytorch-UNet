import os

import numpy as np
import torch
import torch.nn as nn

from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset

from skimage import io

from .utils import chk_mkdir, Logger


class Model:
    def __init__(self, net: nn.Module, loss, optimizer, checkpoint_folder: str,
                 scheduler: torch.optim.lr_scheduler._LRScheduler = None,
                 device: torch.device = torch.device('cpu')):
        """
        Wrapper for PyTorch models.

        Args:
            net: PyTorch model.
            loss: Loss function which you would like to use during training.
            optimizer: Optimizer for the training.
            checkpoint_folder: Folder for saving the results and predictions.
            scheduler: Learning rate scheduler for the optimizer. Optional.
            device: The device on which the model and tensor should be
                located. Optional. The default device is the cpu.

        Attributes:
            net: PyTorch model.
            loss: Loss function which you would like to use during training.
            optimizer: Optimizer for the training.
            checkpoint_folder: Folder for saving the results and predictions.
            scheduler: Learning rate scheduler for the optimizer. Optional.
            device: The device on which the model and tensor should be
                located. Optional.
        """
        self.net = net
        self.loss = loss
        self.optimizer = optimizer
        self.scheduler = scheduler

        self.checkpoint_folder = checkpoint_folder
        chk_mkdir(self.checkpoint_folder)

        # moving net and loss to the selected device
        self.device = device
        self.net.to(device=self.device)
        self.loss.to(device=self.device)

    def fit_batch(self, X_batch, y_batch):
        self.net.train(True)

        X_batch = Variable(X_batch.to(device=self.device))
        y_batch = Variable(y_batch.to(device=self.device))

        # training
        self.optimizer.zero_grad()
        y_out = self.net(X_batch)
        training_loss = self.loss(y_out, y_batch)
        training_loss.backward()
        self.optimizer.step()

        # return the average training loss
        return training_loss.item() / len(X_batch)

    def fit_epoch(self, dataset, n_batch=1, shuffle=False):
        epoch_running_loss = 0
        for X_batch, y_batch, *rest in DataLoader(dataset, batch_size=n_batch, shuffle=shuffle):
            epoch_running_loss += self.fit_batch(X_batch, y_batch)

        del X_batch, y_batch

        return epoch_running_loss / n_batch

    def fit_dataset(self, dataset: Dataset, n_epochs: int, n_batch: int = 1, shuffle: bool = False,
                    val_dataset: Dataset = None, save_freq: int = 100, verbose: bool = False):

        logger = Logger(verbose=verbose)

        self.net.train(True)

        min_loss = np.inf
        for epoch_idx in range(1, n_epochs + 1):
            # doing the epoch
            train_loss = self.fit_epoch(dataset, n_batch=n_batch, shuffle=shuffle)

            # logging the losses
            logs = {'epoch': epoch_idx,
                    'train_loss': train_loss}

            if self.scheduler is not None:
                self.scheduler.step(train_loss)

            if val_dataset is not None:
                val_loss = self.validate_dataset(val_dataset, n_batch=n_batch)
                logs['val_loss'] = val_loss
                if val_loss < min_loss:
                    torch.save(self.net.state_dict(), os.path.join(self.checkpoint_folder, 'model'))
                    min_loss = val_loss
            else:
                if train_loss < min_loss:
                    torch.save(self.net.state_dict(), os.path.join(self.checkpoint_folder, 'model'))
                    min_loss = train_loss

            # recording the losses in the logger
            logger.log(logs)
            # saving model and logs
            if save_freq and (epoch_idx % save_freq == 0):
                epoch_save_path = os.path.join(self.checkpoint_folder, '%d' % epoch_idx)
                chk_mkdir(epoch_save_path)
                torch.save(self.net.state_dict(), os.path.join(epoch_save_path, 'model'))

        self.net.train(False)

        return logger

    def validate_dataset(self, dataset, n_batch=1):
        self.net.train(False)

        total_running_loss = 0
        for batch_idx, (X_batch, y_batch, *rest) in enumerate(DataLoader(dataset, batch_size=n_batch, shuffle=False)):
            X_batch = Variable(X_batch.to(device=self.device))
            y_batch = Variable(y_batch.to(device=self.device))

            y_out = self.net(X_batch)
            training_loss = self.loss(y_out, y_batch)

            total_running_loss += training_loss.item()

        self.net.train(True)

        del X_batch, y_batch

        return total_running_loss / (batch_idx + 1)

    def predict_dataset(self, dataset, export_path):
        self.net.train(False)
        chk_mkdir(export_path)

        for batch_idx, (X_batch, *rest) in enumerate(DataLoader(dataset, batch_size=1)):
            if isinstance(rest[0][0], str):
                image_filename = rest[0][0]
            else:
                image_filename = '%s.png' % str(batch_idx + 1).zfill(3)

            X_batch = Variable(X_batch.to(device=self.device))
            y_out = self.net(X_batch).cpu().data.numpy()

            io.imsave(os.path.join(export_path, image_filename), y_out[0, :, :, :].transpose((1, 2, 0)))

    def predict_batch(self, X_batch):
        self.net.train(False)

        X_batch = Variable(X_batch.to(device=self.device))
        y_out = self.net(X_batch)

        return y_out
