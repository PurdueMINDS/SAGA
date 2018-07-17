"""Neural Network Manager"""
import docopt
import logging
import copy

import sys
import os
import math

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import drawbox
import raw
import flight
import frame
import batch

# must run on Python 3
assert sys.version_info[0] == 3


# /**
#  * Manager
#  */
class AverageMeter(object):
    """Observation Average Value Container"""
    def __init__(self):
        self.reset()

    def reset(self):
        """reset the class"""
        self.sum = 0
        self.cnt = 0
        self.avg = None

    def update(self, val, num):
        """update the class

        Args
        ----
        val : float
            value to update
        num : int
            count number corresponding to the value

        """
        self.sum += val * num
        self.cnt += num
        self.avg = self.sum / self.cnt

class NeuralNetworkManager(object):
    """Neural Network Manager

    It mananges the all processes of a neural network, including training, validation,
    training for man epochs and prediction.

    Before running any process, registration of essential components is necessary.

    1. Model, Criterion, Optimizer

    2. Data Loader: this includes training batch loader and validation batch loader

    3. Shape: this provides an interface between data loader and neural network to reshape
              every batch to fit neural network requirements.

    4. Configurations: e.g. if use cuda, which device to run cuda, ......

    """
    def __init__(self):
        pass

    def register_model(self, model, criterion, optimizer, rnn=False):
        """Register model

        Args
        ----
        model : nn.Module
            neural network model
        criterion : nn.Module
            neural network loss function
        optimizer : torch.optim.Optimizer
            optimizer to update parameters
        rnn : bool
            if model is rnn architecture

        """
        self.model     = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.rnn       = rnn

    def register_loader(self, train_loader, valid_loader):
        """Register data loader

        Args
        ----
        train_loader : frame.FramePairDataLoader or frame.SequencePairDataLoader
            batch loader for training stage
        valid_loader : frame.FramePairDataLoader or frame.SequencePairDataLoader
            batch loader for validation stage

        """
        self.train_loader = train_loader
        self.valid_loader = valid_loader

    def register_shape(self, input_shape, target_shape,
                       input_class=torch.Tensor, target_class=torch.Tensor):
        """Register shape

        Args
        ----
        input_shape : tuple
            shape of input tensor, use -1 for number of batches
        target_shape : tuple
            shape of target tensor, use -1 for number of batches

        """
        assert isinstance(input_shape, tuple) and isinstance(target_shape, tuple)
        self.input_shape  = input_shape
        self.target_shape = target_shape
        self.input_class = input_class
        self.target_class = target_class

    def register_config(self, cuda=False, device=None, print_freq=1):
        """Register configurations

        Args
        ----
        cuda : bool
            if use CUDA to accelerate
        device : int
            CUDA device ID if use CUDA
        print_freq : int
            frequency to print loss

        """
        self.on_cuda = False
        self.cuda = cuda
        self.device = device
        self.print_freq = print_freq

        # transfer model to CUDA
        if self.cuda:
            torch.cuda.set_device(self.device)
            logging.info('Run on CUDA[{}]'.format(torch.cuda.current_device()))
            self.gpu()

    def cpu(self):
        if self.on_cuda:
            self.model = self.model.cpu()
            self.criterion = self.criterion.cpu()
            self.on_cuda = False

    def gpu(self):
        if not self.on_cuda:
            self.model = self.model.cuda()
            self.criterion = self.criterion.cuda()
            self.on_cuda = True

    def fetch_batch(self, loader_data, return_idt=False):
        """Fetch batch pair

        loader_data : object
            a batch instance given by loader class
        return_idt : bool
            if return the identifier of given batch

        """
        # fetch batch pair
        idt, (input, target) = loader_data

        # reshape batch to fit requirements of neural network
        input  = input.copy().reshape(*self.input_shape)
        target = target.copy().reshape(*self.target_shape)

        if not return_idt: return input, target
        else: return input, target, idt

    def train(self, loader):
        """Train for an epoch

        Args
        ----
        loader: frame.FramePairDataLoader or frame.SequencePairDataLoader
            batch loader for training stage

        """
        loss_avg = AverageMeter()

        if self.rnn: flight = None

        for i, data in enumerate(loader):
            # prepare batch pair
            input, target, idt = self.fetch_batch(data, return_idt=True)
            try:
                input  = self.input_class(input)
                target = self.target_class(target)
            except AttributeError:
                input  = torch.Tensor(input)
                target = torch.Tensor(target)
            input_var  = Variable(input)
            target_var = Variable(target)

            if self.cuda:
                input_var  = input_var.cuda()
                target_var = target_var.cuda()

            # train the model by prepared batch pair
            self.optimizer.zero_grad()
            if self.rnn:
                idt_pre = idt.split('_')[:2]
                if flight is None or idt_pre != flight:
                    flight = idt_pre
                    hidden_var = self.model.init_hidden(cuda=self.cuda)
                for itr in hidden_var: itr.detach_()
                output_var, hidden_var = self.model.forward(input_var, hidden_var)
            else:
                output_var = self.model.forward(input_var)
            if self.model.classify:
                output_var = output_var.view(-1, 2)
                target_var = target_var.view(-1)
            loss = self.criterion.forward(output_var, target_var)
            loss.backward()
            self.optimizer.step()

            loss_avg.update(float(loss.data), input.size(0))

    def valid(self, loader):
        """Validate on the loader

        Args
        ----
        loader: frame.FramePairDataLoader or frame.SequencePairDataLoader
            batch loader for validation stage

        """
        loss_avg = AverageMeter()

        if self.rnn: flight = None

        for i, data in enumerate(loader):
            # prepare batch pair
            input, target, idt= self.fetch_batch(data, return_idt=True)
            try:
                input  = self.input_class(input)
                target = self.target_class(target)
            except AttributeError:
                input  = torch.Tensor(input)
                target = torch.Tensor(target)
            input_var  = Variable(input)
            target_var = Variable(target)

            if self.cuda:
                input_var  = input_var.cuda()
                target_var = target_var.cuda()

            # only compute the loss
            if self.rnn:
                idt_pre = idt.split('_')[:2]
                if flight is None or idt_pre != flight:
                    flight = idt_pre
                    hidden_var = self.model.init_hidden(cuda=self.cuda)
                output_var, hidden_var = self.model.forward(input_var, hidden_var)
            else:
                output_var = self.model.forward(input_var)
            if self.model.classify:
                output_var = output_var.view(-1, 2)
                target_var = target_var.view(-1)
            loss = self.criterion.forward(output_var, target_var)

            loss_avg.update(float(loss.data), input.size(0))

        return loss_avg.avg

    def fit(self, num_epochs, best=False):
        """Train registered model for given number of epochs

        Args
        ----
        num_epochs : int
            number of epochs to train
        best : bool
            use the best validation loss rather than the final validation loss

        """
        # initalize buffer for loss of each epoch
        train_loss = self.valid(self.train_loader)
        valid_loss = self.valid(self.valid_loader)

        self.loss   = [(train_loss, valid_loss)]
        best_loss   = valid_loss
        best_params = copy.deepcopy(self.model.state_dict())
        flag        = '(Update)'

        logging.info('Epoch[{}] Train: {:.3f} | Valid: {:.3f} {}'.format(
                        0, train_loss, valid_loss, flag))

        for i in range(num_epochs):
            self.train(self.train_loader)

            train_loss = self.valid(self.train_loader)
            valid_loss = self.valid(self.valid_loader)

            if np.isnan(train_loss) or np.isnan(valid_loss):
                logging.warning('NAN Loss')
                break

            self.loss.append((train_loss, valid_loss))
            if valid_loss < best_loss:
                best_loss   = valid_loss
                best_params = copy.deepcopy(self.model.state_dict())
                flag        = '(Update)'
            else:
                flag = ''

            if (i + 1) % self.print_freq == 0 or i == num_epochs - 1 or len(flag) > 0:
                logging.info('Epoch[{}] Train: {:.3f} | Valid: {:.3f} {}'.format(
                                i + 1, train_loss, valid_loss, flag))

        if best:
            logging.info('Best validation loss: {:.3f}'.format(best_loss))
            self.model.load_state_dict(best_params)

    def predict(self, loader, predict_key, predict_shape, fake=False):
        """Predict given loader by the model

        Args
        ----
        loader: frame.FramePairDataLoader or frame.SequencePairDataLoader
            batch loader for prediction stage
        predict_key : str
            keyword of prediction of the model
        predict_shape : tuple
            shape of prediction of a batch, use -1 for number of batches
            this is useful when predicting multi keywords
        fake: bool
            fake prediction

        """
        # prediction must happen in order
        assert (not hasattr(loader, 'shuffle')) or (not loader.shuffle)

        loss_avg = AverageMeter()

        # initialize buffer for prediction
        prediction_identifier = []
        prediction_data       = []

        if self.rnn: flight = None

        for i, data in enumerate(loader):
            # prepare batch pair
            input, target, idt = self.fetch_batch(data, return_idt=True)

            if fake:
                prediction_data.append(target.reshape(*predict_shape))
                prediction_identifier.append(idt)
            else:
                try:
                    input  = self.input_class(input)
                    target = self.target_class(target)
                except AttributeError:
                    input  = torch.Tensor(input)
                    target = torch.Tensor(target)
                input_var  = Variable(input)
                target_var = Variable(target)

                if self.cuda:
                    input_var  = input_var.cuda()
                    target_var = target_var.cuda()

                # compute the loss and prediction of prepared batch
                if self.rnn:
                    idt_pre = idt.split('_')[:2]
                    if flight is None or idt_pre != flight:
                        flight = idt_pre
                        hidden_var = self.model.init_hidden(cuda=self.cuda)
                    output_var, hidden_var = self.model.forward(input_var, hidden_var)
                else:
                    output_var = self.model.forward(input_var)
                try:
                    is_classify = self.model.classify
                except AttributeError:
                    is_classify = False
                if is_classify:
                    # bsz = target_var.size(0)
                    output_var = output_var.view(-1, 2)
                    target_var = target_var.view(-1)
                loss = self.criterion.forward(output_var, target_var)

                loss_avg.update(float(loss.data), input.size(0))

                if 'hazard' in predict_key:
                    assert is_classify
                    output_var = F.softmax(output_var, dim=1)
                    output_var = output_var[:, 1].contiguous()
                    output_var = output_var.view(-1, 1)

                # append current prediction to buffer
                if self.cuda:
                    prediction_data.append(output_var.cpu().data.numpy().reshape(*predict_shape))
                else:
                    prediction_data.append(output_var.data.numpy().reshape(*predict_shape))
                prediction_identifier.append(idt)

        # append prediction to given batch loader
        loader.append_key(
            target_key=predict_key,
            identifier=prediction_identifier,
            data=prediction_data)

        if fake:
            logging.info('Prediction loss: {:.3f}'.format(0))
            return loader, 0
        else:
            logging.info('Prediction loss: {:.3f}'.format(loss_avg.avg))
            return loader, loss_avg.avg


# /**
#  * Neural Network
#  */
model_list = ('FC3', 'LSTM3')
rnn_list = ('LSTM3',)

class FC3(nn.Module):
    def __init__(self, num_in=32 * 15, num_out=32 * 1, num_hidden=1024, classify=False):
        super(FC3, self).__init__()

        self.num_in     = num_in
        self.num_out    = num_out
        self.num_hidden = num_hidden
        self.classify   = classify

        self.fc1 = nn.Linear(num_in, num_hidden)
        self.fc2 = nn.Linear(num_hidden, num_hidden)
        self.fc3 = nn.Linear(num_hidden, num_out)

    def forward(self, x):
        x = F.sigmoid(self.fc1(x))
        x = F.tanh(self.fc2(x))
        x = self.fc3(x)
        return x

class LSTM3(nn.Module):
    def __init__(self, num_in=32 * 15, num_out=32 * 1, num_hidden=1024, classify=False):
        super(LSTM3, self).__init__()

        self.num_in     = num_in
        self.num_out    = num_out
        self.num_hidden = num_hidden
        self.classify   = classify

        self.fc1   = nn.Linear(num_in, num_hidden)
        self.lstm2 = nn.LSTM(num_hidden, num_hidden)
        self.fc3   = nn.Linear(num_hidden, num_out)

        self.lstm2_dim = num_hidden

    def forward(self, x, h=None):
        x = F.sigmoid(self.fc1(x))
        x, h = self.lstm2(x, h)
        x = F.tanh(x)
        x = self.fc3(x)
        return x, h

    def init_hidden(self, cuda=False):
        if self.cuda:
            return (Variable(torch.zeros(1, 1, self.lstm2_dim)).cuda(),
                    Variable(torch.zeros(1, 1, self.lstm2_dim)).cuda())
        else:
            return (Variable(torch.zeros(1, 1, self.lstm2_dim)),
                    Variable(torch.zeros(1, 1, self.lstm2_dim)))
