#   Copyright 2018 Jianfei Gao, Leonardo Teixeira, Bruno Ribeiro.
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
"""Process regular flight data classes to aligned and framed data classes"""
import docopt
import logging
import copy

import sys
import os
import math

import numpy as np
import fastdtw
import pywt

import torch

import drawbox
import raw
import flight
from flight import FlightDiffRecoverData

# must run on Python 3
assert sys.version_info[0] == 3


# /**
#  * Class Basis
#  */
class BasePairData(object):
    """Class basis for dataset of two types as input and target

    It is just a virtual class basis, not an entity class. Avoid construct any variables
    through this class.
    It only provides a virtual class basis for data classes of two types. One as input,
    and one as target. Each of them can be G1000 or phone data. It can hold several dates
    and several flights, and make sure that input and target are one-to-one.

    In addition, data normalization and denormalization are also implemented in the class.
    They are not essential part (but highly recommend to use for better performance in NN),
    so they are optinal functions. The normalization parameters will be computed and stored
    in the class.

    """
    def __init__(self):
        super(BasePairData, self).__init__()

        self.normal = None

    def __getitem__(self, idx):
        if isinstance(idx, int):
            return self.identifier[idx], (self.input_set[idx], self.target_set[idx])
        elif isinstance(idx, str):
            for i, identifier in enumerate(self.identifier):
                if identifier == idx:
                    return self.identifier[i], (self.input_set[i], self.target_set[i])
            return None, (None, None)

    def __len__(self):
        return len(self.identifier)

    def clone(self):
        """Return a clone of the class"""
        return copy.deepcopy(self)

    def lock(self):
        """Lock all data to avoid modification"""
        for data in self.input_set: data.setflags(write=False)
        for data in self.target_set: data.setflags(write=False)
        self.input_set = tuple(self.input_set)
        self.target_set = tuple(self.target_set)
        self.input_keys = tuple(self.input_keys)
        self.target_keys = tuple(self.target_keys)
        self.identifier = tuple(self.identifier)
        self.locked = True

    def unlock(self):
        """Unlock all data for modification"""
        self.input_set = list(self.input_set)
        self.target_set = list(self.target_set)
        self.input_keys = list(self.input_keys)
        self.target_keys = list(self.target_keys)
        self.identifier = list(self.identifier)
        for data in self.input_set: data.setflags(write=True)
        for data in self.target_set: data.setflags(write=True)
        self.locked = False

    def save(self, path):
        """Save data from the class

        Args
        ----
        path : str
            path to save all containing data

        """
        dirname = os.path.dirname(path)

        if not os.path.isdir(dirname): os.makedirs(dirname)
        torch.save({
                'normal'     : self.normal,
                'input_set'  : self.input_set,
                'target_set' : self.target_set,
                'input_keys' : self.input_keys,
                'target_keys': self.target_keys,
                'identifier' : self.identifier,
            }, path)

    def load(self, path):
        """Load data to the class

        Args
        ----
        path : str
            path to load all data to contain

        """
        assert os.path.isfile(path)

        save_dict = torch.load(path)

        self.normal = save_dict['normal']

        self.input_set = save_dict['input_set']
        self.target_set = save_dict['target_set']
        self.input_keys = save_dict['input_keys']
        self.target_keys = save_dict['target_keys']
        self.identifier = save_dict['identifier']

        # lock data and identifier to avoid modification
        self.lock()

    def description(self):
        """Return a description of current class status

        Returns
        -------
        lines : str
            a description of containing data

        Describe number of keywords, number of flights, identifier and shape of each
        flight data input-target pair.

        """
        # it is recommended, but not necessary, to implement in all child classes
        # it may vary on different child classes
        raise NotImplementedError

    def plot(self, dir):
        """Plot all data

        Args
        ----
        dir : str
            directory to save all plots

        For each identifier, plot each column (keyword) of its data together in the same
        figure, but plot two figures for input and target.

        """
        # it is recommended, but not necessary, to implement in all child classes
        # it may vary on different child classes
        raise NotImplementedError

    def from_(self, input_set, target_set, input_keys, target_keys, identifier):
        """Initialize the class directly from its data components

        Args
        ----
        input_set: list or tuple
            a list or tuple of numpy.array data for input
        input_keys: list or tuple
            a list or tuple of string keywords for input
        target_set: list or tuple
            a list or tuple of numpy.array data for target
        target_keys: list or tuple
            a list or tuple of string keywords for target
        identifier: list or tuple
            a list or tuple of string identifier

        """
        # input_set and target_set must be a list or tuple of numpy.array (2D or 3D, ...) \
        # input_keys and target_keys must be a list or tuple of string and match number of columns \
        # of element in corresponding dataset \
        # identifier must be a list or tuple of string and match length of both dataset
        assert isinstance(input_set, list) or isinstance(input_set, tuple)
        assert isinstance(target_set, list) or isinstance(target_set, tuple)
        assert isinstance(input_keys, list) or isinstance(input_keys, tuple)
        assert isinstance(target_keys, list) or isinstance(target_keys, tuple)
        assert isinstance(identifier, list) or isinstance(identifier, tuple)
        assert len(identifier) == len(input_set)
        assert len(identifier) == len(target_set)
        for itr in input_set:
            assert isinstance(itr, np.ndarray)
            assert len(itr.shape) == 2 or len(itr.shape) == 3
            assert len(input_keys) == itr.shape[-1]
        for itr in target_set:
            assert isinstance(itr, np.ndarray)
            assert len(itr.shape) == 2 or len(itr.shape) == 3
            assert len(target_keys) == itr.shape[-1]

        # clone data
        self.input_set = copy.deepcopy(input_set)
        self.input_keys = copy.deepcopy(input_keys)
        self.target_set = copy.deepcopy(target_set)
        self.target_keys = copy.deepcopy(target_keys)
        self.identifier = copy.deepcopy(identifier)

        self.lock()

    class Normalization(object):
        def __init__(self):
            self.input_keys  = None
            self.input_mean  = None
            self.input_std   = None
            self.target_keys = None
            self.target_mean = None
            self.target_std  = None

    def normalize(self, set_normal=None):
        """Normalize input and target data separately

        Args
        ----
        set_normal : None or BasePairData.Normalization
            normalize with given argument or compute new normalization

        """
        def validate(normal):
            # validate mean
            assert len(normal.input_mean.shape) == 1 and len(normal.target_mean.shape) == 1
            assert len(normal.input_mean) == len(self.input_keys)
            assert len(normal.target_mean) == len(self.target_keys)

            # validate standard deviation
            assert len(normal.input_std.shape) == 1 and len(normal.target_std.shape) == 1
            assert len(normal.input_std) == len(self.input_keys)
            assert len(normal.target_std) == len(self.target_keys)

        # check input and target data form validation
        assert self.normal is None

        for data in self.input_set:
            assert isinstance(data, np.ndarray)
            assert len(data.shape) == 2 or len(data.shape) == 3
            assert len(self.input_keys) == data.shape[-1]

        for data in self.target_set:
            assert isinstance(data, np.ndarray)
            assert len(data.shape) == 2 or len(data.shape) == 3
            assert len(self.target_keys) == data.shape[-1]

        if set_normal:
            # use given normalization settings
            normal = set_normal
        else:
            # initialize normalization settings
            normal = self.Normalization()

            # compute input normalization settings
            input_all_data = np.concatenate(
                [data.reshape(-1, len(self.input_keys)) for data in self.input_set],
                axis=0)
            normal.input_keys = copy.deepcopy(self.input_keys)
            normal.input_mean = np.mean(input_all_data, axis=0)
            normal.input_std  = np.std(input_all_data, axis=0)

            # compute target normalization settings
            target_all_data  = np.concatenate(
                [data.reshape(-1, len(self.target_keys)) for data in self.target_set],
                axis=0)
            normal.target_keys = copy.deepcopy(self.target_keys)
            normal.target_mean = np.mean(target_all_data, axis=0)
            normal.target_std  = np.std(target_all_data, axis=0)

        validate(normal)

        logging.info('Normalization Settings')
        logging.info(' -- {} input means, and {} input standard deviations'.format(
                        len(normal.input_mean), len(normal.input_std)))
        logging.info(' -- {} target means, and {} target standard deviations'.format(
                        len(normal.target_mean), len(normal.target_std)))

        self.normal = normal

        # unlock to normalize
        self.unlock()

        # normalize input data
        for data in self.input_set:
            data -= normal.input_mean
            data /= normal.input_std

        # normalize target data
        for data in self.target_set:
            data -= normal.target_mean
            data /= normal.target_std

        # reconstruct subclasses and lock after normalize
        if hasattr(self, 'reconstruct'): self.reconstruct()
        self.lock()

    def denormalize(self, set_normal=None):
        """Denormalize input and target data separately

        Args
        ----
        set_normal : None or BasePairData.Normalization
            denormalize with given argument or with embedded one

        """
        if set_normal: self.normal = set_normal
        assert isinstance(self.normal, self.Normalization)

        # locate normalization columns according to keywords
        input_col_ids  = []
        target_col_ids = []
        for key in self.input_keys:
            for i, itr in enumerate(self.normal.input_keys):
                if key == itr or key == '*{}'.format(itr):
                    input_col_ids.append(i)
                    break
        for key in self.target_keys:
            for i, itr in enumerate(self.normal.target_keys):
                if key == itr or key == '*{}'.format(itr):
                    target_col_ids.append(i)
                    break

        # generate proper normalization settings
        normal = self.Normalization()
        normal.input_keys  = copy.deepcopy(self.input_keys)
        normal.input_mean  = self.normal.input_mean[input_col_ids]
        normal.input_std   = self.normal.input_std[input_col_ids]
        normal.target_keys = copy.deepcopy(self.target_keys)
        normal.target_mean = self.normal.target_mean[target_col_ids]
        normal.target_std  = self.normal.target_std[target_col_ids]

        # unlock to normalize
        self.unlock()

        # normalize input data
        for i, data in enumerate(self.input_set):
            shape = data.shape
            data = data.reshape(-1, len(self.input_keys))
            data *= self.normal.input_std
            data += self.normal.input_mean
            self.input_set[i] = data.reshape(*shape)

        # normalize target data
        for i, data in enumerate(self.target_set):
            shape = data.shape
            data = data.reshape(-1, len(self.target_keys))
            data *= self.normal.target_std
            data += self.normal.target_mean
            self.target_set[i] = data.reshape(*shape)

        # reconstruct subclasses and lock after normalize
        if hasattr(self, 'reconstruct'): self.reconstruct()
        self.lock()

        self.normal = None


# /**
#  * Sequence Input-Target Flight Data Container
#  */
class FlightSequencePairData(BasePairData):
    """Flight Sequence Pair Data Container

    Input and target sequence alignment and interpolation is implemented in the class.
    It will accept two flight sequence dataset, one of which is input, the other is
    target. Then, it will do dynamic time wrapping on (altitude, longitude) to align
    each input and target pair. If necessary, it will interpolate on both input and
    target sequence to forcely make sequences of the same pair have the same length.

    It can also execute prune simutaneously on input and target sequences. It will also
    take responsible for normalization and denormalization if necessary.

    It is KEY DATA CONTAINER in this project. It is a terminal between processed raw
    data and formalized training and validation data.


        flight_data <--> seq_pair_data <--> frame_pair_data <--> batch_data
              |                                                       |
        interface                                                interface
        with users                                               with NN

    Although we still have 'raw_data --> flight_data', raw_data indeed is invisble after
    loading from files and converting into flight data. Indeed, if we apply interpolation,
    we will even lose some information when convert back from seq_pair_data to flight_data.

    """
    def __init__(self, entity=None, entity_input=None, entity_target=None):
        if isinstance(entity, tuple) or isinstance(entity, list):
            assert len(entity) == 5
            self.from_(entity[0], entity[1], entity[2], entity[3], entity[4])
            self.normal = None
        elif isinstance(entity, dict):
            assert len(entity) == 5
            self.from_(
                entity['input_set'], entity['input_keys'],
                entity['target_set'], entity['target_keys'],
                entity['identifier'])
            self.normal = None
        elif isinstance(entity, str) and os.path.isfile(entity): self.load(entity)
        elif isinstance(entity_input , flight.FlightPruneData) and \
             isinstance(entity_target, flight.FlightPruneData):
            self.from_prune(prune_input=entity_input, prune_target=entity_target)
            self.normal = None
        else:
            raise NotImplementedError

    def save(self, path):
        """Save data from the class

        Args
        ----
        path : str
            path to save all containing data

        """
        dirname = os.path.dirname(path)

        if not os.path.isdir(dirname): os.makedirs(dirname)
        torch.save({
                'normal'     : self.normal,
                'input_set'  : self.input_set,
                'target_set' : self.target_set,
                'input_keys' : self.input_keys,
                'target_keys': self.target_keys,
                'identifier' : self.identifier,
            }, path)

    def load(self, path):
        """Load data to the class

        Args
        ----
        path : str
            path to load all data to contain

        """
        assert os.path.isfile(path)

        save_dict = torch.load(path)

        self.normal = save_dict['normal']

        self.input_set = save_dict['input_set']
        self.target_set = save_dict['target_set']
        self.input_keys = save_dict['input_keys']
        self.target_keys = save_dict['target_keys']
        self.identifier = save_dict['identifier']

        # lock data and identifier to avoid modification
        self.reconstruct()
        self.lock()

    def description(self):
        """Return a description of current class status

        Returns
        -------
        lines : str
            a description of containing data

        Describe number of keywords, number of flights, identifier and shape of each
        flight data input-target pair.

        """
        lines = ''

        # description header
        lines += 'Flight Sequence Pair Data: ({} keywords ---> {} keywords), ' \
                 '{} flights\n'.format(len(self.input_keys), len(self.target_keys), len(self))

        if self.normal is None: lines += 'Not Normalized\n'
        else: lines += 'Normalized\n'

        # description for each flight data
        for i in range(len(self)):
            identifier = self.identifier[i]
            shape_input  = self.input_set[i].shape
            shape_target = self.target_set[i].shape

            # data shape must be #datapoints \times #keys
            assert len(shape_input) == 2 and len(shape_target) == 2

            lines += ' -- {:10s}: {:6d} x {:2d} ---> {:6d} x {:2d}\n'.format(
                        identifier, shape_input[0], shape_input[1], shape_target[0], shape_target[1])

        return lines

    def plot(self, dir):
        """Plot all data

        Args
        ----
        dir : str
            directory to save all plots

        For each identifier, plot each column (keyword) of its data together in the same
        figure, but plot two figures for input and target.

        """
        logging.info('Plot Input')
        self.meta_input.plot(os.path.join(dir, 'input'))

        logging.info('Plot Target')
        self.meta_target.plot(os.path.join(dir, 'target'))

    def plot_match_criterion(self, dir):
        """Plot input and target data together on matching columns

        Args
        ----
        dir : str
            directory to save all plots

        For each identifier, plot each criterion column (keyword) of its input-target data
        together in the same figure.

        """
        # get column ID of matching keys
        def get_col_id(key, from_keys):
            for i, itr in enumerate(from_keys):
                if itr == key:
                    return i
            return None

        # transfer an identifier to a figure title
        def identifier_to_title(identifier):
            date, flight_num = identifier.split('_')

            # Formalize date string
            if len(date) == 6:
                # format: MMDDYY
                year  = int(date[4:6])
                month = int(date[0:2])
                day   = int(date[2:4])
            else:
                raise NotImplementedError

            date = '%02d/%02d/%02d' % (month, day, year)

            # Formalize flight number
            flight = 'Flight %02d' % (int(flight_num))

            return '{}, {}'.format(date, flight)

        # transfer a keyword to an ax subtitle
        def key_to_subtitle(key):
            return self.LONG_KEYS[key]

        if not os.path.isdir(dir): os.makedirs(dir)

        # plot each flight
        for i, identifier in enumerate(self.identifier):
            title = identifier_to_title(identifier)
            fig = drawbox.EqualGridFigure(
                num_rows=len(self.match_keys), num_cols=1,
                ax_height=10, ax_width=80,
                title=title, font_size=40)

            logging.info('Criterion Plot \'{}\''.format(title))

            # plot each keyword
            for j, key in enumerate(self.match_keys):
                subtitle = key_to_subtitle(key)
                fig.subtitle(row_id=j, col_id=0, subtitle=subtitle)

                # logging.info(' -- Plot \'{}\''.format(subtitle))

                alpha = 0.5

                col_id = get_col_id(key, from_keys=self.input_keys)
                fig.lineplot(
                    row_id=j, col_id=0,
                    x_data=range(len(self.input_set[i][:, col_id])),
                    y_data=self.input_set[i][:, col_id],
                    color='C%d' % (j // 3), alpha=alpha,
                    label='Input')

                col_id = get_col_id(key, from_keys=self.target_keys)
                fig.lineplot(
                    row_id=j, col_id=0,
                    x_data=range(len(self.target_set[i][:, col_id])),
                    y_data=self.target_set[i][:, col_id],
                    color='C%d' % (j // 3 + 1), alpha=alpha,
                    label='Target')

            # plot legend for axes with prediction
            fig.legend()

            fig.save_fig(os.path.join(dir, 'png', '{}.png'.format(identifier)), close=False)
            fig.save_fig(os.path.join(dir, 'pdf', '{}.pdf'.format(identifier)), close=True)

    # flight keywords to their long version
    LONG_KEYS = flight.FlightExtensionData.LONG_KEYS

    def from_(self, input_set, target_set, input_keys, target_keys, identifier):
        """Initialize the class directly from its data components

        Args
        ----
        input_set: list or tuple
            a list or tuple of numpy.array data for input
        input_keys: list or tuple
            a list or tuple of string keywords for input
        target_set: list or tuple
            a list or tuple of numpy.array data for target
        target_keys: list or tuple
            a list or tuple of string keywords for target
        identifier: list or tuple
            a list or tuple of string identifier

        """
        super(FlightSequencePairData, self).from_(
            input_set, target_set, input_keys, target_keys, identifier)
        self.reconstruct()

    def from_prune(self, prune_input, prune_target):
        """Initialize the class from pruned flight data classes

        Args
        ----
        prune_input : flight.FlightPruneData
            pruned flight data class to initialize input part
        prune_target : flight.FlightPruneData
            pruned flight data class to initialize target part

        This is a clone initialization.

        """
        # can only construct from two pruned flight data classes
        assert isinstance(prune_input , flight.FlightPruneData) and \
               isinstance(prune_target, flight.FlightPruneData)

        # input and target identifier must match exactly
        assert prune_target.identifier == prune_input.identifier

        # clone data
        self.input_set   = copy.deepcopy(prune_input.dataset)
        self.target_set  = copy.deepcopy(prune_target.dataset)
        self.input_keys  = copy.deepcopy(prune_input.keys)
        self.target_keys = copy.deepcopy(prune_target.keys)
        self.identifier  = copy.deepcopy(prune_target.identifier)

        # reconstruct subclasses and lock
        self.reconstruct()
        self.lock()

    def reconstruct(self):
        """Reconstruct subclasses with class variables"""
        self.meta_input  = flight.FlightPruneData((self.input_set , self.input_keys , self.identifier))
        self.meta_target = flight.FlightPruneData((self.target_set, self.target_keys, self.identifier))

    def relink(self):
        """relink class variables to subclasses"""
        self.input_set   = self.meta_input.dataset
        self.target_set  = self.meta_target.dataset
        self.input_keys  = self.meta_input.keys
        self.target_keys = self.meta_target.keys
        self.identifier  = self.meta_target.identifier

    def length_dict(self):
        """Get the length of input and target for each flight

        Returns
        -------
        len_dict : dict
            a dictionary of length of input and target for each flight

        """
        len_dict = {}
        for i in range(len(self)):
            len_dict[self.identifier[i]] = {
                'input' : len(self.input_set[i]),
                'target': len(self.target_set[i]),
            }
        return len_dict

    def align(self, match_keys, normal=True):
        """Align input and target by criterion from match_keys

        Args
        ----
        match_keys : tuple
            a tuple of keywords to match input and target
        normal : bool
            normalize alignment criterion to the same scale

        """
        # find columns to match input and target
        input_col_ids  = [i for i, key in enumerate(self.input_keys)  if key in match_keys]
        target_col_ids = [i for i, key in enumerate(self.target_keys) if key in match_keys]

        # must find all columns to match
        assert len(match_keys) == len(input_col_ids)
        assert len(match_keys) == len(target_col_ids)

        self.match_keys = match_keys

        # initialize matching relationship buffer
        dists = []
        paths = []

        # for each identifier, builds matching relationship between input and target
        for i in range(len(self)):
            # focus only on columns to match
            flight_input  = self.input_set[i]
            flight_target = self.target_set[i]

            criterion_input  = flight_input[:, input_col_ids]
            criterion_target = flight_target[:, target_col_ids]

            if normal:
                mean_input  = np.mean(criterion_input , axis=0)
                mean_target = np.mean(criterion_target, axis=0)
                criterion_input  -= mean_input
                criterion_target -= mean_target
                std_input  = np.std(criterion_input , axis=0)
                std_target = np.std(criterion_target, axis=0)
                criterion_input  /= std_input
                criterion_target /= std_target
            else:
                logging.warning('Disable normalized alignment, may overfit on large scake criterions')

            # find matching path
            dist, path = fastdtw.fastdtw(criterion_input, criterion_target)

            # save to buffer
            dists.append(dist)
            paths.append(path)

            logging.info('Flight Pair \'{}\' has {} matchings with cost of {:.2f}'.format(
                            self.identifier[i], len(path), dist))

        # save to the class
        self.dists = dists
        self.paths = paths

    def interpolate(self):
        """Interpolate input and target data for each identifier to the same length"""
        # interpolate input or target segment
        def interp_subdata(subdata):
            return np.mean(subdata, axis=0)

        # unlock to interpolate
        self.unlock()

        # for each identifier, interpolate to match input and target
        for i in range(len(self)):
            # initialize subpath segment buffer
            path = self.paths[i]
            subpath = []

            # find all subpath segment that has length one on either input or target
            ptr = 0
            while ptr < len(path):
                begin_input, begin_target = path[ptr]

                # if is still in the same segment, continue to next step in the path
                while ptr < len(path) and \
                      (begin_input == path[ptr][0] or begin_target == path[ptr][1]):
                    ptr += 1

                end_input, end_target = path[ptr - 1]

                subpath.append(((begin_input, end_input), (begin_target, end_target)))

            logging.info('Flight Pair \'{}\' simutaneously downwards to length {} '
                         '(input: {}, target: {})'.format(
                            self.identifier[i], len(subpath),
                            len(self.input_set[i]), len(self.target_set[i])))

            # initialize interpolated input buffer and target buffer
            new_input  = []
            new_target = []

            # for each segment, interpolate so that it has length one on both input and target
            for subseq in subpath:
                # fetch a segment
                (begin_input, end_input), (begin_target, end_target) = subseq
                sub_input  = self.input_set[i][begin_input:end_input + 1, :]
                sub_target = self.target_set[i][begin_target:end_target + 1, :]

                # either input or target has length one
                assert len(sub_input) == 1 or len(sub_target) == 1

                # interpolate segment which has length over one
                if len(sub_input) == 1:
                    new_input.append(sub_input)
                    new_target.append(interp_subdata(sub_target))
                elif len(sub_target) == 1:
                    new_input.append(interp_subdata(sub_input))
                    new_target.append(sub_target)
                else:
                    raise NotImplementedError

            # update input and target datasets
            self.input_set[i]  = np.vstack(new_input)
            self.target_set[i] = np.vstack(new_target)

            # input and target should match on length
            assert len(self.input_set[i]) == len(self.target_set[i])

        # reconstruct subclasses and lock after interpolate
        self.reconstruct()
        self.lock()

    def align_and_interpolate(self, match_keys):
        """Execute align and interpolate together

        Args
        ----
        match_keys : tuple
            a tuple of keywords to match input and target

        """
        self.align(match_keys)
        self.interpolate()

    def distribute(self):
        """Distribute the data to make every day appears in turn

        Original Case:
            Day1_F1 Day1_F2 Day2_F1 ... ...

        Distribute Case:
            Day1_F1 Day2_F1 ... DAY1_F2 ...

        """
        dist_indices = []

        dist_criterion = []
        for i, itr in enumerate(self.identifier):
            date, fid = itr.split('_')
            dist_criterion.append((fid, date, i))

        dist_criterion = sorted(dist_criterion, key=lambda x: (x[0], x[1]))

        self.unlock()

        self.identifier = [self.identifier[itr[2]] for itr in dist_criterion]
        self.input_set  = [self.input_set[itr[2]] for itr in dist_criterion]
        self.target_set = [self.target_set[itr[2]] for itr in dist_criterion]

        self.reconstruct()
        self.lock()

    def prune_keys(self,
                   input_remain_keys=None , input_discard_keys=None ,
                   target_remain_keys=None, target_discard_keys=None):
        """Prune keywords and their data columns for input and target separately

        Args
        ----
        input_remain_keys : list
            keywords to remain for input
        input_discard_keys : list
            keywords to discard for input
        target_remain_keys : list
            keywords to remain for target
        target_discard_keys : list
            keywords to discard for target

        It will activate prune_keys function on input and target flight prune data
        classes separately.
        Pay attention that prune will recreate data, so it is essential to relink shorcut
        for each data.

        """
        # unlock to prune
        self.unlock()

        self.meta_input.prune_keys(remain_keys=input_remain_keys, discard_keys=input_discard_keys)
        self.meta_target.prune_keys(remain_keys=target_remain_keys, discard_keys=target_discard_keys)

        # relink and lock
        self.relink()
        self.lock()

    def prune_identifier(self, remain_identifier=None, discard_identifier=None):
        """Prune keywords and their data columns for input and target simutaneously

        Args
        ----
        remain_identifier : list
            identifier to remain
        discard_identifier : list
            identifier to discard

        Only one of two arguments is necessary. If give both, only remain_identifier will be
        accepted.

        """
        # unlock to prune
        self.unlock()

        self.meta_input.prune_identifier(remain_identifier=remain_identifier, discard_identifier=discard_identifier)
        self.meta_target.prune_identifier(remain_identifier=remain_identifier, discard_identifier=discard_identifier)

        # relink and lock after prune
        self.relink()
        self.lock()

    def update_target(self, meta_target):
        # unlock to prune
        self.unlock()

        self.meta_target = meta_target

        # relink and lock
        self.relink()
        self.lock()



# /**
#  * Frame Input-Target Flight Data Container
#  */
class FlightFramePairData(BasePairData):
    """Flight Frame Pair Data Container

    It will divide each sequence pair into overlapping frames which will be easy for
    neural network to accept.

    <TODO>
    From time domain to frequency domain
    </TODO>

    """
    def __init__(self, entity, input_win_len=None, target_win_len=None,
                 input_win_offset=None, target_win_offset=None,
                 input_win_offset_rate=None, target_win_offset_rate=None,
                 input_pad='repeat_base', target_pad='repeat_base'):
        if isinstance(entity, tuple) or isinstance(entity, list):
            assert len(entity) == 6
            self.from_(entity[0], entity[1], entity[2], entity[3], entity[4], entity[5])
            self.normal = None
        elif isinstance(entity, dict):
            assert len(entity) == 6
            self.from_(
                entity['input_set'], entity['input_keys'],
                entity['target_set'], entity['target_keys'],
                entity['identifier'], entity['window_dict'])
            self.normal = None
        elif isinstance(entity, str) and os.path.isfile(entity): self.load(entity)
        elif isinstance(entity, FlightSequencePairData):
            self.input_win_len          = input_win_len
            self.input_win_offset       = input_win_offset
            self.input_win_offset_rate  = input_win_offset_rate
            self.input_pad              = input_pad
            self.target_win_len         = target_win_len
            self.target_win_offset      = target_win_offset
            self.target_win_offset_rate = target_win_offset_rate
            self.target_pad             = target_pad
            self.from_seq_pair(seq_pair=entity)
            self.normal = None
        else:
            raise NotImplementedError

    def save(self, path):
        """Save data from the class

        Args
        ----
        path : str
            path to save all containing data

        """
        dirname = os.path.dirname(path)

        if not os.path.isdir(dirname): os.makedirs(dirname)
        torch.save({
                'window_dict': {
                    'input_win_len'         : self.input_win_len,
                    'input_win_offset'      : self.input_win_offset,
                    'input_win_offset_rate' : self.input_win_offset_rate,
                    'input_pad'             : self.input_pad,
                    'target_win_len'        : self.target_win_len,
                    'target_win_offset'     : self.target_win_offset,
                    'target_win_offset_rate': self.target_win_offset_rate,
                    'target_pad'            : self.target_pad,
                },
                'input_set'  : self.input_set,
                'target_set' : self.target_set,
                'input_keys' : self.input_keys,
                'target_keys': self.target_keys,
                'identifier' : self.identifier,
            }, path)

    def load(self, path):
        """Load data to the class

        Args
        ----
        path : str
            path to load all data to contain

        """
        assert os.path.isfile(path)

        save_dict = torch.load(path)

        self.input_win_len = save_dict['window_dict']['input_win_len']
        self.input_win_offset = save_dict['window_dict']['input_win_offset']
        self.input_win_offset_rate = save_dict['window_dict']['input_win_offset_rate']
        self.input_pad = save_dict['window_dict']['input_pad']
        self.target_win_len = save_dict['window_dict']['target_win_len']
        self.target_win_offset = save_dict['window_dict']['target_win_offset']
        self.target_win_offset_rate = save_dict['window_dict']['target_win_offset_rate']
        self.target_pad = save_dict['window_dict']['target_pad']

        self.input_set = save_dict['input_set']
        self.target_set = save_dict['target_set']
        self.input_keys = save_dict['input_keys']
        self.target_keys = save_dict['target_keys']
        self.identifier = save_dict['identifier']

        # lock data and identifier to avoid modification
        self.lock()

    def description(self):
        """Return a description of current class status

        Returns
        -------
        lines : str
            a description of containing data

        Describe number of keywords, number of flights, identifier and shape of each
        flight data input-target pair.

        """
        lines = ''

        # description header
        lines += 'Flight Sequence Frame Data: ({} keywords ---> {} keywords), ' \
                 '{} flights\n'.format(len(self.input_keys), len(self.target_keys), len(self))
        lines += 'Input  Configuration: window length of {} ' \
                 'with shift offset of {} [{}] (pad: {})\n'.format(
                    self.input_win_len, self.input_win_offset, self.input_win_offset_rate,
                    self.input_pad)
        lines += 'Target Configuration: window length of {} ' \
                 'with shift offset of {} [{}] (pad: {})\n'.format(
                    self.target_win_len, self.target_win_offset, self.target_win_offset_rate,
                    self.target_pad)

        # description for each flight data
        for i in range(len(self)):
            identifier = self.identifier[i]
            shape_input  = self.input_set[i].shape
            shape_target = self.target_set[i].shape

            # data shape must be #num_frames \times #frame_length \times #keys
            assert len(shape_input) == 3 and len(shape_target) == 3

            lines += ' -- {:10s}: {:6d} x {:3d} x {:2d} ---> {:6d} x {:3d} x {:2d}\n'.format(
                        identifier,
                        shape_input[0] , shape_input[1] , shape_input[2] ,
                        shape_target[0], shape_target[1], shape_target[2])

        return lines

    def from_(self, input_set, target_set, input_keys, target_keys, identifier, window_dict):
        """Initialize the class directly from its data components

        Args
        ----
        input_set: list or tuple
            a list or tuple of numpy.array data for input
        input_keys: list or tuple
            a list or tuple of string keywords for input
        target_set: list or tuple
            a list or tuple of numpy.array data for target
        target_keys: list or tuple
            a list or tuple of string keywords for target
        identifier: list or tuple
            a list or tuple of string identifier
        window_dict: dict
            a dictionary of window settings

        """
        super(FlightFramePairData, self).from_(
            input_set, target_set, input_keys, target_keys, identifier)

        self.input_win_len          = window_dict['input_win_len']
        self.input_win_offset       = window_dict['input_win_offset']
        self.input_win_offset_rate  = window_dict['input_win_offset_rate']
        self.input_pad              = window_dict['input_pad']
        self.target_win_len         = window_dict['target_win_len']
        self.target_win_offset      = window_dict['target_win_offset']
        self.target_win_offset_rate = window_dict['target_win_offset_rate']
        self.target_pad             = window_dict['target_pad']

    def from_seq_pair(self, seq_pair):
        """Initialize the class from flight sequence pair data class

        Args
        ----
        seq_pair : FlightSequencePairData
            flight sequence pair data class to initialize

        This is a clone initialization.

        """
        # can only construct from flight sequence pair data class
        assert isinstance(seq_pair, FlightSequencePairData)

        # clone data
        seq_input_set    = copy.deepcopy(seq_pair.input_set)
        seq_target_set   = copy.deepcopy(seq_pair.target_set)
        self.input_keys  = copy.deepcopy(seq_pair.input_keys)
        self.target_keys = copy.deepcopy(seq_pair.target_keys)
        self.identifier  = copy.deepcopy(seq_pair.identifier)

        # transfer sequence to frame
        frame_input_set  = []
        frame_target_set = []
        for i in range(len(self.identifier)):
            frame_input_set.append(self.seq_to_frame(
                seq_input_set[i], win_len=self.input_win_len,
                win_offset=self.input_win_offset, win_offset_rate=self.input_win_offset_rate,
                pad=self.input_pad))
            frame_target_set.append(self.seq_to_frame(
                seq_target_set[i], win_len=self.target_win_len,
                win_offset=self.target_win_offset, win_offset_rate=self.target_win_offset_rate,
                pad=self.target_pad))
        self.input_set  = frame_input_set
        self.target_set = frame_target_set

        self.lock()

    def to_seq_pair(self, seq_len_dict):
        """Extract data in the class into a sequence pair data class"""
        # transform frame set to sequence set
        seq_input_set  = []
        seq_target_set = []
        for i in range(len(self)):
            seq_input_set.append(self.frame_to_seq(
                self.input_set[i], win_len=self.input_win_len,
                win_offset=self.input_win_offset, win_offset_rate=self.input_win_offset_rate,
                seq_len=seq_len_dict[self.identifier[i]]['input']))
            seq_target_set.append(self.frame_to_seq(
                self.target_set[i], win_len=self.target_win_len,
                win_offset=self.target_win_offset, win_offset_rate=self.target_win_offset_rate,
                seq_len=seq_len_dict[self.identifier[i]]['target']))

        return FlightSequencePairData(
            entity=(
                seq_input_set, seq_target_set,
                self.input_keys, self.target_keys, self.identifier))

    def seq_to_frame(self, seq, win_len, win_offset=None, win_offset_rate=None, pad='repeat_base'):
        """Transform a sequence to a windowed frame

        Args
        ----
        seq : numpy.array
            sequence-like data to transform
        win_len : int
            window length
        win_offset : int
            window offset
        win_offset_rate : float
            window offset rate
            win_offset = int(math.ceil(win_offset_rate * win_len))
        pad : str
            method to pad when out of index

        Returns
        -------
        frame_buffer : numpy.array
            transformed frame-like data

        """
        def required_len(seq_len, win_len, win_offset, is_pad=True):
            if is_pad: num_wins = int(math.ceil((seq_len - win_len) / win_offset)) + 1
            else: num_wins = int(math.floor((seq_len - win_len) / win_offset)) + 1
            return win_len + (num_wins - 1) * win_offset

        # win_offset must be specified
        assert (not win_offset is None) or (not win_offset_rate is None)

        if win_offset is None: win_offset = int(math.ceil(win_len * win_offset_rate))

        # transfer seq into buffer according to pad method
        if pad == None:
            # compute buffer length without padding
            buffer_len = required_len(len(seq), win_len, win_offset, is_pad=False)

            # pad nothing, and discard tail elements
            sequence_buffer = seq[0:buffer_len]
        else:
            # compute buffer length without padding
            buffer_len = required_len(len(seq), win_len, win_offset, is_pad=True)
            pad_len = buffer_len - len(seq)

            # if need padding
            if pad_len > 0:# configure padding
                if pad == 'repeat_base':
                    seq_pad = np.repeat(seq[[-1]], pad_len, axis=0)
                elif pad == 'zero':
                    seq_pad = np.zeros(shape=tuple([pad_len] + list(seq.shape[1:])))
                else:
                    raise NotImplementedError
                seq_pad = seq_pad.astype(seq.dtype)

                # concatenate
                sequence_buffer = np.concatenate([seq, seq_pad], axis=0)
            else:
                sequence_buffer = seq[:]

        # get indices base of each window
        indices_base = np.arange(0, len(sequence_buffer) - win_len + 1, win_offset, dtype=np.intp)
        indices_base = np.repeat(indices_base.reshape(-1, 1), win_len, axis=1)

        # get indices offset inside a window from its base
        indices_offset = np.arange(0, win_len, dtype=np.intp)
        indices_offset = np.repeat(indices_offset.reshape(1, -1), len(indices_base), axis=0)

        logging.info('{} datapoints ({} from {} datapoints) to {} dataframes '
                     'of length {}, offset {}'.format(
                        len(sequence_buffer), pad, len(seq), len(indices_base),
                        win_len, win_offset))

        return sequence_buffer[indices_base + indices_offset]

    def frame_to_seq(self, frame, win_len, win_offset=None, win_offset_rate=None, seq_len=None):
        """Transform a windowed frame to a sequence

        Args
        ----
        frame : numpy.array
            frame-like data to transform
        win_len : int
            window length
        win_offset : int
            window offset
        win_offset_rate : float
            window offset rate
            win_offset = int(math.ceil(win_offset_rate * win_len))
        seq_len : int
            length of wanted sequence

        Returns
        -------
        seq_buffer : numpy.array
            transformed sequence-like data

        """
        # win_offset must be specified
        assert (not win_offset is None) or (not win_offset_rate is None)

        if win_offset is None: win_offset = int(math.ceil(win_len * win_offset_rate))

        # compute buffer length and allocate buffer
        buffer_len = win_len + (len(frame) - 1) * win_offset

        sequence_buffer = np.zeros(shape=tuple([buffer_len] + list(frame.shape[2:])))
        cnt_buffer = np.zeros(shape=sequence_buffer.shape)

        cnt_unit = np.ones(shape=tuple([win_len] + list(cnt_buffer.shape[1:])))

        # overlap-add all frame windows together
        for i in range(len(frame)):
            begin = i * win_offset
            end   = begin + win_len
            sequence_buffer[begin:end] += frame[i]
            cnt_buffer[begin:end] += cnt_unit

        # average overlapping sequence
        sequence_buffer /= cnt_buffer

        logging.info('{} dataframes of length {}, offset {} to {} datapoints '
                     '(require {} datapoints)'.format(
                        len(frame), win_len, win_offset, len(sequence_buffer),
                        seq_len))

        if seq_len is None: return sequence_buffer[:]
        else: return sequence_buffer[0:seq_len]

    def time_to_fourier(self, half=True):
        pass

    def fourier_to_time(self, freq, half=True):
        pass

    def time_to_haar(self, concat =True):
        """Haas wavelet discrete wave transform (haar dwt)

        Args
        ----
        concat : bool
            if concatenate approximation and detail together

        Returns
        -------
        haar : FlightFramePairData
            if concat is True, return concatenated haar domain
        approx, detail : FlightFramePairData
            if concat is False, return approximation and detail separately

        It is recommended to use this embedded function as external function.

            freq_domain = frame.FlightFrameDataPair.time_to_haar(time_domain, concat=True)

        But you can still use it as member function of a variable entity.

            freq_domain = time_domain.time_to_haar(concat=True)

        """
        # initialize buffer for haar transform
        approx_input  = []
        approx_target = []
        detail_input  = []
        detail_target = []
        haar_input    = []
        haar_target   = []

        for i in range(len(self)):
            input_x  = self.input_set[i]
            target_x = self.target_set[i]

            # dwt on frame, so swap dimension to put frame_length to last dimension
            input_x  = np.swapaxes(input_x , 1, 2)
            target_x = np.swapaxes(target_x, 1, 2)

            # haar transform
            assert input_x.shape[-1] % 2 == 0 and target_x.shape[-1] % 2 == 0
            input_a , input_d  = pywt.dwt(input_x , 'haar')
            target_a, target_d = pywt.dwt(target_x, 'haar')

            # swap dimension back
            input_a  = np.swapaxes(input_a , 1, 2)
            input_d  = np.swapaxes(input_d , 1, 2)
            target_a = np.swapaxes(target_a, 1, 2)
            target_d = np.swapaxes(target_d, 1, 2)

            if concat:
                haar_input.append(np.concatenate([input_a, input_d], axis=1))
                haar_target.append(np.concatenate([target_a, target_d], axis=1))
                logging.info('Time domain {}, {} to Haar frequency domain {}, {}'.format(
                                'x'.join([str(d) for d in self.input_set[i].shape]),
                                'x'.join([str(d) for d in self.target_set[i].shape]),
                                'x'.join([str(d) for d in haar_input[i].shape]),
                                'x'.join([str(d) for d in haar_target[i].shape])))
            else:
                approx_input.append(input_a)
                detail_input.append(input_d)
                approx_target.append(target_a)
                detail_target.append(target_d)
                logging.info('Time domain {}, {} to Haar Approximation frequency domain {}, {}'.format(
                                'x'.join([str(d) for d in self.input_set[i].shape]),
                                'x'.join([str(d) for d in self.target_set[i].shape]),
                                'x'.join([str(d) for d in approx_input[i].shape]),
                                'x'.join([str(d) for d in approx_target[i].shape])))
                logging.info('Time domain {}, {} to Haar Detail frequency domain {}, {}'.format(
                                'x'.join([str(d) for d in self.input_set[i].shape]),
                                'x'.join([str(d) for d in self.target_set[i].shape]),
                                'x'.join([str(d) for d in detail_input[i].shape]),
                                'x'.join([str(d) for d in detail_target[i].shape])))

        # construct window setting dict
        win_dict = {
            'input_win_len'         : self.input_win_len         ,
            'input_win_offset'      : self.input_win_offset      ,
            'input_win_offset_rate' : self.input_win_offset_rate ,
            'input_pad'             : self.input_pad             ,
            'target_win_len'        : self.target_win_len        ,
            'target_win_offset'     : self.target_win_offset     ,
            'target_win_offset_rate': self.target_win_offset_rate,
            'target_pad'            : self.target_pad            ,
        }

        if concat:
            haar = FlightFramePairData(entity=(
                haar_input, haar_target,
                self.input_keys, self.target_keys,
                self.identifier, win_dict))
            return haar
        else:
            approx = FlightFramePairData(entity=(
                approx_input, approx_target,
                self.input_keys, self.target_keys,
                self.identifier, win_dict))
            detail = FlightFramePairData(entity=(
                detail_input, detail_target,
                self.input_keys, self.target_keys,
                self.identifier, win_dict))
            return approx, detail

    def haar_to_time(freq, concat=True):
        """Haas wavelet inverse discrete wave transform (haar idwt)

        Args
        ----
        freq : FlightFramePairData or tuple
            if concat is True, it is FlightFramePairData
            if concat is False, it is a pair of FlightFramePairData
        concat : bool
            if concatenate approximation and detail together

        Returns
        -------
        time : FlightFramePairData
            time domain data

        It is not an embedded function, so it must be used as external function.

            time_domain = frame.FlightFrameDataPair.haar_to_time(freq_domain, concat=True)

        """
        if concat:
            haar = freq
            num = len(haar)
            input_keys  = haar.input_keys
            target_keys = haar.target_keys
            identifier  = haar.identifier
            win_dict = {
                'input_win_len'         : haar.input_win_len         ,
                'input_win_offset'      : haar.input_win_offset      ,
                'input_win_offset_rate' : haar.input_win_offset_rate ,
                'input_pad'             : haar.input_pad             ,
                'target_win_len'        : haar.target_win_len        ,
                'target_win_offset'     : haar.target_win_offset     ,
                'target_win_offset_rate': haar.target_win_offset_rate,
                'target_pad'            : haar.target_pad            ,
            }
        else:
            # window setting should match
            approx, detail = freq
            assert approx.identifier             == detail.identifier
            assert approx.input_keys             == detail.input_keys
            assert approx.target_keys            == detail.target_keys
            assert approx.input_win_len          == detail.input_win_len
            assert approx.input_win_offset       == detail.input_win_offset
            assert approx.input_win_offset_rate  == detail.input_win_offset_rate
            assert approx.input_pad              == detail.input_pad
            assert approx.target_win_len         == detail.target_win_len
            assert approx.target_win_offset      == detail.target_win_offset
            assert approx.target_win_offset_rate == detail.target_win_offset_rate
            assert approx.target_pad             == detail.target_pad
            num = len(approx)
            input_keys  = approx.input_keys
            target_keys = approx.target_keys
            identifier  = approx.identifier
            win_dict = {
                'input_win_len'         : approx.input_win_len         ,
                'input_win_offset'      : approx.input_win_offset      ,
                'input_win_offset_rate' : approx.input_win_offset_rate ,
                'input_pad'             : approx.input_pad             ,
                'target_win_len'        : approx.target_win_len        ,
                'target_win_offset'     : approx.target_win_offset     ,
                'target_win_offset_rate': approx.target_win_offset_rate,
                'target_pad'            : approx.target_pad            ,
            }

        # initialize time domain buffer
        input_set  = []
        target_set = []

        for i in range(num):
            if concat:
                input_h  = haar.input_set[i]
                target_h = haar.target_set[i]
                input_a  = input_h[:, :input_h.shape[1] // 2, :]
                input_d  = input_h[:, input_h.shape[1] // 2:, :]
                target_a = target_h[:, :target_h.shape[1] // 2, :]
                target_d = target_h[:, target_h.shape[1] // 2:, :]
            else:
                input_a  = approx.input_set[i]
                input_d  = detail.input_set[i]
                target_a = approx.target_set[i]
                target_d = detail.target_set[i]

            # dwt on frame, so swap dimension to put frame_length to last dimension
            input_a  = np.swapaxes(input_a , 1, 2)
            input_d  = np.swapaxes(input_d , 1, 2)
            target_a = np.swapaxes(target_a, 1, 2)
            target_d = np.swapaxes(target_d, 1, 2)

            # haar transform back
            input_x  = pywt.idwt(input_a , input_d , 'haar')
            target_x = pywt.idwt(target_a, target_d, 'haar')

            # swap dimension back
            input_x  = np.swapaxes(input_x , 1, 2)
            target_x = np.swapaxes(target_x, 1, 2)

            input_set.append(input_x)
            target_set.append(target_x)

            if concat:
                logging.info('Haar frequency domain {}, {} to Time domain {}, {}'.format(
                                'x'.join([str(d) for d in haar.input_set[i].shape]),
                                'x'.join([str(d) for d in haar.target_set[i].shape]),
                                'x'.join([str(d) for d in input_set[i].shape]),
                                'x'.join([str(d) for d in target_set[i].shape])))
            else:
                logging.info('Haar Approximation frequency domain {}, {}'.format(
                                'x'.join([str(d) for d in approx.input_set[i].shape]),
                                'x'.join([str(d) for d in approx.target_set[i].shape])))
                logging.info('Haar Detail frequency domain {}, {}'.format(
                                'x'.join([str(d) for d in detail.input_set[i].shape]),
                                'x'.join([str(d) for d in detail.target_set[i].shape])))
                logging.info('> Transform Time domain {}, {}'.format(
                                'x'.join([str(d) for d in input_set[i].shape]),
                                'x'.join([str(d) for d in target_set[i].shape])))

        return FlightFramePairData(entity=(
            input_set, target_set,
            input_keys, target_keys,
            identifier, win_dict))


# /**
#  * +-----------+
#  * | Testament |
#  * +-----------+
#  */


# /**
#  * +-------------------+
#  * | Console Interface |
#  * +-------------------+
#  */
