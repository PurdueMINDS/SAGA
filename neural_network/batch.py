"""Process aligned and framed data classes to neural network batch loader class"""
import docopt
import logging
import copy

import sys
import os
import math

import numpy as np

import torch

import drawbox
import raw
import flight
import frame

# must run on Python 3
assert sys.version_info[0] == 3


# /**
#  * Flight Timestep Data Batch Container
#  */
class TimestepPairDataLoader(frame.BasePairData):
    """Iteratable Flight Timestep Pair Data Batch Loader

    It is the interface between processed data and neural network.
    It supports iterator and index with or without shuffling.

    Pay attention that for RNN consecutivity, the class only supports ordered fetching and
    will automatically pad the batch without enough time steps. By current, it only supports
    batch size of 1.

    In addition, since frame data can also be regarded as sequence data, this loader should
    support to initialize from both sequence pairs and frame pairs.

    """
    def __init__(self, entity, timestep):
        if isinstance(entity, str) and os.path.isfile(entity): self.load(entity)
        elif isinstance(entity, frame.FlightSequencePairData) or \
             isinstance(entity, frame.FlightFramePairData):
            self.timestep = timestep
            self.from_pair(entity)
        else:
            raise NotImplementedError

    def __getitem__(self, idx):
        if isinstance(idx, int):
            return self.identifier[idx], \
                   (self.input_set[idx], self.target_set[idx])
        elif isinstance(idx, str):
            for i, identifier in enumerate(self.identifier):
                if identifier == idx:
                    return self.identifier[i], (self.input_set[i], self.target_set[i])
            return None, (None, None)

    class Iter(object):
        def __init__(self, array):
            self.array = array

            self.ptr = 0

        def __iter__(self):
            return self

        def __next__(self):
            if self.ptr == len(self.array):
                raise StopIteration

            data = self.array[self.ptr]
            self.ptr += 1

            return data

    def __iter__(self):
        return self.Iter(self)

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
                'timestep'     : self.timestep,
                'appendix'     : self.appendix,
                'num_timesteps': self.num_timesteps_dict,
                'input_set'    : self.input_set,
                'target_set'   : self.target_set,
                'input_keys'   : self.input_keys,
                'target_keys'  : self.target_keys,
                'identifier'   : self.identifier,
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

        self.timestep = save_dict['timestep']

        self.appendix = save_dict['appendix']
        self.num_timesteps_dict = save_dict['num_timesteps']

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

        Describe number of keywords, number of batches, and shape of each batch.

        """
        pass

    def from_pair(self, pair):
        """Initialize the class from flight sequence pair data class

        Args
        ----
        pair : frame.FlightSequencePairData or frame.FlightFramePairData
            flight pair data to initialize the class

        It will divide each flight data into batches with timestep in order.

        """
        # can only construct from flight sequence pair data class
        assert isinstance(pair, frame.FlightSequencePairData) or \
               isinstance(pair, frame.FlightFramePairData)
        # clone data
        flight_input  = copy.deepcopy(pair.input_set)
        flight_target = copy.deepcopy(pair.target_set)
        self.input_keys = copy.deepcopy(pair.input_keys)
        self.target_keys = copy.deepcopy(pair.target_keys)
        flight_identifier = copy.deepcopy(pair.identifier)

        # save additional settings into appendix
        if isinstance(pair, frame.FlightFramePairData):
            self.appendix = {
                'window_dict': {
                    'input_win_len'         : pair.input_win_len,
                    'input_win_offset'      : pair.input_win_offset,
                    'input_win_offset_rate' : pair.input_win_offset_rate,
                    'input_pad'             : pair.input_pad,
                    'target_win_len'        : pair.target_win_len,
                    'target_win_offset'     : pair.target_win_offset,
                    'target_win_offset_rate': pair.target_win_offset_rate,
                    'target_pad'            : pair.target_pad,
                },
            }
        else:
            self.appendix = None

        # initialize timestep pair data buffer
        timestep_input      = []
        timestep_target     = []
        timestep_identifier = []
        num_timesteps_dict = {}

        # divide each flight input and target pair into timesteps
        for i in range(len(flight_identifier)):
            # fetch a pair of input and target
            input_data  = flight_input[i]
            target_data = flight_target[i]
            identifier  = flight_identifier[i]

            logging.info('Divide \'{}\' ({} x {} x {}, {} x {} x {}) into timesteps ({})'.format(
                            identifier,
                            input_data.shape[0], input_data.shape[1], input_data.shape[2],
                            target_data.shape[0], target_data.shape[1], target_data.shape[2],
                            self.timestep))

            # generate all timesteps of the fetched pair
            cnt = 0
            for begin in range(0, len(target_data) - self.timestep):
                end = begin + self.timestep

                timestep_input.append(input_data[begin:end])
                timestep_target.append(target_data[begin:end])
                timestep_identifier.append('{}_{}'.format(identifier, cnt))
                cnt += 1

            logging.info(' -- Generate {} timesteps'.format(cnt))

            # record number of batches of the same flight
            num_timesteps_dict[identifier] = cnt

        # save to the class
        self.input_set  = timestep_input
        self.target_set = timestep_target
        self.identifier = timestep_identifier

        self.num_timesteps_dict = num_timesteps_dict

        self.lock()

    def to_seq_pair(self):
        """Extract data in the class into a sequence pair data class"""
        assert self.appendix is None

    def to_frame_pair(self):
        """Extract data in the class into a frame pair data class"""
        assert self.appendix is not None

        # initialize frame data buffer
        frame_input      = []
        frame_target     = []
        frame_identifier = []

        # traverse all batches to construct flight data
        ptr = 0
        while ptr < len(self):
            # fetch current flight information
            current_flight = self.flight_identifier(self.identifier[ptr])
            num_batches = self.num_timesteps_dict[current_flight]

            # initialize current flight data buffer
            input_buffer  = []
            target_buffer = []

            for i in range(num_batches):
                # batches must be ordered
                assert self.identifier[ptr + i] == '{}_{}'.format(current_flight, i)

                # append to flight buffer
                input_buffer.append(self.input_set[ptr + i])
                target_buffer.append(self.target_set[ptr + i])

            # move to next flight
            ptr += num_batches

            # make sure that all batches of current flight have been fetched
            assert ptr >= len(self) or (not current_flight in self.identifier[ptr])

            # append to frame buffer
            frame_input.append(np.concatenate(input_buffer, axis=0))
            frame_target.append(np.concatenate(target_buffer, axis=0))
            frame_identifier.append(current_flight)

        return frame.FlightFramePairData((
            frame_input, frame_target,
            self.input_keys, self.target_keys,
            frame_identifier, self.appendix['window_dict'],
        ))

    def flight_identifier(self, batch_idt):
        """Get flight identifier out of batch identifier

        Args
        ----
        batch_itd : str
            batch identifier

        Returns
        -------
        identifier : str
            identifier of the flight to which the batch belongs

        """
        entities = batch_idt.split('_')

        # batch identifier must be date_flight_batch
        assert len(entities) == 3

        return '{}_{}'.format(entities[0], entities[1])

    def append_key(self, target_key, identifier, data):
        """Append new keyword to the class

        Args
        ----
        target_key : str
            keyword to append
        identifier : list
            a list of identifier waiting to append
        data : list
            a list of data to append to the class

        This function is designed for prediction. When using neural network to predict a
        keyword, it will generate a list of prediction for each batch (identifier here
        corresponds to batch, not flight). By put this list of prediction into this function,
        it can append prediction to each batch, so that it can regenerate other types of
        flight classes with the prediction.

        """
        # each appending batch should have identifier
        assert len(identifier) == len(data)

        # appending keyword should not conflict with other keywords
        assert not target_key in self.target_keys

        logging.info('Append {} batches to {} column of target batches'.format(len(data), target_key))

        # unlock to append
        self.unlock()

        for i in range(len(self)):
            # appending identifier should match those of the class in order
            assert identifier[i] == self.identifier[i]

            # append data to the class
            self.target_set[i] = np.concatenate([self.target_set[i], data[i]], axis=2)

        self.target_keys.append(target_key)

        # lock after append
        self.lock()

class FramePairDataLoader(frame.BasePairData):
    """Iteratable Flight Frame Pair Data Batch Loader

    It is the interface between processed data and neural network.
    It supports iterator and index with or without shuffling. It also provides functions
    to check if two batches are consecutive for RNN architecture.

    """
    def __init__(self, entity, batch_size=None, shuffle=False, drop_last=False):
        if isinstance(entity, str) and os.path.isfile(entity): self.load(entity)
        elif isinstance(entity, frame.FlightFramePairData):
            self.batch_size = batch_size
            self.shuffle    = shuffle
            self.drop_last  = drop_last
            self.from_frame_pair(entity)
        else:
            raise NotImplementedError

    def __getitem__(self, idx):
        if isinstance(idx, int):
            return self.identifier[self.idx_link[idx]], \
                   (self.input_set[self.idx_link[idx]], self.target_set[self.idx_link[idx]])
        elif isinstance(idx, str):
            for i, identifier in enumerate(self.identifier):
                if identifier == idx:
                    return self.identifier[i], (self.input_set[i], self.target_set[i])
            return None, (None, None)

    class Iter(object):
        def __init__(self, array):
            self.array = array
            self.array.refresh_idx_link()

            self.ptr = 0

        def __iter__(self):
            return self

        def __next__(self):
            if self.ptr == len(self.array):
                raise StopIteration

            data = self.array[self.ptr]
            self.ptr += 1

            return data

    def __iter__(self):
        return self.Iter(self)

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
                'batch_dict' : {
                    'batch_size': self.batch_size,
                    'shuffle'   : self.shuffle,
                    'drop_last' : self.drop_last,
                },
                'appendix'   : self.appendix,
                'num_batches': self.num_batches_dict,
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

        self.batch_size = save_dict['batch_dict']['batch_size']
        self.shuffle = save_dict['batch_dict']['shuffle']
        self.drop_last = save_dict['batch_dict']['drop_last']

        self.appendix = save_dict['appendix']
        self.num_batches_dict = save_dict['num_batches']

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

        Describe number of keywords, number of batches, and shape of each batch.

        """
        lines = ''

        # description header
        lines += 'Flight Pair Batch Data: ({} keywords ---> {} keywords), ' \
                 '{} batches\n'.format(len(self.input_keys), len(self.target_keys), len(self))
        lines += 'Configuration: batch size of {}{}{}\n'.format(
                    self.batch_size,
                    ', shuffle' if self.shuffle else '',
                    ', drop last' if self.drop_last else '')

        # description for batches of each flight
        cnt = 0
        for identifier in self.num_batches_dict:
            num_batches = self.num_batches_dict[identifier]
            lines += ' -- \'{}\' has {} batches\n'.format(identifier, num_batches)
            cnt += num_batches

        # it should only have batches from all flights
        assert len(self) == cnt

        return lines

    def from_frame_pair(self, frame_pair):
        """Initialize the class from flight frame pair data class

        Args
        ----
        frame_pair : frame.FlightFramePairData
            flight frame pair data to initialize the class

        It will divide each flight data into batches, and will drop last incomplete batch
        according to drop_last setting of the class.

        """
        # clone data
        flight_input  = copy.deepcopy(frame_pair.input_set)
        flight_target = copy.deepcopy(frame_pair.target_set)
        self.input_keys = copy.deepcopy(frame_pair.input_keys)
        self.target_keys = copy.deepcopy(frame_pair.target_keys)
        flight_identifier = copy.deepcopy(frame_pair.identifier)

        # save additional settings into appendix
        self.appendix = {
            'window_dict': {
                'input_win_len'         : frame_pair.input_win_len,
                'input_win_offset'      : frame_pair.input_win_offset,
                'input_win_offset_rate' : frame_pair.input_win_offset_rate,
                'input_pad'             : frame_pair.input_pad,
                'target_win_len'        : frame_pair.target_win_len,
                'target_win_offset'     : frame_pair.target_win_offset,
                'target_win_offset_rate': frame_pair.target_win_offset_rate,
                'target_pad'            : frame_pair.target_pad,
            },
        }

        # input and target must be one-to-one pairs
        assert len(flight_identifier) == len(flight_input)
        assert len(flight_identifier) == len(flight_target)

        # initialize batch pair data buffer
        batch_input      = []
        batch_target     = []
        batch_identifier = []
        num_batches_dict = {}

        # divide each flight input and target pair into batches
        for i in range(len(flight_identifier)):
            # fetch a pair of input and target
            input_data  = flight_input[i]
            target_data = flight_target[i]
            identifier  = flight_identifier[i]

            logging.info('Divide \'{}\' ({} x {} x {}, {} x {} x {}) into batches ({}{})'.format(
                            identifier,
                            input_data.shape[0], input_data.shape[1], input_data.shape[2],
                            target_data.shape[0], target_data.shape[1], target_data.shape[2],
                            self.batch_size,
                            ', drop last' if self.drop_last else ''))

            # generate all batches of the fetched pair
            begin = 0
            cnt = 0
            while begin < len(target_data):
                end = begin + self.batch_size
                if end > len(target_data): end = len(target_data)

                # if is complete batch or do not drop last incomplete batch, append to buffer
                if begin + self.batch_size == end or (not self.drop_last):
                    batch_input.append(input_data[begin:end])
                    batch_target.append(target_data[begin:end])
                    batch_identifier.append('{}_{}'.format(identifier, cnt))
                    cnt += 1

                # move to next batch
                begin = end

            logging.info(' -- Generate {} batches'.format(cnt))

            # record number of batches of the same flight
            num_batches_dict[identifier] = cnt

        # save to the class
        self.input_set  = batch_input
        self.target_set = batch_target
        self.identifier = batch_identifier

        self.num_batches_dict = num_batches_dict

        # refresh index link on construction
        self.refresh_idx_link()

        self.lock()

    def to_frame_pair(self):
        """Extract data in the class into a frame pair data class"""
        if self.drop_last:
            logging.warning('Reconstruct from drop last batches')
            raise NotImplementedError

        # initialize frame data buffer
        frame_input      = []
        frame_target     = []
        frame_identifier = []

        # traverse all batches to construct flight data
        ptr = 0
        while ptr < len(self):
            # fetch current flight information
            current_flight = self.flight_identifier(self.identifier[ptr])
            num_batches = self.num_batches_dict[current_flight]

            # initialize current flight data buffer
            input_buffer  = []
            target_buffer = []

            for i in range(num_batches):
                # batches must be ordered
                assert self.identifier[ptr + i] == '{}_{}'.format(current_flight, i)

                # append to flight buffer
                input_buffer.append(self.input_set[ptr + i])
                target_buffer.append(self.target_set[ptr + i])

            # move to next flight
            ptr += num_batches

            # make sure that all batches of current flight have been fetched
            assert ptr >= len(self) or (not current_flight in self.identifier[ptr])

            # append to frame buffer
            frame_input.append(np.concatenate(input_buffer, axis=0))
            frame_target.append(np.concatenate(target_buffer, axis=0))
            frame_identifier.append(current_flight)

        return frame.FlightFramePairData((
            frame_input, frame_target,
            self.input_keys, self.target_keys,
            frame_identifier, self.appendix['window_dict'],
        ))

    def refresh_idx_link(self):
        """Shuffle index link if necessary

        If the class does not shuffle data, it will create ordered link.

        """
        self.idx_link = np.arange(len(self))
        if self.shuffle: np.random.shuffle(self.idx_link)

    def flight_identifier(self, batch_idt):
        """Get flight identifier out of batch identifier

        Args
        ----
        batch_itd : str
            batch identifier

        Returns
        -------
        identifier : str
            identifier of the flight to which the batch belongs

        """
        entities = batch_idt.split('_')

        # batch identifier must be date_flight_batch
        assert len(entities) == 3

        return '{}_{}'.format(entities[0], entities[1])

    def is_same_flight_batch(self, batch_idt1, batch_idt2):
        """Check if two batch identifier are from the same flight

        Args
        ----
        batch_itd1 : str
            one batch identifier
        batch_itd2 : str
            another batch identifier

        Returns
        -------
        flag : bool
            if two batch identifier are from the same flight

        """
        entities1 = batch_idt1.split('_')
        entities2 = batch_idt2.split('_')

        # batch identifier must be date_flight_batch
        assert len(entities1) == 3 and len(entities2) == 3

        return (entities1[0], entities1[1]) == (entities2[0], entities2[1])

    def is_consecutive_batch(self, batch_idt1, batch_idt2):
        """Check if two batch identifier are consecutive

        Args
        ----
        batch_itd1 : str
            one batch identifier
        batch_itd2 : str
            another batch identifier

        Returns
        -------
        flag : bool
            if two batch identifier are consecutive

        """
        entities1 = batch_idt1.split('_')
        entities2 = batch_idt2.split('_')

        # batch identifier must be date_flight_batch
        assert len(entities1) == 3 and len(entities2) == 3

        # fetch batch ID
        batch_id1 = int(entities1[2])
        batch_id2 = int(entities2[2])

        return (entities1[0], entities1[1]) == (entities2[0], entities2[1]) and \
               (batch_idt1 + 1 == batch_idt2) or (batch_idt1 == batch_idt2 + 1)

    def append_key(self, target_key, identifier, data):
        """Append new keyword to the class

        Args
        ----
        target_key : str
            keyword to append
        identifier : list
            a list of identifier waiting to append
        data : list
            a list of data to append to the class

        This function is designed for prediction. When using neural network to predict a
        keyword, it will generate a list of prediction for each batch (identifier here
        corresponds to batch, not flight). By put this list of prediction into this function,
        it can append prediction to each batch, so that it can regenerate other types of
        flight classes with the prediction.

        """
        # each appending batch should have identifier
        assert len(identifier) == len(data)

        # appending keyword should not conflict with other keywords
        assert not target_key in self.target_keys

        logging.info('Append {} batches to {} column of target batches'.format(len(data), target_key))

        # unlock to append
        self.unlock()

        for i in range(len(self)):
            # appending identifier should match those of the class in order
            assert identifier[i] == self.identifier[i]

            # append data to the class
            self.target_set[i] = np.concatenate([self.target_set[i], data[i]], axis=2)

        self.target_keys.append(target_key)

        # lock after append
        self.lock()


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
