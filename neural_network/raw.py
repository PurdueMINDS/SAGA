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
"""Load raw data into classes"""
import docopt
import logging
import copy

import sys
import os
import re
import math
import pandas as pd

import numpy as np

import torch

import constant
import drawbox

# must run on Python 3
assert sys.version_info[0] == 3


# /**
#  * Class Basis
#  */
class BaseData(object):
    """Class basis for dataset of one type

    It is just a virtual class basis, not an entity class. Avoid construct any variables
    through this class.
    It only provides a virtual class basis for data classes of only one type, e.g. only
    G1000 data, only phone data. It can hold several dates and several flights, only resource
    type is limited to one.

    """
    def __init__(self):
        super(BaseData, self).__init__()

    def __getitem__(self, idx):
        if isinstance(idx, int):
            return self.identifier[idx], self.dataset[idx]
        elif isinstance(idx, str):
            for i, identifier in enumerate(self.identifier):
                if identifier == idx:
                    return identifier, self.dataset[i]
            return None, None

    def __len__(self):
        return len(self.identifier)

    def clone(self):
        """Return a clone of the class"""
        return copy.deepcopy(self)

    def lock(self):
        """Lock all data to avoid modification"""
        for data in self.dataset: data.setflags(write=False)
        self.dataset = tuple(self.dataset)
        self.keys = tuple(self.keys)
        self.identifier = tuple(self.identifier)
        self.locked = True

    def unlock(self):
        """Unlock all data for modification"""
        self.dataset = list(self.dataset)
        self.keys = list(self.keys)
        self.identifier = list(self.identifier)
        for data in self.dataset: data.setflags(write=True)
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
                'dataset'   : self.dataset,
                'keys'      : self.keys,
                'identifier': self.identifier,
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

        self.dataset = save_dict['dataset']
        self.keys = save_dict['keys']
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
        flight data.

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

        For each identifier, plot each row (keyword) of its data together in the same
        figure.

        """
        # it is recommended, but not necessary, to implement in all child classes
        # it may vary on different child classes
        raise NotImplementedError

    def from_(self, dataset, keys, identifier):
        """Initialize the class directly from its data components

        Args
        ----
        dataset: list or tuple
            a list or tuple of numpy.array data
        keys: list or tuple
            a list or tuple of string keywords
        identifier: list or tuple
            a list or tuple of string identifier

        """
        # dataset must be a list or tuple of 2D numpy.array \
        # keys must be a list or tuple of string and match number of columns of element in dataset \
        # identifier must be a list or tuple of string and match length of dataset
        assert isinstance(dataset, list) or isinstance(dataset, tuple)
        assert isinstance(keys, list) or isinstance(keys, tuple)
        assert isinstance(identifier, list) or isinstance(identifier, tuple)
        assert len(identifier) == len(dataset)
        for itr in dataset:
            assert isinstance(itr, np.ndarray)
            assert len(itr.shape) == 2
            assert len(keys) == itr.shape[-1]

        # clone data
        self.dataset = copy.deepcopy(dataset)
        self.keys = copy.deepcopy(keys)
        self.identifier = copy.deepcopy(identifier)

        self.lock()


# /**
#  * Raw Data Container
#  */
class RawG1000Data(BaseData):
    """Raw G1000 Data container

    It contains all raw G1000 data in numpy.array according to flight date and number.
    It also contains a list of keywords to record relationship between flight data rows
    and keywords.
    After loading data, freeze every thing to avoid modification.

    Collector may falsely put record files to wrong date folder, so the class will discard
    those files whose date column do not cooperate with date folder.

    """
    def __init__(self, entity, want=None):
        if isinstance(entity, tuple) or isinstance(entity, list):
            assert len(entity) == 3
            self.from_(entity[0], entity[1], entity[2])
        elif isinstance(entity, dict):
            assert len(entity) == 3
            self.from_(entity['dataset'], entity['keys'], entity['identifier'])
        elif os.path.isfile(entity): self.load(entity)
        elif os.path.isdir(entity): self.from_folder(entity, want=want)
        else: raise NotImplementedError

    def description(self):
        """Return a description of current class status
        
        Returns
        -------
        lines : str
            a description of containing data

        Describe number of keywords, number of flights, identifier and shape of each
        flight data.

        """
        lines = ''
        
        # description header
        lines += 'Raw G1000 Data: {} keywords, {} flights\n'.format(len(self.keys), len(self))

        # description for each flight data
        for i, data in enumerate(self.dataset):
            identifier = self.identifier[i]
            shape = data.shape

            # data shape must be #datapoints \times #keys
            assert len(shape) == 2

            lines += ' -- {:10s}: {:6d} x {:2d}\n'.format(identifier, shape[0], shape[1])

        return lines

    def plot(self, dir):
        """Plot all data

        Args
        ----
        dir : str
            directory to save all plots

        For each identifier, plot each column (keyword) of its data together in the same
        figure.

        """
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
                num_rows=len(self.keys), num_cols=1,
                ax_height=10, ax_width=80,
                title=title, font_size=40)
            
            logging.info('Plot \'{}\''.format(title))

            # plot each keyword
            for j, key in enumerate(self.keys):
                subtitle = key_to_subtitle(key)
                fig.subtitle(row_id=j, col_id=0, subtitle=subtitle)

                # logging.info(' -- Plot \'{}\''.format(subtitle))

                fig.lineplot(
                    row_id=j, col_id=0,
                    x_data=range(len(self.dataset[i][:, j])),
                    y_data=self.dataset[i][:, j],
                    color='C%d' % (j // 3))

            fig.save_fig(os.path.join(dir, 'png', '{}.png'.format(identifier)), close=False)
            fig.save_fig(os.path.join(dir, 'pdf', '{}.pdf'.format(identifier)), close=True)

    # G1000 data dict keywords to raw csv file keywords
    KEYS_DICT = {
        'date'   : ('Date'     , 0 ),
        'time'   : ('Time'     , 1 ),
        'utc'    : ('TimeZone' , 2 ),
        'alt'    : ('AltGPS'   , 42),
        'lat'    : ('Latitude' , 4 ),
        'long'   : ('Longitude', 5 ),
        'pitch'  : ('Pitch'    , 13),
        'roll'   : ('Roll'     , 14),
        'heading': ('HDG'      , 17),
    }

    # G1000 keywords to their long version
    LONG_KEYS = {
        'time'   : 'Time'     ,
        'alt'    : 'Altitude' ,
        'lat'    : 'Latitude' ,
        'long'   : 'Longitude',
        'pitch'  : 'Pitch'    ,
        'roll'   : 'Roll'     ,
        'heading': 'Heading'  ,
    }

    def from_file(self, path):
        """Load a csv file into a data dict

        Args
        ----
        path : str
            path to the csv file to load

        Returns
        -------
        date_file : str
            file recording date
        data_dict : dict
            a dict of keyword data

        """
        logging.info('Load G1000 file: \'{}\''.format(path))

        # create an empty data dict
        data_dict = {
            'time'   : [],
            'alt'    : [],
            'lat'    : [],
            'long'   : [],
            'pitch'  : [],
            'roll'   : [],
            'heading': [],
        }

        # load file into line list in buffer
        extension = os.path.splitext(path)[1]
        if extension == '.csv':
            # replace non-utf-8 character by a valid character
            file = open(path, 'r', errors='replace')
            file_lines = file.readlines()
            file.close()
        elif extension == '.xlsx':
            buffer = pd.read_excel(path).to_csv(sep=',', index=False)
            buffer = buffer.split('\n')[:-1]
            file_lines = ['{}\n'.format(content) for content in buffer]

        # read column unit header line (1 line)
        unit_header_line = file_lines[0]
        unit_header_line = unit_header_line.strip()
        col_units = re.split(r',\s*', unit_header_line)

        # read column name header line (1 line)
        name_header_line = file_lines[1]
        name_header_line = name_header_line.strip()
        col_names = re.split(r',\s*', name_header_line)

        assert len(col_names) == len(col_units)

        # check keyword existance
        for key in data_dict:
            assert key in self.KEYS_DICT
            assert key == 'time' or \
                   self.KEYS_DICT[key][0] == col_names[self.KEYS_DICT[key][1]]

        # read each data line in order
        date_file = None
        collecting_idx = []
        for idx, line in enumerate(file_lines[2:]):
            line_entities = re.split(r',\s*', line)
            assert len(col_names) == len(line_entities)

            # check if is collecting keyword data
            collecting = True
            for key in data_dict:
                if len(line_entities[self.KEYS_DICT[key][1]]) == 0:
                    collecting = False
                    break
            if not collecting:
                 # only record the first consecutive data segment
                if len(collecting_idx) > 0: break
                else: continue

            # before collecting date, collect date for future check
            date_col = line_entities[self.KEYS_DICT['date'][1]]
            date_col = date_col[0:10]
            date_ele = re.split(r'/|-', date_col)
            assert len(date_ele) == 3
            if len(date_ele[0]) == 4:
                date_col = '{}/{}/{}'.format(date_ele[1], date_ele[2], date_ele[0])
            elif len(date_ele[2]) == 4:
                date_col = '{}/{}/{}'.format(date_ele[0], date_ele[1], date_ele[2])
            if date_file is None: date_file = date_col
            else: assert date_file == date_col

            # before collecting time, collect time zone to check
            utc_col = line_entities[self.KEYS_DICT['utc'][1]]
            hour, minute = utc_col.split(':')
            hour, minute = int(hour), int(minute)
            # assert hour == -5

            # if is colllecting, save all keyword data, and track line index
            for key in data_dict:
                # collect time data
                if key == 'time':
                    time_col = line_entities[self.KEYS_DICT[key][1]]
                    hour, minute, second = time_col.split(':')
                    hour, minute, second = int(hour), int(minute), int(second)
                    time_val = hour * 3600 + minute * 60 + second

                    # value of time should increase
                    if len(data_dict[key]) > 0 and time_val < data_dict[key][-1]:
                        logging.warning("line {}: {} decreases to {}".format(
                                            idx, data_dict[key][-1], time_val))

                    data_dict[key].append(time_val)
                else:
                    data_dict[key].append(float(line_entities[self.KEYS_DICT[key][1]]))
            collecting_idx.append(idx)

        # collecting indices should be consectutive
        assert len(collecting_idx) == max(collecting_idx) - min(collecting_idx) + 1
        for key in data_dict:
            assert len(collecting_idx) == len(data_dict[key])

        logging.info('[G1000 file] : {} keywords, {} datapoints (at {})'.format(
                        len(data_dict), len(collecting_idx), date_file))

        return date_file, data_dict

    def from_date_folder(self, dir, date):
        """Load all csv files recorded on the date

        Args
        ----
        dir : str
            root directory to all G1000 csv files
        date : str
            date folder name to load

        Returns
        -------
        keys : list
            a list of keywords
        data : list
            a list of numpy.array data on the date

        """
        # check if date given by folder is same as date given by file
        def is_same_date(date_folder, date_file):
            # - Folder Date:
            #   0 1 2 3 4 5
            #   M M D D Y Y
            # - File Date:
            #   0 1 2 3 4 5 6 7 8 9
            #   M M / D D / Y Y Y Y
            folder_date_pair = (int(date_folder[4:6]), int(date_folder[0:2]), int(date_folder[2:4]))
            file_date_pair = date_file.split('/')
            file_date_pair = (int(file_date_pair[2][2:4]), int(file_date_pair[0]), int(file_date_pair[1]))
            return folder_date_pair == file_date_pair

        date_dir = os.path.join(dir, date)

        logging.info('Load G1000 date {} folder: \'{}\''.format(date, date_dir))

        # directory should exist
        assert os.path.isdir(dir)
        assert os.path.isdir(date_dir)

        keys = None
        data = []

        for filename in sorted(os.listdir(date_dir)):
            path = os.path.join(date_dir, filename)

            date_file, meta_data_dict = self.from_file(path)
            meta_keys = meta_data_dict.keys()

            # ignore file which does not fit the folder date
            if not is_same_date(date_folder=date, date_file=date_file):
                logging.warning('folder date {} and file date {} are not the same'.format(
                    date, date_file))
                continue

            # keywords should agree on all data
            if keys is None: keys = meta_keys
            assert keys == meta_keys

            # convert to list
            meta_data_list = [meta_data_dict[key] for key in keys]

            # convert to numpy.array
            # #datapoints \times #keys
            meta_data_array = np.transpose(np.array(meta_data_list))
            data.append(meta_data_array)

        # must read any keyword data
        assert not keys is None
        assert len(data) > 0

        logging.info('[G1000 date folder] : {} keywords, {} flights'.format(len(keys), len(data)))

        return keys, data

    def from_folder(self, dir, want=None):
        """Load all csv files to the class

        Args
        ----
        dir : str
            root directory to all G1000 csv files
        want : list
            wanting date

        """
        logging.info('Load G1000 folder: \'{}\''.format(dir))

        # directory should exist
        assert os.path.isdir(dir)

        self.keys = None
        self.dataset = []
        self.identifier = []

        load_list = sorted(os.listdir(dir)) if want is None else want
        for date in load_list:
            date_keys, date_data = self.from_date_folder(dir, date)

            # keywords should agree on all data
            if self.keys is None: self.keys = date_keys
            assert self.keys == date_keys

            # finally save data to the class, and identify it by date and flight number
            for i, flight_data in enumerate(date_data):
                self.dataset.append(flight_data)
                self.identifier.append('{}_{}'.format(date, '%02d' % i))

        # must read any keyword data
        assert not self.keys is None
        assert len(self.dataset) > 0

        # lock data and identifier to avoid modification
        self.lock()

class RawPhoneData(BaseData):
    """Raw Phone Data container

    It contains all raw phone data in numpy.array according to flight date. Since phone
    data do not differentiate different flights on the same date, it will identify data
    according to flight number.
    It also contains a list of keywords to record relationship between flight data rows
    and keywords.
    After loading data, freeze every thing to avoid modification.

    """
    def __init__(self, entity, want=None):
        if isinstance(entity, tuple) or isinstance(entity, list):
            assert len(entity) == 3
            self.from_(entity[0], entity[1], entity[2])
        elif isinstance(entity, dict):
            assert len(entity) == 3
            self.from_(entity['dataset'], entity['keys'], entity['identifier'])
        elif os.path.isfile(entity): self.load(entity)
        elif os.path.isdir(entity): self.from_folder(entity, want=want)
        else: raise NotImplementedError

    def description(self):
        """Return a description of current class status
        
        Returns
        -------
        lines : str
            a description of containing data

        Describe number of keywords, number of flights, identifier and shape of each
        flight data.

        """
        lines = ''
        
        # description header
        lines += 'Raw Phone Data: {} keywords, {} flights\n'.format(len(self.keys), len(self))

        # description for each flight data
        for i, data in enumerate(self.dataset):
            identifier = self.identifier[i]
            shape = data.shape

            # data shape must be #datapoints \times #keys
            assert len(shape) == 2

            lines += ' -- {:10s}: {:6d} x {:2d}\n'.format(identifier, shape[0], shape[1])

        return lines

    def plot(self, dir):
        """Plot all data

        Args
        ----
        dir : str
            directory to save all plots

        For each identifier, plot each column (keyword) of its data together in the same
        figure.

        """
        # transfer an identifier to a figure title
        def identifier_to_title(identifier):
            # identifier only contans date
            date = identifier

            # Formalize date string
            if len(date) == 6:
                # format: MMDDYY
                year  = int(date[4:6])
                month = int(date[0:2])
                day   = int(date[2:4])
            else:
                raise NotImplementedError

            date = '%02d/%02d/%02d' % (month, day, year)

            return '{}'.format(date)

        # transfer a keyword to an ax subtitle
        def key_to_subtitle(key):
            return self.LONG_KEYS[key]

        if not os.path.isdir(dir): os.makedirs(dir)

        # plot each flight
        for i, identifier in enumerate(self.identifier):
            title = identifier_to_title(identifier)
            fig = drawbox.EqualGridFigure(
                num_rows=len(self.keys), num_cols=1,
                ax_height=10, ax_width=80,
                title=title, font_size=40)
            
            logging.info('Plot \'{}\''.format(title))

            # plot each keyword
            for j, key in enumerate(self.keys):
                subtitle = key_to_subtitle(key)
                fig.subtitle(row_id=j, col_id=0, subtitle=subtitle)

                # logging.info(' -- Plot \'{}\''.format(subtitle))

                fig.lineplot(
                    row_id=j, col_id=0,
                    x_data=range(len(self.dataset[i][:, j])),
                    y_data=self.dataset[i][:, j],
                    color='C%d' % (j // 3))

            fig.save_fig(os.path.join(dir, 'png', '{}.png'.format(identifier)), close=False)
            fig.save_fig(os.path.join(dir, 'pdf', '{}.pdf'.format(identifier)), close=True)

    # phone data dict keywords to raw text file keywords
    KEYS_DICT = {
        'time' : ('Time'                , 1 ),
        'alt'  : ('Altitude'            , 2 ),
        'lat'  : ('Latitude'            , 3 ),
        'long' : ('Longitude'           , 4 ),
        'accmx': ('Accelerometer X-axis', 6 ),
        'accmy': ('Accelerometer Y-axis', 7 ),
        'accmz': ('Accelerometer Z-axis', 8 ),
        'gyrox': ('Gyroscope X-axis'    , 9 ),
        'gyroy': ('Gyroscope Y-axis'    , 10),
        'gyroz': ('Gyroscope Z-axis'    , 11),
    }

    # phone keywords to their long version
    LONG_KEYS = {
        'time' : 'Time'                ,
        'alt'  : 'Altitude'            ,
        'lat'  : 'Latitude'            ,
        'long' : 'Longitude'           ,
        'accmx': 'Accelerometer X-axis',
        'accmy': 'Accelerometer Y-axis',
        'accmz': 'Accelerometer Z-axis',
        'gyrox': 'Gyroscope X-axis'    ,
        'gyroy': 'Gyroscope Y-axis'    ,
        'gyroz': 'Gyroscope Z-axis'    ,
    }

    # time zone shift to G1000
    HOUR_SHIFT = {
        'galaxy': 11,
        'pixel' : 1 ,
    }

    def from_file(self, path):
        """Load a text file into a data dict

        Args
        ----
        path : str
            path to the text file to load

        Returns
        -------
        data_dict : dict
            a dict of keyword data

        """
        # detect phone type, 'root/phone/date/file'
        self.phone = os.path.basename(os.path.dirname(os.path.dirname(path)))

        # phone type must support time zone shift
        assert self.phone in self.HOUR_SHIFT

        logging.info('Load phone ({}) file: \'{}\''.format(self.phone, path))

        # create an empty data dict
        data_dict = {
            'time' : [],
            'alt'  : [],
            'lat'  : [],
            'long' : [],
            'accmx': [],
            'accmy': [],
            'accmz': [],
            'gyrox': [],
            'gyroy': [],
            'gyroz': [],
        }

        # load file into line list in buffer
        file = open(path, 'r')
        file_lines = file.readlines()
        file.close()

        num_cols = None

        # read each data line in order
        collecting_idx = []
        for idx, line in enumerate(file_lines):
            # an empty line should represent ending
            if len(line) == 0:
                assert idx == len(file_lines) - 1
                break

            # for new app record, convert it to old version
            line_entities = re.split(r'\s+', line[:-1])
            if len(line_entities) == 15:
                app_version = 1
                line_entities = [line_entities[0], line_entities[2]] + line_entities[4:]
            elif len(line_entities) == 13:
                app_version = 0

            if num_cols is None: num_cols = len(line_entities)

            # each row should record same number of entities
            assert num_cols == len(line_entities), "{}".format(len(line_entities))

            # save all keyword data, and track line index
            for key in data_dict:
                # collect time data
                if key == 'time':
                    time_col = line_entities[self.KEYS_DICT[key][1]]
                    hour, minute, second = time_col.split(':')
                    hour, minute, second = int(hour), int(minute), int(second)

                    # shift time zone (only old version by current)
                    if app_version == 0:
                        hour = (hour + self.HOUR_SHIFT[self.phone]) % 24
                    
                    time_val = hour * 3600 + minute * 60 + second

                    # value of time should increase
                    if len(data_dict[key]) > 0 and time_val < data_dict[key][-1]:
                        logging.warning("line {}: {} decreases to {}".format(
                                            idx, data_dict[key][-1], time_val))

                    data_dict[key].append(time_val)
                elif key == 'alt':
                    data_dict[key].append(float(line_entities[self.KEYS_DICT[key][1]]) * 3.280839895)
                else:
                    data_dict[key].append(float(line_entities[self.KEYS_DICT[key][1]]))
            collecting_idx.append(idx)

        # collecting indices should be consectutive
        assert len(collecting_idx) == max(collecting_idx) - min(collecting_idx) + 1
        for key in data_dict:
            assert len(collecting_idx) == len(data_dict[key])

        logging.info('[phone file] : {} keywords, {} datapoints'.format(len(data_dict), len(collecting_idx)))

        return data_dict

    def from_date_folder(self, dir, date):
        """Load one (must be one) text file recorded on the date

        Args
        ----
        dir : str
            root directory to one phone text file
        date : str
            date folder name to load

        Returns
        -------
        keys : list
            a list of keywords
        data : list
            a list of numpy.array data on the date

        """
        date_dir = os.path.join(dir, date)

        logging.info('Load phone date {} folder: \'{}\''.format(date, date_dir))

        # directory should exist
        assert os.path.isdir(dir)
        assert os.path.isdir(date_dir)

        # directory should only have one text file
        assert len(os.listdir(date_dir)) == 1

        keys = None
        data = []

        for filename in sorted(os.listdir(date_dir)):
            path = os.path.join(date_dir, filename)

            meta_data_dict = self.from_file(path)
            meta_keys = meta_data_dict.keys()

            # keywords should agree on all data
            if keys is None: keys = meta_keys
            assert keys == meta_keys

            # convert to list
            meta_data_list = [meta_data_dict[key] for key in keys]

            # convert to numpy.array
            # #datapoints \times #keys
            meta_data_array = np.transpose(np.array(meta_data_list))
            data.append(meta_data_array)

        # must read any keyword data
        assert not keys is None
        assert len(data) > 0

        logging.info('[phone date folder] : {} keywords, {} flights'.format(len(keys), len(data)))

        return keys, data

    def from_folder(self, dir, want=None):
        """Load all text files to the class

        Args
        ----
        dir : str
            root directory to all phone text files
        want : list
            wanting date

        """
        logging.info('Load phone folder: \'{}\''.format(dir))

        # directory should exist
        assert os.path.isdir(dir)

        self.keys = None
        self.dataset = []
        self.identifier = []

        load_list = sorted(os.listdir(dir)) if want is None else want
        for date in load_list:
            date_keys, date_data = self.from_date_folder(dir, date)

            # keywords should agree on all data
            if self.keys is None: self.keys = date_keys
            assert self.keys == date_keys

            # all flights on the same date are concatenated together
            assert len(date_data) == 1
            flight_data = date_data[0]

            # finally save data to the class, and identify it by date
            self.dataset.append(flight_data)
            self.identifier.append('{}'.format(date))

        # must read any keyword data
        assert not self.keys is None
        assert len(self.dataset) > 0

        # lock data and identifier to avoid modification
        self.lock()

class RawStratuxData(BaseData):
    """Raw Stratux Data container

    It contains all raw stratux data in numpy.array according to flight date. Since stratux
    data do not differentiate different flights on the same date, it will identify data
    according to flight number.
    It also contains a list of keywords to record relationship between flight data rows
    and keywords.
    After loading data, freeze every thing to avoid modification.

    """
    def __init__(self, entity):
        if isinstance(entity, tuple) or isinstance(entity, list):
            assert len(entity) == 3
            self.from_(entity[0], entity[1], entity[2])
        elif isinstance(entity, dict):
            assert len(entity) == 3
            self.from_(entity['dataset'], entity['keys'], entity['identifier'])
        elif os.path.isfile(entity): self.load(entity)
        elif os.path.isdir(entity): self.from_folder(entity)
        else: raise NotImplementedError

    def description(self):
        """Return a description of current class status
        
        Returns
        -------
        lines : str
            a description of containing data

        Describe number of keywords, number of flights, identifier and shape of each
        flight data.

        """
        lines = ''
        
        # description header
        lines += 'Raw Stratux Data: {} keywords, {} flights\n'.format(len(self.keys), len(self))

        # description for each flight data
        for i, data in enumerate(self.dataset):
            identifier = self.identifier[i]
            shape = data.shape

            # data shape must be #datapoints \times #keys
            assert len(shape) == 2

            lines += ' -- {:10s}: {:6d} x {:2d}\n'.format(identifier, shape[0], shape[1])

        return lines

    def plot(self, dir):
        """Plot all data

        Args
        ----
        dir : str
            directory to save all plots

        For each identifier, plot each column (keyword) of its data together in the same
        figure.

        """
        # transfer an identifier to a figure title
        def identifier_to_title(identifier):
            # identifier only contans date
            date = identifier

            # Formalize date string
            if len(date) == 6:
                # format: MMDDYY
                year  = int(date[4:6])
                month = int(date[0:2])
                day   = int(date[2:4])
            else:
                raise NotImplementedError

            date = '%02d/%02d/%02d' % (month, day, year)

            return '{}'.format(date)

        # transfer a keyword to an ax subtitle
        def key_to_subtitle(key):
            return self.LONG_KEYS[key]

        if not os.path.isdir(dir): os.makedirs(dir)

        # plot each flight
        for i, identifier in enumerate(self.identifier):
            title = identifier_to_title(identifier)
            fig = drawbox.EqualGridFigure(
                num_rows=len(self.keys), num_cols=1,
                ax_height=10, ax_width=80,
                title=title, font_size=40)
            
            logging.info('Plot \'{}\''.format(title))

            # plot each keyword
            for j, key in enumerate(self.keys):
                subtitle = key_to_subtitle(key)
                fig.subtitle(row_id=j, col_id=0, subtitle=subtitle)

                # logging.info(' -- Plot \'{}\''.format(subtitle))

                fig.lineplot(
                    row_id=j, col_id=0,
                    x_data=range(len(self.dataset[i][:, j])),
                    y_data=self.dataset[i][:, j],
                    color='C%d' % (j // 3))

            fig.save_fig(os.path.join(dir, 'png', '{}.png'.format(identifier)), close=False)
            fig.save_fig(os.path.join(dir, 'pdf', '{}.pdf'.format(identifier)), close=True)

    # phone data dict keywords to raw text file keywords
    KEYS_DICT = {
        'time'   : ('GPSTime'        , 19),
        'alt'    : ('GPSAltitudeMSL' , 11),
        'lat'    : ('GPSLatitude'    , 1 ),
        'long'   : ('GPSLongitude'   , 2 ),
        'pitch'  : ('AHRSPitch'      , 28),
        'roll'   : ('AHRSRoll'       , 29),
        'heading': ('AHRSGyroHeading', 30),
    }

    # phone keywords to their long version
    LONG_KEYS = {
        'time'   : 'Time'     ,
        'alt'    : 'Altitude' ,
        'lat'    : 'Latitude' ,
        'long'   : 'Longitude',
        'pitch'  : 'Pitch'    ,
        'roll'   : 'Roll'     ,
        'heading': 'Heading'  ,
    }

    # time zone shift to G1000
    HOUR_SHIFT = {
        'summer': -4,
        'winter': -5,
        'none'  : 0,
    }

    def from_file(self, path):
        """Load a text file into a data dict

        Args
        ----
        path : str
            path to the text file to load

        Returns
        -------
        data_dict : dict
            a dict of keyword data

        """
        logging.info('Load Stratux file: \'{}\''.format(path))

        # create an empty data dict
        data_dict = {
            'time'   : [],
            'alt'    : [],
            'lat'    : [],
            'long'   : [],
            'pitch'  : [],
            'roll'   : [],
            'heading': [],
        }

        # load file into line list in buffer
        file = open(path, 'r')
        file_lines = file.readlines()
        file.close()

        # read column name header line (1 line)
        name_header_line = file_lines[0]
        name_header_line = name_header_line.strip()
        col_names = re.split(r',\s*', name_header_line)
        col_names = [name[1:-1] for name in col_names]

        # check keyword existance
        for key in data_dict:
            assert key in self.KEYS_DICT
            assert self.KEYS_DICT[key][0] == col_names[self.KEYS_DICT[key][1]]

        num_cols = None

        # read each data line in order
        collecting_idx = []
        for idx, line in enumerate(file_lines[1:]):
            # an empty line should represent ending
            if len(line) == 0:
                assert idx == len(file_lines) - 1
                break

            # for new app record, convert it to old version
            line_entities = re.split(r'\",\"', line.strip()[1:-1])

            if num_cols is None: num_cols = len(line_entities)

            if num_cols > len(line_entities):
                logging.warning('Sudden Stopping Record at line {}'.format(idx + 2))
                break

            # each row should record same number of entities
            assert num_cols == len(line_entities), "{}: {}".format(idx, len(line_entities))

            # stratux may not really record
            if line_entities[self.KEYS_DICT['lat'][1]] == '0.0' or \
               line_entities[self.KEYS_DICT['long'][1]] == '0.0':
                continue

            # save all keyword data, and track line index
            for key in data_dict:
                # collect time data
                if key == 'time':
                    time_col = line_entities[self.KEYS_DICT[key][1]]
                    date, time = time_col.split('T')
                    hour, minute, second = time[:-1].split(':')
                    hour, minute, second = int(hour), int(minute), float(second[:-1])
                    second = int(round(second))

                    # check if summer time or not
                    year, month, day = date.split('-')
                    year, month, day = int(year), int(month), int(day)
                    if (year, month, day) == (1, 1, 1):
                        period = 'none'
                    elif constant.SUMMER[year][0] <= (month, day) and \
                         constant.SUMMER[year][1] >= (month, day):
                        period = 'summer'
                    else:
                        period = 'winter'

                    # shift time period zone
                    hour = (hour + self.HOUR_SHIFT[period]) % 24
                    
                    time_val = hour * 3600 + minute * 60 + second

                    # value of time should increase
                    if len(data_dict[key]) > 0 and time_val < data_dict[key][-1]:
                        logging.warning("line {}: {} decreases to {}".format(
                                            idx, data_dict[key][-1], time_val))

                    data_dict[key].append(time_val)
                else:
                    data_dict[key].append(float(line_entities[self.KEYS_DICT[key][1]]))
            collecting_idx.append(idx)

        # collecting indices should be consectutive
        assert len(collecting_idx) == max(collecting_idx) - min(collecting_idx) + 1
        for key in data_dict:
            assert len(collecting_idx) == len(data_dict[key])

        logging.info('[Stratux file] : {} keywords, {} datapoints'.format(len(data_dict), len(collecting_idx)))

        return data_dict

    def from_folder(self, dir):
        """Load all text files to the class

        Args
        ----
        dir : str
            root directory to all stratux text files

        """
        logging.info('Load Stratux folder: \'{}\''.format(dir))

        # directory should exist
        assert os.path.isdir(dir)

        self.keys = None
        self.dataset = []
        self.identifier = []

        for filename in sorted(os.listdir(dir)):
            path = os.path.join(dir, filename)
            
            date, time = filename.split('.')[0].split('-')
            date = date.split('_')
            date = "{}{}{}".format(date[1], date[2], date[0][2:4])
            
            meta_data_dict = self.from_file(path)
            meta_keys = meta_data_dict.keys()

            # keywords should agree on all data
            if self.keys is None: self.keys = meta_keys
            assert self.keys == meta_keys

            # convert to list
            meta_data_list = [meta_data_dict[key] for key in self.keys]

            # convert to numpy.array
            # #datapoints \times #keys
            meta_data_array = np.transpose(np.array(meta_data_list))

            # finally save data to the class, and identify it by date
            self.dataset.append(meta_data_array)
            self.identifier.append('{}'.format(date))

        # must read any keyword data
        assert not self.keys is None
        assert len(self.dataset) > 0

        # lock data and identifier to avoid modification
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