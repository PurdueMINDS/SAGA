"""Process raw data classes to regular flight data classes"""
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

# must run on Python 3
assert sys.version_info[0] == 3


# /**
#  * Raw (Flight) Data Container with Extension Data Columns
#  */
class FlightDiffRecoverData(raw.BaseData):
    """Recovery Flight Difference Reference Data container

    This is an external data container for future flight data recovery from difference.
    For instance, if you have a column of difference of step 5, and want to recover
    original column, this will provide the initial state for you to recover.

    It will provide two mode to saving recovery data

    1. 'big' (default): N elements will be stored for each identifier.

    <TODO>
    2. 'small' : Only step elements will be stored for each identifier.
    </TODO>

    """
    def __init__(self, key, step=None, mode='small'):
        # configure memory cost mode of the class
        self.step = step
        self.mode = mode
        assert self.mode == 'big' or (not self.step is None)

        self.dataset = []
        self.keys = [key]
        self.identifier = []

        # the class should only have one keyword
        assert len(self.keys) == 1

        self.lock()

    def append(self, identifier, data):
        """Append new (identifier, data) pair to the class

        Args
        ----
        identifier : str
            identifer of data to append
        data : numpy.array
            data to append

        """
        # data columns must fit keywords
        assert data.shape[1] == len(self.keys)
        assert self.mode == 'big' or (self.step == data.shape[0])

        self.unlock()

        self.dataset.append(data)
        self.identifier.append(identifier)

        self.lock()

    def recover(self, diff_idt, diff_data):
        """Interface of recovery according to different modes

        Args
        ----
        diff_idt : str
            identifier of difference data to recover
        diff_data : numpy.array:
            difference data to recover

        Returns
        -------
        org_data : numpy.array
            origin data recovered from difference data

        """
        recover_ = getattr(self, 'recover_{}'.format(self.mode))
        return recover_(diff_idt, diff_data)

    def recover_big(self, diff_idt, diff_data):
        """Recovery from difference in big mode

        Args
        ----
        diff_idt : str
            identifier of difference data to recover
        diff_data : numpy.array:
            difference data to recover

        Returns
        -------
        org_data : numpy.array
            origin data recovered from difference data

        reference : numpy.ndarray(N, 1) (from the class)
        difference: numpy.ndarray(N, 1) (args)
        origin    : numpy.ndarray(N, 1) (returns)

        origin = reference .- difference

        """
        _, ref_data = self[diff_idt]

        # recovery reference should fit difference
        assert not ref_data is None
        assert diff_data.shape == ref_data.shape

        return ref_data - diff_data

    def recover_small(self, diff_idt, diff_data):
        """Recovery from difference in small mode

        Args
        ----
        diff_idt : str
            identifier of difference data to recover
        diff_data : numpy.array:
            difference data to recover

        Returns
        -------
        org_data : numpy.array
            origin data recovered from difference data

        reference : numpy.ndarray(step, 1) (from the class)
        difference: numpy.ndarray(N, 1) (args)
        origin    : numpy.ndarray(N, 1) (returns)

        origin[N - 1 * step:N - 0 * step] = reference                         .- difference[N - 1 * step:N - 0 * step]
        origin[N - 2 * step:N - 1 * step] = origin[N - 1 * step:N - 0 * step] .- difference[N - 2 * step:N - 1 * step]
        ......

        """
        _, ref_data = self[diff_idt]

        assert not ref_data is None
        assert self.step == ref_data.shape[0]

        ans_data = copy.deepcopy(diff_data)
        begin = len(ans_data) - self.step
        end   = len(ans_data)
        while end > 0:
            if begin < 0:
                ref_data = ref_data[-begin:]
                begin = 0

            ans_data[begin:end] = ref_data - diff_data[begin:end]
            ref_data = ans_data[begin:end]

            begin -= self.step
            end   -= self.step
        return ans_data


class FlightExtensionData(raw.BaseData):
    """Flight Extension Data container

    It is initialized from raw G1000 or phone data.
    The class only supports raw data augmentation. It can compute new data columns through
    provided data data columns.
    Currently, it supports two types:

    1. Difference Augmentation
        Compute difference of a column and recover original column from difference.

    2. Trigonometrics Augmentation
        Compute trigonometrics function (sin or cos) on a column and recover original
        column from both sin and cos.

    Pay attention that for precision, difference will NOT divide by step size.

    """
    def __init__(self, entity):
        # always initialize recovery data as None
        self.recovery = None

        if isinstance(entity, tuple) or isinstance(entity, list):
            assert len(entity) == 3
            self.from_(entity[0], entity[1], entity[2])
        elif isinstance(entity, dict):
            assert len(entity) == 3
            self.from_(entity['dataset'], entity['keys'], entity['identifier'])
        elif isinstance(entity, str) and os.path.isfile(entity): self.load(entity)
        elif isinstance(entity, raw.RawG1000Data): self.from_raw(entity)
        elif isinstance(entity, raw.RawPhoneData): self.from_raw(entity)
        elif isinstance(entity, raw.RawStratuxData): self.from_raw(entity)
        else: raise NotImplementedError

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
                'recovery'  : self.recovery,
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

        self.recovery = save_dict['recovery']

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

        Describe number of dates, number of keywords, number of flights, identifier,
        number of flights on each date and shape of each flight data.

        """
        lines = ''

        # description for number of flights on each date
        num_dict = self.num_date_flights()
        if not num_dict is None:
            lines += 'Flight with Extension Data: {} dates\n'.format(len(num_dict))
            for date in num_dict: lines += ' -- {:6s}: {:2d}\n'.format(date, num_dict[date])
        else:
            logging.warning('This is constructed from Raw Phone Data, No Flight Number')

        # description header
        lines += 'Flight with Extension Data: {} keywords, {} flights\n'.format(len(self.keys), len(self))

        # description for each flight data
        for i, data in enumerate(self.dataset):
            identifier = self.identifier[i]
            shape = data.shape

            # data shape must be #datapoints \times #keys
            assert len(shape) == 2

            lines += ' -- {:10s}: {:6d} x {:2d}\n'.format(identifier, shape[0], shape[1])

        return lines

    def plot(self, dir,
             target_label=None, prediction_label=None):
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
            # initialize values of identifier entities
            date = None
            flight_num = None

            # set value according to identifier
            entities = identifier.split('_')
            if len(entities) == 1: date = entities[0]
            elif len(entities) == 2: date, flight_num = entities
            else: raise NotImplementedError

            # Formalize date string
            if not date is None:
                if len(date) == 6:
                    # format: MMDDYY
                    year  = int(date[4:6])
                    month = int(date[0:2])
                    day   = int(date[2:4])
                else:
                    raise NotImplementedError

                date = '%02d/%02d/%02d' % (month, day, year)

            # Formalize flight number
            if not flight_num is None:
                flight = 'Flight %02d' % (int(flight_num))

            if len(entities) == 1: return '{}'.format(date)
            elif len(entities) == 2: return '{}, {}'.format(date, flight)
            else: raise NotImplementedError

        # transfer a keyword to an ax subtitle
        def key_to_subtitle(key):
            return self.LONG_KEYS[key]

        if not os.path.isdir(dir): os.makedirs(dir)

        # exclude prediction from keywords
        base_keys = [key for key in self.keys if key[0] != '*']
        pred_keys = [key for key in self.keys if key[0] == '*']

        # plot each flight
        for i, identifier in enumerate(self.identifier):
            title = identifier_to_title(identifier)
            fig = drawbox.EqualGridFigure(
                num_rows=len(base_keys), num_cols=1,
                ax_height=10, ax_width=80,
                title=title, font_size=40)

            logging.info('Plot \'{}\''.format(title))

            # plot each base keyword
            for j, key in enumerate(base_keys):
                subtitle = key_to_subtitle(key)
                fig.subtitle(row_id=j, col_id=0, subtitle=subtitle,
                             x_label='Time (Second)',
                             y_label="{} ({})".format(subtitle, self.KEY_UNITS[key]))

                # logging.info(' -- Plot \'{}\''.format(subtitle))

                col_id = self.get_col_id(key)
                alpha = 0.8 if '*{}'.format(key) in pred_keys else 1
                vmin, vmax = self.KEY_LIMITS[key] if key in self.KEY_LIMITS else None, None
                fig.lineplot(
                    row_id=j, col_id=0,
                    x_data=range(len(self.dataset[i][:, col_id])),
                    y_data=self.dataset[i][:, col_id],
                    color='C%d' % (j // 3), alpha=alpha, vmin=vmin, vmax=vmax,
                    label='Target' if target_label is None else target_label)

            # plot each prediction keyword
            for j, key in enumerate(pred_keys):
                subtitle = key_to_subtitle(key[1:])

                # logging.info(' -- Prediction Plot \'{}\''.format(subtitle))

                # find ax to plot prediction
                base_j = None
                for base_idx, base_itr in enumerate(base_keys):
                    if base_itr == key[1:]:
                        base_j = base_idx
                        break

                assert not base_j is None

                col_id = self.get_col_id(key)
                alpha = 0.4 if key[1:] in base_keys else 1
                vmin, vmax = self.KEY_LIMITS[key[1:]] if key[1:] in self.KEY_LIMITS else None, None
                fig.lineplot(
                    row_id=base_j, col_id=0,
                    x_data=range(len(self.dataset[i][:, col_id])),
                    y_data=self.dataset[i][:, col_id],
                    color='C%d' % (base_j // 3 + fig.cnt[base_j][0] * 3), alpha=alpha, vmin=vmin, vmax=vmax,
                    label='Prediction' if prediction_label is None else prediction_label)

            # plot legend for axes with prediction
            fig.legend()

            fig.save_fig(os.path.join(dir, 'png', '{}.png'.format(identifier)), close=False)
            fig.save_fig(os.path.join(dir, 'pdf', '{}.pdf'.format(identifier)), close=True)

    # flight keywords to their long version
    LONG_KEYS = {
        'time'       : 'Time'                  ,
        'alt'        : 'Altitude'              ,
        'lat'        : 'Latitude'              ,
        'long'       : 'Longitude'             ,
        'spd_alt'    : 'Speed Altitude'        ,
        'spd_lat'    : 'Speed Latitude'        ,
        'spd_long'   : 'Speed Longitude'       ,
        'acc_alt'    : 'Acceleration Altitude' ,
        'acc_lat'    : 'Acceleration Latitude' ,
        'acc_long'   : 'Acceleration Longitude',
        'accmx'      : 'Accelerometer X-axis'  ,
        'accmy'      : 'Accelerometer Y-axis'  ,
        'accmz'      : 'Accelerometer Z-axis'  ,
        'gyrox'      : 'Gyroscope X-axis'      ,
        'gyroy'      : 'Gyroscope Y-axis'      ,
        'gyroz'      : 'Gyroscope Z-axis'      ,
        'pitch'      : 'Pitch'                 ,
        'roll'       : 'Roll'                  ,
        'heading'    : 'Heading'               ,
        'spd_pitch'  : 'Speed Pitch'           ,
        'spd_roll'   : 'Speed Roll'            ,
        'spd_heading': 'Speed Heading'         ,
        'sin_pitch'  : 'SIN Value Pitch'       ,
        'sin_roll'   : 'SIN Value Roll'        ,
        'sin_heading': 'SIN Value Heading'     ,
        'cos_pitch'  : 'COS Value Pitch'       ,
        'cos_roll'   : 'COS Value Roll'        ,
        'cos_heading': 'COS Value Heading'     ,
        'spd_gd'     : 'Speed Ground'          ,
        'hazard'     : 'Hazardous State'       ,
    }

    # flight plot ranges of keywords
    KEY_LIMITS = {
        'pitch'      : (-30, 30),
        'roll'       : (-60, 60),
        'heading'    : (0, 360),
        'sin_pitch'  : (-1, 1),
        'sin_roll'   : (-1, 1),
        'sin_heading': (-1, 1),
        'cos_pitch'  : (-1, 1),
        'cos_roll'   : (-1, 1),
        'cos_heading': (-1, 1),
    }

    # flight plot units of keywords
    KEY_UNITS = {
        'time'       : 'Second'            ,
        'alt'        : 'Feet'              ,
        'lat'        : 'Degree'            ,
        'long'       : 'Degree'            ,
        'spd_alt'    : 'Feet / 5 Seconds'  ,
        'spd_lat'    : 'Degree / 5 Seconds',
        'spd_long'   : 'Degree / 5 Seconds',
        'acc_alt'    : 'Feet / Second^2 '  ,
        'acc_lat'    : 'Degree / Second^2' ,
        'acc_long'   : 'Degree / Second^2' ,
        'accmx'      : 'Meter / Second^2'  ,
        'accmy'      : 'Meter / Second^2'  ,
        'accmz'      : 'Meter / Second^2'  ,
        'gyrox'      : 'Degree / Second'   ,
        'gyroy'      : 'Degree / Second'   ,
        'gyroz'      : 'Degree / Second'   ,
        'pitch'      : 'Degree'            ,
        'roll'       : 'Degree'            ,
        'heading'    : 'Degree'            ,
        'spd_pitch'  : 'Degree / 5 Seconds',
        'spd_roll'   : 'Degree / 5 Seconds',
        'spd_heading': 'Degree / 5 Seconds',
        'sin_pitch'  : None                ,
        'sin_roll'   : None                ,
        'sin_heading': None                ,
        'cos_pitch'  : None                ,
        'cos_roll'   : None                ,
        'cos_heading': None                ,
        'spd_gd'     : 'Degree / 5 Seconds',
        'hazard'     : None                ,
    }

    def from_raw(self, raw_data):
        """Initialize the class from raw data classes

        Args
        ----
        raw_data : raw.Raw*Data
            raw data class to initialize

        This is a clone initialization.

        """
        # can only construct from two raw data classes
        assert isinstance(raw_data, raw.RawG1000Data) or \
               isinstance(raw_data, raw.RawPhoneData) or \
               isinstance(raw_data, raw.RawStratuxData)

        # clone data
        self.dataset = copy.deepcopy(raw_data.dataset)
        self.keys = copy.deepcopy(raw_data.keys)
        self.identifier = copy.deepcopy(raw_data.identifier)

        self.lock()

    def get_col_id(self, key):
        """Get the column ID of specific key

        Args
        ----
        key : str
            keyword to locate

        Returns
        -------
        id : int
            column ID of specific key

        """
        for i, itr in enumerate(self.keys):
            if itr == key:
                return i
        return None

    def set_recovery(self, recovery):
        """Set recovery data by given recovery data

        Args
        ----
        recovery : dict
            a dictionary of recovery data

        """
        # can set recovery data only when the class does not have any
        assert self.recovery is None

        self.recovery = recovery

    def append_num_diff(self, key, new_key=None, step=1, pad='repeat_base'):
        """Append decimal difference of data from column key to column new_key

        Args
        ----
        key : str
            keyword of column to compute sin value
        new_key : str
            keyword of column to append computation result
        step : int
            size of index gap to compute difference
        pad : str
            method to pad when out of index

        """
        if new_key is None: new_key = 'diff{}_{}'.format(step, key)

        # locate column for specific key
        col_id = self.get_col_id(key)
        logging.info('Append decimal difference of column {}, key \'{}\' '
                     'to new key \'{}\''.format(col_id, key, new_key))

        # appending basis must exist
        assert not col_id is None

        # unlock to append
        self.unlock()

        # for difference, save additional information for recovery
        recover_data = FlightDiffRecoverData(new_key, step=step)

        # append new column data to each identifier
        for i, meta_data in enumerate(self.dataset):
            # get appending basis
            base_data = meta_data[:, col_id]
            base_data = base_data.reshape(-1, 1)

            minuend_base = base_data[step:, :]
            minuend_pad  = np.zeros(shape=(step, 1))

            # configure padding
            if pad == 'repeat_base':
                minuend_pad.fill(base_data[-1, 0])
            elif pad == 'zero':
                minuend_pad.fill(0)
            else:
                raise NotImplementedError

            # reconstruct appending basis
            minuend = np.concatenate([minuend_base, minuend_pad], axis=0)
            subtend = base_data[:, :]

            # save information for recovery
            recover_data.append(self.identifier[i], minuend[-step:])

            # recovery data should fit the order of the class
            assert len(recover_data.identifier) == i + 1
            assert len(recover_data.dataset) == i + 1

            # compute new column
            new_data = minuend - subtend

            # append to tail column of specific identifier
            self.dataset[i] = np.concatenate([meta_data, new_data], axis=1)

        # append new column keyword
        self.keys.append(new_key)

        # lock after append
        self.lock()

        # append recovery data
        if self.recovery is None: self.recovery = {new_key: recover_data}
        else: self.recovery[new_key] = recover_data

    def append_deg_diff(self, key, new_key=None, step=1, pad='repeat_base'):
        """Append decimal difference of data from column key to column new_key

        Args
        ----
        key : str
            keyword of column to compute sin value
        new_key : str
            keyword of column to append computation result
        step : int
            size of index gap to compute difference
        pad : str
            method to pad when out of index

        """
        if new_key is None: new_key = 'diff{}_{}'.format(step, key)

        # locate column for specific key
        col_id = self.get_col_id(key)
        logging.info('Append decimal difference of column {}, key \'{}\' '
                     'to new key \'{}\''.format(col_id, key, new_key))

        # appending basis must exist
        assert not col_id is None

        # unlock to append
        self.unlock()

        # for difference, save additional information for recovery
        recover_data = FlightDiffRecoverData(new_key, step=step)

        # append new column data to each identifier
        for i, meta_data in enumerate(self.dataset):
            # get appending basis
            base_data = meta_data[:, col_id]
            base_data = base_data.reshape(-1, 1)

            minuend_base = base_data[step:, :]
            minuend_pad = np.zeros(shape=(step, 1))

            # configure padding
            if pad == 'repeat_base':
                minuend_pad.fill(base_data[-1, 0])
            elif pad == 'zero':
                minuend_pad.fill(0)
            else:
                raise NotImplementedError

            # reconstruct appending basis
            minuend = np.concatenate([minuend_base, minuend_pad], axis=0)
            subtend = base_data[:, :]

            # save information for recovery
            recover_data.append(self.identifier[i], minuend[-step:])

            # recovery data should fit the order of the class
            assert len(recover_data.identifier) == i + 1
            assert len(recover_data.dataset) == i + 1

            # compute new column
            new_data = np.zeros(shape=base_data.shape)
            for j in range(len(new_data)):
                # get potential degree difference
                diff = [
                    minuend[j] - subtend[j],
                    (minuend[j] + 360) - subtend[j],
                    minuend[j] - (subtend[j] + 360),
                ]

                # select the one with smallest absolute value
                new_data[j] = diff[np.argmin(np.fabs(diff))]

            # append to tail column of specific identifier
            self.dataset[i] = np.concatenate([meta_data, new_data], axis=1)

        # append new column keyword
        self.keys.append(new_key)

        # lock after append
        self.lock()

        # append recovery data
        if self.recovery is None: self.recovery = {new_key: recover_data}
        else: self.recovery[new_key] = recover_data

    def append_deg_sin(self, key, new_key=None):
        """Append sin value of data from column key to column new_key

        Args
        ----
        key : str
            keyword of column to compute sin value
        new_key : str
            keyword of column to append computation result

        Trigonometric functions compute on radian. For degree input, it should compute by

            rad = deg * \frac{\pi}{180},
            val = sin(rad).

        """
        if new_key is None: new_key = 'sin_{}'.format(key)

        # locate column for specific key
        col_id = self.get_col_id(key)
        logging.info('Append degree sin value of column {}, key \'{}\' '
                     'to new key \'{}\''.format(col_id, key, new_key))

        # appending basis must exist
        assert not col_id is None

        # unlock to append
        self.unlock()

        # append new column data to each identifier
        for i, meta_data in enumerate(self.dataset):
            # get appending basis
            base_data = meta_data[:, col_id]
            base_data = base_data.reshape(-1, 1)

            # compute new column
            new_data = np.sin(base_data * math.pi / 180)

            # append to tail column of specific identifier
            self.dataset[i] = np.concatenate([meta_data, new_data], axis=1)

        # append new column keyword
        self.keys.append(new_key)

        # lock after append
        self.lock()

    def append_deg_cos(self, key, new_key=None):
        """Append cos value of data from column key to column new_key

        Args
        ----
        key : str
            keyword of column to compute cos value
        new_key : str
            keyword of column to append computation result

        Trigonometric functions compute on radian. For degree input, it should compute by

            rad = deg * \frac{\pi}{180},
            val = cos(rad).

        """
        if new_key is None: new_key = 'cos_{}'.format(key)

        # locate column for specific key
        col_id = self.get_col_id(key)
        logging.info('Append degree cos value of column {}, key \'{}\' '
                     'to new key \'{}\''.format(col_id, key, new_key))

        # appending basis must exist
        assert not col_id is None

        # unlock to append
        self.unlock()

        # append new column data to each identifier
        for i, meta_data in enumerate(self.dataset):
            # get appending basis
            base_data = meta_data[:, col_id]
            base_data = base_data.reshape(-1, 1)

            # compute new column
            new_data = np.cos(base_data * math.pi / 180)

            # append to tail column of specific identifier
            self.dataset[i] = np.concatenate([meta_data, new_data], axis=1)

        # append new column keyword
        self.keys.append(new_key)

        # lock after append
        self.lock()

    def append_rev_num_diff(self, key, new_key=None):
        """Append difference recovery data from column key to column new_key

        Args
        ----
        key : str
            keyword of column to compute sin value
        new_key : str
            keyword of column to append computation result

        """
        if new_key is None: new_key = 'int_{}'.format(key)

        # locate column for specific key
        col_id = self.get_col_id(key)
        logging.info('Append difference recovery of column {}, key \'{}\' '
                     'to new key \'{}\''.format(col_id, key, new_key))

        # appending basis must exist
        assert not col_id is None

        # unlock to append
        self.unlock()

        # append new column data to each identifier
        for i, meta_data in enumerate(self.dataset):
            # get appending basis
            base_data = meta_data[:, col_id]
            base_data = base_data.reshape(-1, 1)

            # compute new column
            new_data = self.recovery[key.lstrip('*')].recover(self.identifier[i], base_data)

            # append to tail column of specific identifier
            self.dataset[i] = np.concatenate([meta_data, new_data], axis=1)

        # append new column keyword
        self.keys.append(new_key)

        # lock after append
        self.lock()

    def append_rev_deg_diff(self, key, bidirect=False, new_key=None):
        """Append difference recovery data from column key to column new_key

        Args
        ----
        key : str
            keyword of column to compute sin value
        bidirect : bool
            if True, degree in [0, 360), otherwise, degree in (-180, 180]
        new_key : str
            keyword of column to append computation result

        """
        if new_key is None: new_key = 'int_{}'.format(key)

        # locate column for specific key
        col_id = self.get_col_id(key)
        logging.info('Append difference recovery of column {}, key \'{}\' '
                     'to new key \'{}\''.format(col_id, key, new_key))

        # appending basis must exist
        assert not col_id is None
        assert not self.recovery is None

        # unlock to append
        self.unlock()

        # append new column data to each identifier
        for i, meta_data in enumerate(self.dataset):
            # get appending basis
            base_data = meta_data[:, col_id]
            base_data = base_data.reshape(-1, 1)

            # compute new column
            new_data = self.recovery[key.lstrip('*')].recover(self.identifier[i], base_data)
            new_data = new_data % 360

            # if bidirect, degree over 180 will transform into negative degree
            if bidirect: new_data[new_data > 180] -= 360

            # append to tail column of specific identifier
            self.dataset[i] = np.concatenate([meta_data, new_data], axis=1)

        # append new column keyword
        self.keys.append(new_key)

        # lock after append
        self.lock()

    def append_rev_deg_trig(self, sin_key, cos_key, bidirect=False, new_key=None,
                            sin_loss=None, cos_loss=None, scale=1.0):
        """Append interpolation of arcsin and arccos values to column new_key

        Args
        ----
        sin_key : str
            keyword of column to contain sin value
        cos_key : str
            keyword of column to contain cos value
        bidirect : bool
            if True, degree in [0, 360), otherwise, degree in (-180, 180]
        new_key : str
            keyword of column to append computation result
        sin_loss : float
            loss of sin metrics
        cos_loss : float
            loss of cos metrics
        scale : float
            scale to compare sin_loss and cos_loss

        Two trigonometric functions together determine a singular angle. The degree should
        compute by

            arcsin_1, arcsin_2 = arcsin(sin) \in [0, 2\pi],
            arccos_1, arccos_2 = arccos(cos) \in [0, 2\pi],
            arcsin = argmin(|cos(arcsin_i) - cos|), i = 1, 2,
            arccos = argmin(|sin(arccos_i) - sin|), i = 1, 2,
            rad = {
                arcsin, sin_loss < cos_loss * scale
                arccos, cos_loss < sin_loss * scale
                (arcsin + arccos) / 2, o.w.
            }
            deg = rad * \frac{180}{\pi}.

        """
        if new_key is None: new_key = 'arc_{}'.format(sin_key)

        # locate column for specific key
        sin_col_id = self.get_col_id(sin_key)
        cos_col_id = self.get_col_id(cos_key)
        logging.info('Append arcsin and arccos degree value of '
                     'column {}, key \'{}\', column {}, key \'{}\' '
                     'to new key \'{}\''.format(
                        sin_col_id, sin_key, cos_col_id, cos_key, new_key))

        # appending basis must exist
        assert (not sin_col_id is None) and (not cos_col_id is None)

        # unlock to append
        self.unlock()

        # append new column data to each identifier
        for i, meta_data in enumerate(self.dataset):
            # get appending basis
            sin_base_data = meta_data[:, sin_col_id]
            cos_base_data = meta_data[:, cos_col_id]
            sin_base_data = sin_base_data.reshape(-1, 1)
            cos_base_data = cos_base_data.reshape(-1, 1)

            # limit all values in [-1, 1]
            sin_base_data[sin_base_data < -1] = -1
            sin_base_data[sin_base_data > 1]  = 1
            cos_base_data[cos_base_data < -1] = -1
            cos_base_data[cos_base_data > 1]  = 1

            # estimate arc for sin and cos separately
            arcsin_base_data = np.arcsin(sin_base_data)
            arccos_base_data = np.arccos(cos_base_data)

            # update arc for sin according to cos
            for j in range(len(arcsin_base_data)):
                # compute two potential arcsin results
                if arcsin_base_data[j] < 0:
                    arc1 = arcsin_base_data[j] + 2 * math.pi
                    arc2 = (-math.pi - arcsin_base_data[j]) + 2 * math.pi
                else:
                    arc1 = arcsin_base_data[j]
                    arc2 = math.pi - arcsin_base_data[j]

                # trigonometrics of potential arcsin results should not differeniate a lot
                assert math.fabs(math.sin(arc1) - math.sin(arc2)) < 1e-6

                # evaluate potential arcsin results by cos
                dist1 = math.fabs(math.cos(arc1) - cos_base_data[j])
                dist2 = math.fabs(math.cos(arc2) - cos_base_data[j])

                # select the nearest one
                if dist1 > dist2: arcsin_base_data[j] = arc2
                else: arcsin_base_data[j] = arc1

            # update arc for cos according to sin
            for j in range(len(arccos_base_data)):
                # compute two potential arccos results
                arc1 = arccos_base_data[j]
                arc2 = 2 * math.pi - arccos_base_data[j]

                # trigonometrics of potential arccos results should not differeniate a lot
                assert math.fabs(math.cos(arc1) - math.cos(arc2)) < 1e-6

                # evaluate potential arcsin results by cos
                dist1 = math.fabs(math.sin(arc1) - sin_base_data[j])
                dist2 = math.fabs(math.sin(arc2) - sin_base_data[j])

                # select the nearest one
                if dist1 > dist2: arccos_base_data[j] = arc2
                else: arccos_base_data[j] = arc1

            # compute new column
            if sin_loss is not None and cos_loss is not None:
                if sin_loss < cos_loss * scale:
                    new_data = arcsin_base_data
                elif cos_loss < sin_loss * scale:
                    new_data = arccos_base_data
                else:
                    new_data = (arcsin_base_data + arccos_base_data) / 2
            else:
                new_data = (arcsin_base_data + arccos_base_data) / 2
            new_data = new_data * 180 / math.pi

            # if bidirect, degree over 180 will transform into negative degree
            if bidirect: new_data[new_data > 180] -= 360

            # append to tail column of specific identifier
            self.dataset[i] = np.concatenate([meta_data, new_data], axis=1)

        # append new column keyword
        self.keys.append(new_key)

        # lock after append
        self.lock()

    def append_ground_speed(self, spd_lat_key, spd_long_key, new_key=None):
        """Append ground speed from columns spd_lat_key and spd_long_key to column new_key

        Args
        ----
        spd_lat_key : str
            keyword of latitude speed column
        spd_long_key : str
            keyword of longitude speed column
        new_key : str
            keyword of column to append computation result

        """
        if new_key is None: new_key = 'spd_gd'

        # locate column for specific key
        spd_lat_col_id  = self.get_col_id(spd_lat_key)
        spd_long_col_id = self.get_col_id(spd_long_key)
        logging.info('Append square root of square summation of '
                     'column {}, key \'{}\', column {}, key \'{}\' '
                     'to new key \'{}\''.format(
                        spd_lat_col_id, spd_lat_key, spd_long_col_id, spd_long_key, new_key))

        # appending basis must exist
        assert (not spd_lat_col_id is None) and (not spd_long_col_id is None)

        # unlock to append
        self.unlock()

        # append new column data to each identifier
        for i, meta_data in enumerate(self.dataset):
            # get appending basis
            spd_lat_base_data  = meta_data[:, spd_lat_col_id]
            spd_long_base_data = meta_data[:, spd_long_col_id]
            spd_lat_base_data  = spd_lat_base_data.reshape(-1, 1)
            spd_long_base_data = spd_long_base_data.reshape(-1, 1)

            # compute new column
            new_data = np.sqrt(spd_lat_base_data ** 2 + spd_long_base_data ** 2)

            # append to tail column of specific identifier
            self.dataset[i] = np.concatenate([meta_data, new_data], axis=1)

        # append new column keyword
        self.keys.append(new_key)

        # lock after append
        self.lock()

    def append_hazard(self, threshold, roll_key, new_key=None):
        if new_key is None: new_key = 'hazard'

        # locate column for specific key
        roll_col_id = self.get_col_id(roll_key)
        logging.info('Append hazard state of column {}, key \'{}\' '
                     'to new key \'{}\''.format(roll_col_id, roll_key, new_key))

        # appending basis must exist
        assert not roll_col_id is None

        # unlock to append
        self.unlock()

        # append new column data to each identifier
        for i, meta_data in enumerate(self.dataset):
            # get appending basis
            base_data = meta_data[:, roll_col_id]
            base_data = base_data.reshape(-1, 1)

            # compute new column
            new_data = np.zeros(base_data.shape)
            new_data[np.fabs(base_data) >= threshold] = 1

            # append to tail column of specific identifier
            self.dataset[i] = np.concatenate([meta_data, new_data], axis=1)

        # append new column keyword
        self.keys.append(new_key)

        # lock after append
        self.lock()


# /**
#  * Flight Data Container Without Parking Data
#  */
class FlightPruneData(FlightExtensionData):
    """Flight Extension Data container

    It is initialized from flight extension data class. Although it inherits
    FlightExtensionData class which supports data augmentation, it is recommend to
    augment necessary data before initialize this class.
    The class supports data prune and parking-flight detection. It can truncate data
    on flight level, keyword level and data level.

    1. Flight Level Prune
        Discard some flights or only remain specific flights.

    2. Keyword Level Prune
        Discard some keyword columns or only remain specific keyword columns.

    3. Data Level Prune
        For each piece of identifier, data pair, break them into specific number of
        flight identifier, data pair according to parking-flight detection result.

    Pay attention that unlike extension, most of whose extension function is reversible,
    prune function is irreversible. So, be careful to truncate any data on any level.

    """
    def __init__(self, entity):
        # always initialize recovery data as None
        self.recovery = None

        if isinstance(entity, tuple) or isinstance(entity, list):
            assert len(entity) == 3
            self.from_(entity[0], entity[1], entity[2])
        elif isinstance(entity, dict):
            assert len(entity) == 3
            self.from_(entity['dataset'], entity['keys'], entity['identifier'])
        elif isinstance(entity, str) and os.path.isfile(entity): self.load(entity)
        elif isinstance(entity, raw.RawG1000Data): self.from_raw(entity)
        elif isinstance(entity, raw.RawPhoneData): self.from_raw(entity)
        elif isinstance(entity, raw.RawStratuxData): self.from_raw(entity)
        elif isinstance(entity, FlightExtensionData): self.from_extension(entity)
        else: raise NotImplementedError

    def plot_parking_criterion(self, dir, num_flights=None):
        """Plot parking and flight data seperately on criterion columns

        Args
        ----
        dir : str
            directory to save all plots
        num_flights : dict
            number of flights each identifer should have

        For each identifier, plot each criterion column (keyword) of its data together in the same
        figure.

        """
        # transfer an identifier to a figure title
        def identifier_to_title(identifier):
            # initialize values of identifier entities
            date = None
            flight_num = None

            # set value according to identifier
            entities = identifier.split('_')
            if len(entities) == 1: date = entities[0]
            elif len(entities) == 2: date, flight_num = entities

            # Formalize date string
            if not date is None:
                if len(date) == 6:
                    # format: MMDDYY
                    year  = int(date[4:6])
                    month = int(date[0:2])
                    day   = int(date[2:4])
                else:
                    raise NotImplementedError

                date = '%02d/%02d/%02d' % (month, day, year)

            # Formalize flight number
            if not flight_num is None:
                flight = 'Flight %02d' % (int(flight_num))

            # Formalize expect number of flights
            if num_flights: expect = ' (Expect {} Flights)'.format(num_flights[identifier])
            else: expect = ''

            if len(entities) == 1: return '{}{}'.format(date, expect)
            elif len(entities) == 2: return '{}, {}{}'.format(date, flight, expect)

        # transfer a keyword to an ax subtitle
        def key_to_subtitle(key):
            return self.LONG_KEYS[key]

        # plot each flight
        for i, identifier in enumerate(self.identifier):
            title = identifier_to_title(identifier)
            fig = drawbox.EqualGridFigure(
                num_rows=len(self.criterion_keys), num_cols=1,
                ax_height=10, ax_width=80,
                title=title, font_size=40)

            logging.info('Criterion Plot \'{}\''.format(title))

            # plot each criterion keyword
            for j, key in enumerate(self.criterion_keys):
                subtitle = key_to_subtitle(key)
                fig.subtitle(row_id=j, col_id=0, subtitle=subtitle)

                # logging.info(' -- Plot \'{}\''.format(subtitle))

                col_id = self.get_col_id(key)

                # plot parking segments first
                # logging.info(' -- -- Parking')

                for k, (begin, end) in enumerate(self.parking_ranges[i]):
                    fig.lineplot(
                        row_id=j, col_id=0,
                        x_data=range(begin, end),
                        y_data=self.dataset[i][begin:end, col_id],
                        color='C%d' % (j // 3), alpha=1)

                # plot flight segments over parking segments
                # logging.info(' -- -- Flight')

                for k, (begin, end) in enumerate(self.flight_ranges[i]):
                    fig.lineplot(
                        row_id=j, col_id=0,
                        x_data=range(begin, end),
                        y_data=self.dataset[i][begin:end, col_id],
                        color='C%d' % (j // 3 + 1), alpha=1)

            fig.save_fig(os.path.join(dir, 'png', '{}.png'.format(identifier)), close=False)
            fig.save_fig(os.path.join(dir, 'pdf', '{}.pdf'.format(identifier)), close=True)

    def num_date_flights(self):
        """Get the number of flights on each date

        Returns
        -------
        num_dict : dict
            a dictionary of number of flights on each date

        Pay attention that only G1000 data can compute this attribute. For phone data,
        since several flights may concatenate together, we can not differeniate them
        at this class.

        """
        # sample an identifier to decide raw data source
        sample_identifier = self.identifier[0]
        sample_entites = sample_identifier.split('_')

        if len(sample_entites) == 1: raw_source = 'RawPhoneData'
        elif len(sample_entites) == 2: raw_source = 'RawG1000Data'
        else: raise NotImplementedError

        if raw_source == 'RawG1000Data':
            # for G1000 data, number of flights is number of identifier with the same date prefix
            num_dict = {}

            for i, identifier in enumerate(self.identifier):
                identifier_entites = identifier.split('_')

                # G1000 identifier only has date and flight number
                assert len(identifier_entites) == 2
                date, flight_num = identifier_entites

                if len(self.flight_ranges[i]) == 0: continue

                # counter of date prefix increases 1
                if date in num_dict: num_dict[date] += 1
                else: num_dict[date] = 1

            return num_dict
        elif raw_source == 'RawPhoneData':
            # for phone data, number of flights is unknown on each date
            return None

    def time_date_flights(self):
        """Get time ranges of flights on each date

        Returns
        -------
        num_dict : dict
            a dictionary of number of flights on each date

        Pay attention that only G1000 data can compute this attribute. For phone data,
        since several flights may concatenate together, we can not differeniate them
        at this class.

        """
        # sample an identifier to decide raw data source
        sample_identifier = self.identifier[0]
        sample_entites = sample_identifier.split('_')

        if len(sample_entites) == 1: raw_source = 'RawPhoneData'
        elif len(sample_entites) == 2: raw_source = 'RawG1000Data'
        else: raise NotImplementedError

        if raw_source == 'RawG1000Data':
            # for G1000 data, time ranges of flights are flights of identifier with the same date prefix
            num_dict = {}

            for i, identifier in enumerate(self.identifier):
                identifier_entites = identifier.split('_')

                # G1000 identifier only has date and flight number
                assert len(identifier_entites) == 2
                date, flight_num = identifier_entites

                # get time range
                time_id = self.get_col_id('time')
                time_data = self.dataset[i][:, time_id]

                if len(self.flight_ranges[i]) == 0: continue
                assert len(self.flight_ranges[i]) == 1

                begin, end = self.flight_ranges[i][0]
                begin = time_data[begin]
                end   = time_data[end - 1]

                # append time range to the date
                if date in num_dict: num_dict[date].append((begin, end))
                else: num_dict[date] = [(begin, end)]

            return num_dict
        elif raw_source == 'RawPhoneData':
            # for phone data, time ranges of flights are unknown on each date
            return None

    def from_extension(self, extension_data):
        """Initialize the class from extension data classes

        Args
        ----
        extension_data : FlightExtensionData
            extension data class to initialize

        This is a clone initialization.

        """
        # can only construct from two raw data classes
        assert isinstance(extension_data, FlightExtensionData)

        # clone data
        self.recovery = copy.deepcopy(extension_data.recovery)
        self.dataset = copy.deepcopy(extension_data.dataset)
        self.keys = copy.deepcopy(extension_data.keys)
        self.identifier = copy.deepcopy(extension_data.identifier)

        self.lock()

    def prune_keys(self, remain_keys=None, discard_keys=None):
        """Prune keywords and their data columns

        Args
        ----
        remain_keys : list
            keywords to remain
        discard_keys : list
            keywords to discard

        Only one of two arguments is necessary. If give both, only remain_keys will be
        accepted.

        """
        assert (not remain_keys is None) or (not discard_keys is None)

        if remain_keys is None:
            remain_keys = [key for key in self.keys if key not in discard_keys]

        # new keywords must be a subset of current keywords
        assert set(remain_keys) <= set(self.keys)

        # locate column for specific key
        col_ids = [self.get_col_id(key) for key in remain_keys]
        logging.info('Pruning columns ({} > {})'.format(len(self.keys), len(col_ids)))

        # appending basis must exist
        assert (not None in col_ids) and len(col_ids) > 0

        # unlock to prune
        self.unlock()

        # prune data for each identifier
        for i, meta_data in enumerate(self.dataset):
            self.dataset[i] = self.dataset[i][:, col_ids]

        # update keywords
        self.keys = remain_keys

        # lock after prune
        self.lock()

    def prune_identifier(self, remain_identifier=None, discard_identifier=None):
        """Prune keywords and their data columns

        Args
        ----
        remain_identifier : list
            identifier to remain
        discard_identifier : list
            identifier to discard

        Only one of two arguments is necessary. If give both, only remain_identifier will be
        accepted.

        """
        assert (not remain_identifier is None) or (not discard_identifier is None)

        if remain_identifier is None:
            remain_identifier = [identifier for identifier in self.identifier
                                    if identifier not in discard_identifier]

        # new keywords must be a subset of current keywords
        assert set(remain_identifier) <= set(self.identifier)

        # locate column for specific key
        logging.info('Pruning identifier ({} > {})'.format(len(self.identifier), len(remain_identifier)))

        # appending basis must exist
        assert len(remain_identifier) > 0

        # prune to get new data and identifier
        new_dataset = [meta_data for i, meta_data in enumerate(self.dataset)
                            if self.identifier[i] in remain_identifier]
        new_identifier = [identifier for identifier in self.identifier
                            if identifier in remain_identifier]

        # unlock to prune
        self.unlock()

        # prune data and identifier
        self.dataset = new_dataset
        self.identifier = new_identifier

        # lock after prune
        self.lock()

    # Parking detection constant
    ALT_MIN_G1000 = 510
    ALT_MIN_PHONE = 161.963867
    LAT_MIN       = 40.415430
    LAT_MAX       = 40.416338
    LONG_MIN      = -86.936677
    LONG_MAX      = -86.929266
    SPD_GD_LIMIT  = 0.001

    def detect_parking(self, method='time', **detect_args):
        """Interface of parking time detection according to different methods

        Args
        ----
        method : str
            method to detect parking time
        **detect_args : arguments
            arguments to specified method

        """
        detect_parking_ = getattr(self, 'detect_parking_{}'.format(method))
        return detect_parking_(**detect_args)

    def detect_parking_time(self, time_flights=None):
        """Parking time detection accoring to recording time

        Args
        ----
        time_flights : dict
            time range of flights each identifer should have

        Only use recording time information (time,) to detect parking time.
        It follows following formula to check parking time

            not (
                time_flights[idt][0] < time[idt] &
                time[idt] < time_flights[idt][1]
            )

        Pay attention that there will be several time range under the same idt
        of time_flights.

        """
        time_id   = self.get_col_id('time')
        spd_gd_id = self.get_col_id('spd_gd')

        # criterion column must exist
        assert (not time_id is None) and (not spd_gd_id is None)

        logging.info('Detect segments parking at airport according to time (ground speed)')

        # initialize parking and flight segment range buffer
        self.criterion_keys = ('alt', 'lat', 'long', 'time', 'spd_gd')
        self.parking_ranges = []
        self.flight_ranges  = []

        # compute range of each segment
        for i, meta_data in enumerate(self.dataset):
            time_data   = meta_data[:, time_id]
            spd_gd_data = meta_data[:, spd_gd_id]

            if time_flights is None:
                if np.max(spd_gd_data) < self.SPD_GD_LIMIT:
                    # if requirement is not specified and ground speed is small \
                    # the whole data will be parking
                    parking_ranges = [(0, len(time_data))]
                    flight_ranges  = []
                else:
                    # if requirement is not specified and ground speed is not small \
                    # the whole data will be flight
                    parking_ranges = []
                    flight_ranges  = [(0, len(time_data))]
            else:
                parking_ranges = []
                flight_ranges  = []

                # get fitting time range according to give requirement
                for (begin, end) in time_flights[self.identifier[i]]:
                    indices = np.where((begin <= time_data) & (time_data <= end))

                    # since time may flash back, only the first one will matter
                    indices = indices[0]
                    if len(indices) > 0:
                        assert indices[-1] + 1 - indices[0] == len(indices), \
                            "{}".format(self.identifier[i])
                        flight_ranges.append((indices[0], indices[-1] + 1))
                    else:
                        logging.warning('No matching from {} to {} for {}'.format(
                                            begin, end, self.identifier[i]))

                # generate parking time according to flight time
                flight_ranges = sorted(flight_ranges, key=lambda x: x[0])
                if flight_ranges[0][0] > 0:
                    parking_ranges.append((0, flight_ranges[0][0]))
                for j in range(len(flight_ranges) - 1):
                    parking_ranges.append((flight_ranges[j][1], flight_ranges[j + 1][0]))
                if flight_ranges[-1][1] < len(self.dataset[i]):
                    parking_ranges.append((flight_ranges[-1][1], len(self.dataset[i])))

            self.parking_ranges.append(parking_ranges)
            self.flight_ranges.append(flight_ranges)

    def detect_parking_gps1(self):
        """Parking time detection according to 1st order GPS

        Only use GPS information tuple (altitude, latitude, longitude) to detect parking time.
        It follows following formula to check parking time

            LAT_MIN < lat < LAT_MAX &
            LONG_MIN < long < LONG_MAX

        Then, divide data of each identifier into consecutive segments, e.g. indicator array

            0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19
            T T T T F F F F F F  F  F  T  T  T  T  F  T  T  T

        will be divide into following segments

            parking: [0, 4], [12, 16], [17, 20]
            flight : [4, 12], [16, 17]

        """
        alt_id  = self.get_col_id('alt')
        lat_id  = self.get_col_id('lat')
        long_id = self.get_col_id('long')

        # criterion column must exist
        assert (not alt_id is None) and (not lat_id is None) and \
               (not long_id is None)

        logging.info('Detect segments parking at airport according to 1st GPS')

        # initialize parking and flight segment range buffer
        self.criterion_keys = ('alt', 'lat', 'long')
        self.parking_ranges = []
        self.flight_ranges  = []

        # compute range of each segment
        for i, meta_data in enumerate(self.dataset):
            alt_data  = meta_data[:, alt_id]
            lat_data  = meta_data[:, lat_id]
            long_data = meta_data[:, long_id]

            # get True or False array to indicate if at airport
            indicator = (self.LAT_MIN < lat_data) & (lat_data < self.LAT_MAX) & \
                        (self.LONG_MIN < long_data) & (long_data < self.LONG_MAX)

            parking_ranges = []
            flight_ranges  = []

            # collect begin and end index of all parking and flight segments
            state = None
            for j, itr in enumerate(indicator):
                if state is None:
                    # if is the first element, initialize a new segment
                    begin = j
                    state = itr
                elif itr != state or j == len(indicator) - 1:
                    # if reach a new segment or last element, update segment
                    end = j
                    if state: parking_ranges.append((begin, end))
                    else: flight_ranges.append((begin, end))

                    # initialize a new segment
                    begin = j
                    state = itr

            self.parking_ranges.append(parking_ranges)
            self.flight_ranges.append(flight_ranges)

    def detect_parking_gps2(self):
        """Parking time detection according to 2nd order GPS

        Use GPS information tuple (altitude, latitude, longitude), speed of previous tuple,
        and ground speed to detect parking time.
        It follows following formula to check parking time

            spd_gd > SPD_GD_MIN

        Then, divide data of each identifier into consecutive segments, e.g. indicator array

            0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19
            T T T T F F F F F F  F  F  T  T  T  T  F  T  T  T

        will be divide into following segments

            parking: [0, 4], [12, 16], [17, 20]
            flight : [4, 12], [16, 17]

        <TODO>
        Allow offset
        </TODO>

        """
        spd_gd_id  = self.get_col_id('spd_gd')

        # criterion column must exist
        assert not spd_gd_id is None

        logging.info('Detect segments parking at airport according to 2nd GPS')

        # initialize parking and flight segment range buffer
        self.criterion_keys = ('alt', 'lat', 'long', 'spd_gd')
        self.parking_ranges = []
        self.flight_ranges  = []

        # compute range of each segment
        for i, meta_data in enumerate(self.dataset):
            spd_gd_data  = meta_data[:, spd_gd_id]

            # get True or False array to indicate if at airport
            indicator = spd_gd_data < self.SPD_GD_MIN

            parking_ranges = []
            flight_ranges  = []

            # collect begin and end index of all parking and flight segments
            state = None
            for j, itr in enumerate(indicator):
                if state is None:
                    # if is the first element, initialize a new segment
                    begin = j
                    state = itr
                elif itr != state or j == len(indicator) - 1:
                    # if reach a new segment or last element, update segment
                    end = j
                    if state: parking_ranges.append((begin, end))
                    else: flight_ranges.append((begin, end))

                    # initialize a new segment
                    begin = j
                    state = itr

            self.parking_ranges.append(parking_ranges)
            self.flight_ranges.append(flight_ranges)

    def is_wierd_parking(self, num_flights=None):
        """Check if detected parking and flight segments are problematic

        Args
        ----
        num_flights : dict
            number of flights each identifer should have

        It will check segments detected by detecting functions by

        1. number of flight segments >= expect flight number, for each identifier

        """
        # number of flight segments >= expect flight number, for each identifier
        if not num_flights is None:
            flag_num_flights = True
            for i in range(len(self)):
                if len(self.flight_ranges[i]) < num_flights[self.identifier[i]]:
                    logging.warning('{} have {} flight segments, but expect {}'.format(
                                        self.identifier[i], len(self.flight_ranges[i]),
                                        num_flights[self.identifier[i]]))
                    flag_num_flights = False
                if not flag_num_flights: return True

        # pass all check, no wierd parking segment
        return False

    def refine_parking(self, num_flights):
        """Refine detected parking and flight segments

        Args
        ----
        num_flights : dict
            number of flights each identifer should have

        Pay attention that detecting functions may not provide exactly want we want. For example,
        it may have more flights than expectation because thresholds are not precise. So, we
        need to refine segments to fit out requirements

        We select the longest num_flights flight segments as flights we required, and forcely
        regard all remaining segments (ignore already detected parking segments) as new parking
        segments, and update.

        """
        logging.info('Refine parking and flight segments from last detection')

        for i in range(len(self)):
            flight_ranges = self.flight_ranges[i]

            # sort flight ranges by their length
            flight_ranges = sorted(flight_ranges, key=lambda x: x[1] - x[0], reverse=True)

            # must provide at least num_flights flight ranges to select from
            assert len(flight_ranges) >= num_flights[self.identifier[i]]

            # select longest num_flights flight ranges
            flight_ranges = flight_ranges[0:num_flights[self.identifier[i]]]

            # sort selected flight ranges by their time order
            flight_ranges = sorted(flight_ranges, key=lambda x: x[0])

            # regard all remaining segments as parking segments
            parking_ranges = []
            if flight_ranges[0][0] > 0:
                parking_ranges.append((0, flight_ranges[0][0]))
            for j in range(len(flight_ranges) - 1):
                parking_ranges.append((flight_ranges[j][1], flight_ranges[j + 1][0]))
            if flight_ranges[-1][1] < len(self.dataset[i]):
                parking_ranges.append((flight_ranges[-1][1], len(self.dataset[i])))

            self.parking_ranges[i] = parking_ranges
            self.flight_ranges[i]  = flight_ranges

    def prune_parking(self):
        """Prune parking data and remain flight data for each identifier

        Break data into several individual flight data, and discard all the other data
        for each identifier.

        """
        # initialize counter of flights for each date
        cnt_of_date = {}

        # initialize new buffer of identifier and dataset
        new_identifier = []
        new_dataset = []

        for i in range(len(self)):
            # formalize identifier
            identifier_entites = self.identifier[i].split('_')

            if len(identifier_entites) == 1: date = identifier_entites[0]
            elif len(identifier_entites): date, _ = identifier_entites
            else: raise NotImplementedError

            # clean counter of flights on new date
            if not date in cnt_of_date: cnt_of_date[date] = 0

            for flight_range in self.flight_ranges[i]:
                # append new data and identifier to buffer
                new_dataset.append(self.dataset[i][flight_range[0]:flight_range[1], :])
                new_identifier.append('{}_{}'.format(date, '%02d' % cnt_of_date[date]))

                # increase counter of flights on specific date
                cnt_of_date[date] += 1

                logging.info('Divide {} to {}'.format(self.identifier[i], new_identifier[-1]))

        # unlock to prune
        self.unlock()

        # update data and identifier
        self.dataset = new_dataset
        self.identifier = new_identifier

        # lock after prune
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
