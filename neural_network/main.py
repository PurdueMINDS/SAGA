"""Console Interface

Usage:
    main.py <input-dir> <output-dir>
            [--phone <phone>]
            [--update] [--eval]
            [--freq <wt>] [--threshold <T>] [--batch-size <batch-size>]
            [--win <L> (--offset <L> | --rate <R>)]
            [--no-normal] [--fake] [--best] [--limit <N>]
            [--model <name>] [--epochs <num>] [--lr <lr>] [--print-freq <N>]
            [(--cuda --device <N>)]
            [(--trig | --diff | --direct)] [--keyword <keyword>] [--stratux <lv>]
            [--no-plot] [--preview-plot] [--try <N>]

IO Options:
    <input-dir>         Directory of input to load data
    <output-dir>        Directory of output to save generated entities
    --phone <phone>     Phone type to specify [default: pixel]

Data Options:
    --update                    Indicator to regenerate flight data from raw data
    --eval                      Evaluate processed data
    --freq <wt>                 Wavelet to discrete wave transform for frequency domain
                                If not specified, use time domain
    --threshold <T>             Roll threshold of hazardous states [default: 45]
    --batch-size <batch-size>   Batch size of data loader for training, validation and prediction
                                [default: 16]
    --win <L>                   Feature window length
    --offset <L>                Feature window offset
    --rate <R>                  Feature window offset rate
    --no-normal                 Disable dataset normalization
    --limit <N>                 Limit number of samples to use
    --bptt <N>                  Back propagate through time

Training Options:
    --best              After training, only remain the one with best validation loss
    --fake              Do fake training, in other words, let target to be prediction
    --model <name>      Name of the model to train [default: FC3]
    --epochs <num>      Number of epochs to train [default: 1]
    --lr <lr>           Initial learning rate [default: 0.01]
    --print-freq <N>    Frequency to print loss [default: 1]
    --cuda              Enable GPU
    --device <N>        Specify GPU device ID [default: 0]

Prediction Options:
    --trig                  Use trigonometrics as midterm prediction
    --diff                  Use difference as midterm prediction
    --direct                Predict directly
    --keyword <keyword>     Keyword to predict [default: heading]
    --stratux <lv>          Specify stratux input level

Other Options:
    --no-plot       Forbid drawing plots
    --preview-plot  Preview raw data plots on specific keywords (Code Specified)
    --try <N>       Try several times to get the best result [default: 1]

"""
import docopt
import logging
import copy

import sys
import os
import math

import numpy as np

import torch
import torch.nn as nn

import drawbox
import raw
import flight
import frame
import batch
import nnet
import constant

from evaluate import evaluate_label_diff, evaluate_abs_diff

# must run on Python 3
assert sys.version_info[0] == 3


# /**
#  * Data Processing
#  */
def data_meta(input_dir, output_dir, phone_type, args=None):
    """Generate meta data

    Args
    ----
    input_dir : str
        root directory to load data
    output_dir : str
        root directory to save data
    phone_type : str
        phone type as input
    args : dict
        global arguments

    Returns
    -------
    ext_phone_data : flight.FlightExtensionData
        extended flight data of the phone as input
    ext_g1000_data : flight.FlightExtensionData
        extended flight data of the g1000 as target

    It will load input and target data from raw files, and extend all potential columns for
    future use.

    """
    def g1000_meta():
        # load raw data for g1000
        raw_g1000_data = raw.RawG1000Data('{}/g1000'.format(input_dir))
        ext_g1000_data = flight.FlightExtensionData(raw_g1000_data)

        # extend GPS speed for g1000
        ext_g1000_data.append_num_diff(key='alt' , new_key='spd_alt' , step=5, pad='repeat_base')
        ext_g1000_data.append_num_diff(key='lat' , new_key='spd_lat' , step=5, pad='repeat_base')
        ext_g1000_data.append_num_diff(key='long', new_key='spd_long', step=5, pad='repeat_base')
        ext_g1000_data.append_ground_speed(spd_lat_key='spd_lat', spd_long_key='spd_long', new_key='spd_gd')

        # extend angle speed for g1000
        ext_g1000_data.append_num_diff(key='pitch'  , new_key='spd_pitch'  , step=5, pad='repeat_base')
        ext_g1000_data.append_num_diff(key='roll'   , new_key='spd_roll'   , step=5, pad='repeat_base')
        ext_g1000_data.append_deg_diff(key='heading', new_key='spd_heading', step=5, pad='repeat_base')

        # extend sin angle value for g1000
        ext_g1000_data.append_deg_sin(key='pitch'  , new_key='sin_pitch'  )
        ext_g1000_data.append_deg_sin(key='roll'   , new_key='sin_roll'   )
        ext_g1000_data.append_deg_sin(key='heading', new_key='sin_heading')

        # extend cos angle value for g1000
        ext_g1000_data.append_deg_cos(key='pitch'  , new_key='cos_pitch'  )
        ext_g1000_data.append_deg_cos(key='roll'   , new_key='cos_roll'   )
        ext_g1000_data.append_deg_cos(key='heading', new_key='cos_heading')

        # save extension data for g1000
        ext_g1000_data.save('{}/g1000.extension'.format(output_dir))
        return ext_g1000_data

    def phone_meta():
        # load raw data for phone
        if phone_type == 'stratux':
            raw_phone_data = raw.RawStratuxData('{}/stratux'.format(input_dir))
        else:
            raw_phone_data = raw.RawPhoneData('{}/{}'.format(input_dir, phone_type))
        ext_phone_data = flight.FlightExtensionData(raw_phone_data)

        # extend GPS speed for phone
        ext_phone_data.append_num_diff(key='alt' , new_key='spd_alt' , step=5, pad='repeat_base')
        ext_phone_data.append_num_diff(key='lat' , new_key='spd_lat' , step=5, pad='repeat_base')
        ext_phone_data.append_num_diff(key='long', new_key='spd_long', step=5, pad='repeat_base')
        ext_phone_data.append_ground_speed(spd_lat_key='spd_lat', spd_long_key='spd_long', new_key='spd_gd')

        # extend GPS acceleration for phone
        ext_phone_data.append_num_diff(key='spd_alt' , new_key='acc_alt' , step=1, pad='repeat_base')
        ext_phone_data.append_num_diff(key='spd_lat' , new_key='acc_lat' , step=1, pad='repeat_base')
        ext_phone_data.append_num_diff(key='spd_long', new_key='acc_long', step=1, pad='repeat_base')

        # save extension data for phone
        ext_phone_data.save('{}/{}.extension'.format(output_dir, phone_type))
        return ext_phone_data

    ext_phone_data = phone_meta()
    ext_g1000_data = g1000_meta()

    return (ext_phone_data, ext_g1000_data)

def data_flight(input_dir, output_dir, phone_type, args=None):
    """Generate flight data

    Args
    ----
    input_dir : str
        root directory to load data
    output_dir : str
        root directory to save data
    phone_type : str
        phone type as input
    args : dict
        global arguments

    Returns
    -------
    seq_pair_data : frame.FlightSequencePairData
        sequence pair data after all data processing

    It will load extended data, align and truncate data to remain only pure and necessary
    flight data, and normalize data for neural network.

    """
    # load extension data
    ext_g1000_data = flight.FlightExtensionData('{}/g1000.extension'.format(input_dir))
    ext_phone_data = flight.FlightExtensionData('{}/{}.extension'.format(input_dir, phone_type))

    # generate prune data
    prn_g1000_data = flight.FlightPruneData(ext_g1000_data)
    prn_phone_data = flight.FlightPruneData(ext_phone_data)

    # only focus on identifier intersection between g1000 and phone
    g1000_date = set([itr.split('_')[0] for itr in prn_g1000_data.identifier])
    phone_date = set(prn_phone_data.identifier)
    share_date = g1000_date & phone_date
    union_date = g1000_date | phone_date
    share_date, union_date = list(sorted(share_date)), list(sorted(union_date))

    for date in union_date:
        if date in share_date:
            logging.info("Detect Date - \033[32;1m{}\033[0m".format(date))
        elif date in g1000_date:
            logging.warning("Detect Date - \033[31;1m{}\033[0m (G1000)".format(date))
        elif date in phone_date:
            logging.warning("Detect Date - \033[31;1m{}\033[0m ({})".format(date, phone_type))
        else:
            raise NotImplementedError

    # discard data not in the intersection
    g1000_discard = [itr for itr in prn_g1000_data.identifier if itr.split('_')[0] not in share_date]
    prn_g1000_data.prune_identifier(discard_identifier=g1000_discard)
    prn_phone_data.prune_identifier(remain_identifier=share_date)

    # plot preview
    if args['--preview-plot']:
        prev_g1000_data = prn_g1000_data.clone()
        prev_phone_data = prn_phone_data.clone()
        prev_g1000_data.prune_keys(remain_keys=['alt', 'lat', 'long', 'time'])
        prev_phone_data.prune_keys(remain_keys=['alt', 'lat', 'long', 'time'])
        prev_g1000_data.plot('{}/preview/g1000'.format(output_dir))
        prev_phone_data.plot('{}/preview/{}'.format(output_dir, phone_type))

    # detect pure flight data for g1000 according to given requirement (no)
    prn_g1000_data.prune_identifier(discard_identifier=constant.HIZARD_FLIGHTS)
    prn_g1000_data.detect_parking(method='time')

    # detect pure flight data for phone according to given requirement
    phone_requirment = prn_g1000_data.time_date_flights()
    prn_phone_data.prune_identifier(remain_identifier=phone_requirment.keys())
    prn_phone_data.detect_parking(method='time', time_flights=phone_requirment)

    # plot parking criterion
    if not args['--no-plot']:
        prn_g1000_data.plot_parking_criterion('{}/park/g1000'.format(output_dir))
        prn_phone_data.plot_parking_criterion('{}/park/{}'.format(output_dir, phone_type))

    # prune parking data for both phone and g1000
    prn_g1000_data.prune_parking()
    prn_phone_data.prune_parking()

    # check if there are missing record from phone records
    g1000_idt = set(prn_g1000_data.identifier)
    phone_idt = set(prn_phone_data.identifier)
    share_idt = g1000_idt & phone_idt
    union_idt = g1000_idt | phone_idt
    share_idt, union_idt = list(sorted(share_idt)), list(sorted(union_idt))

    for idt in union_idt:
        if idt in share_idt:
            logging.info("Valid Record: \033[32;1m{}\033[0m".format(idt))
        elif idt in g1000_idt:
            logging.warning("Redundant Record: \033[31;1m{}\033[0m (G1000)".format(idt))
        elif idt in phone_idt:
            logging.warning("Redundant Record: \033[31;1m{}\033[0m ({})".format(idt, phone_type))
        else:
            raise NotImplementedError

    # It is possible phone data record less flights than g1000 on the same date
    # (e.g. not enough battery)
    prn_g1000_data.prune_identifier(remain_identifier=prn_phone_data.identifier)

    # align prune data
    seq_pair_data = frame.FlightSequencePairData(entity_input=prn_phone_data, entity_target=prn_g1000_data)
    seq_pair_data.align_and_interpolate(match_keys=('alt', 'lat', 'long'))
    seq_pair_data.distribute()

    # plot alignment criterion
    if not args['--no-plot']:
        seq_pair_data.plot_match_criterion('{}/wrap/{}_g1000'.format(output_dir, phone_type))

    # save sequence data
    seq_pair_data.save('{}/{}_g1000.sequence'.format(output_dir, phone_type))

    return seq_pair_data

def data_batch(input_dir, output_dir, phone_type,
               target_key, rnn=False, batch_size=16, shuffle=False,
               select_rate=(0.0, 1.0, 'large'),
               window_config=None,
               set_normal=None,
               return_len_dict=False, args=None):
    """Generate batch loader

    Args
    ----
    input_dir : str
        root directory to load data
    output_dir : str
        root directory to save data
    phone_type : str
        phone type as input
    target_key : str
        target keyword
    batch_size : int
        batch size
    shuffle : bool
        if should shuffle batch loader
    select_rate : tuple
        proportion to select from original data
        large mode will extend select range on both head and tail
        small mode will truncate select range on both head and tail
    window_config : dict
        configuration of frame window
    set_normal : None or frame.BasePairData.Normalization
        normalize with given argument or return new normalization
    return_len_dict : bool
        if return length dict for future conversion back to sequence
    args : dict
        global arguments

    Returns
    -------
    batch_loader : batch.FramePairDataLoader
        batch loader
    normal : frame.BasePairData.Normalization
        normalization parameters
    len_dict : dict
        dict of length of each sequence data

    It will load sequence pair data, and convert into batches.
    It will discard useless keywords, and can truncate flights to generate different data loader.

    """
    # load sequence data
    seq_pair_data = frame.FlightSequencePairData('{}/{}_g1000.sequence'.format(input_dir, phone_type))

    if args['--limit'] is not None:
        seq_pair_data.prune_identifier(seq_pair_data.identifier[:args['--limit']])
    else:
        pass

    # extend hazardous state for g1000
    # It must locate after alignment and interpolation
    if args['--keyword'] == 'hazard':
        meta_target = seq_pair_data.meta_target
        meta_target.append_hazard(threshold=args['--threshold'], roll_key='roll')
        seq_pair_data.update_target(meta_target)

    # remain only necessary keywords
    if args['--stratux'] is None:
        seq_pair_data.prune_keys(
            input_remain_keys=constant.INPUT_KEYWORDS,
            target_remain_keys=(target_key,))
    else:
        lv = args['--stratux']
        logging.info("Stratux Level {} Batch".format(lv))
        if args['--keyword'] == 'hazard':
            input_key = 'roll'
        else:
            input_key = target_key.split('_')[1]
        if lv == 0: input_remain_keys = (input_key,)
        elif lv == 1: input_remain_keys = ('alt', 'lat', 'long', input_key)
        elif lv == 2: input_remain_keys = ('alt', 'lat', 'long', 'pitch', 'roll', 'heading')
        else: raise NotImplementedError
        seq_pair_data.prune_keys(
            input_remain_keys=input_remain_keys,
            target_remain_keys=(target_key,))

    # remain only selective range of flights
    num_flights = len(seq_pair_data)
    begin, end  = select_rate[0:2]
    if select_rate[2] == 'large':
        begin = int(math.floor(num_flights * begin))
        end   = int(math.ceil(num_flights * end))
    elif select_rate[2] == 'small':
        begin = int(math.ceil(num_flights * begin))
        end   = int(math.floor(num_flights * end))
    else:
        raise NotImplementedError
    if begin == end:
        if end == num_flights:
            begin -= 1
        else:
            end += 1

    seq_pair_data.prune_identifier(
        remain_identifier=seq_pair_data.identifier[begin:end])

    # normalize sequence data on time domain
    if not args['--freq'] and not args['--no-normal']:
        if set_normal: seq_pair_data.normalize(set_normal)
        else: seq_pair_data.normalize()
        normal = seq_pair_data.normal
    else:
        normal = None

    # divide sequence data into frames
    frame_pair_data = frame.FlightFramePairData(
        seq_pair_data,
        input_win_len =window_config['input'] ['length'],
        target_win_len=window_config['target']['length'],
        input_win_offset =window_config['input'] ['offset_length'],
        target_win_offset=window_config['target']['offset_length'],
        input_win_offset_rate =window_config['input'] ['offset_rate'],
        target_win_offset_rate=window_config['target']['offset_rate'],
        input_pad =window_config['input'] ['padding'],
        target_pad=window_config['target']['padding'])

    # transform to frequency domain and normalize
    if args['--freq']:
        if args['--freq'] == 'haar':
            frame_pair_data = frame.FlightFramePairData.time_to_haar(
                frame_pair_data, concat=True)
            if not args['--no-normal']:
                if set_normal: frame_pair_data.normalize(set_normal)
                else: frame_pair_data.normalize()
                normal = frame_pair_data.normal
            else:
                normal = None
        else:
            raise NotImplementedError

    # generate batch loader
    if rnn:
        batch_loader = batch.TimestepPairDataLoader(frame_pair_data, timestep=batch_size)
    else:
        batch_loader = batch.FramePairDataLoader(
            frame_pair_data, batch_size=batch_size, shuffle=shuffle, drop_last=False)

    if not return_len_dict: return batch_loader, normal
    else: return batch_loader, normal, seq_pair_data.length_dict()

def evaluate(flight_input, flight_target):
    """Evaluate between G1000 and phone GPS data

    flight_input : flight.FlightPruneData
        phone data
    flight_target : flight.FlightPruneData
        g1000 data

    """
    STRATUX_LIST = (
        '020918_00', '020918_01', '021318_00', '021318_01', '021318_02',
        '021418_00', '022718_00', '030318_00', '030318_01', '030318_02',
        '030318_03', '030418_00', '030418_01', '032618_00', '040718_00',
        '040718_01', '040718_02', '041018_00', '042018_00', '042018_01',
        '042018_02', '042718_00', '042718_01', '042718_02',
    )

    flight_std = flight_target.clone()
    flight_obv = flight_input.clone()
    # flight_std.prune_identifier(remain_identifier=STRATUX_LIST)
    # flight_obv.prune_identifier(remain_identifier=STRATUX_LIST)

    evaluate_abs_diff(
        standard=flight_std, std_key='alt',
        observe=flight_obv , obv_key='alt')
    evaluate_abs_diff(
        standard=flight_std, std_key='lat',
        observe=flight_obv , obv_key='lat')
    evaluate_abs_diff(
        standard=flight_std, std_key='long',
        observe=flight_obv , obv_key='long')


# /**
#  * Model Training
#  */
def keyword_model(key, model, input_dir, fake=False, args=None):
    """Train a model for given keyword

    Args
    ----
    key : str
        target keyword to train
    model : str
        model name to train
    input_dir : str
        root directory to load data
    fake : bool
        fake prediction
    args:
        global arguments

    Returns
    -------
    monitor : nnet.NeuralNetworkManager
        manager of the training model
    normal : frame.BasePairData.Normalization
        normalization parameters of training data

    """
    logging.info('Train for \'{}\' from {} to g1000'.format(key, args['--phone']))
    logging.info(' -- Batch size: {}'.format(args['--batch-size']))

    # generate data loader
    if model in nnet.rnn_list:
        train_loader, normal = data_batch(
            input_dir, args['<output-dir>'],
            args['--phone'], target_key=key,
            batch_size=args['--batch-size'], rnn=True,
            window_config=WINDOW_CONFIG,
            select_rate=(0.0, 0.6, 'large'), args=args)
        valid_loader, normal = data_batch(
            input_dir, args['<output-dir>'],
            args['--phone'], target_key=key,
            batch_size=args['--batch-size'], rnn=True,
            window_config=WINDOW_CONFIG, set_normal=normal,
            select_rate=(0.6, 0.8, 'small'), args=args)
    else:
        train_loader, normal = data_batch(
            input_dir, args['<output-dir>'],
            args['--phone'], target_key=key,
            batch_size=args['--batch-size'], shuffle=True,
            window_config=WINDOW_CONFIG,
            select_rate=(0.0, 0.6, 'large'), args=args)
        valid_loader, normal = data_batch(
            input_dir, args['<output-dir>'],
            args['--phone'], target_key=key,
            batch_size=args['--batch-size'], shuffle=False,
            window_config=WINDOW_CONFIG, set_normal=normal,
            select_rate=(0.6, 0.8, 'small'), args=args)

    if key == 'hazard':
        for i, itr in enumerate(train_loader.target_keys):
            if itr == 'hazard':
                tid = i
                break
        total = 0
        num_0 = 0
        num_1 = 0
        for data in train_loader.target_set:
            tdata = data[:, :, tid]
            _total = tdata.size
            _num_1 = tdata.sum()
            _num_0 = _total - _num_1
            total += _total
            num_1 += _num_1
            num_0 += _num_0

    # initialize training stage
    monitor = nnet.NeuralNetworkManager()
    win_in = WINDOW_CONFIG['input']['length']
    win_out = WINDOW_CONFIG['target']['length']
    if args['--stratux'] is None:
        num_feats = 15
    else:
        lv = args['--stratux']
        if lv == 0: num_feats = 1
        elif lv == 1: num_feats = 4
        elif lv == 2: num_feats = 6
        else: raise NotImplementedError
    if model in nnet.rnn_list:
        bptt = args['--batch-size']
        bsz = 1
        monitor.register_shape((-1, bsz, win_in * num_feats), (-1, bsz, win_out * 1),
            target_class=torch.LongTensor if key == 'hazard' else torch.Tensor)
    else:
        bsz = args['--batch-size']
        monitor.register_shape((-1, win_in * num_feats), (-1, win_out * 1),
            target_class=torch.LongTensor if key == 'hazard' else torch.Tensor)
    monitor.register_loader(train_loader, valid_loader)

    if fake: return monitor, normal

    # register training settings
    architect = nnet.__dict__[model](
        num_in=win_in * num_feats,
        num_out=win_out * 2 if key == 'hazard' else win_out * 1,
        classify=(key == 'hazard'))
    if key == 'hazard':
        criterion = torch.nn.CrossEntropyLoss(weight=torch.Tensor([total / num_0, total / num_1]))
    else:
        criterion = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(architect.parameters(), lr=args['--lr'])

    monitor.register_model(architect, criterion, optimizer, rnn=model in nnet.rnn_list)
    monitor.register_config(
        cuda=args['--cuda'], device=args['--device'],
        print_freq=args['--print-freq'])

    # train the model
    monitor.fit(num_epochs=args['--epochs'], best=args['--best'])

    return monitor, normal


# /**
#  * Model Prediction
#  */
def keyword_prediction(key, input_dir, monitor, normal, fake=False, args=None):
    """Predict batches for given keyword

    Args
    ----
    key : str
        target keyword to predict
    input_dir : str
        root directory to load data
    monitor : nnet.NeuralNetworkManager
        neural network manager for prediction
    normal : frame.BasePairData.Normalization
        normalization settings

    Returns
    -------
    predict_target : flight.FlightPruneData
        flight data containing the prediction column
    loss_avg : float
        average loss of all predictions

    """
    logging.info('Predict for \'{}\' from {} to g1000'.format(key, args['--phone']))
    logging.info(' -- Batch size: {}'.format(args['--batch-size']))

    # generate data loader and predict by the model
    if monitor.rnn:
        predict_loader, normal, len_dict = data_batch(
            input_dir, args['<output-dir>'],
            args['--phone'], target_key=key,
            batch_size=args['--batch-size'], rnn=True,
            window_config=WINDOW_CONFIG, set_normal=normal,
            select_rate=(0.8, 1.0, 'large'),
            return_len_dict=True, args=args)
        predict_loader, loss_avg = monitor.predict(predict_loader, '*{}'.format(key), (-1, WINDOW_CONFIG['target']['length'], 1), fake=fake)
    else:
        predict_loader, normal, len_dict = data_batch(
            input_dir, args['<output-dir>'],
            args['--phone'], target_key=key,
            batch_size=args['--batch-size'], shuffle=False,
            window_config=WINDOW_CONFIG, set_normal=normal,
            select_rate=(0.8, 1.0, 'large'),
            return_len_dict=True, args=args)
        predict_loader, loss_avg = monitor.predict(predict_loader, '*{}'.format(key), (-1, WINDOW_CONFIG['target']['length'], 1), fake=fake)

    # extract flight data
    predict_frame = predict_loader.to_frame_pair()
    if args['--freq']:
        if args['--freq'] == 'haar':
            predict_frame = frame.FlightFramePairData.haar_to_time(
                predict_frame, concat=True)
            if not args['--no-normal']: predict_frame.denormalize(normal)
        else:
            raise NotImplementedError

    predict_seq = predict_frame.to_seq_pair(len_dict)
    if not args['--freq'] and not args['--no-normal']: predict_seq.denormalize(normal)

    predict_target = predict_seq.meta_target

    return predict_target, loss_avg

def concat_prediction(prediction1, prediction2):
    """Concatenate prediction data

    Args
    ----
    prediction1 : flight.FlightPruneData
    prediction2 : flight.FlightPruneData

    Returns
    -------
    prediction3 : flight.FlightPruneData

    """
    # link data
    dataset1    = prediction1.dataset
    keys1       = prediction1.keys
    identifier1 = prediction1.identifier

    dataset2    = prediction2.dataset
    keys2       = prediction2.keys
    identifier2 = prediction2.identifier

    # predictions should match each other
    assert identifier1 == identifier2

    # keywords should not overlap
    assert set(keys1) & set(keys2) == set()

    # concatenate data
    dataset    = []
    identifier = []
    for i in range(len(dataset1)):
        # identifier should match each other
        assert identifier1[i] == identifier2[i]

        dataset.append(np.concatenate([dataset1[i], dataset2[i]], axis=1))
        identifier.append(identifier1[i])

    keys = keys1 + keys2

    return flight.FlightPruneData((dataset, keys, identifier))


# /**
#  * Real Process
#  */
def trigonometrics(key, model, standard_data, input_dir, output_dir, fake=False, args=None):
    """Trigonometrics Prediction

    Args
    ----
    key : str
        target keyword to predict
    model : str
        model name to train
    standard_data : frame.FlightSequencePairData
        standard data to compare with prediction
    input_dir : str
        root directory to load data
    output_dir : str
        root directory to save data
    fake : bool
        fake prediction
    args : dict
        global arguments

    Returns
    -------
    predict : flight.FlightPruneData
        flight data containing the prediction column and the target (standard) column

    This process will use trigonometrics prediction as neural network outputs.
    That's to say, one model to predict sin, the other model to predict cos, then use sin
    and cos to recover angle.

    """
    # only support predict pitch or roll or heading
    assert key in ('pitch', 'roll', 'heading')

    # train the models
    sin_monitor, sin_normal = keyword_model(key='sin_{}'.format(key), model=model, input_dir=input_dir, fake=fake, args=args)
    cos_monitor, cos_normal = keyword_model(key='cos_{}'.format(key), model=model, input_dir=input_dir, fake=fake, args=args)

    # predict by the models
    sin_prediction, sin_loss = keyword_prediction(
        key='sin_{}'.format(key), input_dir=input_dir, monitor=sin_monitor, normal=sin_normal, fake=fake, args=args)
    cos_prediction, cos_loss = keyword_prediction(
        key='cos_{}'.format(key), input_dir=input_dir, monitor=cos_monitor, normal=cos_normal, fake=fake, args=args)
    prediction = concat_prediction(sin_prediction, cos_prediction)

    # fetch standard target data column
    std_prediction = standard_data.meta_target.clone()
    std_prediction.prune_identifier(remain_identifier=prediction.identifier)
    std_prediction.prune_keys(remain_keys=['{}'.format(key)])
    prediction = concat_prediction(prediction, std_prediction)

    # convert from trigonometrics prediction to final prediction
    prediction.append_rev_deg_trig(
        sin_key='*sin_{}'.format(key), cos_key='*cos_{}'.format(key),
        new_key='*{}'.format(key),
        bidirect=False if key == 'heading' else True,
        sin_loss=sin_loss, cos_loss=cos_loss)

    # only remain final prediction
    prediction.prune_keys(remain_keys=['*{}'.format(key), key])

    logging.info('sin loss: {:.3f}, cos loss: {:.3f}'.format(sin_loss, cos_loss))

    # evaluate
    logging.info('\033[31;1mEvaluation Loss: {:.6f}\033[0m'.format(min(sin_loss, cos_loss)))

    sin_monitor.cpu()
    cos_monitor.cpu()

    return {'sin': sin_monitor, 'cos': cos_monitor}, \
           {'sin': sin_normal, 'cos': cos_normal}, \
           min(sin_loss, cos_loss), prediction

def difference(key, model, standard_data, input_dir, output_dir, fake=False, args=None):
    """Difference Prediction

    Args
    ----
    key : str
        target keyword to predict
    model : str
        model name to train
    standard_data : frame.FlightSequencePairData
        standard data to compare with prediction
    input_dir : str
        root directory to load data
    output_dir : str
        root directory to save data
    fake : bool
        fake prediction
    args : dict
        global arguments

    Returns
    -------
    predict : flight.FlightPruneData
        flight data containing the prediction column and the target (standard) column

    This process will use trigonometrics prediction as neural network outputs.
    That's to say, one model to predict sin, the other model to predict cos, then use sin
    and cos to recover angle.

    """
    # only support predict pitch or roll or heading
    assert key in ('pitch', 'roll', 'heading')

    # train the models
    spd_monitor, spd_normal = keyword_model(key='spd_{}'.format(key), model=model, input_dir=input_dir, fake=fake, args=args)

    # predict by the models
    spd_prediction, spd_loss = keyword_prediction(
        key='spd_{}'.format(key), input_dir=input_dir, monitor=spd_monitor, normal=spd_normal, fake=fake, args=args)
    prediction = spd_prediction.clone()

    # fetch standard target data column
    std_prediction = standard_data.meta_target.clone()
    std_prediction.prune_identifier(remain_identifier=prediction.identifier)
    std_prediction.prune_keys(remain_keys=['{}'.format(key)])
    prediction = concat_prediction(prediction, std_prediction)

    # convert from difference prediction to final prediction
    prediction.set_recovery(standard_data.meta_target.recovery)
    prediction.append_rev_deg_diff(
        key='*spd_{}'.format(key),
        new_key='*{}'.format(key),
        bidirect=False if key == 'heading' else True)

    # only remain final prediction
    prediction.prune_keys(remain_keys=[
        '*spd_{}'.format(key), 'spd_{}'.format(key),
        '*{}'.format(key), key])

    logging.info('spd loss: {:.3f}'.format(spd_loss))

    # evaluate
    logging.info('\033[31;1mEvaluation Loss: {:.6f}\033[0m'.format(spd_loss))

    spd_monitor.cpu()

    return {'spd': spd_monitor}, \
           {'spd': spd_normal}, \
           spd_loss, prediction

def hazardous(key, model, input_dir, output_dir, fake=False, args=None):
    """Hazardous State Prediction

    Args
    ----
    key : str
        target keyword to predict
    model : str
        model name to train
    input_dir : str
        root directory to load data
    output_dir : str
        root directory to save data
    fake : bool
        fake prediction
    args : dict
        global arguments

    Returns
    -------
    predict : flight.FlightPruneData
        flight data containing the prediction column and the target (standard) column

    This process will use trigonometrics prediction as neural network outputs.
    That's to say, one model to predict sin, the other model to predict cos, then use sin
    and cos to recover angle.

    """
    # only support predict hazardous state
    assert key == 'hazard'

    # train the models
    hzd_monitor, hzd_normal = keyword_model(key='hazard', model=model, input_dir=input_dir, fake=fake, args=args)

    # predict by the models
    hzd_prediction, hzd_loss = keyword_prediction(
        key='hazard', input_dir=input_dir, monitor=hzd_monitor, normal=hzd_normal, fake=fake, args=args)
    prediction = hzd_prediction.clone()

    # convert from probability prediction to final label
    # prediction.replace_hazard_to_label(key='hazard'.format(key))

    # only remain final prediction
    prediction.prune_keys(remain_keys=['*{}'.format(key), key])

    logging.info('hazard loss: {:.3f}'.format(hzd_loss))

    # evaluate
    logging.info('\033[31;1mEvaluation Loss: {:.6f}\033[0m'.format(hzd_loss))

    hzd_monitor.cpu()

    return {'hazard': hzd_monitor}, \
           {'hazard': hzd_normal}, \
           hzd_loss, prediction


# /**
#  * Console Interface
#  */
def parse_args():
    from schema import Schema, Use, And, Or

    args = docopt.docopt(__doc__, version='SAGA Project Ver 4.2')

    requirements = {
        '--phone'     : And(Use(str), lambda x: x in ('galaxy', 'pixel', 'stratux'),
                            error='Phone type only support \'galaxy\', \'pixel\' and \'stratux\''),
        '--threshold' : And(Use(int), lambda x: x > 0,
                            error='Roll threshold should be integer > 0'),
        '--freq'      : Or(None,
                           And(Use(str), lambda x: x in ('haar'),
                                error='Wavelet only support \'haar\'')),
        '--batch-size': And(Use(int), lambda x: x > 0,
                            error='Batch size should be integer > 0'),
        '--win'       : Or(None, And(Use(int), lambda x: x > 0),
                            error='Feature window length should be integer > 0'),
        '--offset'    : Or(None, And(Use(int), lambda x: x > 0),
                           error='Feature window offset should be integer > 0'),
        '--rate'      : Or(None, And(Use(float), lambda x: (0 < x) & (x <= 1)),
                           error='Feature window offset rate should be float in (0, 1]'),
        '--limit'     : Or(None, And(Use(int), lambda x: x > 0),
                           error='Batch size should be integer > 0'),
        '--model'     : And(Use(str), lambda x: x in nnet.model_list,
                            error='Model not available'),
        '--epochs'    : And(Use(int), lambda x: x >= 0,
                            error='Number of epochs should be integer >= 0'),
        '--lr'        : And(Use(float), lambda x: x > 0,
                            error='Learning rate should be float > 0'),
        '--print-freq': And(Use(int), lambda x: x > 0,
                            error='Print frequency should be integer > 0'),
        '--device'    : And(Use(int), lambda x: x >= 0,
                            error='CUDA device ID should be integer >= 0'),
        '--keyword'   : And(Use(str), lambda x: x in ('pitch', 'roll', 'heading', 'hazard'),
                            error='Only predict \'pitch\' or \'roll\' or \'heading\' or \'hazard\''),
        '--stratux'   : Or(None,
                           And(Use(int), lambda x: x >= 0,
                               error='Stratux input level should be integer >= 0')),
        '--try'       : And(Use(int), lambda x: x > 0,
                            error='Number of trials should be integer > 0'),
        object        : object,
    }
    args = Schema(requirements).validate(args)

    # midterm prediction must be fixed
    assert not (args['--trig'] and args['--diff'] and args['--direct'])
    assert args['--keyword'] != 'hazard' or (args['--direct'] and args['--no-normal'])
    assert args['--phone'] != 'stratux' or args['--stratux'] is not None

    global WINDOW_CONFIG
    if args['--win'] is not None:
        WINDOW_CONFIG = {
            'input': {
                'length': args['--win'],
                'offset_length': args['--offset'], 'offset_rate': args['--rate'],
                'padding': 'repeat_base',
            },
            'target': {
                'length': args['--win'],
                'offset_length': args['--offset'], 'offset_rate': args['--rate'],
                'padding': 'repeat_base',
            },
        }
    else:
        WINDOW_CONFIG = constant.WINDOW_CONFIG

    return args

def config_log():
    logging.basicConfig(
        stream=sys.stdout, level=logging.INFO,
        format='[%(asctime)s] '
               '\033[1m%(levelname)s\033[0m :'
               ' %(message)s',
        datefmt='%y-%m-%d %H:%M:%S')
    logger = logging.getLogger(__name__)
    return logger

def main():
    args   = parse_args()
    logger = config_log()

    # regenerate flight data from raw data
    if args['--update']:
        input_data, target_data = data_meta(
            args['<input-dir>'], args['<output-dir>'], args['--phone'], args=args)
        standard_data = data_flight(
            args['<output-dir>'], args['<output-dir>'], args['--phone'], args=args)
        standard_data.meta_input.set_recovery(input_data.recovery)
        standard_data.meta_target.set_recovery(target_data.recovery)
        batch_input_dir = args['<output-dir>']
    else:
        input_data = flight.FlightExtensionData(
            '{}/{}.extension'.format(args['<input-dir>'], args['--phone']))
        target_data = flight.FlightExtensionData(
            '{}/g1000.extension'.format(args['<input-dir>']))
        standard_data = frame.FlightSequencePairData(
            '{}/{}_g1000.sequence'.format(args['<input-dir>'], args['--phone']))
        standard_data.meta_input.set_recovery(input_data.recovery)
        standard_data.meta_target.set_recovery(target_data.recovery)
        batch_input_dir = args['<input-dir>']

    # evaluate processed data
    if args['--eval']:
        evaluate(standard_data.meta_input, standard_data.meta_target)

    # real process
    feat   = args['--win']
    key    = args['--keyword']
    mname  = args['--model']
    output = args['<output-dir>']
    phone  = args['--phone']
    for i in range(args['--try']):
        if key == 'hazard':
            model, normal, loss, prediction = hazardous(
                key, mname, batch_input_dir, output,
                fake=args['--fake'], args=args)
        elif args['--trig']:
            model, normal, loss, prediction = trigonometrics(
                key, mname, standard_data, batch_input_dir, output,
                fake=args['--fake'], args=args)
        elif args['--diff']:
            model, normal, loss, prediction = difference(
                key, mname, standard_data, batch_input_dir, output,
                fake=args['--fake'], args=args)
        else: continue
        if i == 0 or loss < best_loss:
            logging.info('Save better model')
            best_loss = loss

            if args['--stratux'] is None:
                pass
            else:
                phone = "stratux-{}".format(args['--stratux'])
            if key == 'hazard':
                key = "hazard-{}".format(args['--threshold'])
            else:
                pass
            feat = "win-{}".format(feat)
            name = "{}.{}.{}.{}".format(phone, key, feat, mname)

            if 'hazard' in key:
                result = evaluate_label_diff(
                    standard=prediction, std_key='hazard',
                    observe=prediction , obv_key='*hazard')
            else:
                result = evaluate_abs_diff(
                    standard=prediction, std_key=key,
                    observe=prediction , obv_key="*{}".format(key))

            torch.save(model, './model/{}.model'.format(name))
            torch.save(normal, './model/{}.normal'.format(name))
            torch.save(loss, './model/{}.loss'.format(name))
            torch.save(result, './model/{}.result'.format(name))

            prediction.save('{}/predict/{}/predict.tar'.format(output, name))
            if not args['--no-plot']:
                prediction.plot(
                    '{}/predict/{}'.format(output, name),
                    prediction_label="{} + Neural Network (Prediction)".format(
                                        phone[0].upper() + phone[1:].lower()),
                    target_label="G1000 (Target)")
        else:
            logging.info('Ignore worse model')

if __name__ == '__main__':
    main()
