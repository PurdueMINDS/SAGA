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
"""

Usage:
    main.py <input-dir>
            [--phone <phone>] [--model <prefix>] [--stratux <lv>]
            (--test-key <key>)... [--threshold <T>]
            [--all] [--hazard-success]


Options:
    <input-dir>             Directory of input to load data [default: ./output]
    --phone <phone>         Phone type to specify [default: pixel]
    --model <prefix>        Model prefix
    --stratux <lv>          Specify stratux model level
    --test-key <key>        Keyword to test
    --threshold <T>         Roll threshold of hazardous state [default: 45]
    --all                   Predict all data regardless of training
    --hazard-success        Show successful hazardous state prediction

"""
import docopt
from schema import Schema, Use, And, Or
import os
import numpy as np
import torch

import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
COLORS = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)
import matplotlib.lines as mlines

import constant
import flight
import frame
import batch

args = docopt.docopt(__doc__, version='SAGA Project Test Ver 1.0')

requirements = {
    '--phone'    : And(Use(str), lambda x: x in ('galaxy', 'pixel', 'stratux'),
                      error='Phone type only support \'galaxy\', \'pixel\' and \'stratux\''),
    '--stratux'  : Or(None, Use(int),
                     error='Stratux input level should be integer'),
    '--test-key' : And(Use(list), lambda x: set(x) <= set(['pitch', 'roll', 'heading']),
                      error='Keyword should be \'pitch\', \'roll\', \'heading\''),
    '--threshold': And(Use(int), lambda x: x > 0,
                      error='Stratux input level should be integer'),
    object       : object,
}
args = Schema(requirements).validate(args)

STRATUX_LIST = (
    '020918_00', '020918_01', '021318_00', '021318_01', '021318_02',
    '021418_00', '022718_00', '030318_00', '030318_01', '030318_02',
    '030318_03', '030418_00', '030418_01', '032618_00', '040718_00',
    '040718_01', '040718_02', '041018_00', '042018_00', '042018_01',
    '042018_02', '042718_00', '042718_01', '042718_02',
)

STRATUX_TEST_LIST = (
    '030318_02', '030318_03', '040718_02', '042018_02', '042718_02',
)

PHONE_TEST_LIST = (
    '021318_03', '030318_03', '030318_04', '040718_02', '040718_03',
    '042018_02', '042718_02', '111417_02', '120117_02', '120217_02',
)

TEST_LIST_0 = ('030318_03', '042018_02', '040718_02', '042718_02') # Overlapping
TEST_LIST_1 = ('111417_02', '120117_02', '021318_03') # Have Hazard State
TEST_LIST = PHONE_TEST_LIST
# if args['--phone'] != 'stratux': TEST_LIST += TEST_LIST_1

def prune_key_id(prune, key):
    assert isinstance(prune, flight.FlightPruneData)

    col_id = None
    for i, itr in enumerate(prune.keys):
        if itr == key:
            col_id = i
            break
    assert col_id is not None

    return col_id

input_data = flight.FlightExtensionData(
    '{}/{}.extension'.format(args['<input-dir>'], args['--phone']))
target_data = flight.FlightExtensionData(
    '{}/g1000.extension'.format(args['<input-dir>']))
standard_data = frame.FlightSequencePairData(
    '{}/{}_g1000.sequence'.format(args['<input-dir>'], args['--phone']))
standard_data.meta_input.set_recovery(input_data.recovery)
standard_data.meta_target.set_recovery(target_data.recovery)
batch_input_dir = args['<input-dir>']

def predict(key, args):
    def concat_prediction(prediction1, prediction2):
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

    if args['--stratux'] is not None and args['--stratux'] < 0:
        seq_pair_data = frame.FlightSequencePairData(
            '{}/{}_g1000.sequence'.format(args['<input-dir>'], args['--phone']))
        time_col = seq_pair_data.meta_target.clone()
        time_col.prune_keys(remain_keys=['time'])
        if not args['--all']: time_col.prune_identifier(remain_identifier=TEST_LIST)
        seq_pair_data.prune_keys(
            input_remain_keys=(key,),
            target_remain_keys=(key,))
        if not args['--all']:
            seq_pair_data.prune_identifier(remain_identifier=TEST_LIST)
        prediction = seq_pair_data.meta_input.clone()
        prediction.prune_keys(remain_keys=['{}'.format(key)])
        std_prediction = seq_pair_data.meta_target.clone()
        std_prediction.prune_keys(remain_keys=['{}'.format(key)])
        prediction.keys = tuple(['*{}'.format(key)])
        prediction = concat_prediction(prediction, std_prediction)
        return prediction, time_col

    if args['--stratux'] is None: name = key
    else: name = "{}.lv{}".format(key, args['--stratux'])
    model = torch.load('{}/{}.{}.model'.format(args['--model'], args['--phone'], name))
    normal = torch.load('{}/{}.{}.normal'.format(args['--model'], args['--phone'], name))

    model['sin'].cuda = False
    model['cos'].cuda = False

    def data_batch(
        input_dir, phone_type, target_key, set_normal,
        batch_size=16, shuffle=False,
        window_config=constant.WINDOW_CONFIG):
        # load sequence data
        seq_pair_data = frame.FlightSequencePairData('{}/{}_g1000.sequence'.format(
                                                        input_dir, phone_type))
        time_col = seq_pair_data.meta_target.clone()
        time_col.prune_keys(remain_keys=['time'])
        if not args['--all']: time_col.prune_identifier(remain_identifier=TEST_LIST)

        # remain only necessary keywords
        if args['--stratux'] is None:
            seq_pair_data.prune_keys(
                input_remain_keys=constant.INPUT_KEYWORDS,
                target_remain_keys=(target_key,))
        else:
            lv = args['--stratux']
            input_key = target_key.split('_')[1]
            if lv == 0: input_remain_keys = (input_key,)
            elif lv == 1: input_remain_keys = ('alt', 'lat', 'long', input_key)
            elif lv == 2: input_remain_keys = ('alt', 'lat', 'long', 'pitch', 'roll', 'heading')
            else: raise NotImplementedError
            seq_pair_data.prune_keys(
                input_remain_keys=input_remain_keys,
                target_remain_keys=(target_key,))

        if not args['--all']:
            seq_pair_data.prune_identifier(remain_identifier=TEST_LIST)

        # normalize sequence data on time domain
        seq_pair_data.normalize(set_normal)

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

        batch_loader = batch.FramePairDataLoader(
            frame_pair_data, batch_size=batch_size, shuffle=shuffle, drop_last=False)

        return batch_loader, seq_pair_data.length_dict(), time_col

    sin_loader, sin_len_dict, sin_time_col = data_batch(
        args['<input-dir>'], args['--phone'], "sin_{}".format(key), normal['sin'])
    sin_loader, sin_loss = model['sin'].predict(sin_loader, '*sin_{}'.format(key), (-1, 32, 1), fake=False)
    sin_frame = sin_loader.to_frame_pair()
    sin_seq = sin_frame.to_seq_pair(sin_len_dict)
    sin_seq.denormalize(normal['sin'])
    sin_target = sin_seq.meta_target

    cos_loader, cos_len_dict, cos_time_col = data_batch(
        args['<input-dir>'], args['--phone'], "cos_{}".format(key), normal['cos'])
    cos_loader, cos_loss = model['cos'].predict(cos_loader, '*cos_{}'.format(key), (-1, 32, 1), fake=False)
    cos_frame = cos_loader.to_frame_pair()
    cos_seq = cos_frame.to_seq_pair(cos_len_dict)
    cos_seq.denormalize(normal['cos'])
    cos_target = cos_seq.meta_target

    prediction = concat_prediction(sin_target, cos_target)

    # fetch standard target data column
    global standard_data
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
    return prediction, sin_time_col

def evaluate_abs_diff(standard, std_key, observe, obv_key):
    assert isinstance(standard, flight.FlightPruneData) and \
           isinstance(observe , flight.FlightPruneData)

    assert standard.identifier == observe.identifier

    std_id = prune_key_id(standard, std_key)
    obv_id = prune_key_id(observe , obv_key)

    std_list  = []
    obv_list  = []
    diff_list = []

    for i in range(len(standard)):
        std_col  = standard[i][1][:, std_id]
        obv_col  = observe[i][1][:, obv_id]
        diff_col = np.fabs(std_col - obv_col)
        if 'heading' in std_key or 'heading' in obv_key:
            diff2_col = np.fabs(360 - diff_col)
            diff_col = np.minimum(diff_col, diff2_col)
        std_list.append(std_col)
        obv_list.append(obv_col)
        diff_list.append(diff_col)

    std  = np.concatenate(std_list)
    obv  = np.concatenate(obv_list)
    diff = np.concatenate(diff_list)

    min_val   = np.min(std)
    max_val   = np.max(std)
    min_diff  = np.min(diff)
    max_diff  = np.max(diff)
    mean_diff = np.mean(diff)
    std_diff  = np.std(diff)

    print('\033[32;1mStandard ({}) vs Observation ({})\033[0m'.format(std_key, obv_key))
    print('{:8.4f} {:8.4f}, {:8.4f} {:8.4f}, {:8.4f} {:8.4f}'.format(
            min_val, max_val, min_diff, max_diff, mean_diff, std_diff))

def evaluate_hazard(eval_data, threshold=45):
    if args['--phone'] == 'stratux': name = "{}.lv{}".format(args['--phone'], args['--stratux'])
    else: name = args['--phone']
    folder = "test/hazard.{}".format(name)
    if args['--all']: folder = "{}.all".format(folder)
    else: folder = "{}.test".format(folder)
    if not os.path.isdir(folder): os.makedirs(folder)
    std_id = prune_key_id(eval_data, 'roll')
    obv_id = prune_key_id(eval_data, '*roll')
    file = open("{}/r{}.csv".format(folder, args['--threshold']), 'w')
    file.write('Flight,Hazard-Fail%,#Hazard-Fail,#Hazard,Safe-Fail%,#Safe-Fail,#Safe\n')
    for i in range(len(eval_data)):
        idt = eval_data.identifier[i]
        std_col = eval_data[i][1][:, std_id]
        obv_col = eval_data[i][1][:, obv_id]

        std_safe   = set(np.where(np.fabs(std_col) < threshold)[0])
        std_hazard = set(np.where(np.fabs(std_col) >= threshold)[0])
        obv_safe   = set(np.where(np.fabs(obv_col) < threshold)[0])
        obv_hazard = set(np.where(np.fabs(obv_col) >= threshold)[0])

        both_safe = std_safe & obv_safe
        both_hazard = std_hazard & obv_hazard
        should_safe_but_not = std_safe - obv_safe
        should_hazard_but_not = std_hazard - obv_hazard

        num_both_safe = len(both_safe)
        num_both_hazard = len(both_hazard)
        num_should_safe_but_not = len(should_safe_but_not)
        num_should_hazard_but_not = len(should_hazard_but_not)
        total_safe = num_both_safe + num_should_safe_but_not
        total_hazard = num_both_hazard + num_should_hazard_but_not
        if total_safe > 0:
            error_rate_safe = "{:7.3f}".format(num_should_safe_but_not / total_safe * 100)
        else:
            error_rate_safe = '-'
        if total_hazard > 0:
            success_rate_hazard = "{:7.3f}".format(num_both_hazard / total_hazard * 100)
            error_rate_hazard = "{:7.3f}".format(num_should_hazard_but_not / total_hazard * 100)
        else:
            success_rate_hazard = 100
            error_rate_hazard = '-'

        print(eval_data.identifier[i])
        print(num_both_hazard, num_should_hazard_but_not)
        print(num_should_safe_but_not, num_both_safe)

        file.write("{},{},{},{},{},{},{}\n".format(
            idt,
            '' if total_hazard == 0 else num_should_hazard_but_not / total_hazard * 100,
            num_should_hazard_but_not, total_hazard,
            '' if total_safe == 0 else num_should_safe_but_not / total_safe * 100,
            num_should_safe_but_not, total_safe))

        title = "{}/{}/{}, Flight {}".format(idt[4:6], idt[0:2], idt[2:4], idt[7:9])
        title += ", Roll Threshold: {}".format(args['--threshold'])
        if args['--hazard-success']:
            title += "\nHazard-Success: {}% ({} / {})".format(
                success_rate_hazard, num_both_hazard, total_hazard)
        else:
            title += "\nHazard-Failure: {}% ({} / {})".format(
                error_rate_hazard, num_should_hazard_but_not, total_hazard)
            title += "\nSafe-Failure  : {}% ({} / {})".format(
                error_rate_safe, num_should_safe_but_not, total_safe)

        label = args['--phone'][0].upper() + args['--phone'][1:].lower() + " + NN"

        fig, ax = plt.subplots(1, 1, figsize=(20, 6))
        fig.suptitle(title)
        ax.set_xlabel('Time Index')
        ax.set_ylabel('Roll (Degree)')
        ax.plot(range(len(std_col)), std_col, label='G1000', color=COLORS['purple'], alpha=0.5)
        ax.plot(range(len(obv_col)), obv_col, label=label  , color=COLORS['green'] , alpha=0.5)
        ax.axhline(y=threshold , color=COLORS['black'], alpha=0.25)
        ax.axhline(y=-threshold, color=COLORS['black'], alpha=0.25)
        legend1 = ax.legend(loc='upper right')
        if args['--hazard-success']:
            for succ in both_hazard:
                ax.plot((succ, succ), (std_col[succ], obv_col[succ]), color=COLORS['blue'], alpha=0.25, linestyle='--')
                ax.scatter(succ, std_col[succ], color=COLORS['blue'], alpha=0.5)
                ax.scatter(succ, obv_col[succ], color=COLORS['blue'], alpha=0.5)
            legend2 = ax.legend(loc='upper left', handles=[
                mlines.Line2D([], [], color=COLORS['blue'], alpha=0.25, linestyle='--', marker='o', label='Hazard-Success')
            ])
        else:
            for err in should_hazard_but_not:
                ax.plot((err, err), (std_col[err], obv_col[err]), color=COLORS['red'], alpha=0.25, linestyle='--')
                ax.scatter(err, std_col[err], color=COLORS['red'], alpha=0.5)
                ax.scatter(err, obv_col[err], color=COLORS['red'], alpha=0.5)
            for err in should_safe_but_not:
                ax.plot((err, err), (std_col[err], obv_col[err]), color=COLORS['orange'], alpha=0.25, linestyle='--')
                ax.scatter(err, std_col[err], color=COLORS['orange'], alpha=0.5, marker='o')
                ax.scatter(err, obv_col[err], color=COLORS['orange'], alpha=0.5, marker='o')
            legend2 = ax.legend(loc='upper left', handles=[
                mlines.Line2D([], [], color=COLORS['red'], alpha=0.25, linestyle='--', marker='o', label='Hazard-Failure'),
                mlines.Line2D([], [], color=COLORS['orange'], alpha=0.25, linestyle='--', marker='o', label='Safe-Failure')
            ])
        ax.add_artist(legend1)
        ax.add_artist(legend2)
        fig.savefig('{}/r{}-{}.png'.format(folder, args['--threshold'], idt))
        plt.close(fig)
    file.close()

def timefmt(num):
    num = int(num)
    hour = num // 3600
    minute = (num % 3600) // 60
    second = (num % 3600) % 60
    return "%02d:%02d:%02d" % (hour, minute, second)

def save_pure_prediction(prediction, time, key):
    output = prediction.clone()
    output.prune_keys(remain_keys=["*{}".format(key)])
    target = prediction.clone()
    target.prune_keys(remain_keys=[key])
    buffer = {}
    assert output.identifier == time.identifier
    assert target.identifier == time.identifier
    for i, idt in enumerate(prediction.identifier):
        keyout = list(output.dataset[i].reshape(-1))
        keytar = list(target.dataset[i].reshape(-1))
        timeout = list(time.dataset[i].reshape(-1))
        assert len(keyout) == len(timeout)
        assert len(keytar) == len(timeout)
        buffer[idt] = {
            'time'  : [timefmt(itr) for itr in timeout],
            'output': keyout,
            'target': keytar,
        }

    folder = 'test_temp'
    if not os.path.isdir(folder): os.makedirs(folder)
    for idt in buffer.keys():
        time_seqs = buffer[idt]['time']
        output_seqs = buffer[idt]['output']
        target_seqs = buffer[idt]['target']
        file = open(os.path.join(folder, "{}.csv".format(idt)), 'w')
        for time, output, target in zip(time_seqs, output_seqs, target_seqs):
            # file.write("{},{}\n".format(time, output))
            file.write("{},{},{}\n".format(time, output, target))
        file.close()

for key in args['--test-key']:
    pred, time_col = predict(key, args)
    # save_pure_prediction(pred, time_col, key=key)
    # if key == 'roll': evaluate_hazard(pred, threshold=args['--threshold'])
    evaluate_abs_diff(standard=pred, std_key=key,
                      observe=pred , obv_key="*{}".format(key))
