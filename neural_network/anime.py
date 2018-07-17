"""
Usage:
    anime.py <dir> <phone> [--model <phn>] [--start <ptr>] [--rate <num>]

Options:
    <dir>           Root directory of data to predict
    <phone>         Smartphone to generate animation
    --model <phn>   Specify phone type of the model
    --start <ptr>   Specify animation starting second [default: 0]
    --rate <num>    Specify number of animation frames per second [default: 20]

"""
import os
import sys
import docopt
from schema import Schema, Use, And, Or
import logging

import numpy as np

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib import animation

import time

import torch

import constant
import raw
import flight
import frame
import batch
import nnet

def process_data(dir, phone, date):
    def process_g1000():
        raw_g1000_data = raw.RawG1000Data('{}/g1000'.format(dir), want=[date])
        ext_g1000_data = flight.FlightExtensionData(raw_g1000_data)
    
        ext_g1000_data.append_num_diff(key='alt' , new_key='spd_alt' , step=5, pad='repeat_base')
        ext_g1000_data.append_num_diff(key='lat' , new_key='spd_lat' , step=5, pad='repeat_base')
        ext_g1000_data.append_num_diff(key='long', new_key='spd_long', step=5, pad='repeat_base')
        ext_g1000_data.append_ground_speed(spd_lat_key='spd_lat', spd_long_key='spd_long', new_key='spd_gd')
    
        ext_g1000_data.append_num_diff(key='pitch'  , new_key='spd_pitch'  , step=5, pad='repeat_base')
        ext_g1000_data.append_num_diff(key='roll'   , new_key='spd_roll'   , step=5, pad='repeat_base')
        ext_g1000_data.append_deg_diff(key='heading', new_key='spd_heading', step=5, pad='repeat_base')
    
        ext_g1000_data.append_deg_sin(key='pitch'  , new_key='sin_pitch'  )
        ext_g1000_data.append_deg_sin(key='roll'   , new_key='sin_roll'   )
        ext_g1000_data.append_deg_sin(key='heading', new_key='sin_heading')
    
        ext_g1000_data.append_deg_cos(key='pitch'  , new_key='cos_pitch'  )
        ext_g1000_data.append_deg_cos(key='roll'   , new_key='cos_roll'   )
        ext_g1000_data.append_deg_cos(key='heading', new_key='cos_heading')
    
        return ext_g1000_data

    def process_phone():
        raw_phone_data = raw.RawPhoneData('{}/{}'.format(dir, phone), want=[date])
        ext_phone_data = flight.FlightExtensionData(raw_phone_data)
    
        ext_phone_data.append_num_diff(key='alt' , new_key='spd_alt' , step=5, pad='repeat_base')
        ext_phone_data.append_num_diff(key='lat' , new_key='spd_lat' , step=5, pad='repeat_base')
        ext_phone_data.append_num_diff(key='long', new_key='spd_long', step=5, pad='repeat_base')
        ext_phone_data.append_ground_speed(spd_lat_key='spd_lat', spd_long_key='spd_long', new_key='spd_gd')
    
        ext_phone_data.append_num_diff(key='spd_alt' , new_key='acc_alt' , step=1, pad='repeat_base')
        ext_phone_data.append_num_diff(key='spd_lat' , new_key='acc_lat' , step=1, pad='repeat_base')
        ext_phone_data.append_num_diff(key='spd_long', new_key='acc_long', step=1, pad='repeat_base')
    
        return ext_phone_data

    ext_g1000_data = process_g1000()
    ext_phone_data = process_phone()

    prn_g1000_data = flight.FlightPruneData(ext_g1000_data)
    prn_phone_data = flight.FlightPruneData(ext_phone_data)

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

    prn_g1000_data.prune_identifier(discard_identifier=constant.HIZARD_FLIGHTS)
    prn_g1000_data.detect_parking(method='time')

    phone_requirment = prn_g1000_data.time_date_flights()
    prn_phone_data.prune_identifier(remain_identifier=phone_requirment.keys())
    prn_phone_data.detect_parking(method='time', time_flights=phone_requirment)

    prn_g1000_data.prune_parking()
    prn_phone_data.prune_parking()

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

    prn_g1000_data.prune_identifier(remain_identifier=prn_phone_data.identifier)

    seq_pair_data = frame.FlightSequencePairData(entity_input=prn_phone_data, entity_target=prn_g1000_data)
    seq_pair_data.align_and_interpolate(match_keys=('alt', 'lat', 'long'))
    seq_pair_data.distribute()

    return seq_pair_data

def data_batch(seq_pair_data, target_key, batch_size,
               set_normal=None, rnn=False):
    seq_pair_data.prune_keys(
        input_remain_keys=constant.INPUT_KEYWORDS,
        target_remain_keys=(target_key,))
    
    if set_normal is None: pass
    else: seq_pair_data.normalize(set_normal)

    window_config = constant.WINDOW_CONFIG
    frame_pair_data = frame.FlightFramePairData(
        seq_pair_data,
        input_win_len=window_config['input']['length'],
        target_win_len=window_config['target']['length'],
        input_win_offset=window_config['input']['offset_length'],
        target_win_offset=window_config['target']['offset_length'],
        input_win_offset_rate=window_config['input']['offset_rate'],
        target_win_offset_rate=window_config['target']['offset_rate'],
        input_pad=window_config['input']['padding'],
        target_pad=window_config['target']['padding'])

    if rnn:
        batch_loader = batch.TimestepPairDataLoader(frame_pair_data, timestep=batch_size)
    else:
        batch_loader = batch.FramePairDataLoader(
            frame_pair_data, batch_size=batch_size, shuffle=False, drop_last=False)
    
    return batch_loader, seq_pair_data.length_dict()

def keyword_prediction(key, seq_data, monitor, normal, fake=False):
    # generate data loader and predict by the model
    if monitor.rnn:
        predict_loader, len_dict = data_batch(
            seq_data, target_key=key, batch_size=16, rnn=True, set_normal=normal)
        predict_loader, loss_avg = monitor.predict(predict_loader, '*{}'.format(key), (1, -1, 32, 1), fake=fake)
    else:
        predict_loader, len_dict = data_batch(
            seq_data, target_key=key, batch_size=16, rnn=False, set_normal=normal)
        predict_loader, loss_avg = monitor.predict(predict_loader, '*{}'.format(key), (-1, 32, 1), fake=fake)

    # extract flight data
    predict_frame = predict_loader.to_frame_pair()
    predict_seq = predict_frame.to_seq_pair(len_dict)
    predict_seq.denormalize(normal)
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

def predict_trig(sequence, key, dir):
    model  = torch.load("{}.model".format(dir))
    normal = torch.load("{}.normal".format(dir))
    sin_monitor, cos_monitor = model['sin'] , model['cos']
    sin_normal , cos_normal  = normal['sin'], normal['cos']

    if not torch.cuda.is_available():
        sin_monitor.cuda = False
        cos_monitor.cuda = False

    # predict by the models
    sin_prediction, sin_loss = keyword_prediction(
        "sin_{}".format(key), sequence.clone(), monitor=sin_monitor, normal=sin_normal)
    cos_prediction, cos_loss = keyword_prediction(
        "cos_{}".format(key), sequence.clone(), monitor=cos_monitor, normal=cos_normal)
    prediction = concat_prediction(sin_prediction, cos_prediction)

    # fetch standard target data column
    std_prediction = sequence.meta_target.clone()
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
    return prediction

class Animation():
    def __init__(self, anime_data, ptr=0):
        self.phone = anime_data['phone']

        self.g1000_altitude  = anime_data['g1000_altitude']
        self.g1000_latitude  = anime_data['g1000_latitude']
        self.g1000_longitude = anime_data['g1000_longitude']
        self.g1000_pitch     = anime_data['g1000_pitch']
        self.g1000_roll      = anime_data['g1000_roll']
        self.g1000_heading   = anime_data['g1000_heading']
        self.phone_altitude  = anime_data['phone_altitude']
        self.phone_latitude  = anime_data['phone_latitude']
        self.phone_longitude = anime_data['phone_longitude']
        self.phone_pitch     = anime_data['phone_pitch']
        self.phone_roll      = anime_data['phone_roll']
        self.phone_heading   = anime_data['phone_heading']
        
        self.ptr = ptr
        self.limit = len(self.g1000_altitude)

        # a figure instance to plot on
        self.figure = plt.figure(figsize=(11, 9))
        self.ax_gps     = plt.subplot2grid((9, 11), (0, 0), rowspan=5, colspan=5)
        self.ax_alt     = plt.subplot2grid((9, 11), (0, 6), rowspan=5, colspan=5)
        self.ax_pitch   = plt.subplot2grid((9, 11), (6, 0), rowspan=3, colspan=3, polar=True)
        self.ax_roll    = plt.subplot2grid((9, 11), (6, 4), rowspan=3, colspan=3, polar=True)
        self.ax_heading = plt.subplot2grid((9, 11), (6, 8), rowspan=3, colspan=3, polar=True)

        C0 = 'red'
        C1 = 'blue'

        self.ax_gps.set_title('GPS')
        self.ax_alt.set_title('Altitude')

        self.ax_gps.set_xlabel('Latitude (Degree)')
        self.ax_alt.set_xlabel('Time (Seconds)')
        self.ax_pitch.set_xlabel('Pitch (Degree)')
        self.ax_roll.set_xlabel('Roll (Degree)')
        self.ax_heading.set_xlabel('Heading (Degree)')
        
        self.ax_gps.set_ylabel('Longitude (Degree)')
        self.ax_alt.set_ylabel('Altitude (Feet)')

        self.ax_pitch.set_xlim(-np.pi / 2, np.pi / 2)
        self.ax_roll.set_xlim(-np.pi / 2, np.pi / 2)

        self.ax_pitch.set_ylim(0, 1.2)
        self.ax_roll.set_ylim(0, 1.2)
        self.ax_heading.set_ylim(0, 1.2)

        self.ax_pitch.get_yaxis().set_visible(False)
        self.ax_roll.get_yaxis().set_visible(False)
        self.ax_heading.get_yaxis().set_visible(False)

        self.ax_pitch.set_theta_zero_location('E')
        self.ax_roll.set_theta_zero_location('N')
        self.ax_heading.set_theta_zero_location('N')

        self.ax_pitch.set_xticks(np.array([0, -10, -20, -30, np.nan, 30, 20, 10]) / 180 * np.pi * 4.5)
        self.ax_roll.set_xticks(np.array([0, -20, -40, -60, np.nan, 60, 40, 20]) / 180 * np.pi * 2.25)

        self.ax_pitch.set_xticklabels([
            r'$0\degree$', r'$-10\degree$', r'$-20\degree$', r'$-30\degree$', '',
            r'$30\degree$', r'$20\degree$', r'$10\degree$'])
        self.ax_roll.set_xticklabels([
            r'$0\degree$', r'$-20\degree$', r'$-40\degree$', r'$-60\degree$', '',
            r'$60\degree$', r'$40\degree$', r'$20\degree$'])

        self.g1000_gps_all, = self.ax_gps.plot(
            self.g1000_latitude, self.g1000_longitude,
            color=C0, alpha=0.1, linewidth=2.0)
        self.g1000_gps_pass, = self.ax_gps.plot(
            self.g1000_latitude[:self.ptr], self.g1000_longitude[:self.ptr],
            color=C0, alpha=0.4, linewidth=2.0)
        self.g1000_gps_now, = self.ax_gps.plot(
            [], [], color=C0, alpha=0.5, marker='o')

        self.phone_gps_all, = self.ax_gps.plot(
            self.phone_latitude, self.phone_longitude,
            color=C1, alpha=0.1, linewidth=2.0)
        self.phone_gps_pass, = self.ax_gps.plot(
            self.phone_latitude[:self.ptr], self.phone_longitude[:self.ptr],
            color=C1, alpha=0.4, linewidth=2.0)
        self.phone_gps_now, = self.ax_gps.plot(
            [], [], color=C1, alpha=0.5, marker='o')

        self.g1000_alt_all, = self.ax_alt.plot(
            np.arange(0, self.limit), self.g1000_altitude,
            color=C0, alpha=0.1, linewidth=2.0)
        self.g1000_alt_pass, = self.ax_alt.plot(
            np.arange(0, self.ptr), self.g1000_altitude[:self.ptr],
            color=C0, alpha=0.4, linewidth=2.0)

        self.phone_alt_all, = self.ax_alt.plot(
            np.arange(0, self.limit), self.phone_altitude,
            color=C1, alpha=0.1, linewidth=2.0)
        self.phone_alt_pass, = self.ax_alt.plot(
            np.arange(0, self.ptr), self.phone_altitude[:self.ptr],
            color=C1, alpha=0.4, linewidth=2.0)

        self.g1000_pitch_arw = self.ax_pitch.annotate('',
            xy=(0, 1), xytext=(0, 0),
            arrowprops=dict(arrowstyle="->", color=C0, alpha=0.5, linewidth=2.0, linestyle='--'))
        self.g1000_roll_arw = self.ax_roll.annotate('',
            xy=(0, 1), xytext=(0, 0),
            arrowprops=dict(arrowstyle="->", color=C0, alpha=0.5, linewidth=2.0, linestyle='--'))
        self.g1000_heading_arw = self.ax_heading.annotate('',
            xy=(0, 1), xytext=(0, 0),
            arrowprops=dict(arrowstyle="->", color=C0, alpha=0.5, linewidth=2.0, linestyle='--'))

        self.phone_pitch_arw = self.ax_pitch.annotate('',
            xy=(0, 1), xytext=(0, 0),
            arrowprops=dict(arrowstyle="->", color=C1, alpha=0.5, linewidth=2.0, linestyle='-'))
        self.phone_roll_arw = self.ax_roll.annotate('',
            xy=(0, 1), xytext=(0, 0),
            arrowprops=dict(arrowstyle="->", color=C1, alpha=0.5, linewidth=2.0, linestyle='-'))
        self.phone_heading_arw = self.ax_heading.annotate('',
            xy=(0, 1), xytext=(0, 0),
            arrowprops=dict(arrowstyle="->", color=C1, alpha=0.5, linewidth=2.0, linestyle='-'))
        
        self.title1 = self.ax_gps.text(
            0.5, 0.95, self.pass_time(), fontsize=12,
            transform=self.ax_gps.transAxes,
            horizontalalignment='center', verticalalignment='center',
            bbox=dict(facecolor='white', alpha=0.5))
        self.title2 = self.ax_alt.text(
            0.5, 0.95, self.pass_time(), fontsize=12,
            transform=self.ax_alt.transAxes,
            horizontalalignment='center', verticalalignment='center',
            bbox=dict(facecolor='white', alpha=0.5))

        self.legend = self.figure.legend(
            handles=[
                plt.plot([], linestyle="--", color=C0)[0],
                plt.plot([], linestyle="-", color=C1)[0]
            ],
            labels=[self.phone, 'G1000'],
            loc='upper center', framealpha=0.5)

    def pass_time(self):
        hour   = self.ptr // 3600
        minute = (self.ptr % 3600) // 60
        second = (self.ptr % 3600) % 60
        return "%02d:%02d:%02d" % (hour, minute, second)

    def plot(self, this):
        self.g1000_gps_pass.set_data(
            self.g1000_latitude[:self.ptr], self.g1000_longitude[:self.ptr])
        self.phone_gps_pass.set_data(
            self.phone_latitude[:self.ptr], self.phone_longitude[:self.ptr])
        
        self.g1000_gps_now.set_data(
            self.g1000_latitude[self.ptr], self.g1000_longitude[self.ptr])
        self.phone_gps_now.set_data(
            self.phone_latitude[self.ptr], self.phone_longitude[self.ptr])

        self.g1000_alt_pass.set_data(
            np.arange(0, self.ptr), self.g1000_altitude[:self.ptr])
        self.phone_alt_pass.set_data(
            np.arange(0, self.ptr), self.phone_altitude[:self.ptr])

        self.g1000_pitch_arw.xy   = (self.g1000_pitch[self.ptr] / 180 * np.pi * 9, 1)
        self.g1000_roll_arw.xy    = (self.g1000_roll[self.ptr] / 180 * np.pi, 1)
        self.g1000_heading_arw.xy = (self.g1000_heading[self.ptr] / 180 * np.pi, 1)

        self.g1000_pitch_arw.xytext   = (0, 0)
        self.g1000_roll_arw.xytext    = (0, 0)
        self.g1000_heading_arw.xytext = (0, 0)

        self.phone_pitch_arw.xy   = (self.phone_pitch[self.ptr] / 180 * np.pi * 9, 1)
        self.phone_roll_arw.xy    = (self.phone_roll[self.ptr] / 180 * np.pi, 1)
        self.phone_heading_arw.xy = (self.phone_heading[self.ptr] / 180 * np.pi, 1)

        self.phone_pitch_arw.xytext   = (0, 0)
        self.phone_roll_arw.xytext    = (0, 0)
        self.phone_heading_arw.xytext = (0, 0)

        self.ptr += 1
        if self.ptr == self.limit:
            self.title1.set_text('Finish')
            self.title2.set_text('Finish')
        else:
            self.title1.set_text(self.pass_time())
            self.title2.set_text(self.pass_time())

        self.ptr = min(self.ptr, self.limit - 1)

        return self.g1000_gps_pass, self.g1000_gps_now, \
               self.phone_gps_pass, self.phone_gps_now, \
               self.g1000_alt_pass, \
               self.phone_alt_pass, \
               self.g1000_pitch_arw, self.g1000_roll_arw, self.g1000_heading_arw, \
               self.phone_pitch_arw, self.phone_roll_arw, self.phone_heading_arw, \
               self.title1, self.title2

    def onClick(self, event):
        if self.running:
            self.anime.event_source.stop()
            self.running = False
        else:
            self.anime.event_source.start()
            self.running = True

    def render(self, rate):
        self.running = True
        self.figure.canvas.mpl_connect('button_press_event', self.onClick)
        self.anime = animation.FuncAnimation(
            self.figure, self.plot, interval=rate, blit=True)
        plt.show()

def main():
    logging.basicConfig(
        stream=sys.stdout, \
        level=logging.INFO, \
        format='%(message)s', \
        datefmt='%y-%m-%d %H:%M:%S')

    args = docopt.docopt(__doc__, version='SAGA PyAnimation 2.2')
    if args['--model'] is None: args['--model'] = args['<phone>']

    requirements = {
        '<dir>'   : And(Use(str), lambda x: os.path.isdir(x),
                        error='Data directory must exist'),
        '<phone>' : And(Use(str), lambda x: x in ('galaxy', 'pixel'),
                        error='Phone type only supports \'galaxy\' or \'pixel\''),
        '--model' : And(Use(str), lambda x: x in ('galaxy', 'pixel'),
                        error='Phone type only supports \'galaxy\' or \'pixel\''),
        '--start' : And(Use(int), lambda x: x >= 0,
                        error='Starting second must be interger >= 0'),
        '--rate'  : And(Use(int), lambda x: x > 0,
                        error='Number of frames per second must be integer > 0'),
    }
    args = Schema(requirements).validate(args)

    dir    = args['<dir>']
    phone  = args['<phone>']

    TEST = {
        '021318': ['03'],
        '030318': ['04'],
        '040718': ['03'],
        '120117': ['02'],
    }

    date_list = sorted(os.listdir(os.path.join(dir, phone)))
    assert set(date_list) >= set(TEST)
    date_list = list(sorted(set(date_list) & set(TEST)))
    if len(date_list) > 1:
        date = input("\033[34;1mSelect a Date\033[0;m ({}): ".format(', '.join(date_list)))
        assert date in date_list
    elif len(date_list) == 1:
        date = date_list[0]
        print("\033[34;1mSelect a Date\033[0;m ({}): {}".format(', '.join(date_list), date))
    else: raise NotImplementedError

    sequence = process_data(dir, phone, date)

    flight_list = [idt.split('_')[-1] for idt in sequence.identifier]
    assert set(flight_list) >= set(TEST[date])
    flight_list = list(sorted(set(flight_list) & set(TEST[date])))
    if len(flight_list) > 1:
        flight = input("\033[34;1mSelect a Flight\033[0;m ({}): ".format(', '.join(flight_list)))
        assert flight in flight_list
    elif len(flight_list) == 1:
        flight = flight_list[0]
        print("\033[34;1mSelect a Flight\033[0;m ({}): {}".format(', '.join(flight_list), flight))
    else: raise NotImplementedError

    remain = ["{}_{}".format(date, flight)]
    sequence.prune_identifier(remain_identifier=remain)
    assert len(sequence) == 1

    def get_col(flight, key):
        assert len(flight) == 1
        id = flight.get_col_id(key)
        assert id is not None
        _, all_data = flight[0]
        return all_data[:, id]

    g1000_altitude  = get_col(sequence.meta_target, 'alt')
    g1000_latitude  = get_col(sequence.meta_target, 'lat')
    g1000_longitude = get_col(sequence.meta_target, 'long')
    g1000_pitch     = get_col(sequence.meta_target, 'pitch')
    g1000_roll      = get_col(sequence.meta_target, 'roll')
    g1000_heading   = get_col(sequence.meta_target, 'heading')

    # print(g1000_altitude .shape)
    # print(g1000_latitude .shape)
    # print(g1000_longitude.shape)
    # print(g1000_pitch    .shape)
    # print(g1000_roll     .shape)
    # print(g1000_heading  .shape)

    mphone = args['--model']
    pitch_data   = predict_trig(sequence, 'pitch', "./model/{}.pitch".format(mphone))
    roll_data    = predict_trig(sequence, 'roll', "./model/{}.roll".format(mphone))
    heading_data = predict_trig(sequence, 'heading', "./model/{}.heading".format(mphone))

    phone_altitude  = get_col(sequence.meta_input, 'alt')
    phone_latitude  = get_col(sequence.meta_input, 'lat')
    phone_longitude = get_col(sequence.meta_input, 'long')
    phone_pitch     = get_col(pitch_data, '*pitch')
    phone_roll      = get_col(roll_data, '*roll')
    phone_heading   = get_col(heading_data, '*heading')

    # print(phone_altitude .shape)
    # print(phone_latitude .shape)
    # print(phone_longitude.shape)
    # print(phone_pitch    .shape)
    # print(phone_roll     .shape)
    # print(phone_heading  .shape)

    anime = Animation(
        anime_data={
            'phone': phone[0].upper() + phone[1:].lower(),
            'g1000_altitude' : g1000_altitude ,
            'g1000_latitude' : g1000_latitude ,
            'g1000_longitude': g1000_longitude,
            'g1000_pitch'    : g1000_pitch    ,
            'g1000_roll'     : g1000_roll     ,
            'g1000_heading'  : g1000_heading  ,
            'phone_altitude' : phone_altitude ,
            'phone_latitude' : phone_latitude ,
            'phone_longitude': phone_longitude,
            'phone_pitch'    : phone_pitch    ,
            'phone_roll'     : phone_roll     ,
            'phone_heading'  : phone_heading  ,
        },
        ptr=args['--start'])
    anime.render(rate=1000 // args['--rate'])

    # Pay attention that using this part will influence plotting font size etc.
    # pitch_data.plot('./anime/{}.pitch'.format(phone),
    #                 prediction_label="{} + Neural Network (Prediction)".format(
    #                                     phone[0].upper() + phone[1:].lower()),
    #                 target_label="G1000 (Target)")
    # roll_data.plot('./anime/{}.roll'.format(phone),
    #                prediction_label="{} + Neural Network (Prediction)".format(
    #                                     phone[0].upper() + phone[1:].lower()),
    #                target_label="G1000 (Target)")
    # heading_data.plot('./anime/{}.heading'.format(phone),
    #                   prediction_label="{} + Neural Network (Prediction)".format(
    #                                     phone[0].upper() + phone[1:].lower()),
    #                   target_label="G1000 (Target)")

if __name__ == '__main__':
    main()