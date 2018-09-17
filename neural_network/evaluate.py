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
"""Evaluate two flight data"""
import docopt
import logging
import copy

import sys
import os
import math

import numpy as np

import flight

# must run on Python 3
assert sys.version_info[0] == 3


def prune_key_id(prune, key):
    """Get the ID of given keyword

    Args
    ----
    prune : flight.FlightPruneData
        dataset to find the keyword
    key : str
        keyword to find column ID

    """
    assert isinstance(prune, flight.FlightPruneData)

    col_id = None
    for i, itr in enumerate(prune.keys):
        if itr == key:
            col_id = i
            break
    assert col_id is not None

    return col_id

def evaluate_label_diff(standard, std_key, observe, obv_key):
    """Evalute the label difference between two dataset on the given column

    Args
    ----
    standard : flight.FlightPruneData
        standard dataset
    std_key : str
        keyword of column of standard dataset
    observe : flight.FlightPruneData
        observed dataset to compare with standard dataset
    obv_key : str
        keyword of column of observed dataset to compare

    """
    assert isinstance(standard, flight.FlightPruneData) and \
           isinstance(observe , flight.FlightPruneData)

    assert standard.identifier == observe.identifier

    std_id = prune_key_id(standard, std_key)
    obv_id = prune_key_id(observe , obv_key)

    global_both_safe      = 0
    global_both_hazard    = 0
    global_safe_but_not   = 0
    global_hazard_but_not = 0

    results = {}

    for i in range(len(standard)):
        std_col  = standard[i][1][:, std_id]
        obv_col  = observe[i][1][:, obv_id]

        std_hazard = set(np.where(std_col >= 0.5)[0])
        std_safe   = set(np.where(std_col < 0.5)[0])
        obv_hazard = set(np.where(obv_col >= 0.5)[0])
        obv_safe   = set(np.where(obv_col < 0.5)[0])

        both_safe = len(std_safe & obv_safe)
        both_hazard = len(std_hazard & obv_hazard)
        safe_but_not = len(std_safe - obv_safe)
        hazard_but_not = len(std_hazard - obv_hazard)

        results[standard.identifier[i]] = (both_hazard, hazard_but_not, safe_but_not, both_safe)
        
        logging.info('{}: \033[34;1m{:4d}\033[0m \033[31;1m{:4d}\033[0m \033[31;1m{:4d}\033[0m \033[32;1m{:4d}\033[0m'.format(
                        standard.identifier[i],
                        both_hazard, hazard_but_not, safe_but_not, both_safe))

        global_both_safe += both_safe
        global_both_hazard += both_hazard
        global_safe_but_not += safe_but_not
        global_hazard_but_not += hazard_but_not

    results['global'] = (global_both_hazard, global_hazard_but_not, global_safe_but_not, global_both_safe)

    logging.info('\033[32;1mStandard ({}) vs Observation ({})\033[0m'.format(std_key, obv_key))
    logging.info('\033[34;1m{:4d}\033[0m \033[31;1m{:4d}\033[0m \033[31;1m{:4d}\033[0m \033[32;1m{:4d}\033[0m'.format(
                    global_both_hazard, global_hazard_but_not, global_safe_but_not, global_both_safe))

    return results

def evaluate_abs_diff(standard, std_key, observe, obv_key):
    """Evalute the absolute difference between two dataset on the given column

    Args
    ----
    standard : flight.FlightPruneData
        standard dataset
    std_key : str
        keyword of column of standard dataset
    observe : flight.FlightPruneData
        observed dataset to compare with standard dataset
    obv_key : str
        keyword of column of observed dataset to compare

    """
    assert isinstance(standard, flight.FlightPruneData) and \
           isinstance(observe , flight.FlightPruneData)

    assert standard.identifier == observe.identifier

    std_id = prune_key_id(standard, std_key)
    obv_id = prune_key_id(observe , obv_key)

    results = {}
    std_all = []
    obv_all = []
    diff_all = []

    for i in range(len(standard)):
        std_col  = standard[i][1][:, std_id]
        obv_col  = observe[i][1][:, obv_id]
        diff_col = np.fabs(std_col - obv_col)

        min_val  = np.min(std_col)
        max_val  = np.max(std_col)
        min_diff = np.min(diff_col)
        max_diff = np.max(diff_col)
        avg_diff = np.average(diff_col)

        results[standard.identifier[i]] = (min_val, max_val, min_diff, max_diff, avg_diff)

        # logging.info('{}: Value between {:.6f} and {:.6f}'
        #              ', Absolute differnce between {:.6f} and {:.6f}'
        #              ' (average: {:.6f})'.format(
        #                 standard.identifier[i],
        #                 min_val, max_val, min_diff, max_diff, avg_diff))

        std_all.append(std_col)
        obv_all.append(obv_col)
        diff_all.append(diff_col)

    std_all = np.concatenate(std_all)
    obv_all = np.concatenate(obv_all)
    diff_all = np.concatenate(diff_all)

    global_min_val = np.min(std_all)
    global_max_val = np.max(std_all)
    global_min_diff = np.min(diff_all)
    global_max_diff = np.max(diff_all)
    global_mean_diff = np.mean(diff_all)
    global_std_diff = np.std(diff_all)

    results['global'] = (
        global_min_val, global_max_val, global_min_diff, global_max_diff,
        global_mean_diff, global_std_diff)

    logging.info('\033[32;1mStandard ({}) vs Observation ({})\033[0m'.format(std_key, obv_key))
    logging.info('Value between {:.4f} and {:.4f}'
                 ', Absolute differnce between {:.4f} and {:.4f}'
                 ' (average: {:.4f}, {:.4f})'.format(
                    global_min_val , global_max_val,
                    global_min_diff, global_max_diff,
                    global_mean_diff, global_std_diff))

    return results