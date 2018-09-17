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
"""Summer Time Period

Set summer time period for stratux data to shift time zone.

"""
SUMMER = {
    2017: ((3, 12), (11, 5)),
    2018: ((3, 11), (11, 4)),
    2019: ((3, 10), (11, 3)),
}

"""Minimum Ground Speed

Set minimum ground speed, above which will be regarded as flight.
The default class embedded value is 0.0005, and is empirically designed.

"""
GROUND_SPEED_THRESHOLD = 0.0004


"""Problematic Flights

Set problematic flights which are provided commonly, but in trouble. For instance,

1. There are some flights provided in G1000 records which are not real flight data.

2. Some time stratux will run out of battery, and give partial flight data which will
crash alignment process.

"""
HIZARD_FLIGHTS = ('112017_02', '112817_01', '022718_01', '030418_03')


"""Input Keywords

Necessary keywords of input in short version.
By current, it is defined by five types:

1. GPS
2. Speed of GPS (Not divided by time step)
3. Acceleration of GPS (Not divided by time step)
4. Accelerometer
5. Gyroscope

"""
INPUT_KEYWORDS = (
    'alt', 'lat', 'long',
    'spd_alt', 'spd_lat', 'spd_long',
    'acc_alt', 'acc_lat', 'acc_long',
    'accmx', 'accmy', 'accmz',
    'gyrox', 'gyroy', 'gyroz',
)


"""Window Configuration

Window configuration for dividing sequences into frames.
It should contain enough information for generating frames. For instance,

1. It should have window length for input and target;
2. It should have information to compute window offset length for input and target;
3. It should have padding method for input and target.

"""
WINDOW_CONFIG = {
    'input': {
        'length': 32,
        'offset_length': None, 'offset_rate': 0.4,
        'padding': 'repeat_base',
    },
    'target': {
        'length': 32,
        'offset_length': None, 'offset_rate': 0.4,
        'padding': 'repeat_base',
    },
}