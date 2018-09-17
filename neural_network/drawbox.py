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
"""Draw plots"""
import os
import math

import numpy as np

import pandas as pd

# forbid GUI
import matplotlib as mpl
mpl.use('Agg')

import matplotlib.pyplot as plt
import seaborn as sns


# /**
#  * Multi Plots Container
#  */
class EqualGridFigure(object):
    """Distribute all axes with similar attributes with grids of same size

    It can have multi axes in one figure, but all axes will have the same height and width,
    so they should describe data of similar attribute.

    """
    def __init__(self, num_rows, num_cols, ax_height, ax_width, title=None, font_size=None):
        super(EqualGridFigure, self).__init__()

        # global settings
        if not font_size is None: mpl.rcParams.update({'font.size': font_size})

        # figure size
        self.num_rows = num_rows
        self.num_cols = num_cols
        self.fig, self.axes = plt.subplots(
            num_rows, num_cols,
            figsize=(ax_width * num_cols, max(ax_height * num_rows, 20)))

        # figure title
        if not title is None:
            if font_size is None: self.fig.suptitle(title)
            else: self.fig.suptitle(title, fontsize=font_size * 2)

        # buffer for each plot
        self.cnt = [[0 for cl in range(self.num_cols)] for rw in range(self.num_rows)]

    def __getitem__(self, idx_pair):
        assert len(idx_pair) == 2

        row_id, col_id = idx_pair

        if self.num_rows == 1:
            if self.num_cols == 1: return self.axes
            else: return self.axes[col_id]
        else:
            if self.num_cols == 1: return self.axes[row_id]
            else: return self.axes[row_id][col_id]

    def close(self):
        plt.close(self.fig)

    def subtitle(self, row_id, col_id, subtitle,
                 x_label=None, y_label=None):
        """Set ax title

        Args
        ----
        row_id, col_id : int
            indices to specify the exact ax
        subtitle : str
            title for specified ax
        x_label : str
            title for x-axis
        y_label : str
            title for y-axis

        """
        ax = self[row_id, col_id]
        ax.set_title(subtitle)
        if x_label is not None: ax.set_xlabel(x_label)
        if y_label is not None: ax.set_ylabel(y_label)

    def save_fig(self, path, close=True):
        """Save figure of all axes

        Args
        ----
        path : str
            path to save the figure
        close : bool
            if close figure after saving

        """
        dirname = os.path.dirname(path)
        _, extname = os.path.splitext(path)
        extname = extname[1:]

        if not os.path.isdir(dirname): os.makedirs(dirname)
        self.fig.savefig(path, format=extname, dpi='figure')

        if close: self.close()

    def lineplot(self, row_id, col_id, x_data, y_data,
                 label=None, color=None, alpha=None, marker=None, linestyle=None,
                 vmin=None, vmax=None):
        """Line plot

        Args
        ----
        row_id, col_id : int
            indices to specify the exact ax
        x_data, y_data : <1D-array-like>
            data for x-axis and y-axis
        label : str
            label of data
        color : str
            specify color to plot
        marker : str
            point style to plot
        linestyle : str
            line style to plot
        vmin : float
            min value of data to plot
        vmax : float
            max value of data to plot

        Draw a line in ax (row_id, col_id).
        It can draw a line on ax which already has something.

        """
        ax = self[row_id, col_id]

        # settings
        label = label or 'Line {}'.format(self.cnt[row_id][col_id])
        color = color or 'C{}'.format(self.cnt[row_id][col_id])
        alpha = alpha or 1
        marker = marker or ','
        linestyle = linestyle or '-'

        ax.plot(x_data, y_data, label=label, \
                color=color, alpha=alpha, \
                marker=marker, \
                linewidth=6.0, linestyle=linestyle)
        if vmin: ax.set_ylim(ymin=vmin)
        if vmax: ax.set_ylim(ymax=vmax)

        self.cnt[row_id][col_id] += 1

    def legend(self):
        """Place legend"""
        for rw in range(self.num_rows):
            for cl in range(self.num_cols):
                ax = self[rw, cl]
                if self.cnt[rw][cl] > 1: ax.legend()
