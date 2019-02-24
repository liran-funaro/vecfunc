"""
Plot a vecfunc in various plot types according to the vecfunc dimension.

Author: Liran Funaro <liran.funaro@gmail.com>

Copyright (C) 2006-2018 Liran Funaro

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""
import numpy as np
from matplotlib import pylab as plt
import seaborn as sns

# noinspection PyUnresolvedReferences
from mpl_toolkits.mplot3d import Axes3D


SURF_COLORS = [
    [0.36392157, 0.34039216, 0.65176472, 1.],
    [0.82588236, 0.21725491, 0.25333333, 1.],
    [0.0, 0.5, 0.58509806, 1.],
    [0.59909266, 0.70385237, 0.64633605, 1.],
    [0.87892349, 0.04370627, 0.51461746, 1.],
    [0.49803922, 0.78823531, 0.49803922, 1.],
    [0.63952327, 0.37039601, 0.18738947, 1.],
    [0.2, 0.5, 0, 1.],
    [0.6675894, 0.71557095, 0.72679739, 1.],
    [0.1, 0.1, 0.5, 1.],
    [0.5, 0, 0, 1.],
    [0.9, 0.9, 0.1, 1.],
    [0, 0, 0, 1.],
    [0.99, 0.1, 0.1, 1.],
    [0.1, 0.2, 0.2, 1.],
]


def visualize_vector(vec, *args, force_wire_2d=False, **kwargs):
    if vec.ndim == 1:
        return visualize_1d_vector(vec, *args, **kwargs)
    elif vec.ndim == 2 and not force_wire_2d:
        return visualize_2d_vector(vec, *args, **kwargs)
    elif vec.ndim == 2:
        return visualize_2d_vector_wire(vec, *args, **kwargs)
    elif vec.ndim == 3:
        return visualize_3d_vector(vec, *args, **kwargs)
    else:
        raise Exception("Cannot plot vector of ndim: %s" % vec.ndim)


def visualize_1d_vector(vec, d_vals=None, d_keys=None, d_limits=None, val_key=None, val_limits=None, **kwargs):
    if d_vals is None:
        d_vals = np.arange(len(vec)),
    plt.plot(d_vals[0], vec, **kwargs)
    if d_keys is not None:
        plt.xlabel(d_keys[0])
    if d_limits is not None:
        plt.xlim(d_limits[0])
    if val_key is not None:
        plt.ylabel(val_key)
    if val_limits is not None:
        plt.ylim(val_limits)


def visualize_2d_vector(vec, d_vals=None, d_keys=None, d_limits=None, val_key=None, val_limits=None, d_ticks=None,
                        annotate=True, ticks_format=None, fmt='0.2f', mask_zero=False, mask=None, **kwargs):
    if d_vals is None:
        d_vals = np.arange(vec.shape[0]), np.arange(vec.shape[1])

    if val_limits is not None:
        kwargs['vmin'] = val_limits[0]
        kwargs['vmax'] = val_limits[1]

    if mask is None and mask_zero:
        mask = np.isclose(vec, 0)

    if annotate:
        d_vals = np.arange(vec.shape[0])+0.5, np.arange(vec.shape[1])+0.5
        if mask is not None:
            mask = np.transpose(mask)
        sns.heatmap(np.transpose(vec), annot=annotate, cmap='coolwarm', fmt=fmt, linewidths=.05, mask=mask)
    else:
        cmap = plt.cm.coolwarm
        if mask is not None:
            cmap.set_bad(color='white')
            vec = np.ma.masked_where(mask, vec)
        plt.pcolor(d_vals[0], d_vals[1], np.transpose(vec), cmap=cmap, **kwargs)

    if d_keys is not None:
        plt.xlabel(d_keys[0])
        plt.ylabel(d_keys[1])
    if d_limits is not None:
        plt.xlim(d_limits[0])
        plt.ylim(d_limits[1])
    if d_ticks is not None:
        if ticks_format is None:
            ticks_format = lambda d: d
        plt.xticks(d_vals[0], map(ticks_format, d_ticks[0]), rotation='vertical')
        plt.yticks(d_vals[1], map(ticks_format, d_ticks[1]), rotation='horizontal')
    if not annotate:
        cbar = plt.colorbar()
        if val_key is not None:
            cbar.set_label(val_key, rotation=270)


def visualize_2d_vector_wire(vec, d_vals=None, d_keys=None, d_limits=None, val_key=None, val_limits=None, d_ticks=None,
                             view_init=None, **kwargs):
    ax = plt.gca(projection='3d')
    ax.patch.set_facecolor('white')
    ax.w_xaxis.set_pane_color((1, 1, 1, 1.0))
    ax.w_yaxis.set_pane_color((1, 1, 1, 1.0))
    ax.w_zaxis.set_pane_color((1, 1, 1, 1.0))

    if d_vals is None:
        d_vals = np.arange(vec.shape[0]), np.arange(vec.shape[1]),

    sample_mesh = np.meshgrid(np.arange(vec.shape[0]), np.arange(vec.shape[1]), indexing='ij')

    x = d_vals[0]
    y = d_vals[1]
    X, Y = np.meshgrid(x, y, indexing='ij')
    Z = vec[sample_mesh]
    ax.plot_wireframe(X, Y, Z, antialiased=False,  **kwargs)

    if d_keys is not None:
        ax.set_xlabel(d_keys[0])
        ax.set_ylabel(d_keys[1])
    if d_limits is not None:
        ax.set_xlim(d_limits[0])
        ax.set_ylim(d_limits[1])
    if d_ticks is not None:
        ax.set_xticklabels(d_ticks[0])
        ax.set_yticklabels(d_ticks[1])
    if val_key is not None:
        ax.set_zlabel(val_key)
    if val_limits is not None:
        ax.set_zlim(val_limits)

    if view_init is not None:
        ax.view_init(view_init)
    return ax


def visualize_3d_vector(vec, d_vals=None, d_keys=None, d_limits=None, val_key=None, val_limits=None, d_ticks=None,
                        view_init=None, colors=None, is_scatter=False, **kwargs):
    ax = plt.gca(projection='3d')
    ax.patch.set_facecolor('white')
    ax.w_xaxis.set_pane_color((1, 1, 1, 1.0))
    ax.w_yaxis.set_pane_color((1, 1, 1, 1.0))
    ax.w_zaxis.set_pane_color((1, 1, 1, 1.0))

    cur_min_val = float("inf")
    cur_max_val = 0

    if d_vals is None:
        d_vals = np.arange(vec.shape[0]), np.arange(vec.shape[1]), np.arange(vec.shape[1])

    sample_mesh = np.meshgrid(np.arange(vec.shape[1]), np.arange(vec.shape[2]), indexing='ij')

    if colors is None:
        colors = plt.get_cmap("Set1")(np.linspace(0, 1, len(vec)))
    for i, sub_vec in enumerate(vec):
        label = str(d_vals[0][i])
        if d_keys is not None:
            label = "%s %s" % (d_keys[0], label)

        x = d_vals[1]
        y = d_vals[2]
        X, Y = np.meshgrid(x, y, indexing='ij')
        Z = sub_vec[sample_mesh]
        if is_scatter:
            ax.scatter(X, Y, Z, label=label, color=colors[i], **kwargs)
        else:
            ax.plot_wireframe(X, Y, Z, antialiased=False, label=label, color=colors[i], **kwargs)

        cur_min_val = min(cur_min_val, min(min(z) for z in Z))
        cur_max_val = max(cur_max_val, max(max(z) for z in Z))

    if d_keys is not None:
        ax.set_xlabel(d_keys[1])
        ax.set_ylabel(d_keys[2])
    if d_limits is not None:
        ax.set_xlim(d_limits[0])
        ax.set_ylim(d_limits[1])
    if d_ticks is not None:
        ax.set_xticklabels(d_ticks[0])
        ax.set_yticklabels(d_ticks[1])
    if val_key is not None:
        ax.set_zlabel(val_key)
    if val_limits is not None:
        ax.set_zlim(val_limits)

    ax.legend(loc="upper left", fancybox=False, shadow=False, frameon=False)
    if view_init is not None:
        ax.view_init(view_init)
    return ax
