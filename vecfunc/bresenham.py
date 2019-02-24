"""
N-D Bresenham line algo
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
BRESENHAM'S LINE ALGORITHM IN N-DIMENSIONS (PYTHON RECIPE)
Given start and end point, produce a list of points through which line (or ray) will traverse.

Author: Vikas Dhiman
Edited: Liran Funaro <liran.funaro@gmail.com>

Taken from:
http://code.activestate.com/recipes/578112-bresenhams-line-algorithm-in-n-dimensions/

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


def _bresenham_line_slope(slope):
    """
    Normalize slope for Bresenham's line algorithm.
    >>> s = np.array([[-2, -2, -2, 0]])
    >>> _bresenham_line_slope(s)
    array([[-1., -1., -1.,  0.]])
    >>> s = np.array([[0, 0, 0, 0]])
    >>> _bresenham_line_slope(s)
    array([[ 0.,  0.,  0.,  0.]])
    >>> s = np.array([[0, 0, 9, 0]])
    >>> _bresenham_line_slope(s)
    array([[ 0.,  0.,  1.,  0.]])
    """
    scale = np.amax(np.abs(slope), axis=1).reshape(-1, 1)
    zero_slope = (scale == 0).all(1)
    scale[zero_slope] = np.ones(1)
    normalized_slope = np.array(slope, dtype=np.double) / scale
    normalized_slope[zero_slope] = np.zeros(slope[0].shape)
    return normalized_slope


def _bresenham_lines(start, end, max_iter=None):
    """
    Returns npts lines of length max_iter each. (npts x max_iter x dimension)
    >>> _bresenham_lines([[3, 1, 9, 0], [0, 0, 3, 0]], np.zeros(4), max_iter=None)
    array([[[ 3,  1,  8,  0],
            [ 2,  1,  7,  0],
            [ 2,  1,  6,  0],
            [ 2,  1,  5,  0],
            [ 1,  0,  4,  0],
            [ 1,  0,  3,  0],
            [ 1,  0,  2,  0],
            [ 0,  0,  1,  0],
            [ 0,  0,  0,  0]],
    <BLANKLINE>
           [[ 0,  0,  2,  0],
            [ 0,  0,  1,  0],
            [ 0,  0,  0,  0],
            [ 0,  0, -1,  0],
            [ 0,  0, -2,  0],
            [ 0,  0, -3,  0],
            [ 0,  0, -4,  0],
            [ 0,  0, -5,  0],
            [ 0,  0, -6,  0]]])
    """
    if max_iter is None:
        max_iter = np.amax(np.amax(np.abs(end - start), axis=1))
    npts, dim = start.shape
    nslope = _bresenham_line_slope(end - start)

    # steps to iterate on
    stepseq = np.arange(1, max_iter + 1)
    stepmat = np.tile(stepseq, (dim, 1)).T

    # some hacks for broadcasting properly
    bline = start[:, np.newaxis, :] + nslope[:, np.newaxis, :] * stepmat

    # Approximate to nearest int
    return np.rint(bline).astype(int)


def bresenham_line(start, end, max_iter=None):
    """
    Returns a list of points from (start, end] by ray tracing a line b/w the
    points.
    Parameters:
        start: An array of start points (number of points x dimension)
        end:   An end points (1 x dimension)
            or An array of end point corresponding to each start point
                (number of points x dimension)
        max_iter: Max points to traverse. if -1, maximum number of required
                  points are traversed
    Returns:
        linevox (n x dimension) A cumulative array of all points traversed by
        all the lines so far.
    >>> bresenham_line([[3, 1, 9, 0],[0, 0, 3, 0]], np.zeros(4), max_iter=None)
    array([[ 3,  1,  8,  0],
           [ 2,  1,  7,  0],
           [ 2,  1,  6,  0],
           [ 2,  1,  5,  0],
           [ 1,  0,  4,  0],
           [ 1,  0,  3,  0],
           [ 1,  0,  2,  0],
           [ 0,  0,  1,  0],
           [ 0,  0,  0,  0],
           [ 0,  0,  2,  0],
           [ 0,  0,  1,  0],
           [ 0,  0,  0,  0],
           [ 0,  0, -1,  0],
           [ 0,  0, -2,  0],
           [ 0,  0, -3,  0],
           [ 0,  0, -4,  0],
           [ 0,  0, -5,  0],
           [ 0,  0, -6,  0]])
    """
    start = np.array(start, dtype=int)
    if start.ndim == 1:
        start = np.array([start], dtype=int)
    end = np.array(end, dtype=int)
    # Return the points as a single array
    return _bresenham_lines(start, end, max_iter).reshape(-1, start.shape[-1])
