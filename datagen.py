# -*- coding: utf-8 -*-
"""
Created on Wed Feb  9 08:15:59 2022

@author: mlsol
"""
import torch as t
import numpy as np
from numpy import linalg as la
from matplotlib.collections import LineCollection

wa = [.157 + .148j, .60 + .170j, .30 + .120j]  # (zs) the complex point zp[i]
zp = [.147 + .143j, .90 + .040j, .30 + .120j]  # (wa) will be in wa[i]

# transformation parameters
a = la.det([[zp[0] * wa[0], wa[0], 1],
                [zp[1] * wa[1], wa[1], 1],
                [zp[2] * wa[2], wa[2], 1]]);

b = la.det([[zp[0] * wa[0], zp[0], wa[0]],
                [zp[1] * wa[1], zp[1], wa[1]],
                [zp[2] * wa[2], zp[2], wa[2]]]);

c = la.det([[zp[0], wa[0], 1],
                [zp[1], wa[1], 1],
                [zp[2], wa[2], 1]]);

d = la.det([[zp[0] * wa[0], zp[0], 1],
                [zp[1] * wa[1], zp[1], 1],
                [zp[2] * wa[2], zp[2], 1]]);

def generate_mobius_grid(f, upper, lower, res=20):
    grid_x, grid_y = np.meshgrid(np.linspace(lower, upper, res), np.linspace(lower, upper, res))
    z = np.stack((grid_x, grid_y), axis=-1)
    z = z[..., 0] + 1j * z[..., 1]
    fz = f(z, a, b, c, d)
    return z, fz, grid_x, grid_y


def plot_grid(x, y, ax, **kwargs):
    segs1 = np.stack((x, y), axis=2)
    segs2 = segs1.transpose(1, 0, 2)
    ax.add_collection(LineCollection(segs1, **kwargs))
    ax.add_collection(LineCollection(segs2, **kwargs))
    ax.autoscale()
    return ax


def plot_mobius_tranform_comparison(Y_true, Y_pred, grid_shape, ax, figsize, **kwargs):
  
  x = Y_true.numpy().reshape(grid_shape).real
  y = Y_true.numpy().reshape(grid_shape).imag
  plot_grid(x, y, ax=ax, color='r', alpha=.4, **kwargs)

  x = Y_pred.numpy().reshape(grid_shape).real
  y = Y_pred.numpy().reshape(grid_shape).imag
  plot_grid(x, y, ax=ax, **kwargs)
  
  return ax
  