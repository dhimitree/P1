from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from mpl_toolkits.axesgrid1 import make_axes_locatable

#Parameters/Variables

eps = 8.854e-12
E0 = 100
Ez = -E0
W = 20e-3
L = 20e-3
t_p = 1e-3
gap = 0
d = 5e-6
v = 1
R = 1

#Grid parameters (spacing and overall system placement)
#Values will be changed based on output
h = 0.5e-3
space = 20e-3
H = 50e-3
n_pos = 20 #number of shutter positions (laplace solves)

YS_CROSS_SECTION = 0 #(0 = shutter over plate 1)

FIG_DIR = Path("figures")

TOL = 1e-6

#Creating the grid and geometry functions for the project
#create the grid
def make_grid():
    ny = int(round((2 * W + gap + 2 * space)/h)) + 1
    nz = int(round(H/h)) + 1
    y = -space + h * np.arange(ny)
    z = h * np.arange(nz)
    return y, z

#create and place the plates
def rect_mask(y, z, y0, z0):
    in_y = (y >= y0 - TOL) & (y <= y0+ W + TOL)
    in_z = (z >= z0 - TOL) & (z <= z0 + t_p + TOL)
    return np.outer(in_y, in_z)

def geometry(y, z, ys):
    p1 = rect_mask(y, z, 0, 0)
    p2 = rect_mask(y, z, W + gap, 0)
    if gap == 0:
        p2 &= ~p1
    sh = rect_mask(y, z, ys, t_p +d)
    return p1, p2, sh
