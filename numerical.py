from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from mpl_toolkits.axes_grid1 import make_axes_locatable

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

#Laplace solver
NEIGHBORS = [[0, 1], [0, -1], [1, 0], [-1, 0]] #4 NEIGHBOIRS for 2D sim

def neighbor_index(jj, ii, dj, di, ny):
    nj = jj + dj
    ni = ii + di
    ni = np.where(ni < 0, 1, ni)
    ni = np.where(ni > ny - 1, ny -2, ni)
    return nj, ni

def solve_laplace(y, z, ys):
    p1, p2, sh = geometry(y, z, ys)
    return solve_with_conductors(y, z, p1 | p2 | sh)

def solve_with_conductors(y, z, conductors):
    nz, ny = len(z), len(y)

    fixed = conductors.copy()
    fixed[0, :] = True
    fixed[-1, :] = True
    v_fix = np.zeros((nz, ny))
    v_fix[-1, :] = -Ez * H

    free = ~fixed
    jj, ii = np.nonzero(free)
    n = jj.size
    idx = np.full((nz,ny), -1, dtype=np.int64)
    idx[jj, ii] = np.arange(n)

    rows = [np.arange(n)]
    cols = [np.arange(n)]
    vals = [4 * np.ones(n)]
    b = np.zeros(n)

    for dj, di in NEIGHBORS:
        nj, ni = neighbor_index(jj, ii, dj, di, ny)
        nb_free = free[nj, ni]
        rows.append(np.nonzero(nb_free)[0])
        cols.append(idx[nj[nb_free], ni[nb_free]])
        vals.append(-1 * np.ones(nb_free.sum()))
        np.add.at(b, np.nonzero(-nb_free)[0], v_fix[nj[~nb_free], ni[~nb_free]])

    A = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(n, n))
    V= v_fix.copy()
    V[jj, ii] = spla.spsolve(A, b)
    return V

def conductor_charge(V, mask):
    ny = V.shape[1]
    jj, ii = np.nonzero(mask)
    q = 0
    for dj, di in NEIGHBORS:
        nj, ni = neighbor_index(jj, ii, dj, di, ny)
        q += np.sum(V[jj[ok], ii[ok]] - V[nj[ok], ni[ok]])
    return eps * q * L

def plate_charges(V, y, z, ys):
    p1, p2, _ = geometry(y, z, ys)
    return conductor_charge(V, p1), conductor_charge(V, p2)

#Analytical model from Part 1
#Exposed area of plate spans from point a to b with the shuter spanning (ys, ys + W) as ys changes as plate slides.

def exposed_area(ys, a, b):
    overlap = np.clip(np.minimum(ys + W, b) - np.maximum(ys, a), 0, None)
    return L * ((b - a) - overlap)

#application of Gauss' Law for the exposed area then solved for charge Q.
def analytical_charges(ys):
    Q1 = eps * Ez * exposed_area(ys, 0, W)
    Q2 = eps * Ez * exposed_area(ys, W + gap, 2 * W + gap)
    return Q1, Q2