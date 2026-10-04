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