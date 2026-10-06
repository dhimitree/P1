import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.colors import LogNorm, SymLogNorm
from numerical import (E0, W, t_p, gap, d, h, TOL, FIG_DIR, make_grid, geometry, solve_laplace)

SNAPSHOTS = {
    "Shutter over left plate": 0.0,
    "Shutter in the middle": W / 2,
    "Shutter over right plate": W + gap,
}

Y_LIM = (-8e-3, 2 * W + gap + 8e-3)
Z_LIM = (0.0, 2 * t_p + d + 8e-3)

def numerical_E(V):
    dVdz, dVdy = np.gradient(V, h, h)
    return np.hypot(dVdy, dVdz)

def analytical_E(y, z, ys):
    z_stop = np.zeros_like(y)                                   
    z_stop[(y >= -TOL) & (y <= 2 * W + gap + TOL)] = t_p        
    z_stop[(y >= ys - TOL) & (y <= ys + W + TOL)] = 2 * t_p + d 
    return np.where(z[:, None] > z_stop[None, :] + TOL, E0, 0.0)

def main():
    y, z = make_grid()

    V = solve_laplace(y, z, 0.0)     
    E = numerical_E(V)

    E_ana = analytical_E(y, z, 0.0)

    print("values:", np.unique(E_ana))
    plt.imshow(E_ana, origin="lower")
    plt.colorbar(label="|E| (V/m)")
    plt.title("Test: analytical |E|")
    plt.show()

    print("max |E|:", E.max())
    plt.imshow(E, origin="lower", vmax=150)
    plt.colorbar(label="|E| (V/m)")
    plt.title("Test: numerical |E|")
    plt.show()

if __name__ == "__main__":
    main()