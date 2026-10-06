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

def draw_plates(ax, ys):
    for y0, z0, color in [(0, 0, "0.8"), (W + gap, 0, "0.8"), (ys, t_p + d, "0.5")]:
        ax.add_patch(Rectangle((y0 * 1e3, z0 * 1e3), W * 1e3, t_p * 1e3,
                               facecolor=color, edgecolor="k", lw=0.8))
    ax.set_xlim(Y_LIM[0] * 1e3, Y_LIM[1] * 1e3)
    ax.set_ylim(Z_LIM[0] * 1e3, Z_LIM[1] * 1e3)
    ax.set_aspect("equal")
    ax.set_ylabel("z (mm)")

def analytical_E(y, z, ys):
    z_stop = np.zeros_like(y)                                   
    z_stop[(y >= -TOL) & (y <= 2 * W + gap + TOL)] = t_p        
    z_stop[(y >= ys - TOL) & (y <= ys + W + TOL)] = 2 * t_p + d 
    return np.where(z[:, None] > z_stop[None, :] + TOL, E0, 0.0)

def main():
    y, z = make_grid()

    fig_n, axs_n = plt.subplots(3, 1, figsize=(10, 7.5), layout="constrained")
    fig_d, axs_d = plt.subplots(3, 1, figsize=(10, 7.5), layout="constrained")


    for (name, ys), ax_n, ax_d in zip(SNAPSHOTS.items(), axs_n, axs_d):
        ys = np.round(ys / h) * h                
        V = solve_laplace(y, z, ys)
        E_num = numerical_E(V)
        dE = E_num - analytical_E(y, z, ys)
        p1, p2, sh = geometry(y, z, ys)
        dE[p1 | p2 | sh] = 0                     
        dE[0, :] = 0                            

        im_n = ax_n.pcolormesh(y * 1e3, z * 1e3, E_num, shading="gouraud", cmap="viridis", vmin=0, vmax=1.5 * E0)
        draw_plates(ax_n, ys)
        im_d = ax_d.pcolormesh(y * 1e3, z * 1e3, dE, shading="gouraud", cmap="viridis", vmin=-E0, vmax=E0)

        for ax in (ax_n, ax_d):
            draw_plates(ax, ys)
            ax.set_title(f"{name} (ys = {ys * 1e3:.0f} mm)")

    plt.show()

if __name__ == "__main__":
    main()