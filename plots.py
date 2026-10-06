import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.colors import LogNorm, SymLogNorm
from numerical import (E0, W, t_p, gap, v, d, h, TOL, FIG_DIR, make_grid, geometry, solve_laplace)

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
    z_stop[(y >= -TOL) & (y <= W + TOL)] = t_p        
    z_stop[(y >= W + gap - TOL) & (y <= 2 * W + gap + TOL)] = t_p
    return np.where(z[:, None] > z_stop[None, :] + TOL, E0, 0.0)

def current(ys, Q1, Q2):
    td = W+ gap
    T = 2 * t_p / d
    dQ1 = np.gradient(Q1, ys, edge_order=2)
    dQ2 = np.gradient(Q2, ys, edge_order=2)
    t = np.concatenate([ys / v, T / 2 + (td - ys[::-1]) / v])
    I1 = np.concatenate([v * dQ1, -v * dQ1[::-1]])
    I2 = np.concatenate([v * dQ2, -v * dQ2[::-1]])
    return t, I1, I2

def main(): 
    FIG_DIR.mkdir(exist_ok=True)
    y, z = make_grid()

    fig_n, axs_n = plt.subplots(3, 1, figsize=(10, 7.5), layout="constrained")
    fig_d, axs_d = plt.subplots(3, 1, figsize=(10, 7.5), layout="constrained")
    fig_nl, axs_nl = plt.subplots(3, 1, figsize=(10, 7.5), layout="constrained")
    fig_dl, axs_dl = plt.subplots(3, 1, figsize=(10, 7.5), layout="constrained")

    for (name, ys), ax_n, ax_d, ax_nl, ax_dl in zip(SNAPSHOTS.items(), axs_n, axs_d, axs_nl, axs_dl):
        ys = np.round(ys / h) * h                
        V = solve_laplace(y, z, ys)
        E_num = numerical_E(V)
        dE = E_num - analytical_E(y, z, ys)
        p1, p2, sh = geometry(y, z, ys)
        dE[p1 | p2 | sh] = 0                     
        dE[0, :] = 0                            

        im_n = ax_n.pcolormesh(y * 1e3, z * 1e3, E_num, shading="gouraud", cmap="viridis", vmin=0, vmax=1.5 * E0)
        im_d = ax_d.pcolormesh(y * 1e3, z * 1e3, dE, shading="gouraud", cmap="viridis", vmin=-E0, vmax=E0)

        im_nl = ax_nl.pcolormesh(y * 1e3, z * 1e3, np.clip(E_num,0.1, None), shading="gouraud", cmap="viridis", norm=LogNorm(vmin=0.1, vmax=1.5 * E0))
        im_dl = ax_dl.pcolormesh(y * 1e3, z * 1e3, dE, shading="gouraud", cmap="viridis", norm=SymLogNorm(linthresh=1 * E0, vmin=-E0, vmax=E0))

        for ax in (ax_n, ax_d, ax_nl, ax_dl):
            draw_plates(ax, ys)
            ax.set_title(f"{name} (ys = {ys * 1e3:.0f} mm)")
    
    axs_n[-1].set_xlabel("y (mm)")
    axs_d[-1].set_xlabel("y (mm)")
    axs_nl[-1].set_xlabel("y (mm)")
    axs_dl[-1].set_xlabel("y (mm)")
    fig_n.colorbar(im_n, ax=axs_n, label="|E| numerical (V/m)")
    fig_d.colorbar(im_d, ax=axs_d, label="|E| numerical − |E| analytical (V/m)")
    fig_nl.colorbar(im_nl, ax=axs_nl, label="|E| numerical (V/m)")
    fig_dl.colorbar(im_dl, ax=axs_dl, label="|E| numerical − |E| analytical (V/m)")
    fig_n.suptitle("Numerical E-field magnitude")
    fig_d.suptitle("Difference: numerical − analytical")
    fig_nl.suptitle("Numerical E-field magnitude (log scale)")
    fig_dl.suptitle("Difference: numerical − analytical (symmetric log scale)")

    fig_n.savefig(FIG_DIR / "E_numerical_snapshots.png", dpi=200)
    fig_d.savefig(FIG_DIR / "E_difference_snapshots.png", dpi=200)
    fig_nl.savefig(FIG_DIR / "E_numerical_log.png", dpi=200)
    fig_dl.savefig(FIG_DIR / "E_difference_log.png", dpi=200)
    plt.show()

if __name__ == "__main__":
    main()