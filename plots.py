import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.colors import LogNorm, SymLogNorm
from numerical import (E0, W, t_p, gap, v, d, h, TOL, FIG_DIR, R, n_pos, make_grid, geometry, solve_laplace, plate_charges, analytical_charges, to_time_domain)

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
    z_stop[(y >= ys - TOL) & (y <= ys + W + TOL)] = 2 * t_p + d
    return np.where(z[:, None] > z_stop[None, :] + TOL, E0, 0.0)

def current(ys, Q1, Q2):
    td = W+ gap
    T = 2 * td / v
    dQ1 = np.gradient(Q1, ys, edge_order=2)
    dQ2 = np.gradient(Q2, ys, edge_order=2)
    t = np.concatenate([ys / v, T / 2 + (td - ys[::-1]) / v])
    I1 = np.concatenate([v * dQ1, -v * dQ1[::-1]])
    I2 = np.concatenate([v * dQ2, -v * dQ2[::-1]])
    return t, I1, I2

def line_plots():
    y, z = make_grid()
    ys = np.unique(np.round(np.linspace(0, W + gap, n_pos) / h) * h)
    Q1 = np.zeros_like(ys)
    Q2 = np.zeros_like(ys)
    for k, pos in enumerate(ys):
        V = solve_laplace(y, z, pos)
        Q1[k], Q2[k] = plate_charges(V, y, z, pos)

    ys_a = np.linspace(0, W + gap, 2001)
    Q1_a, Q2_a = analytical_charges(ys_a)

    t, I1, I2 = current(ys, Q1, Q2)
    t_a, I1_a, I2_a = current(ys_a, Q1_a, Q2_a)
    Vout, Vout_a = R * (I2 - I1), R * (I2_a - I1_a)
    T = 2 * (W + gap) / v

    Q1_t, Q2_t = np.concatenate([Q1, Q1[::-1]]), np.concatenate([Q2, Q2[::-1]])
    Q1_at, Q2_at = np.concatenate([Q1_a, Q1_a[::-1]]), np.concatenate([Q2_a, Q2_a[::-1]])

    plt.figure(figsize=(9, 4))
    plt.plot(t_a * 1e3, Vout_a * 1e3, "k--", label="Analytical")
    plt.plot(t * 1e3, Vout * 1e3, "o-", markersize=3, label="Numerical")
    plt.xlim(0, T * 1e3)
    plt.xlabel("Time (ms)")
    plt.ylabel("$V_{out}$ (mV)")
    plt.title("Transimpedance Amplifier Voltage")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "vout_period.png", dpi=200)

    plt.figure(figsize=(9, 4))
    plt.plot(t_a * 1e3, I1_a * 1e12, "--", color="tab:red", label="$I_1$ analytical")
    plt.plot(t_a * 1e3, I2_a * 1e12, "--", color="tab:orange", label="$I_2$ analytical")
    plt.plot(t * 1e3, I1 * 1e12, "o-", markersize=3, color="tab:red", label="$I_1$ numerical")
    plt.plot(t * 1e3, I2 * 1e12, "o-", markersize=3, color="tab:orange", label="$I_2$ numerical")
    plt.xlim(0, T * 1e3)
    plt.xlabel("Time (ms)")
    plt.ylabel("Current (pA)")
    plt.title("Sense Plate Currents")
    plt.grid(alpha=0.3)
    plt.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.18), frameon=False)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "currents_period.png", dpi=200)

    plt.figure(figsize=(9, 4))
    plt.plot(t_a * 1e3, Q1_at * 1e12, "--", color="tab:red", label="$Q_1$ analytical")
    plt.plot(t_a * 1e3, Q2_at * 1e12, "--", color="tab:orange", label="$Q_2$ analytical")
    plt.plot(t * 1e3, Q1_t * 1e12, "o-", markersize=3, color="tab:red", label="$Q_1$ numerical")
    plt.plot(t * 1e3, Q2_t * 1e12, "o-", markersize=3, color="tab:orange", label="$Q_2$ numerical")
    plt.xlim(0, T * 1e3)
    plt.xlabel("Time (ms)")
    plt.ylabel("Charge (pC)")
    plt.title("Sense Plate Charge")
    plt.grid(alpha=0.3)
    plt.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.18), frameon=False)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "charges_period.png", dpi=200)

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

    line_plots()
    plt.show()

if __name__ == "__main__":
    main()