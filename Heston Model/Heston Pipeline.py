"""
Heston_Pipeline.py
==================
Static 3D implied-volatility surface of the Heston Stochastic Volatility model.

Single full-screen 3D implied-volatility surface  IV(K, T)  
as the instantaneous variance v_t evolves through the CIR dynamics:

    dS_t = r S_t dt + sqrt(v_t) S_t dW^1_t
    dv_t = kappa (theta - v_t) dt + xi sqrt(v_t) dW^2_t
    corr(dW^1, dW^2) = rho

We simulate v_t under the CIR SDE (full-truncation scheme), then compute the
implied-vol surface for the final state using a Heston-inspired parametric
form that captures the ATM variance term structure + skew + smile.

Pipeline: SIMULATE v_t  ->  BUILD IV SURFACE  ->  RENDER
Resolution: 1920x1080
Dependencies: pip install numpy matplotlib
"""

import os, time, warnings
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

warnings.filterwarnings("ignore")

# ═══════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════
CONFIG = {
    "RESOLUTION": (1920, 1080),
    "OUTPUT_FILE": "Heston_Output.png",
    "LOG_FILE": "heston_pipeline.log",
}

# Heston model parameters (equity-like)
HESTON = {
    "S0":     100.0,
    "r":      0.02,
    "v0":     0.040,     # initial variance (20% vol)
    "kappa":  2.5,       # mean reversion speed
    "theta":  0.070,     # long-run variance (26.5% vol)
    "xi":     0.75,      # vol of vol
    "rho":    -0.72,     # leverage effect (negative for equities)
    "T_horizon": 1.5,    # 1.5 years of variance path
    "dt_sim": 0.002,
    "n_snaps": 140,      # variance path points for plotting
    "seed": 7,

    # IV surface grid
    "n_K":   55,
    "n_T":   55,
    "K_lo":  0.60,   # fraction of S0
    "K_hi":  1.45,
    "T_lo":  0.05,
    "T_hi":  1.00,
}

# ─── THEME ────────────────────────────────────────────────────────
THEME = {
    "BG":         "#000000",
    "PANEL_BG":   "#0a0a0a",
    "GRID":       "#222222",
    "TEXT":       "#ffffff",
    "TEXT_DIM":   "#aaaaaa",
    "ORANGE":     "#ff9500",
    "ORANGE_HOT": "#ff6b00",
    "YELLOW":     "#ffd400",
    "PINK":       "#ff2a9e",
    "RED":        "#ff3050",
    "CYAN":       "#00f2ff",
    "FONT":       "Arial",
}

# magma — dark purple -> red -> orange -> pink-yellow
CMAP = cm.get_cmap("magma")


# ═══════════════════════════════════════════════════════════════════
# UTILITIES
# ═══════════════════════════════════════════════════════════════════

def log(msg):
    ts = time.strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line)
    try:
        with open(CONFIG["LOG_FILE"], "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        pass





# ═══════════════════════════════════════════════════════════════════
# MODULE 1 — VARIANCE PATH + IV SURFACE
# ═══════════════════════════════════════════════════════════════════

def simulate_variance_path():
    """CIR (Cox-Ingersoll-Ross) path for instantaneous variance v_t."""
    p = HESTON
    rng = np.random.default_rng(p["seed"])
    n = int(p["T_horizon"] / p["dt_sim"])
    dt = p["dt_sim"]

    v = np.zeros(n + 1)
    v[0] = p["v0"]
    # full-truncation Euler
    for i in range(n):
        v_plus = max(v[i], 0.0)
        dw = rng.standard_normal() * np.sqrt(dt)
        v[i + 1] = v[i] + p["kappa"] * (p["theta"] - v_plus) * dt \
                   + p["xi"] * np.sqrt(v_plus) * dw
    v = np.maximum(v, 1e-6)

    # downsample to snapshot count
    idx = np.linspace(0, n, p["n_snaps"]).astype(int)
    return idx * dt, v[idx]


def build_iv_surface(v_t):
    """Heston-inspired implied-vol surface for a given instantaneous variance.

    Captures:
      - mean-reverting term structure of variance (exact Heston form)
      - skew ~ rho * xi (negative for equities -> left skew)
      - smile (quadratic in log-moneyness, scaled by xi^2)
    """
    p = HESTON
    S0    = p["S0"]
    kappa = p["kappa"]
    theta = p["theta"]
    xi    = p["xi"]
    rho   = p["rho"]

    K = np.linspace(p["K_lo"] * S0, p["K_hi"] * S0, p["n_K"])
    T = np.linspace(p["T_lo"],      p["T_hi"],      p["n_T"])
    K_g, T_g = np.meshgrid(K, T, indexing="xy")
    k = np.log(K_g / S0)

    # ATM integrated variance  w_atm(T) = theta*T + (v_t - theta)*(1 - e^{-kT})/k
    w_atm = theta * T_g + (v_t - theta) * (1 - np.exp(-kappa * T_g)) / kappa
    w_atm = np.maximum(w_atm, 1e-6)
    atm_iv = np.sqrt(w_atm / T_g)

    # skew and smile (Heston-inspired approximations)
    sqrtT = np.sqrt(T_g + 0.04)
    skew  = rho * xi * 0.85 / sqrtT
    smile = xi * xi * 0.20 / sqrtT

    iv = atm_iv * (1.0 + skew * k + smile * k * k)
    iv = np.clip(iv, 0.05, 1.2)
    return K, T, iv


# ═══════════════════════════════════════════════════════════════════
# MODULE 2 — RENDERING
# ═══════════════════════════════════════════════════════════════════

def render_image(K_arr, T_arr, IV, v_path, t_sim, out_path):
    try:
        elev, azim, dist, salpha = 28, -50, 1.1, 0.95
        v_now = v_path[-1]
        t_now = t_sim[-1]

        # ── figure ────────────────────────────────────────────────
        fig = plt.figure(figsize=(19.2, 10.8), dpi=100, facecolor=THEME["BG"])
        ax = fig.add_axes([0.02, 0.02, 0.96, 0.83],
                          projection="3d", computed_zorder=False)
        ax.set_facecolor(THEME["BG"])

        pane = (0.02, 0.02, 0.02, 1)
        ax.xaxis.set_pane_color(pane)
        ax.yaxis.set_pane_color(pane)
        ax.zaxis.set_pane_color(pane)
        for a in (ax.xaxis, ax.yaxis, ax.zaxis):
            a._axinfo["grid"]["color"]     = (0.13, 0.13, 0.13, 0.6)
            a._axinfo["grid"]["linewidth"] = 0.4

        K_g, T_g = np.meshgrid(K_arr, T_arr, indexing="xy")

        # main surface
        ax.plot_surface(
            K_g, T_g, IV,
            cmap=CMAP, alpha=salpha,
            rstride=1, cstride=1,
            edgecolor=(1.0, 0.30, 0.55, 0.18),   # hot pink wireframe
            linewidth=0.28,
            antialiased=True, zorder=1,
        )

        # floor heatmap shadow
        z_floor = max(0.02, IV.min() - 0.08)
        try:
            ax.contourf(K_g, T_g, IV, zdir="z", offset=z_floor,
                        cmap=CMAP, alpha=0.32, levels=14)
        except Exception:
            pass

        # ATM ridge line (K = S0 slice across all T)
        S0 = HESTON["S0"]
        atm_col = np.argmin(np.abs(K_arr - S0))
        ax.plot([S0] * len(T_arr), T_arr, IV[:, atm_col],
                color=THEME["YELLOW"], lw=3.2, alpha=1.0, zorder=15)

        # Front-of-surface (shortest expiry) — the live "smile"
        ax.plot(K_arr, np.full_like(K_arr, T_arr[0]), IV[0, :],
                color=THEME["ORANGE_HOT"], lw=3.2, alpha=1.0, zorder=14)

        # Strike marker on the floor
        ax.plot([S0, S0], [T_arr[0], T_arr[-1]], [z_floor, z_floor],
                color=THEME["RED"], lw=1.3, alpha=0.6, ls="--", zorder=2)

        # axes styling
        ax.set_xlabel("STRIKE  K", fontsize=12, fontweight="bold",
                      color=THEME["TEXT_DIM"], labelpad=14, fontfamily=THEME["FONT"])
        ax.set_ylabel("EXPIRY  T", fontsize=12, fontweight="bold",
                      color=THEME["TEXT_DIM"], labelpad=14, fontfamily=THEME["FONT"])
        ax.set_zlabel(r"$\sigma_{imp}$", fontsize=13, fontweight="bold",
                      color=THEME["TEXT_DIM"], labelpad=12, fontfamily=THEME["FONT"])
        ax.tick_params(axis="both", colors=THEME["TEXT_DIM"], labelsize=9)

        x_lo = K_arr[0]
        x_hi = K_arr[-1]
        y_hi = T_arr[-1]

        ax.set_xlim(x_lo, x_hi)
        ax.set_ylim(T_arr[0], y_hi)
        ax.set_zlim(z_floor, float(IV.max()) * 1.10)
        ax.set_box_aspect([1.6 * dist, 1.0 * dist, 0.9 * dist])
        ax.view_init(elev=elev, azim=azim)

        # ── small v_t inset (bottom-left) ─────────────────────────
        ax_ins = fig.add_axes([0.045, 0.06, 0.22, 0.16])
        ax_ins.set_facecolor("#0a0a0a")
        for sp in ax_ins.spines.values():
            sp.set_color("#333333"); sp.set_linewidth(0.5)
        ax_ins.tick_params(axis="both", colors=THEME["TEXT_DIM"],
                           labelsize=7, direction="in", length=3)
        ax_ins.yaxis.grid(True, lw=0.3, alpha=0.4, color="#1f1f1f")
        sqrt_v = np.sqrt(v_path)
        ax_ins.fill_between(t_sim, 0, sqrt_v,
                            color=THEME["PINK"], alpha=0.18)
        ax_ins.plot(t_sim, sqrt_v,
                    color=THEME["PINK"], lw=1.2)
        ax_ins.axhline(np.sqrt(HESTON["theta"]), color=THEME["YELLOW"],
                       lw=0.8, ls="--", alpha=0.75)
        ax_ins.scatter([t_sim[-1]], [sqrt_v[-1]],
                       s=28, color=THEME["YELLOW"], edgecolor="white",
                       linewidth=0.5, zorder=10)
        ax_ins.set_xlim(0, t_sim[-1])
        ax_ins.set_ylim(0, max(sqrt_v.max() * 1.1, 0.4))
        ax_ins.set_title(r"$\sqrt{v_t}$  (instantaneous vol)",
                         fontsize=8, color=THEME["TEXT_DIM"],
                         fontfamily=THEME["FONT"], loc="left", pad=2)

        # ── Title bar ─────────────────────────────────────────────
        fig.text(0.50, 0.955, "THE HESTON MODEL",
                 ha="center", va="center", fontsize=30, fontweight="bold",
                 color=THEME["ORANGE"], fontfamily=THEME["FONT"])
        fig.text(0.50, 0.918,
                 r"$dS_t = r S_t\, dt + \sqrt{v_t}\, S_t\, dW^{1}_t$"
                 "          "
                 r"$dv_t = \kappa(\theta - v_t)\, dt + \xi\sqrt{v_t}\, dW^{2}_t$",
                 ha="center", va="center", fontsize=14,
                 color=THEME["TEXT"], fontfamily=THEME["FONT"])
        fig.text(0.50, 0.886,
                 r"STOCHASTIC VOLATILITY    "
                 r"$\kappa = 2.5$    "
                 r"$\theta = 0.07$    "
                 r"$\xi = 0.75$    "
                 r"$\rho = -0.72$",
                 ha="center", va="center", fontsize=11, fontweight="bold",
                 color=THEME["TEXT_DIM"], fontfamily=THEME["FONT"])

        # ── HUD ───────────────────────────────────────────────────
        atm_iv = IV[0, atm_col]
        fig.text(0.97, 0.89,
                 f"t = {t_now:4.2f}y    "
                 f"v_t = {v_now:5.3f}    "
                 f"ATM IV = {atm_iv:5.1%}",
                 ha="right", va="center", fontsize=12, fontweight="bold",
                 color=THEME["YELLOW"], fontfamily=THEME["FONT"])
        fig.text(0.28, 0.89,
                 "YELLOW = ATM RIDGE\nORANGE = FRONT SMILE",
                 ha="left", va="center", fontsize=9, fontweight="bold",
                 color=THEME["TEXT_DIM"], fontfamily=THEME["FONT"])

        # ── Footer ────────────────────────────────────────────────
        fig.text(0.98, 0.012, "@quant.traderr",
                 ha="right", va="bottom", fontsize=10,
                 color=THEME["TEXT_DIM"], fontfamily=THEME["FONT"], alpha=0.6)

        fig.savefig(out_path, dpi=100, facecolor=THEME["BG"])
        plt.close(fig)
        log(f"Saved image to: {out_path}")
        return True

    except Exception:
        import traceback
        traceback.print_exc()
        return False


# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════

def main():
    t0 = time.time()
    log("=" * 60)
    log("Heston Pipeline START")
    log("=" * 60)

    log("Simulating CIR variance path v_t ...")
    t_sim, v_path = simulate_variance_path()
    log(f"  {len(v_path)} snapshots   v range = [{v_path.min():.4f}, {v_path.max():.4f}]")

    log("Building Heston-parametric IV surface for the final snapshot ...")
    K_arr, T_arr, IV = build_iv_surface(v_path[-1])
    log(f"  IV surface built (range {IV.min():.3f} - {IV.max():.3f})")

    log("Rendering static image ...")
    render_image(K_arr, T_arr, IV, v_path, t_sim, CONFIG["OUTPUT_FILE"])
    log(f"Pipeline complete in {time.time() - t0:.1f}s")
    log("=" * 60)


if __name__ == "__main__":
    main()
