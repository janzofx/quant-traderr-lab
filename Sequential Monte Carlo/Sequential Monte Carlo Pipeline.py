"""
Particle_Filter_Pipeline.py
===========================
Cinematic 3D visualization of a Sequential Monte Carlo (Particle) Filter tracking
the Gordon-Salmond-Smith 1993 nonlinear benchmark:

    x_{t+1} = 0.5 x_t + 25 x_t / (1 + x_t^2) + 8 cos(1.2 t) + v_t,  v_t ~ N(0, Q)
    y_t     = x_t^2 / 20 + w_t,                                     w_t ~ N(0, R)

The quadratic observation makes the posterior BIMODAL (can't distinguish +/- x),
which is exactly where particle filters shine vs Kalman/Gaussian methods.

Left panel  : 3D particle cloud  (x=time, y=particle value, z=weight)
Right panel : 4-stack 2D — State vs posterior · Obs · ESS · Uncertainty

Pipeline: SIMULATE + FILTER -> RENDER
Resolution: 1920x1080
Dependencies: pip install numpy matplotlib
"""

import os, time, warnings
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

warnings.filterwarnings("ignore")

# ═══════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════
CONFIG = {
    "RESOLUTION": (1920, 1080),
    "OUTPUT_FILE": "Particle_Filter_Output.png",
    "LOG_FILE": "pf_pipeline.log",
}

# Particle filter + model parameters
PF = {
    "N_particles":  300,
    "N_display":    150,     # random subsample for visualization
    "T":            100,
    "Q":            10.0,    # process noise variance
    "R":            1.0,     # observation noise variance
    "prior_var":    5.0,
    "ess_thresh":   0.5,     # resample when ESS < N * this
    "seed":         42,
}

# ─── THEME ────────────────────────────────────────────────────────
THEME = {
    "BG":         "#000000",
    "PANEL_BG":   "#0a0a0a",
    "GRID":       "#222222",
    "GRID_ALT":   "#1f1f1f",
    "SPINE":      "#333333",
    "TEXT":       "#ffffff",
    "TEXT_DIM":   "#aaaaaa",
    "TEXT_SEC":   "#c0c0c0",
    "ORANGE":     "#ff9500",
    "YELLOW":     "#ffd400",
    "CYAN":       "#00f2ff",
    "GREEN":      "#00ff7f",
    "RED":        "#ff3050",
    "PINK":       "#ff2a9e",
    "BLUE":       "#00bfff",
    "FONT":       "Arial",
}

# particles colored by weight — viridis has great legibility on black
CMAP_W = cm.get_cmap("viridis")


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
# MODULE 1 — MODEL + PARTICLE FILTER
# ═══════════════════════════════════════════════════════════════════

def f_state(x, t):
    """Nonlinear state transition (Gordon 1993)."""
    return 0.5 * x + 25 * x / (1 + x * x) + 8 * np.cos(1.2 * t)


def h_obs(x):
    """Nonlinear (quadratic) observation — produces bimodal posteriors."""
    return x * x / 20.0


def simulate_truth():
    """Sample ground-truth state and noisy observations."""
    p = PF
    rng = np.random.default_rng(p["seed"])
    n = p["T"]

    x = np.zeros(n)
    y = np.zeros(n)
    x[0] = rng.normal(0, np.sqrt(p["prior_var"]))
    y[0] = h_obs(x[0]) + rng.normal(0, np.sqrt(p["R"]))

    for t in range(1, n):
        x[t] = f_state(x[t - 1], t) + rng.normal(0, np.sqrt(p["Q"]))
        y[t] = h_obs(x[t]) + rng.normal(0, np.sqrt(p["R"]))

    return x, y


def _systematic_resample(weights, rng):
    N = len(weights)
    positions = (rng.random() + np.arange(N)) / N
    cdf = np.cumsum(weights)
    cdf[-1] = 1.0
    return np.searchsorted(cdf, positions)


def run_particle_filter(y_obs):
    """Bootstrap particle filter with systematic resampling.

    Stores the cloud at every time step for final visualization.
    """
    p = PF
    rng = np.random.default_rng(p["seed"] + 1)
    N = p["N_particles"]
    n = p["T"]

    particles_hist = np.zeros((n, N))
    weights_hist   = np.zeros((n, N))
    post_mean      = np.zeros(n)
    post_std       = np.zeros(n)
    ess_hist       = np.zeros(n)
    resampled      = np.zeros(n, dtype=bool)

    # init from prior
    particles = rng.normal(0, np.sqrt(p["prior_var"]), N)
    weights   = np.full(N, 1.0 / N)

    for t in range(n):
        if t > 0:
            particles = f_state(particles, t) + rng.normal(0, np.sqrt(p["Q"]), N)

        # weight update: log-likelihood of y_t | x_i
        y_pred = h_obs(particles)
        log_w = -0.5 * (y_obs[t] - y_pred) ** 2 / p["R"]
        log_w -= log_w.max()                          # stabilize
        w = np.exp(log_w) * weights
        w /= w.sum() + 1e-15
        weights = w

        particles_hist[t] = particles
        weights_hist[t]   = weights
        post_mean[t]      = np.sum(weights * particles)
        post_std[t]       = np.sqrt(np.sum(weights * (particles - post_mean[t]) ** 2))
        ess               = 1.0 / (np.sum(weights ** 2) + 1e-15)
        ess_hist[t]       = ess

        if ess < p["ess_thresh"] * N:
            idx = _systematic_resample(weights, rng)
            particles = particles[idx]
            weights   = np.full(N, 1.0 / N)
            resampled[t] = True

    return dict(
        particles = particles_hist,
        weights   = weights_hist,
        mean      = post_mean,
        std       = post_std,
        ess       = ess_hist,
        resampled = resampled,
    )


# ═══════════════════════════════════════════════════════════════════
# MODULE 2 — RENDERING
# ═══════════════════════════════════════════════════════════════════

def _style_2d(ax, xlabel=False):
    ax.set_facecolor(THEME["PANEL_BG"])
    for sp in ax.spines.values():
        sp.set_color(THEME["SPINE"]); sp.set_linewidth(0.6)
    ax.tick_params(axis="both", colors=THEME["TEXT_DIM"], labelsize=9,
                   direction="in", length=4)
    ax.yaxis.grid(True, lw=0.3, alpha=0.4, color=THEME["GRID_ALT"])
    ax.xaxis.grid(False)
    if not xlabel:
        ax.tick_params(axis="x", labelbottom=False)


def render_static(pf, truth, obs, yr, disp_idx):
    fp = CONFIG["OUTPUT_FILE"]
    try:
        elev, azim, dist, salpha = 34, 260, 1.12, 0.95

        T = PF["T"]
        n_visible = T
        sl = slice(0, n_visible)

        fig = plt.figure(figsize=(19.2, 10.8), dpi=100, facecolor=THEME["BG"])
        gs = gridspec.GridSpec(
            4, 2, width_ratios=[1.15, 1],
            height_ratios=[1.2, 1, 1, 1.2],
            hspace=0.30, wspace=0.24,
            left=0.04, right=0.97, top=0.86, bottom=0.06,
        )

        # ═══ LEFT: 3D particle cloud ══════════════════════════════
        ax = fig.add_subplot(gs[:, 0], projection="3d", computed_zorder=False)
        ax.set_facecolor(THEME["BG"])
        pane = (0.02, 0.02, 0.02, 1)
        ax.xaxis.set_pane_color(pane)
        ax.yaxis.set_pane_color(pane)
        ax.zaxis.set_pane_color(pane)
        for a in (ax.xaxis, ax.yaxis, ax.zaxis):
            a._axinfo["grid"]["color"]     = (0.13, 0.13, 0.13, 0.55)
            a._axinfo["grid"]["linewidth"] = 0.4

        # particle cloud — ALL visible time steps at once
        particles = pf["particles"][sl][:, disp_idx]   # (n_vis, N_display)
        weights   = pf["weights"  ][sl][:, disp_idx]

        # build per-step scatter arrays
        n_v  = particles.shape[0]
        Nd   = particles.shape[1]
        tt   = np.tile(np.arange(n_v)[:, None], (1, Nd)).ravel()
        xx   = particles.ravel()
        # normalise weights per time step so colors pop
        w_norm = weights / (weights.max(axis=1, keepdims=True) + 1e-15)
        ww   = w_norm.ravel()
        # z-height shows weight, scaled
        zz   = ww * 2.2

        # age fade — older particles dimmer
        age = np.tile(np.arange(n_v)[:, None], (1, Nd)).ravel()
        age_norm = age / max(n_v - 1, 1)
        alpha_arr = 0.15 + 0.80 * age_norm

        sc_colors = CMAP_W(ww)
        sc_colors[:, 3] = alpha_arr * salpha
        sizes = 6 + 36 * ww ** 1.5

        ax.scatter(tt, xx, zz,
                   c=sc_colors, s=sizes,
                   edgecolors="none", zorder=3, depthshade=False)

        # ground-truth state (bold white line on the floor, z=0)
        t_full = np.arange(T)
        ax.plot(t_full[sl], truth[sl], np.zeros(n_visible),
                color=THEME["WHITE"] if False else "#ffffff",
                lw=2.6, alpha=1.0, zorder=15)

        # posterior mean (cyan, slightly above floor)
        ax.plot(t_full[sl], pf["mean"][sl], np.full(n_visible, 0.05),
                color=THEME["CYAN"], lw=2.0, alpha=0.9, zorder=14)

        # resample events: yellow vertical pulses
        resample_idx = np.where(pf["resampled"][sl])[0]
        for r in resample_idx:
            ax.plot([r, r],
                    [pf["mean"][r], pf["mean"][r]],
                    [0, 2.4],
                    color=THEME["YELLOW"], lw=1.2, alpha=0.45, zorder=10)

        # axes
        ax.set_xlabel("TIME  t", fontsize=12, fontweight="bold",
                      color=THEME["TEXT_DIM"], labelpad=14, fontfamily=THEME["FONT"])
        ax.set_ylabel(r"STATE  $x_t$", fontsize=12, fontweight="bold",
                      color=THEME["TEXT_DIM"], labelpad=14, fontfamily=THEME["FONT"])
        ax.set_zlabel(r"WEIGHT  $w^{(i)}$", fontsize=11, fontweight="bold",
                      color=THEME["TEXT_DIM"], labelpad=8, fontfamily=THEME["FONT"])
        ax.tick_params(axis="both", colors=THEME["TEXT_DIM"], labelsize=8)

        ax.set_xlim(0, T)
        ax.set_ylim(*yr["state"])
        ax.set_zlim(0, 2.6)
        ax.set_box_aspect([1.55 * dist, 1.0 * dist, 0.65 * dist])
        ax.view_init(elev=elev, azim=azim)
        ax.set_title("PARTICLE CLOUD", fontsize=14, fontweight="bold",
                     color=THEME["CYAN"], fontfamily=THEME["FONT"], pad=8)

        # ═══ RIGHT: 4 panels ══════════════════════════════════════

        # Panel 1 — True state vs posterior with uncertainty band
        a1 = fig.add_subplot(gs[0, 1])
        _style_2d(a1)
        a1.fill_between(t_full[sl],
                        pf["mean"][sl] - 2 * pf["std"][sl],
                        pf["mean"][sl] + 2 * pf["std"][sl],
                        color=THEME["CYAN"], alpha=0.18, linewidth=0)
        a1.plot(t_full[sl], truth[sl], color="#ffffff", lw=1.3, label="truth")
        a1.plot(t_full[sl], pf["mean"][sl], color=THEME["CYAN"],
                lw=1.1, ls="--", label="posterior mean")
        a1.set_xlim(0, T); a1.set_ylim(*yr["state"])
        a1.set_title(r"Truth  vs  $\hat{x}_t = E[x_t \mid y_{1:t}]$   "
                     r"(band = $\pm 2\sigma$)",
                     fontsize=11, fontweight="bold",
                     color=THEME["TEXT_SEC"], loc="left", pad=4)
        a1.legend(loc="upper left", fontsize=8, facecolor=THEME["BG"],
                  edgecolor=THEME["SPINE"], labelcolor=THEME["TEXT_SEC"],
                  framealpha=0.85, ncol=2)

        # Panel 2 — Observations (yellow dots + line)
        a2 = fig.add_subplot(gs[1, 1])
        _style_2d(a2)
        a2.plot(t_full[sl], obs[sl], color=THEME["YELLOW"],
                lw=0.9, alpha=0.9)
        a2.scatter(t_full[sl], obs[sl], s=12, color=THEME["YELLOW"],
                   alpha=0.85, edgecolors="none")
        a2.set_xlim(0, T); a2.set_ylim(*yr["obs"])
        a2.set_title(r"Observations   $y_t = x_t^2 / 20 + w_t$",
                     fontsize=11, fontweight="bold",
                     color=THEME["TEXT_SEC"], loc="left", pad=4)

        # Panel 3 — ESS with resample events
        a3 = fig.add_subplot(gs[2, 1])
        _style_2d(a3)
        a3.plot(t_full[sl], pf["ess"][sl], color=THEME["BLUE"], lw=1.0)
        a3.fill_between(t_full[sl], 0, pf["ess"][sl],
                        color=THEME["BLUE"], alpha=0.12)
        a3.axhline(PF["N_particles"] * PF["ess_thresh"],
                   color=THEME["RED"], lw=0.8, ls="--", alpha=0.7)
        r_pts = np.where(pf["resampled"][sl])[0]
        if len(r_pts) > 0:
            a3.scatter(r_pts, pf["ess"][r_pts],
                       s=22, color=THEME["RED"], edgecolors="none",
                       zorder=5, alpha=0.9)
        a3.set_xlim(0, T); a3.set_ylim(0, PF["N_particles"] * 1.05)
        a3.set_title(r"Effective Sample Size  (red dots = resample events)",
                     fontsize=11, fontweight="bold",
                     color=THEME["TEXT_SEC"], loc="left", pad=4)

        # Panel 4 — Posterior uncertainty
        a4 = fig.add_subplot(gs[3, 1])
        _style_2d(a4, xlabel=True)
        a4.plot(t_full[sl], pf["std"][sl], color=THEME["PINK"], lw=1.1)
        a4.fill_between(t_full[sl], 0, pf["std"][sl],
                        color=THEME["PINK"], alpha=0.15)
        a4.set_xlim(0, T); a4.set_ylim(*yr["std"])
        a4.set_xlabel("Time  t", fontsize=10,
                      color=THEME["TEXT_DIM"], fontfamily=THEME["FONT"])
        a4.set_title(r"Posterior Std  $\sigma_{x_t \mid y}$",
                     fontsize=11, fontweight="bold",
                     color=THEME["TEXT_SEC"], loc="left", pad=4)

        # ═══ Title bar ════════════════════════════════════════════
        fig.text(0.50, 0.955, "PARTICLE FILTER  /  SEQUENTIAL MONTE CARLO",
                 ha="center", va="center", fontsize=26, fontweight="bold",
                 color=THEME["ORANGE"], fontfamily=THEME["FONT"])
        fig.text(0.50, 0.918,
                 r"$p(x_t \mid y_{1:t}) \;\approx\; "
                 r"\sum_{i=1}^{N} w^{(i)}_t\, \delta(x_t - x^{(i)}_t)$"
                 "          "
                 r"$w^{(i)}_t \;\propto\; w^{(i)}_{t-1}\, p(y_t \mid x^{(i)}_t)$",
                 ha="center", va="center", fontsize=13,
                 color=THEME["TEXT"], fontfamily=THEME["FONT"])
        fig.text(0.50, 0.886,
                 r"GORDON 1993 BENCHMARK    "
                 r"$N = 300$    "
                 r"$Q = 10$    "
                 r"$R = 1$    "
                 r"Systematic Resampling   ESS $< N/2$",
                 ha="center", va="center", fontsize=10, fontweight="bold",
                 color=THEME["TEXT_DIM"], fontfamily=THEME["FONT"])

        # ═══ HUD ══════════════════════════════════════════════════
        cur_t   = n_visible - 1
        cur_err = abs(truth[cur_t] - pf["mean"][cur_t])
        n_res   = int(pf["resampled"][sl].sum())
        fig.text(0.97, 0.875,
                 f"t = {cur_t:3d}    "
                 f"|truth - mean| = {cur_err:5.2f}    "
                 f"resamples = {n_res}",
                 ha="right", va="center", fontsize=11, fontweight="bold",
                 color=THEME["CYAN"], fontfamily=THEME["FONT"])

        fig.text(0.98, 0.012, "@quant.traderr",
                 ha="right", va="bottom", fontsize=10,
                 color=THEME["TEXT_DIM"], fontfamily=THEME["FONT"], alpha=0.6)

        fig.savefig(fp, dpi=100, facecolor=THEME["BG"])
        plt.close(fig)

        log(f"Saved image to {fp}")
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
    log("Particle Filter Pipeline START")
    log("=" * 60)

    log("Simulating truth + obs ...")
    truth, obs = simulate_truth()
    log(f"  T = {PF['T']}  |  truth range = [{truth.min():.2f}, {truth.max():.2f}]")

    log("Running bootstrap particle filter ...")
    pf = run_particle_filter(obs)
    rmse = np.sqrt(np.mean((pf["mean"] - truth) ** 2))
    log(f"  N = {PF['N_particles']}  |  RMSE = {rmse:.3f}  "
        f"|  resamples = {int(pf['resampled'].sum())}")

    # stable y-ranges
    all_particles = pf["particles"].ravel()
    p_lo, p_hi = np.percentile(all_particles, [2, 98])
    state_lo = min(truth.min(), p_lo) - 3
    state_hi = max(truth.max(), p_hi) + 3

    yr = {
        "state": (state_lo, state_hi),
        "obs":   (obs.min() - 2, obs.max() + 2),
        "std":   (0, pf["std"].max() * 1.15),
    }

    # fixed random subsample of particle indices for display
    rng = np.random.default_rng(PF["seed"] + 99)
    disp_idx = rng.choice(PF["N_particles"], PF["N_display"], replace=False)

    log("Rendering static visualizations ...")
    render_static(pf, truth, obs, yr, disp_idx)

    log(f"Pipeline complete in {time.time() - t0:.1f}s")
    log("=" * 60)


if __name__ == "__main__":
    main()
