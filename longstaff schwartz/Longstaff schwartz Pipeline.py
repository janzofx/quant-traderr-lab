"""
Longstaff_Schwartz_Pipeline.py
==============================
Static 3D visualization of the Longstaff-Schwartz Method for American option
pricing. We simulate N geometric Brownian motion paths under the risk-neutral
measure, then run the LSM backward iteration:

  for t = T-1, ..., 1:
      regress  discounted future CF  ~  basis(S_t)  on ITM paths
      exercise if  intrinsic(S_t) > continuation_estimate(S_t)

Left panel  : 3D bundle of GBM paths  (t, path_idx, S)  with early-exercise dots
Right panel : 4-stack 2D — Boundary · Live paths · Exercise PnL hist · Price convergence

Dependencies: pip install numpy matplotlib
"""

import os, time, warnings
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

warnings.filterwarnings("ignore")

# ═══════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════
CONFIG = {
    "RESOLUTION": (1920, 1080),
    "OUTPUT_IMAGE": "Longstaff_Schwartz_Output.png",
    "LOG_FILE": "lsm_pipeline.log",
}

# American option + simulation parameters
LSM = {
    "S0":       100.0,
    "K":        100.0,    # strike (ATM)
    "r":        0.05,     # risk-free
    "sigma":    0.32,     # volatility (high so early exercise matters)
    "T":        1.0,      # 1 year maturity
    "n_steps":  50,       # time grid
    "N_paths":  500,      # MC paths for LSM
    "N_display": 60,      # visual subsample of paths
    "seed":     11,
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
    "PALE":       "#88aaff",
    "FONT":       "Arial",
}


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
# MODULE 1 — SIMULATE PATHS + RUN LSM
# ═══════════════════════════════════════════════════════════════════

def simulate_gbm_paths():
    """Risk-neutral GBM paths (antithetic for variance reduction)."""
    p = LSM
    rng = np.random.default_rng(p["seed"])
    n_steps = p["n_steps"]
    N = p["N_paths"]
    dt = p["T"] / n_steps

    Z = rng.standard_normal((N // 2, n_steps))
    Z = np.vstack([Z, -Z])    # antithetic
    if Z.shape[0] < N:
        Z = np.vstack([Z, rng.standard_normal((N - Z.shape[0], n_steps))])

    drift = (p["r"] - 0.5 * p["sigma"] ** 2) * dt
    diff  = p["sigma"] * np.sqrt(dt)
    incr  = drift + diff * Z

    log_paths = np.cumsum(incr, axis=1)
    paths = np.empty((N, n_steps + 1))
    paths[:, 0]  = p["S0"]
    paths[:, 1:] = p["S0"] * np.exp(log_paths)
    return paths


def run_lsm(paths):
    """Longstaff-Schwartz backward iteration for an American PUT."""
    p = LSM
    N, n_total = paths.shape
    n_steps = n_total - 1
    K = p["K"]; r = p["r"]
    dt = p["T"] / n_steps

    cash_flow = np.maximum(K - paths[:, -1], 0.0)
    exercise_t = np.full(N, n_steps, dtype=int)

    boundary = np.full(n_total, np.nan)
    boundary[-1] = K
    european_payoffs = np.maximum(K - paths[:, -1], 0.0)
    european_price = float(np.mean(european_payoffs * np.exp(-r * p["T"])))

    for t in range(n_steps - 1, 0, -1):
        S_t = paths[:, t]
        intrinsic = np.maximum(K - S_t, 0.0)
        itm = intrinsic > 0
        if itm.sum() < 5:
            continue

        future_cf = cash_flow[itm] * np.exp(-r * dt * (exercise_t[itm] - t))
        S_itm = S_t[itm]

        X = np.column_stack([np.ones_like(S_itm), S_itm, S_itm ** 2])
        coefs, *_ = np.linalg.lstsq(X, future_cf, rcond=None)
        cont_est = X @ coefs

        ex_now = intrinsic[itm] > cont_est
        ex_idx = np.where(itm)[0][ex_now]
        cash_flow[ex_idx] = intrinsic[itm][ex_now]
        exercise_t[ex_idx] = t

        S_grid = np.linspace(0.40 * K, K, 200)
        intr_g = K - S_grid
        cont_g = coefs[0] + coefs[1] * S_grid + coefs[2] * S_grid ** 2
        ex_mask = intr_g > cont_g
        if ex_mask.any():
            boundary[t] = float(S_grid[ex_mask].max())

    discounted = cash_flow * np.exp(-r * dt * exercise_t)
    american_price = float(np.mean(discounted))

    return dict(
        paths=paths,
        boundary=boundary,
        exercise_t=exercise_t,
        cash_flow=cash_flow,
        american_price=american_price,
        european_price=european_price,
        early_premium=american_price - european_price,
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


def render_static(lsm, disp_idx, yr, out_path):
    """Render the complete LSM result as a single static image."""
    try:
        elev, azim, dist, salpha = 34, 260, 1.15, 0.95

        paths  = lsm["paths"]
        bnd    = lsm["boundary"]
        ex_t   = lsm["exercise_t"]
        ame_p  = lsm["american_price"]
        eur_p  = lsm["european_price"]
        N, n_total = paths.shape
        n_steps = n_total - 1

        # full boundary visible (all steps solved)
        ex_visible_mask = (ex_t < n_steps)

        fig = plt.figure(figsize=(19.2, 10.8), dpi=100, facecolor=THEME["BG"])
        gs = gridspec.GridSpec(
            4, 2, width_ratios=[1.15, 1],
            height_ratios=[1.2, 1, 1, 1.2],
            hspace=0.30, wspace=0.24,
            left=0.04, right=0.97, top=0.86, bottom=0.06,
        )

        # ═══ LEFT: 3D path bundle ═════════════════════════════════
        ax = fig.add_subplot(gs[:, 0], projection="3d", computed_zorder=False)
        ax.set_facecolor(THEME["BG"])
        pane = (0.02, 0.02, 0.02, 1)
        ax.xaxis.set_pane_color(pane)
        ax.yaxis.set_pane_color(pane)
        ax.zaxis.set_pane_color(pane)
        for a in (ax.xaxis, ax.yaxis, ax.zaxis):
            a._axinfo["grid"]["color"]     = (0.13, 0.13, 0.13, 0.55)
            a._axinfo["grid"]["linewidth"] = 0.4

        t_full = np.arange(n_total)
        for j, p_idx in enumerate(disp_idx):
            S = paths[p_idx]
            y_line = np.full_like(t_full, j, dtype=float)
            if ex_visible_mask[p_idx]:
                ex_step = ex_t[p_idx]
                ax.plot(t_full[:ex_step + 1], y_line[:ex_step + 1], S[:ex_step + 1],
                        color=THEME["RED"], lw=0.9, alpha=0.85 * salpha, zorder=3)
                ax.scatter([ex_step], [j], [S[ex_step]],
                           s=22, color=THEME["RED"],
                           edgecolors="white", linewidths=0.4,
                           zorder=10, depthshade=False)
            else:
                ax.plot(t_full, y_line, S,
                        color=THEME["PALE"], lw=0.7, alpha=0.55 * salpha, zorder=2)

        # exercise boundary as a yellow wall
        bnd_t = np.where(~np.isnan(bnd))[0]
        if len(bnd_t) > 1:
            B = bnd[bnd_t]
            for j in range(0, len(disp_idx), 4):
                y_line = np.full_like(bnd_t, j, dtype=float)
                ax.plot(bnd_t, y_line, B,
                        color=THEME["YELLOW"], lw=1.4, alpha=0.55, zorder=8)
            y_mid = len(disp_idx) / 2.0
            ax.plot(bnd_t, np.full_like(bnd_t, y_mid), B,
                    color=THEME["YELLOW"], lw=3.2, alpha=1.0, zorder=15)

        K_v = LSM["K"]
        for j in range(0, len(disp_idx), 8):
            ax.plot([0, n_steps], [j, j], [K_v, K_v],
                    color=THEME["CYAN"], lw=0.5, alpha=0.25, ls="--", zorder=1)

        ax.set_xlabel("TIME STEP", fontsize=12, fontweight="bold",
                      color=THEME["TEXT_DIM"], labelpad=14, fontfamily=THEME["FONT"])
        ax.set_ylabel("PATH #", fontsize=12, fontweight="bold",
                      color=THEME["TEXT_DIM"], labelpad=14, fontfamily=THEME["FONT"])
        ax.set_zlabel(r"PRICE  $S_t$", fontsize=12, fontweight="bold",
                      color=THEME["TEXT_DIM"], labelpad=12, fontfamily=THEME["FONT"])
        ax.tick_params(axis="both", colors=THEME["TEXT_DIM"], labelsize=8)

        ax.set_xlim(0, n_steps)
        ax.set_ylim(0, len(disp_idx) - 1)
        ax.set_zlim(*yr["S"])
        ax.set_box_aspect([1.55 * dist, 1.0 * dist, 0.85 * dist])
        ax.view_init(elev=elev, azim=azim)
        ax.set_title("PATH BUNDLE  +  EARLY-EXERCISE BOUNDARY",
                     fontsize=13, fontweight="bold",
                     color=THEME["YELLOW"], fontfamily=THEME["FONT"], pad=8)

        # ═══ RIGHT: 4 panels ══════════════════════════════════════

        # Panel 1 — Boundary
        a1 = fig.add_subplot(gs[0, 1])
        _style_2d(a1)
        a1.axhline(LSM["K"], color=THEME["CYAN"], lw=0.9, ls="--", alpha=0.7,
                   label="strike  K")
        a1.plot(t_full, bnd, color=THEME["YELLOW"], lw=1.6,
                label=r"$S^{*}(t)$  exercise boundary")
        a1.fill_between(t_full, np.zeros_like(t_full, dtype=float),
                        bnd, where=~np.isnan(bnd),
                        color=THEME["YELLOW"], alpha=0.10)
        a1.set_xlim(0, n_steps); a1.set_ylim(*yr["S"])
        a1.legend(loc="lower left", fontsize=8, facecolor=THEME["BG"],
                  edgecolor=THEME["SPINE"], labelcolor=THEME["TEXT_SEC"],
                  framealpha=0.85, ncol=2)
        a1.set_title(r"Early-Exercise Boundary  $S^{*}(t)$",
                     fontsize=11, fontweight="bold",
                     color=THEME["TEXT_SEC"], loc="left", pad=4)

        # Panel 2 — Paths still alive
        a2 = fig.add_subplot(gs[1, 1])
        _style_2d(a2)
        alive = np.zeros(n_total, dtype=int)
        for tt in range(n_total):
            alive[tt] = int(np.sum(ex_t > tt))
        a2.plot(t_full, alive, color=THEME["BLUE"], lw=1.0)
        a2.fill_between(t_full, 0, alive, color=THEME["BLUE"], alpha=0.12)
        a2.set_xlim(0, n_steps); a2.set_ylim(0, N * 1.05)
        a2.set_title("Paths still alive   (decrease = early exercise)",
                     fontsize=11, fontweight="bold",
                     color=THEME["TEXT_SEC"], loc="left", pad=4)

        # Panel 3 — Exercise payoff histogram
        a3 = fig.add_subplot(gs[2, 1])
        _style_2d(a3)
        if ex_visible_mask.sum() > 0:
            payoffs = np.maximum(LSM["K"] - paths[ex_visible_mask, ex_t[ex_visible_mask]], 0)
            a3.hist(payoffs, bins=22, range=(0, max(LSM["K"] * 0.55, 1)),
                    color=THEME["RED"], alpha=0.65, edgecolor=THEME["SPINE"],
                    linewidth=0.4)
        a3.set_xlim(0, LSM["K"] * 0.55)
        a3.set_title("Early-Exercise Payoff Histogram",
                     fontsize=11, fontweight="bold",
                     color=THEME["TEXT_SEC"], loc="left", pad=4)

        # Panel 4 — Price comparison
        a4 = fig.add_subplot(gs[3, 1])
        _style_2d(a4, xlabel=True)
        a4.axhline(eur_p, color=THEME["TEXT_DIM"], lw=0.9, ls="--",
                   alpha=0.7, label=f"European  = {eur_p:.3f}")
        a4.axhline(ame_p, color=THEME["GREEN"], lw=0.9, ls="--",
                   alpha=0.7, label=f"American  = {ame_p:.3f}")
        a4.bar(n_steps // 2, ame_p - eur_p, width=4, bottom=eur_p,
               color=THEME["YELLOW"], alpha=0.55,
               label=f"early premium = {ame_p - eur_p:.3f}")
        a4.set_xlim(0, n_steps)
        a4.set_ylim(eur_p - 0.5, ame_p + 1.0)
        a4.set_xlabel("Time step", fontsize=10,
                      color=THEME["TEXT_DIM"], fontfamily=THEME["FONT"])
        a4.legend(loc="upper right", fontsize=8, facecolor=THEME["BG"],
                  edgecolor=THEME["SPINE"], labelcolor=THEME["TEXT_SEC"],
                  framealpha=0.85)
        a4.set_title("Price convergence:  European  ->  American",
                     fontsize=11, fontweight="bold",
                     color=THEME["TEXT_SEC"], loc="left", pad=4)

        # ═══ Title bar ════════════════════════════════════════════
        fig.text(0.50, 0.955, "LONGSTAFF\u2013SCHWARTZ  AMERICAN OPTION",
                 ha="center", va="center", fontsize=26, fontweight="bold",
                 color=THEME["ORANGE"], fontfamily=THEME["FONT"])
        fig.text(0.50, 0.918,
                 r"$V_t \;=\; \max\!\left(\, (K - S_t)^+,\;\;"
                 r"\hat{E}\!\left[\, e^{-r\Delta t}\, V_{t+\Delta t} \mid S_t \,\right]\,\right)$",
                 ha="center", va="center", fontsize=14,
                 color=THEME["TEXT"], fontfamily=THEME["FONT"])
        fig.text(0.50, 0.886,
                 r"AMERICAN PUT    "
                 r"$S_0 = K = 100$    $r = 5\%$    "
                 r"$\sigma = 32\%$    $T = 1\,\mathrm{yr}$    "
                 r"$N = 500$ paths   basis $\{1, S, S^2\}$",
                 ha="center", va="center", fontsize=10, fontweight="bold",
                 color=THEME["TEXT_DIM"], fontfamily=THEME["FONT"])

        # ═══ HUD ══════════════════════════════════════════════════
        n_ex = int(ex_visible_mask.sum())
        early_premium = ame_p - eur_p
        fig.text(0.97, 0.875,
                 f"exercises = {n_ex:3d}    "
                 f"early premium = {early_premium:.3f}",
                 ha="right", va="center", fontsize=11, fontweight="bold",
                 color=THEME["YELLOW"], fontfamily=THEME["FONT"])

        # ═══ Footer ═══════════════════════════════════════════════
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
    log("Longstaff-Schwartz Pipeline START")
    log("=" * 60)

    log("Simulating GBM paths ...")
    paths = simulate_gbm_paths()
    log(f"  N = {paths.shape[0]}  steps = {paths.shape[1] - 1}  "
        f"S range = [{paths.min():.2f}, {paths.max():.2f}]")

    log("Running LSM backward iteration ...")
    lsm = run_lsm(paths)
    log(f"  European put: {lsm['european_price']:.4f}")
    log(f"  American put: {lsm['american_price']:.4f}")
    log(f"  Early-ex premium: {lsm['early_premium']:.4f}")
    n_ex = int(np.sum(lsm['exercise_t'] < paths.shape[1] - 1))
    log(f"  paths exercised early: {n_ex}/{paths.shape[0]}")

    rng = np.random.default_rng(LSM["seed"] + 99)
    disp_idx = rng.choice(LSM["N_paths"], LSM["N_display"], replace=False)

    yr = {
        "S": (max(0, paths.min() * 0.95), paths.max() * 1.05),
    }

    log("Rendering static image ...")
    render_static(lsm, disp_idx, yr, CONFIG["OUTPUT_IMAGE"])
    log(f"Pipeline complete in {time.time() - t0:.1f}s")
    log("=" * 60)


if __name__ == "__main__":
    main()
