"""
GBM_Pipeline.py
================
Project: Quant Trader Lab - Stochastic Processes
Author: quant.traderr (Instagram)
License: MIT

Description:
    Production-ready pipeline for Geometric Brownian Motion visualization.

    Simulates N price paths under GBM:
        dS_t = mu * S_t * dt + sigma * S_t * dW_t

    with Ito-corrected log-price dynamics. Produces a Bloomberg-dark
    static snapshot showing the path fan, mean/percentile bands,
    and the terminal log-normal distribution.

    Pipeline Steps:
    1.  **Simulation**: Vectorized GBM path generation via log-price cumsum.
    2.  **Analysis**: Mean, median, percentile bands, terminal distribution.
    3.  **Static Visualization**: Bloomberg-dark 1920x1080 PNG.

Dependencies:
    pip install numpy matplotlib
"""

import os, time, warnings
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ═══════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════
CONFIG = {
    # Simulation
    "S0":               100.0,
    "MU":               0.08,       # annualized drift
    "SIGMA":            0.25,       # annualized volatility
    "T":                2.0,        # horizon in years
    "DT":               0.002,      # time step
    "N_PATHS":          500,
    "SEED":             42,

    # Output
    "OUTPUT_IMAGE":     "GBM_Output.png",
    "RESOLUTION":       (1920, 1080),
    "DPI":              100,
    "LOG_FILE":         os.path.join(BASE_DIR, "gbm_pipeline.log"),

    # Theme (Bloomberg Dark)
    "BG":               "#0b0b0b",
    "PANEL_BG":         "#0e0e0e",
    "GRID":             "#1a1a1a",
    "TEXT":             "#ffffff",
    "TEXT_DIM":         "#aaaaaa",
    "ORANGE":           "#ff9500",
    "CYAN":             "#00f2ff",
    "MAGENTA":          "#ff1493",
    "YELLOW":           "#ffd400",
    "GREEN":            "#00ff41",
    "FONT":             "Arial",
}


# ═══════════════════════════════════════════════════════════════════
# UTILS
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
# MODULE 1: SIMULATION
# ═══════════════════════════════════════════════════════════════════
def simulate_gbm():
    """Vectorized GBM simulation via Ito-corrected log-price dynamics."""
    S0    = CONFIG["S0"]
    mu    = CONFIG["MU"]
    sigma = CONFIG["SIGMA"]
    T     = CONFIG["T"]
    dt    = CONFIG["DT"]
    n_paths = CONFIG["N_PATHS"]

    np.random.seed(CONFIG["SEED"])
    n_steps = int(T / dt)
    t = np.linspace(0, T, n_steps + 1)

    log(f"[Simulation] {n_paths} paths, {n_steps} steps, T={T}y, dt={dt}")
    log(f"[Simulation] S0={S0}, mu={mu}, sigma={sigma}")

    start = time.time()

    # Brownian increments
    dW = np.random.randn(n_paths, n_steps) * np.sqrt(dt)

    # Ito-corrected log-price increments
    drift = (mu - 0.5 * sigma**2) * dt
    log_S = np.cumsum(drift + sigma * dW, axis=1)

    # Prepend zero so all paths start at S0
    log_S = np.c_[np.zeros(n_paths), log_S]
    S = S0 * np.exp(log_S)

    elapsed = time.time() - start
    log(f"[Simulation] Complete in {elapsed:.2f}s")

    return t, S


# ═══════════════════════════════════════════════════════════════════
# MODULE 2: ANALYSIS
# ═══════════════════════════════════════════════════════════════════
def analyze(t, S):
    """Compute summary statistics across paths."""
    mean_path   = np.mean(S, axis=0)
    median_path = np.median(S, axis=0)
    p5          = np.percentile(S, 5, axis=0)
    p25         = np.percentile(S, 25, axis=0)
    p75         = np.percentile(S, 75, axis=0)
    p95         = np.percentile(S, 95, axis=0)
    final       = S[:, -1]

    log("=== GBM RESULTS ===")
    log(f"  Mean final price:   ${mean_path[-1]:,.2f}")
    log(f"  Median final price: ${median_path[-1]:,.2f}")
    log(f"  5th percentile:     ${p5[-1]:,.2f}")
    log(f"  95th percentile:    ${p95[-1]:,.2f}")
    log(f"  Expected return:    {(mean_path[-1] / CONFIG['S0']) - 1:.1%}")

    # Theoretical check: E[S_T] = S0 * exp(mu * T)
    theoretical = CONFIG["S0"] * np.exp(CONFIG["MU"] * CONFIG["T"])
    log(f"  Theoretical E[S_T]: ${theoretical:,.2f}")

    return {
        "mean": mean_path, "median": median_path,
        "p5": p5, "p25": p25, "p75": p75, "p95": p95,
        "final": final,
    }


# ═══════════════════════════════════════════════════════════════════
# MODULE 3: VISUALIZATION
# ═══════════════════════════════════════════════════════════════════
def visualize(t, S, stats):
    """Render Bloomberg-dark static visualization."""
    log("[Visual] Generating static snapshot...")

    fig = plt.figure(
        figsize=(CONFIG["RESOLUTION"][0] / CONFIG["DPI"],
                 CONFIG["RESOLUTION"][1] / CONFIG["DPI"]),
        dpi=CONFIG["DPI"],
        facecolor=CONFIG["BG"],
    )

    gs = GridSpec(1, 2, width_ratios=[4, 1],
                  left=0.06, right=0.96, top=0.88, bottom=0.10,
                  wspace=0.02)

    # ── Main panel: paths + stats ──
    ax = fig.add_subplot(gs[0])
    ax.set_facecolor(CONFIG["PANEL_BG"])

    # Path fan (low alpha)
    step = max(1, CONFIG["N_PATHS"] // 250)
    for i in range(0, CONFIG["N_PATHS"], step):
        ax.plot(t, S[i], color=CONFIG["CYAN"], alpha=0.04, linewidth=0.5)

    # Confidence bands
    ax.fill_between(t, stats["p5"], stats["p95"],
                    color=CONFIG["CYAN"], alpha=0.05, label="5-95% band")
    ax.fill_between(t, stats["p25"], stats["p75"],
                    color=CONFIG["CYAN"], alpha=0.08, label="25-75% band")

    # Stat lines
    ax.plot(t, stats["mean"], color=CONFIG["ORANGE"],
            linewidth=2.5, label=r"$\mathbb{E}[S_t]$", zorder=10)
    ax.plot(t, stats["median"], color=CONFIG["GREEN"],
            linewidth=1.5, linestyle=":", label="Median", zorder=10)
    ax.plot(t, stats["p95"], color=CONFIG["YELLOW"],
            linewidth=1.2, linestyle="--", label="95th pct", zorder=10)
    ax.plot(t, stats["p5"], color=CONFIG["MAGENTA"],
            linewidth=1.2, linestyle="--", label="5th pct", zorder=10)

    ax.set_xlim(0, CONFIG["T"])
    ax.set_xlabel("Time (years)", color=CONFIG["TEXT_DIM"], fontsize=12)
    ax.set_ylabel("Price S(t)", color=CONFIG["TEXT_DIM"], fontsize=12)
    ax.tick_params(colors=CONFIG["TEXT_DIM"], labelsize=9)
    ax.grid(True, color=CONFIG["GRID"], linewidth=0.3, alpha=0.5)
    for spine in ax.spines.values():
        spine.set_color(CONFIG["GRID"])
        spine.set_linewidth(0.5)

    leg = ax.legend(loc="upper left", fontsize=9,
                    facecolor=CONFIG["BG"], edgecolor=CONFIG["GRID"])
    for text in leg.get_texts():
        text.set_color(CONFIG["TEXT_DIM"])

    # ── Right panel: terminal distribution ──
    ax_hist = fig.add_subplot(gs[1], sharey=ax)
    ax_hist.set_facecolor(CONFIG["PANEL_BG"])

    ax_hist.hist(stats["final"], bins=60, orientation="horizontal",
                 color=CONFIG["CYAN"], alpha=0.6, edgecolor=CONFIG["BG"])
    ax_hist.axhline(np.mean(stats["final"]),
                    color=CONFIG["ORANGE"], linewidth=1.5)
    ax_hist.axhline(np.median(stats["final"]),
                    color=CONFIG["GREEN"], linewidth=1.0, linestyle=":")
    ax_hist.axhline(np.percentile(stats["final"], 5),
                    color=CONFIG["MAGENTA"], linewidth=1.0, linestyle="--")
    ax_hist.axhline(np.percentile(stats["final"], 95),
                    color=CONFIG["YELLOW"], linewidth=1.0, linestyle="--")

    ax_hist.set_xlabel("Density", color=CONFIG["TEXT_DIM"], fontsize=9)
    ax_hist.tick_params(colors=CONFIG["TEXT_DIM"], labelsize=7, labelleft=False)
    ax_hist.grid(True, color=CONFIG["GRID"], linewidth=0.2, alpha=0.3)
    for spine in ax_hist.spines.values():
        spine.set_color(CONFIG["GRID"])
        spine.set_linewidth(0.5)

    # ── Title bar ──
    fig.text(0.50, 0.96,
             "GEOMETRIC BROWNIAN MOTION",
             ha="center", va="center", fontsize=24, fontweight="bold",
             color=CONFIG["ORANGE"], fontfamily=CONFIG["FONT"])

    sigma_pct = int(CONFIG["SIGMA"] * 100)
    mu_pct = int(CONFIG["MU"] * 100)
    fig.text(0.50, 0.93,
             r"$dS_t = \mu S_t \, dt + \sigma S_t \, dW_t$"
             f"          "
             f"$S_0 = {CONFIG['S0']:.0f}$    "
             f"$\\mu = {mu_pct}\\%$    "
             f"$\\sigma = {sigma_pct}\\%$    "
             f"$T = {CONFIG['T']:.0f}y$    "
             f"{CONFIG['N_PATHS']} paths",
             ha="center", va="center", fontsize=11,
             color=CONFIG["TEXT_DIM"], fontfamily=CONFIG["FONT"])

    # ── Footer ──
    fig.text(0.98, 0.012, "@quant.traderr",
             ha="right", va="bottom", fontsize=10,
             color=CONFIG["TEXT_DIM"], fontfamily=CONFIG["FONT"], alpha=0.6)

    # ── HUD (current stats) ──
    fig.text(0.96, 0.88,
             f"E[S_T] = ${stats['mean'][-1]:,.1f}    "
             f"P5 = ${stats['p5'][-1]:,.1f}    "
             f"P95 = ${stats['p95'][-1]:,.1f}",
             ha="right", va="center", fontsize=10, fontweight="bold",
             color=CONFIG["YELLOW"], fontfamily=CONFIG["FONT"])

    # ── Save ──
    out_path = os.path.join(BASE_DIR, CONFIG["OUTPUT_IMAGE"])
    fig.savefig(out_path, dpi=CONFIG["DPI"], facecolor=CONFIG["BG"])
    plt.close(fig)
    log(f"[Visual] Saved to {out_path}")


# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════
def main():
    t0 = time.time()
    log("=== GBM PIPELINE ===")

    # 1. Simulate
    t, S = simulate_gbm()

    # 2. Analyze
    stats = analyze(t, S)

    # 3. Visualize
    visualize(t, S, stats)

    log(f"=== PIPELINE FINISHED ({time.time() - t0:.1f}s) ===")


if __name__ == "__main__":
    main()
