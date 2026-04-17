"""
Kelly_Pipeline.py
==================
Project: Quant Trader Lab - Kelly Criterion Static Visualization
Author: quant.traderr (Instagram)
License: MIT

Description:
    Bloomberg Dark two-panel static visualization of the Kelly
    Criterion. Mirrors the layout used on slide 1 of the Demiurge
    carousel, scaled to 1920x1080 for standalone social-media use
    (reel thumbnail, cross-posts, archive).

    Panel 1 (left) - Expected log-growth dome G(f) vs bet size f.
        Kelly peak marked at f*; ruin zone shaded past 2 f*.

    Panel 2 (right) - Monte Carlo equity paths for a biased coin
        bet (p=0.55, b=2.0) across three leverage regimes:
        Half Kelly, Full Kelly, Over-leveraged (1.6 f*). All three
        see the same sequence of wins/losses, so the divergence
        is purely a consequence of bet sizing.

    Pipeline Steps:
        1. MATH        - Compute Kelly fraction, growth curve.
        2. SIMULATION  - Generate shared coin-flip sequence, evolve
                         three equity paths under fixed fractions.
        3. VISUAL      - Two-panel matplotlib figure on pure black,
                         warm accent palette, Bloomberg-style chrome.

Dependencies:
    pip install numpy matplotlib

Usage:
    python Kelly_Pipeline.py
"""

import os
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patheffects


# --- CONFIGURATION ---

CONFIG = {
    "P":               0.55,          # win probability
    "B":               2.0,            # reward:risk
    "N_TRADES":        2000,
    "INITIAL_CAPITAL": 10_000,
    "SEED":            42,
    "RESOLUTION":      (1920, 1080),  # pixels
    "DPI":             170,
    "OUTPUT_IMAGE":    "Kelly_Output.png",
}

THEME = {
    "BG":         "#000000",
    "PANEL_BG":   "#050505",
    "GRID":       "#1f1f1f",
    "EDGE":       "#2a2a2a",
    "TEXT":       "#ffffff",
    "TEXT_SEC":   "#cccccc",
    "TEXT_MUTED": "#777777",
    "AMBER":      "#f0c8a0",   # growth dome line
    "PEAK":       "#a0d8a0",   # Kelly marker (green)
    "RUIN":       "#ff4060",   # ruin zone (red)
    "BLUE":       "#a0c8e8",   # half Kelly equity
    "GREEN":      "#a0d8a0",   # full Kelly equity
    "PINK":       "#e8b0c8",   # over-leveraged equity
}


def log(msg):
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] {msg}")


# --- MODULE 1: MATH ---

def kelly_fraction(p, b):
    """Discrete-binary Kelly: f* = (bp - q) / b"""
    return (p * b - (1.0 - p)) / b


def growth_curve(p, b, f_max, n=300):
    f = np.linspace(1e-6, f_max, n)
    g = p * np.log(1 + f * b) + (1 - p) * np.log(1 - f)
    return f, g


# --- MODULE 2: SIMULATION ---

def simulate_paths(p, b, kelly_f, n_trades, initial, seed):
    rng = np.random.default_rng(seed)
    outcomes = rng.choice([1, 0], size=n_trades, p=[p, 1 - p])

    regimes = {
        "Half Kelly  (0.5 f*)":  0.5 * kelly_f,
        "Full Kelly  (1.0 f*)":  1.0 * kelly_f,
        "Over  (1.6 f*)":        1.6 * kelly_f,
    }
    paths = {}
    for name, frac in regimes.items():
        eq = [float(initial)]
        for r in outcomes:
            bet = eq[-1] * frac
            eq.append(eq[-1] + bet * b if r == 1 else eq[-1] - bet)
        paths[name] = (np.array(eq), frac)
    return paths


# --- MODULE 3: VISUAL ---

def _style_axes(ax):
    ax.set_facecolor(THEME["PANEL_BG"])
    for spine in ax.spines.values():
        spine.set_color(THEME["EDGE"])
        spine.set_linewidth(0.8)
    ax.tick_params(colors=THEME["TEXT_MUTED"], labelsize=10)
    ax.grid(True, color=THEME["GRID"], linewidth=0.6, alpha=0.9)


def _glow(color):
    return [patheffects.Stroke(linewidth=4, foreground=color, alpha=0.18),
            patheffects.Normal()]


def make_figure(kelly_f, paths, p, b):
    plt.rcParams.update({
        "figure.facecolor":  THEME["BG"],
        "savefig.facecolor": THEME["BG"],
        "text.color":        THEME["TEXT"],
        "font.family":       "DejaVu Sans",
    })

    w_in = CONFIG["RESOLUTION"][0] / CONFIG["DPI"]
    h_in = CONFIG["RESOLUTION"][1] / CONFIG["DPI"]
    fig = plt.figure(figsize=(w_in, h_in), facecolor=THEME["BG"])

    # Title band
    fig.text(0.5, 0.945,
             "KELLY CRITERION  //  THE OPTIMAL BET",
             ha="center", va="center",
             color=THEME["TEXT"], fontsize=22, fontweight="bold",
             family="DejaVu Sans")
    fig.text(0.5, 0.905,
             f"Growth dome + Monte Carlo equity paths   |   "
             f"p = {p:.2f}   b = {b:.1f}   "
             f"f* = {kelly_f:.3f}   |   {CONFIG['N_TRADES']} bets",
             ha="center", va="center",
             color=THEME["TEXT_MUTED"], fontsize=12,
             family="DejaVu Sans")

    # Two panels
    gs = fig.add_gridspec(
        nrows=1, ncols=2,
        left=0.07, right=0.97, top=0.85, bottom=0.10,
        wspace=0.22,
    )
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    _style_axes(ax1)
    _style_axes(ax2)

    # ---- Panel 1: Growth dome ----
    f_max = kelly_f * 2.5
    f_vals, g_vals = growth_curve(p, b, f_max)
    line, = ax1.plot(f_vals, g_vals,
                     color=THEME["AMBER"], linewidth=2.4,
                     label="Expected log growth  G(f)")
    line.set_path_effects(_glow(THEME["AMBER"]))

    ax1.axvline(kelly_f, color=THEME["PEAK"], linestyle="--",
                linewidth=1.4, label=f"Kelly  f* = {kelly_f:.2f}")
    ax1.axvline(2 * kelly_f, color=THEME["RUIN"], linestyle="--",
                linewidth=1.4, label="Zero growth at 2 f*")
    ax1.axhline(0, color=THEME["EDGE"], linewidth=0.6)

    ax1.fill_between(f_vals, g_vals, 0,
                     where=(f_vals > 2 * kelly_f),
                     color=THEME["RUIN"], alpha=0.22,
                     label="Ruin zone")

    ax1.set_title("Expected log growth  vs  bet fraction",
                  color=THEME["TEXT_SEC"], fontsize=13, pad=12,
                  loc="left")
    ax1.set_xlabel("Bet fraction  f", color=THEME["TEXT_MUTED"])
    ax1.set_ylabel("Expected log growth  G(f)",
                   color=THEME["TEXT_MUTED"])
    leg = ax1.legend(loc="lower left",
                     facecolor=THEME["PANEL_BG"],
                     edgecolor=THEME["EDGE"],
                     fontsize=10, labelcolor=THEME["TEXT_SEC"])
    for text in leg.get_texts():
        text.set_color(THEME["TEXT_SEC"])

    # ---- Panel 2: Monte Carlo equity paths ----
    regime_colors = {
        "Half Kelly  (0.5 f*)": THEME["BLUE"],
        "Full Kelly  (1.0 f*)": THEME["GREEN"],
        "Over  (1.6 f*)":       THEME["PINK"],
    }
    for name, (equity, frac) in paths.items():
        color = regime_colors[name]
        line, = ax2.plot(equity, color=color, linewidth=1.4,
                         label=f"{name}   (f={frac:.2%})")
        line.set_path_effects(_glow(color))

    ax2.set_yscale("log")
    ax2.set_title(f"Monte Carlo  //  {CONFIG['N_TRADES']} sequential bets",
                  color=THEME["TEXT_SEC"], fontsize=13, pad=12,
                  loc="left")
    ax2.set_xlabel("Trade  #", color=THEME["TEXT_MUTED"])
    ax2.set_ylabel("Capital  (log scale)",
                   color=THEME["TEXT_MUTED"])
    leg2 = ax2.legend(loc="upper left",
                      facecolor=THEME["PANEL_BG"],
                      edgecolor=THEME["EDGE"],
                      fontsize=10, labelcolor=THEME["TEXT_SEC"])
    for text in leg2.get_texts():
        text.set_color(THEME["TEXT_SEC"])

    # Footer
    fig.text(0.015, 0.025, "@quant.traderr",
             color=THEME["TEXT_MUTED"], fontsize=11,
             family="DejaVu Sans")
    fig.text(0.985, 0.025, "f* = (bp - q) / b",
             color=THEME["TEXT_MUTED"], fontsize=11,
             ha="right", family="DejaVu Sans Mono")

    return fig


# --- MAIN ---

def main():
    t0 = time.time()
    log("=" * 60)
    log("Kelly_Pipeline - Bloomberg Dark Static")
    log("=" * 60)

    kelly_f = kelly_fraction(CONFIG["P"], CONFIG["B"])
    log(f"  p = {CONFIG['P']}  b = {CONFIG['B']}  f* = {kelly_f:.4f}")

    paths = simulate_paths(CONFIG["P"], CONFIG["B"], kelly_f,
                           CONFIG["N_TRADES"], CONFIG["INITIAL_CAPITAL"],
                           CONFIG["SEED"])
    for name, (eq, frac) in paths.items():
        log(f"  {name:22s}  final = {eq[-1]:,.0f}")

    fig = make_figure(kelly_f, paths, CONFIG["P"], CONFIG["B"])

    out_dir = os.path.dirname(os.path.abspath(__file__))
    out_path = os.path.join(out_dir, CONFIG["OUTPUT_IMAGE"])
    fig.savefig(out_path, dpi=CONFIG["DPI"],
                facecolor=THEME["BG"], bbox_inches=None)
    plt.close(fig)

    log(f"Saved: {out_path}")
    log(f"Time:  {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
