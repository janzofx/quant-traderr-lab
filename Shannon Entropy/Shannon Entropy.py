"""
Shannon_Pipeline.py
===================
Project : Quant Trader Lab - Shannon Entropy Market Regime Analysis
Author  : quant.traderr (Instagram)
License : MIT

Description
-----------
A production-ready pipeline for visualizing **rolling Shannon Entropy** on
financial return distributions as a cinematic timelapse.

Shannon Entropy measures information content / uncertainty in a probability
distribution.  Applied to rolling windows of log returns, it reveals:
  - **High entropy** : random / efficient market (returns uniformly spread)
  - **Low entropy**  : trending / predictable regime (returns concentrated)

The pipeline computes entropy at four time scales (21d, 63d, 126d, 252d)
and visualizes the evolving market regime alongside price and the return
distribution histogram.

Pipeline Steps
--------------
    1. **Data Acquisition** : Fetches BTC-USD via yfinance, computes log returns.
    2. **Entropy Engine**   : Rolling Shannon entropy at 4 window sizes.
    3. **Rendering**        : Parallelized matplotlib 4-panel frame generation.
    4. **Compilation**      : Assembles frames into a cinematic MP4 video.

Dependencies
------------
    pip install numpy pandas matplotlib moviepy yfinance
"""

import os
import sys
import time
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
from multiprocessing import Pool, cpu_count

# MoviePy v1/v2 Compatibility
try:
    from moviepy import ImageSequenceClip
    MOVIEPY_V2 = True
except ImportError:
    from moviepy.editor import ImageSequenceClip
    MOVIEPY_V2 = False

warnings.filterwarnings("ignore")

# =============================================================================
# CONFIGURATION
# =============================================================================

CONFIG = {
    # -- Data --
    "TICKER":         "BTC-USD",
    "PERIOD":         "2y",
    "INTERVAL":       "1d",

    # -- Entropy Parameters --
    "WINDOW_SIZES":   [21, 63, 126, 252],
    "WINDOW_LABELS":  ["21d", "63d", "126d", "252d"],
    "N_BINS":         30,
    "SEED":           42,

    # -- Video --
    "FPS":            30,
    "DURATION_SEC":   15,
    "RESOLUTION":     (1920, 1080),

    # -- Paths --
    "TEMP_DIR":       os.path.join(os.path.dirname(__file__), "temp_entropy_frames"),
    "OUTPUT_FILE":    os.path.join(os.path.dirname(__file__), "Shannon_Timelapse.mp4"),
    "LOG_FILE":       os.path.join(os.path.dirname(__file__), "shannon_pipeline.log"),
}

THEME = {
    "BG":        "#0e0e0e",
    "GRID":      "#1f1f1f",
    "SPINE":     "#333333",
    "TEXT":      "#c0c0c0",
    "WHITE":     "#ffffff",
    "FONT":      "DejaVu Sans",

    # Entropy-specific colors
    "PRICE":          "#00d4ff",    # Cyan
    "ENTROPY_SHORT":  "#00d4ff",    # Cyan  (21d)
    "ENTROPY_MED":    "#ff9800",    # Orange (63d)
    "ENTROPY_LONG":   "#ff1493",    # Deep pink (126d)
    "ENTROPY_XLNG":   "#bb66ff",    # Purple (252d)
    "HIST_FILL":      "#00d4ff",    # Cyan histogram fill
    "MAX_ENT":        "#ff0055",    # Red dashed H_max line
}

ENTROPY_COLORS = [
    THEME["ENTROPY_SHORT"],
    THEME["ENTROPY_MED"],
    THEME["ENTROPY_LONG"],
    THEME["ENTROPY_XLNG"],
]

# =============================================================================
# UTILS
# =============================================================================

def log(msg):
    """Timestamped console + file logger."""
    timestamp = time.strftime("%H:%M:%S")
    formatted = f"[{timestamp}] {msg}"
    try:
        print(formatted)
    except UnicodeEncodeError:
        print(formatted.encode("ascii", errors="replace").decode())
    try:
        with open(CONFIG["LOG_FILE"], "a", encoding="utf-8") as f:
            f.write(formatted + "\n")
    except Exception:
        pass


# =============================================================================
# MODULE 1 : DATA + ENTROPY COMPUTATION
# =============================================================================

def fetch_price_data():
    """
    Fetch BTC-USD via yfinance and compute log returns.

    Returns: (prices, log_returns, dates)
    """
    import yfinance as yf

    log(f"[Data] Fetching {CONFIG['TICKER']} ({CONFIG['PERIOD']})...")

    try:
        df = yf.download(CONFIG["TICKER"], period=CONFIG["PERIOD"],
                         interval=CONFIG["INTERVAL"], progress=False)
    except Exception as e:
        log(f"[Error] YF Download failed: {e}. Using synthetic fallback.")
        return _synthetic_fallback()

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)
        log("[Data] Flattened MultiIndex columns.")

    if df.empty:
        log("[Warning] Empty dataframe. Using synthetic fallback.")
        return _synthetic_fallback()

    price = df["Close"].values.flatten().astype(float)
    dates = df.index

    # Log returns
    log_returns = np.diff(np.log(price + 1e-9))
    valid = ~np.isnan(log_returns)
    log_returns = log_returns[valid]
    price = price[1:][valid]       # Align price with returns
    dates = dates[1:][valid]

    log(f"[Data] {len(log_returns)} log returns | "
        f"mean={np.mean(log_returns):.6f}, std={np.std(log_returns):.6f}")
    return price, log_returns, dates


def _synthetic_fallback():
    """Generate synthetic returns if yfinance fails."""
    log("[Data] Generating synthetic returns (fat-tail mixture)...")
    np.random.seed(CONFIG["SEED"])
    n = 500
    normal = np.random.normal(0.0003, 0.025, int(n * 0.85))
    heavy = np.random.standard_t(3, int(n * 0.15)) * 0.04
    returns = np.concatenate([normal, heavy])
    np.random.shuffle(returns)
    returns = returns[:n]
    price = 30000.0 * np.exp(np.cumsum(returns))
    dates = pd.date_range("2024-01-01", periods=n, freq="B")
    return price, returns, dates


def compute_rolling_entropy(log_returns):
    """
    Compute rolling Shannon entropy at multiple window sizes.

    For each window W and time t >= W:
        1. Slice returns[t-W : t]
        2. Histogram with fixed global bin edges
        3. H(X) = -sum(p_i * ln(p_i)) for p_i > 0

    Returns dict with entropy arrays, h_max, bin_edges, events.
    """
    n_bins = CONFIG["N_BINS"]
    h_max = np.log(n_bins)

    log(f"[Entropy] Computing rolling entropy (bins={n_bins}, H_max={h_max:.4f})...")

    # Fixed global bin edges from percentiles (avoids outlier distortion)
    lo = np.percentile(log_returns, 0.5)
    hi = np.percentile(log_returns, 99.5)
    bin_edges = np.linspace(lo, hi, n_bins + 1)

    entropy = {}
    for W in CONFIG["WINDOW_SIZES"]:
        ent_arr = np.full(len(log_returns), np.nan)
        for t in range(W, len(log_returns) + 1):
            window = log_returns[t - W : t]
            counts, _ = np.histogram(window, bins=bin_edges)
            total = counts.sum()
            if total == 0:
                continue
            probs = counts / total
            probs = probs[probs > 0]
            ent_arr[t - 1] = -np.sum(probs * np.log(probs))
        entropy[W] = ent_arr
        valid_count = np.sum(~np.isnan(ent_arr))
        log(f"[Entropy] W={W:3d} | {valid_count} valid points | "
            f"mean H={np.nanmean(ent_arr):.3f}")

    # Detect notable events using 63d entropy
    events = _detect_entropy_events(entropy, log_returns, h_max)

    return {
        "entropy": entropy,
        "h_max": h_max,
        "bin_edges": bin_edges,
        "events": events,
    }


def _detect_entropy_events(entropy, log_returns, h_max):
    """Scan 63d entropy for notable regime events using data-relative thresholds."""
    events = []
    W = 63
    ent = entropy.get(W)
    if ent is None:
        return events

    valid = ent[~np.isnan(ent)]
    if len(valid) < 10:
        return events

    # Use percentile-based thresholds (relative to actual data range)
    low_thresh = np.percentile(valid, 10)
    high_thresh = np.percentile(valid, 90)

    used_zones = []
    min_gap = 60

    def _zone_clear(idx):
        for z in used_zones:
            if abs(idx - z) < min_gap:
                return False
        return True

    n = len(ent)

    # 1. Low entropy regimes (trending — bottom 10%)
    for i in range(W, n):
        if not np.isnan(ent[i]) and ent[i] < low_thresh and _zone_clear(i):
            events.append((i, "Low Entropy"))
            used_zones.append(i)
            if len(events) >= 2:
                break

    # 2. High entropy peaks (random walk — top 10%)
    for i in range(W, n):
        if not np.isnan(ent[i]) and ent[i] > high_thresh and _zone_clear(i):
            events.append((i, "High Entropy"))
            used_zones.append(i)
            if len(events) >= 4:
                break

    # 3. Largest entropy drops (regime shifts)
    drops = []
    for i in range(W + 10, n):
        if not np.isnan(ent[i]) and not np.isnan(ent[i - 10]):
            drops.append((ent[i] - ent[i - 10], i))
    drops.sort()
    for _, i in drops[:3]:
        if _zone_clear(i):
            events.append((i, "Regime Shift"))
            used_zones.append(i)

    events.sort(key=lambda x: x[0])
    log(f"[Events] Detected {len(events)} regime events.")
    return events[:5]


# =============================================================================
# MODULE 2 : RENDERING
# =============================================================================

def draw_flowchart(ax, current_entropy, h_max):
    """Draw Shannon Entropy pipeline flowchart in the top panel."""
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_facecolor(THEME["BG"])

    mid = 0.50
    rng = np.random.RandomState(42)

    # ── Helpers ──────────────────────────────────────────────────────────

    def box(x, y, w, h, ec, label):
        p = FancyBboxPatch(
            (x - w / 2, y - h / 2), w, h,
            boxstyle="round,pad=0.06",
            facecolor=THEME["BG"], edgecolor=ec, linewidth=1.8, zorder=2,
        )
        ax.add_patch(p)
        ax.text(x, y + h / 2 + 0.06, label, ha="center", va="bottom",
                color=ec, fontsize=10, fontweight="bold", zorder=3)

    def arrow(x1, y1, x2, y2, color=THEME["WHITE"], ls="-", lw=1.5):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="-|>", color=color,
                                    lw=lw, linestyle=ls, mutation_scale=15),
                    zorder=4)

    # ── 1. Price Data ────────────────────────────────────────────────────
    box(0.8, mid, 1.1, 0.55, "#888888", "Price Data")
    # Mini price line inside
    px = np.linspace(0.35, 1.25, 40)
    py = mid + 0.08 * np.cumsum(rng.randn(40) * 0.15)
    py = mid + (py - py.mean()) * 0.8
    ax.plot(px, py, color=THEME["PRICE"], linewidth=0.9, alpha=0.6, zorder=3)

    # ── 2. Arrow ──────────────────────────────────────────────────────────
    arrow(1.35, mid, 1.85, mid)

    # ── 3. Log Returns ────────────────────────────────────────────────────
    box(2.6, mid, 1.1, 0.55, THEME["PRICE"], "Log Returns")
    # Mini bar chart inside
    bx = np.linspace(2.20, 3.00, 12)
    bh = rng.randn(12) * 0.06
    for i, (x, h) in enumerate(zip(bx, bh)):
        c = THEME["PRICE"] if h >= 0 else THEME["MAX_ENT"]
        ax.bar(x, h, width=0.05, bottom=mid, color=c, alpha=0.5, zorder=3)

    # ── 4. Arrow ──────────────────────────────────────────────────────────
    arrow(3.15, mid, 3.65, mid)

    # ── 5. Rolling Window ─────────────────────────────────────────────────
    box(4.5, mid, 1.2, 0.55, THEME["ENTROPY_MED"], "Window (W)")
    # Bracket annotation inside
    ax.text(4.5, mid + 0.04, "W = 63", ha="center", va="center",
            color=THEME["ENTROPY_MED"], fontsize=9, fontfamily="monospace",
            alpha=0.8, zorder=3)

    # ── 6. Arrow ──────────────────────────────────────────────────────────
    arrow(5.10, mid, 5.65, mid)

    # ── 7. Histogram Binning ──────────────────────────────────────────────
    box(6.5, mid, 1.2, 0.55, THEME["ENTROPY_LONG"], "Histogram")
    # Mini histogram inside
    hx = np.linspace(6.10, 6.90, 8)
    hh = np.array([0.04, 0.08, 0.14, 0.18, 0.15, 0.10, 0.06, 0.03])
    for x, h in zip(hx, hh):
        ax.bar(x, h, width=0.08, bottom=mid - 0.10, color=THEME["ENTROPY_LONG"],
               alpha=0.45, zorder=3)

    # ── 8. Arrow ──────────────────────────────────────────────────────────
    arrow(7.10, mid, 7.65, mid)

    # ── 9. Entropy H(X) ──────────────────────────────────────────────────
    box(8.5, mid, 1.2, 0.55, THEME["ENTROPY_XLNG"], "H(X)")

    # Live entropy value inside
    if not np.isnan(current_entropy):
        pct = current_entropy / h_max * 100
        ax.text(8.5, mid + 0.05, f"H = {current_entropy:.3f}", ha="center",
                va="center", color=THEME["ENTROPY_XLNG"], fontsize=10,
                fontfamily="monospace", fontweight="bold", alpha=0.9, zorder=5)
        ax.text(8.5, mid - 0.10, f"({pct:.0f}% of max)", ha="center",
                va="center", color=THEME["TEXT"], fontsize=8,
                fontfamily="monospace", alpha=0.6, zorder=5)



def render_worker(args):
    """Parallel worker: renders a single timelapse frame."""
    (frame_idx, data_end_idx, data, total_frames, temp_dir) = args

    try:
        # Unpack data
        prices = data["prices"]
        log_returns = data["log_returns"]
        dates = data["dates"]
        entropy = data["entropy"]
        h_max = data["h_max"]
        bin_edges = data["bin_edges"]
        events = data["events"]

        # Slice to current time
        end = data_end_idx
        p = prices[:end]
        lr = log_returns[:end]
        d = dates[:end]

        # Current 63d entropy for flowchart
        ent_63 = entropy[63]
        current_h = ent_63[end - 1] if end > 0 and not np.isnan(ent_63[end - 1]) else np.nan

        # ── Figure Setup ──────────────────────────────────────────────────
        fig = plt.figure(figsize=(19.2, 10.8), dpi=100, facecolor=THEME["BG"])
        gs = gridspec.GridSpec(
            4, 1,
            height_ratios=[1.0, 3, 1.8, 1.2],
            hspace=0.10,
            left=0.07, right=0.95, top=0.90, bottom=0.08,
        )
        ax_flow  = fig.add_subplot(gs[0])
        ax_price = fig.add_subplot(gs[1])
        ax_ent   = fig.add_subplot(gs[2])
        ax_dist  = fig.add_subplot(gs[3])

        # ── Flowchart ─────────────────────────────────────────────────────
        draw_flowchart(ax_flow, current_h, h_max)

        axes = [ax_price, ax_ent, ax_dist]

        # ── Common Styling (Hawkes pattern) ───────────────────────────────
        for ax in axes:
            ax.set_facecolor(THEME["BG"])
            ax.tick_params(axis="both", which="major", labelsize=12,
                           colors=THEME["TEXT"], direction="in", length=5)
            ax.tick_params(axis="both", which="minor",
                           colors=THEME["TEXT"], direction="in", length=3)
            for spine in ax.spines.values():
                spine.set_color(THEME["SPINE"])
                spine.set_linewidth(0.6)
            ax.yaxis.grid(True, linewidth=0.3, alpha=0.45, color=THEME["GRID"])
            ax.xaxis.grid(False)

        # Hide x-tick labels on upper panels
        plt.setp(ax_price.get_xticklabels(), visible=False)
        plt.setp(ax_ent.get_xticklabels(), visible=False)

        # ── Panel 1: Price ────────────────────────────────────────────────
        x_idx = np.arange(len(prices))

        ax_price.plot(x_idx[:end], p, color=THEME["PRICE"],
                     linewidth=2.0, zorder=3, label=CONFIG["TICKER"])

        # Fixed axis limits
        price_margin = (prices.max() - prices.min()) * 0.08
        ax_price.set_xlim(0, len(prices))
        ax_price.set_ylim(prices.min() - price_margin,
                          prices.max() + price_margin)
        ax_price.set_ylabel("Price (USD)", color=THEME["TEXT"],
                           fontsize=14, fontweight="bold")

        # Annotations (events revealed so far)
        for evt_idx, label in events:
            if evt_idx < end:
                evt_p = prices[evt_idx]
                ax_price.annotate(
                    label,
                    xy=(evt_idx, evt_p),
                    xytext=(evt_idx, evt_p + price_margin * 0.7),
                    fontsize=11, color=THEME["WHITE"], fontweight="bold",
                    ha="center",
                    arrowprops=dict(
                        arrowstyle="-|>", color=THEME["WHITE"],
                        lw=1.3, connectionstyle="arc3,rad=0.15",
                    ),
                    bbox=dict(boxstyle="round,pad=0.3",
                              facecolor=THEME["BG"],
                              edgecolor=THEME["SPINE"], alpha=0.9),
                    zorder=10,
                )

        # ── Panel 2: Entropy Lines ────────────────────────────────────────
        for i, W in enumerate(CONFIG["WINDOW_SIZES"]):
            ent_arr = entropy[W][:end]
            valid = ~np.isnan(ent_arr)
            if np.any(valid):
                ax_ent.plot(x_idx[:end][valid], ent_arr[valid],
                           color=ENTROPY_COLORS[i], linewidth=1.6,
                           alpha=0.95, zorder=3,
                           label=f"H({CONFIG['WINDOW_LABELS'][i]})")

        # H_max reference line
        ax_ent.axhline(h_max, color=THEME["MAX_ENT"], linewidth=1.0,
                       linestyle="--", alpha=0.6, zorder=2,
                       label=r"$H_{max}$")
        ax_ent.text(len(prices) * 0.98, h_max * 1.02,
                    r"$H_{max} = \ln(N)$",
                    color=THEME["MAX_ENT"], fontsize=11, ha="right",
                    alpha=0.7, zorder=5)

        ax_ent.set_xlim(0, len(prices))
        ax_ent.set_ylim(0, h_max * 1.15)
        ax_ent.set_ylabel(r"Shannon Entropy  $H(X)$", color=THEME["TEXT"],
                          fontsize=14, fontweight="bold")

        # ── Panel 3: Return Distribution ──────────────────────────────────
        W_hist = 63
        if end >= W_hist:
            window_ret = log_returns[end - W_hist : end]
            ax_dist.hist(window_ret, bins=bin_edges, density=True,
                        color=THEME["HIST_FILL"], alpha=0.55,
                        edgecolor=THEME["HIST_FILL"], linewidth=0.5,
                        zorder=2, label="Return Dist (63d)")

            # Annotate current entropy on histogram
            if not np.isnan(current_h):
                ax_dist.text(0.97, 0.88,
                            f"H = {current_h:.3f}",
                            transform=ax_dist.transAxes, ha="right",
                            color=THEME["ENTROPY_MED"], fontsize=13,
                            fontfamily="monospace", fontweight="bold",
                            zorder=5)

        ax_dist.set_ylabel("Density", color=THEME["TEXT"],
                          fontsize=14, fontweight="bold")
        ax_dist.set_xlabel("Log Return", color=THEME["TEXT"], fontsize=14)

        # ── Legend (bottom of figure) ─────────────────────────────────────
        h1, l1 = ax_price.get_legend_handles_labels()
        h2, l2 = ax_ent.get_legend_handles_labels()
        h3, l3 = ax_dist.get_legend_handles_labels()
        all_h = h1 + h2 + h3
        all_l = l1 + l2 + l3

        if all_h:
            leg = fig.legend(
                all_h, all_l,
                loc="lower center", ncol=len(all_l),
                fontsize=12, frameon=True, fancybox=False,
                borderpad=0.5, handlelength=2.5, columnspacing=2.0,
                bbox_to_anchor=(0.52, 0.002),
                edgecolor=THEME["SPINE"],
                facecolor=THEME["BG"],
            )
            for txt in leg.get_texts():
                txt.set_color(THEME["WHITE"])

        # ── Title / HUD ──────────────────────────────────────────────────
        progress = frame_idx / max(total_frames - 1, 1) * 100
        fig.suptitle(
            "Shannon Entropy  //  Market Regime Analysis",
            fontsize=20, fontweight="bold", color=THEME["WHITE"],
            y=0.965,
        )
        fig.text(
            0.95, 0.965,
            f"Day {end}/{len(prices)}  |  {progress:.0f}%",
            fontsize=13, color=THEME["TEXT"], ha="right",
            fontfamily="monospace",
        )
        fig.text(
            0.07, 0.940,
            r"$H(X) = -\sum_{i=1}^{n} \, p_i \, \ln(p_i)$",
            fontsize=13, color="#999999", fontfamily="serif",
        )

        # ── Save ──────────────────────────────────────────────────────────
        out_path = os.path.join(temp_dir, f"frame_{frame_idx:04d}.png")
        fig.savefig(out_path, dpi=100, facecolor=THEME["BG"])
        plt.close(fig)
        return True

    except Exception as e:
        print(f"[Error] Frame {frame_idx}: {e}")
        plt.close("all")
        return False


def run_render_manager(data):
    """Manages parallel frame rendering with smart resume."""
    total_frames = CONFIG["FPS"] * CONFIG["DURATION_SEC"]
    n_data = len(data["prices"])
    min_window = max(CONFIG["WINDOW_SIZES"]) + 10  # 262

    # Map frames to data indices (expanding window)
    step = max(1, (n_data - min_window) // total_frames)
    indices = [min(min_window + i * step, n_data) for i in range(total_frames)]
    indices[-1] = n_data

    if not os.path.exists(CONFIG["TEMP_DIR"]):
        os.makedirs(CONFIG["TEMP_DIR"])

    # Smart Resume
    tasks = []
    for i, idx in enumerate(indices):
        frame_path = os.path.join(CONFIG["TEMP_DIR"], f"frame_{i:04d}.png")
        if not os.path.exists(frame_path) or os.path.getsize(frame_path) == 0:
            tasks.append((i, idx, data, total_frames, CONFIG["TEMP_DIR"]))

    if not tasks:
        log("[Render] All frames exist. Skipping render.")
        return

    log(f"[Render] Rendering {len(tasks)} / {total_frames} frames...")
    start_time = time.time()

    if len(tasks) > 10:
        cores = max(1, cpu_count() - 2)
        log(f"[Render] Using Multiprocessing Pool ({cores} cores).")
        try:
            with Pool(processes=cores) as pool:
                results = pool.map(render_worker, tasks, chunksize=2)
            success = sum(1 for r in results if r)
            log(f"[Render] {success}/{len(tasks)} frames rendered.")
        except Exception as e:
            log(f"[Error] Pool crashed: {e}. Falling back to serial.")
            for t_args in tasks:
                render_worker(t_args)
    else:
        log("[Render] Using Serial Processing.")
        for t_args in tasks:
            render_worker(t_args)

    elapsed = time.time() - start_time
    log(f"[Render] Completed in {elapsed:.1f}s")


# =============================================================================
# MODULE 3 : COMPILATION
# =============================================================================

def compile_video():
    """Compile rendered frames into a cinematic MP4."""
    log("[Compile] Assembling video...")

    frames = sorted([
        os.path.join(CONFIG["TEMP_DIR"], f)
        for f in os.listdir(CONFIG["TEMP_DIR"])
        if f.endswith(".png")
    ])

    if not frames:
        log("[Error] No frames found in temp directory!")
        return

    log(f"[Compile] Found {len(frames)} frames.")

    try:
        hold_count = CONFIG["FPS"] * 2
        frames_with_hold = frames + [frames[-1]] * hold_count

        clip = ImageSequenceClip(frames_with_hold, fps=CONFIG["FPS"])
        output = CONFIG["OUTPUT_FILE"]
        clip.write_videofile(
            output, codec="libx264", bitrate="15000k",
            audio=False, logger=None,
        )
        log(f"[Success] Video saved to: {output}")

    except Exception as e:
        log(f"[Error] Compilation failed: {e}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    if os.path.exists(CONFIG["LOG_FILE"]):
        os.remove(CONFIG["LOG_FILE"])

    h_max = np.log(CONFIG["N_BINS"])

    log("=" * 60)
    log("  SHANNON ENTROPY PIPELINE")
    log("  Information Theory // Market Regime Detection")
    log("=" * 60)
    log(f"    Ticker: {CONFIG['TICKER']}")
    log(f"    Window sizes: {CONFIG['WINDOW_SIZES']}")
    log(f"    Histogram bins: {CONFIG['N_BINS']}")
    log(f"    H_max = ln({CONFIG['N_BINS']}) = {h_max:.4f}")

    # 1. Fetch data
    prices, log_returns, dates = fetch_price_data()

    # 2. Compute rolling entropy
    result = compute_rolling_entropy(log_returns)
    result["prices"] = prices
    result["log_returns"] = log_returns
    result["dates"] = dates

    # 3. Render frames
    run_render_manager(result)

    # 4. Compile video
    compile_video()

    log("=" * 60)
    log("  PIPELINE FINISHED")
    log("=" * 60)


if __name__ == "__main__":
    main()
