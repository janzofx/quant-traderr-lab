"""
potts_lattice_pipeline.py
=========================
Project: Quant Trader Lab — Potts Model 3D Lattice
Author: quant.traderr (Instagram)
License: MIT

Description:
    Cinematic 3D Potts model on a cubic lattice, themed for quant finance.
    Each voxel represents a market participant whose "regime state" (Bull,
    Bear, Sideways, Hedged) evolves via Metropolis-Hastings Monte Carlo.
    The lattice is rendered as a translucent cube with a small circle
    annotation highlighting a cluster region.

    Pipeline Steps:
    1.  **Simulation**: 3D Potts lattice (20³ = 8000 spins) with 4 states.
    2.  **Rendering**: PyVista GPU-accelerated off-screen static image generation.

Dependencies:
    pip install numpy pyvista matplotlib pillow
"""

import os
import shutil
import time
import numpy as np
import pyvista as pv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from io import BytesIO

# =============================================================================
# CONFIGURATION
# =============================================================================

CONFIG = {
    # Lattice
    "GRID_SIZE": 20,             # 20x20x20 cubic lattice
    "Q_STATES": 4,               # 4 regime states
    "TEMPERATURE": 0.55,         # Market noise (lower = more clustering/divergence)
    "J_COUPLING": 1.0,           # Coupling between neighbors
    "MC_STEPS": 80000,           # Monte Carlo sweeps
    "NUM_SNAPSHOTS": 60,         # Snapshots captured during simulation
    "SEED": 42,

    # Output Image
    "RESOLUTION": (1920, 1080),
    "OUTPUT_FILE": "Potts_Lattice_Regimes.png",
    "LOG_FILE": "potts_lattice_pipeline.log",

    # Circle annotation (highlight cluster region)
    "CIRCLE_CENTER": (0.55, 0.55, 0.55),  # Normalized position on cube face
    "CIRCLE_RADIUS": 0.12,                 # Small circle
}

# Quant finance regime colors (RGB tuples for PyVista)
REGIME_COLORS = {
    0: (0.00, 0.83, 0.67),   # Bull     — Cyan/Teal
    1: (0.91, 0.26, 0.58),   # Bear     — Magenta
    2: (0.96, 0.65, 0.14),   # Sideways — Amber
    3: (0.49, 0.72, 0.85),   # Hedged   — Steel Blue
}

REGIME_NAMES = ["Bull", "Bear", "Sideways", "Hedged"]

# =============================================================================
# UTILS
# =============================================================================

def log(msg):
    timestamp = time.strftime("%H:%M:%S")
    formatted = f"[{timestamp}] {msg}"
    print(formatted)
    try:
        with open(CONFIG["LOG_FILE"], "a") as f:
            f.write(formatted + "\n")
    except:
        pass

def compute_energy_per_spin(lattice, J):
    """Compute average Potts energy per spin (periodic BC)."""
    energy = 0
    for axis in range(3):
        shifted = np.roll(lattice, 1, axis=axis)
        energy -= J * np.sum(lattice == shifted)
    return energy / lattice.size

REGIME_MPL_COLORS = ['#00d4aa', '#e84393', '#f5a623', '#7eb8da']

# =============================================================================
# MODULE 1: SIMULATION — 3D Potts Lattice
# =============================================================================

def run_potts_simulation():
    """
    Runs a 3D Potts model simulation on an LxLxL cubic lattice.
    Returns a list of snapshot arrays (each LxLxL) at regular intervals.
    """
    np.random.seed(CONFIG["SEED"])
    L = CONFIG["GRID_SIZE"]
    Q = CONFIG["Q_STATES"]
    T = CONFIG["TEMPERATURE"]
    J = CONFIG["J_COUPLING"]
    steps = CONFIG["MC_STEPS"]
    n_snaps = CONFIG["NUM_SNAPSHOTS"]

    log(f"[Sim] Initializing {L}x{L}x{L} lattice ({L**3} spins, Q={Q})...")
    lattice = np.random.randint(0, Q, size=(L, L, L))

    snap_interval = max(1, steps // n_snaps)
    snapshots = [lattice.copy()]

    log(f"[Sim] Running {steps} MC steps (T={T}, J={J})...")
    t0 = time.time()

    for step in range(steps):
        # Pick random site
        x, y, z = np.random.randint(0, L, size=3)
        current_state = lattice[x, y, z]
        proposed_state = np.random.randint(0, Q)
        if proposed_state == current_state:
            continue

        # Calculate energy change from 6 nearest neighbors (periodic BC)
        delta_e = 0
        for dx, dy, dz in [(1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)]:
            nx_, ny_, nz_ = (x+dx) % L, (y+dy) % L, (z+dz) % L
            neighbor = lattice[nx_, ny_, nz_]
            # Potts: E = -J * delta(s_i, s_j)
            delta_e += J * (1 if neighbor == current_state else 0)
            delta_e -= J * (1 if neighbor == proposed_state else 0)

        # Metropolis acceptance
        if delta_e <= 0 or np.random.rand() < np.exp(-delta_e / T):
            lattice[x, y, z] = proposed_state

        if (step + 1) % snap_interval == 0:
            snapshots.append(lattice.copy())

        if (step + 1) % (steps // 5) == 0:
            log(f"[Sim] Step {step+1}/{steps}")

    elapsed = time.time() - t0
    log(f"[Sim] Done in {elapsed:.1f}s. {len(snapshots)} snapshots captured.")

    # Log final regime distribution
    final = snapshots[-1]
    for q in range(Q):
        count = np.sum(final == q)
        pct = count / final.size * 100
        log(f"[Sim] {REGIME_NAMES[q]}: {count} ({pct:.1f}%)")

    # Compute metrics at each snapshot for chart overlays
    log("[Sim] Computing regime & energy histories...")
    regime_history = np.zeros((len(snapshots), Q))
    energy_history = np.zeros(len(snapshots))
    mc_steps_axis = np.linspace(0, steps, len(snapshots))

    for i, snap in enumerate(snapshots):
        for q in range(Q):
            regime_history[i, q] = np.sum(snap == q) / snap.size * 100
        energy_history[i] = compute_energy_per_spin(snap, J)

    return snapshots, regime_history, energy_history, mc_steps_axis


# =============================================================================
# MODULE 2: RENDERING — PyVista GPU
# =============================================================================

def create_circle_annotation(center, radius, normal=(1, 0, 0), n_pts=64):
    """Create a 3D circle mesh for the annotation highlight."""
    theta = np.linspace(0, 2 * np.pi, n_pts, endpoint=False)

    # Build circle in XY plane, then rotate to face the given normal
    cx = radius * np.cos(theta)
    cy = radius * np.sin(theta)
    cz = np.zeros(n_pts)

    # Simple rotation: if normal is (1,0,0), circle is in YZ plane
    normal = np.array(normal, dtype=float)
    normal /= np.linalg.norm(normal)

    # Rodrigues rotation from (0,0,1) to normal
    z_axis = np.array([0.0, 0.0, 1.0])
    v = np.cross(z_axis, normal)
    s = np.linalg.norm(v)
    c = np.dot(z_axis, normal)

    if s < 1e-6:
        if c > 0:
            R = np.eye(3)
        else:
            R = -np.eye(3)
    else:
        vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        R = np.eye(3) + vx + vx @ vx * (1 - c) / (s * s)

    pts = np.column_stack([cx, cy, cz])
    pts = (R @ pts.T).T
    pts += np.array(center)

    # Close the loop
    lines = []
    for i in range(n_pts):
        lines.extend([2, i, (i + 1) % n_pts])

    mesh = pv.PolyData(pts)
    mesh.lines = np.array(lines)
    return mesh


def render_left_panel(n_visible, regime_history, energy_history, mc_steps):
    """Render the full left panel: title, math formulas, two charts, watermark."""
    PANEL_W, PANEL_H = 1056, 1080
    n_pts = len(regime_history)
    n_visible = min(n_visible, n_pts)

    fig = plt.figure(figsize=(PANEL_W / 100, PANEL_H / 100), dpi=100)
    fig.patch.set_facecolor('#0b0b0b')

    # ---- Title ----
    fig.text(0.50, 0.965, 'POTTS MODEL', ha='center', color='#ffffff',
             fontsize=22, fontfamily='monospace', fontweight='bold')
    fig.text(0.50, 0.935, 'MARKET REGIME LATTICE', ha='center', color='#b0b0b0',
             fontsize=14, fontfamily='monospace')

    # ---- Mathematical formulas ----
    fig.text(0.50, 0.900,
             r'$\mathcal{H} = -J \sum_{\langle i,j \rangle} \delta(\sigma_i, \sigma_j)$',
             ha='center', color='#ffffff', fontsize=14)
    fig.text(0.50, 0.870,
             r'$L=20 \quad Q=4 \quad T=0.55 \quad J=1.0$',
             ha='center', color='#666666', fontsize=10, fontfamily='monospace')

    # ---- Top chart: Regime Population ----
    ax1 = fig.add_axes([0.15, 0.50, 0.78, 0.30])
    x_full = mc_steps / 1000
    x = mc_steps[:n_visible] / 1000

    for q in range(4):
        ax1.plot(x_full, regime_history[:, q],
                 color=REGIME_MPL_COLORS[q], alpha=0.12, linewidth=0.7)
        ax1.plot(x, regime_history[:n_visible, q],
                 color=REGIME_MPL_COLORS[q], linewidth=1.8, label=REGIME_NAMES[q])

    ax1.set_facecolor('#0d0d0d')
    ax1.set_title('Regime Population', color='#999999', fontsize=9,
                   fontfamily='monospace', pad=6)
    ax1.set_ylabel('Population %', color='#666666', fontsize=7, fontfamily='monospace')
    ax1.set_xlabel('MC Steps (×10³)', color='#666666', fontsize=7, fontfamily='monospace')
    ax1.set_xlim(0, mc_steps[-1] / 1000)
    ax1.set_ylim(0, 55)
    ax1.tick_params(colors='#555555', labelsize=6)
    ax1.grid(True, color='#1a1a1a', linewidth=0.5)
    for spine in ax1.spines.values():
        spine.set_color('#2a2a2a')
    ax1.legend(loc='upper right', fontsize=6, facecolor='#0b0b0b',
               edgecolor='#333333', labelcolor='#888888')
    if n_visible < n_pts:
        ax1.axvline(x=x[-1], color='#ffffff', alpha=0.25, linewidth=0.7, linestyle='--')

    # ---- Bottom chart: Energy per Spin ----
    ax2 = fig.add_axes([0.15, 0.10, 0.78, 0.30])
    ax2.plot(x_full, energy_history, color='#e84393', alpha=0.12, linewidth=0.7)
    ax2.plot(x, energy_history[:n_visible], color='#e84393', linewidth=1.8)

    ax2.set_facecolor('#0d0d0d')
    ax2.set_title('Energy / Spin', color='#999999', fontsize=9,
                   fontfamily='monospace', pad=6)
    ax2.set_ylabel('E / N', color='#666666', fontsize=7, fontfamily='monospace')
    ax2.set_xlabel('MC Steps (×10³)', color='#666666', fontsize=7, fontfamily='monospace')
    ax2.set_xlim(0, mc_steps[-1] / 1000)
    ax2.tick_params(colors='#555555', labelsize=6)
    ax2.grid(True, color='#1a1a1a', linewidth=0.5)
    for spine in ax2.spines.values():
        spine.set_color('#2a2a2a')
    if n_visible < n_pts:
        ax2.axvline(x=x[-1], color='#ffffff', alpha=0.25, linewidth=0.7, linestyle='--')

    # ---- Watermark ----
    fig.text(0.50, 0.02, '@quant.traderr', ha='center', color='#555555',
             fontsize=9, fontfamily='monospace')

    buf = BytesIO()
    fig.savefig(buf, format='png', facecolor=fig.get_facecolor(), dpi=100)
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert('RGB')


def render_static_image(snapshots, regime_history, energy_history, mc_steps):
    """Renders a single static image using PyVista GPU off-screen rendering."""
    L = CONFIG["GRID_SIZE"]
    W, H = CONFIG["RESOLUTION"]
    output_file = CONFIG["OUTPUT_FILE"]

    PANEL_W = 1056
    CUBE_W = W - PANEL_W  # 864px for the 3D cube

    log(f"[Render] Setting up PyVista ({CUBE_W}x{H}, GPU off-screen)...")
    pv.global_theme.background = '#0b0b0b'
    pv.global_theme.anti_aliasing = "msaa"
    pv.global_theme.font.family = "courier"
    pv.global_theme.font.color = "white"

    plotter = pv.Plotter(window_size=[CUBE_W, H], off_screen=True)

    # --- Build voxel geometry ---
    grid = pv.ImageData(dimensions=(L+1, L+1, L+1))
    grid.origin = (0, 0, 0)
    grid.spacing = (1.0/L, 1.0/L, 1.0/L)

    # Final state colors
    final_snap = snapshots[-1]
    colors = np.zeros((L**3, 3))
    for q, col in REGIME_COLORS.items():
        mask = final_snap.flatten() == q
        colors[mask] = col

    grid.cell_data["RegimeColor"] = (colors * 255).astype(np.uint8)

    # Add the cube mesh
    cube_actor = plotter.add_mesh(
        grid, scalars="RegimeColor", rgb=True,
        show_edges=False, opacity=1.0,
        lighting=True, pbr=True, metallic=0.15, roughness=0.5,
    )

    # --- Wireframe outline of the cube ---
    outline = grid.outline()
    plotter.add_mesh(outline, color="#444444", line_width=2)

    # --- Circle annotation (small highlight on cube face) ---
    circle_center = CONFIG["CIRCLE_CENTER"]
    circle_radius = CONFIG["CIRCLE_RADIUS"]
    circle = create_circle_annotation(
        center=(circle_center[0], circle_center[1], -0.002),
        radius=circle_radius,
        normal=(0, 0, -1),
    )
    circle_actor = plotter.add_mesh(
        circle, color="#ff1493", line_width=3, opacity=0.9,
    )

    # --- Subtle grid floor ---
    floor = pv.Plane(
        center=(0.5, -0.15, 0.5), direction=(0, 1, 0),
        i_size=2.5, j_size=2.5, i_resolution=30, j_resolution=30,
    )
    plotter.add_mesh(floor, style="wireframe", color="#1a1a1a", line_width=0.5)

    # --- Lighting ---
    plotter.remove_all_lights()
    light1 = pv.Light(position=(3, 4, 3), focal_point=(0.5, 0.5, 0.5), intensity=0.8)
    light2 = pv.Light(position=(-2, 3, -1), focal_point=(0.5, 0.5, 0.5), intensity=0.4)
    light3 = pv.Light(position=(0, -2, 3), focal_point=(0.5, 0.5, 0.5), intensity=0.3)
    plotter.add_light(light1)
    plotter.add_light(light2)
    plotter.add_light(light3)

    log("[Render] Generating 3D representation...")

    # --- Camera view similar to final frame of video ---
    angle = 345  # End of the sweep
    rad = np.radians(angle)
    dist = 3.2
    elev = 0.65

    cam_x = 0.5 + dist * np.cos(rad) * np.cos(elev)
    cam_y = 0.5 + dist * np.sin(elev)
    cam_z = 0.5 + dist * np.sin(rad) * np.cos(elev)

    plotter.camera_position = [
        (cam_x, cam_y, cam_z),
        (0.5, 0.5, 0.5),
        (0, 1, 0),
    ]

    # Render off-screen snapshot to numpy array and convert
    img_array = plotter.screenshot()
    right_panel = Image.fromarray(img_array).convert('RGB')
    
    plotter.close()

    log("[Render] Stitching left and right panels...")
    # Stitch panels
    n_vis = len(regime_history)
    left_panel = render_left_panel(n_vis, regime_history, energy_history, mc_steps)
    
    final_frame = Image.new('RGB', (W, H), (11, 11, 11))
    final_frame.paste(left_panel, (0, 0))
    final_frame.paste(right_panel, (PANEL_W, 0))
    final_frame.save(output_file)

    log(f"[Success] Static image saved: {os.path.abspath(output_file)}")

# =============================================================================
# MAIN
# =============================================================================

def main():
    if os.path.exists(CONFIG["LOG_FILE"]):
        os.remove(CONFIG["LOG_FILE"])

    log("=" * 60)
    log("  POTTS MODEL — MARKET REGIME LATTICE")
    log("  3D Cubic Lattice Visualization")
    log("  @quant.traderr")
    log("=" * 60)

    # 1. Simulation
    snapshots, regime_history, energy_history, mc_steps = run_potts_simulation()

    # 2. Render static image (PyVista GPU)
    render_static_image(snapshots, regime_history, energy_history, mc_steps)

    log("=" * 60)
    log("  PIPELINE FINISHED")
    log("=" * 60)


if __name__ == "__main__":
    main()
