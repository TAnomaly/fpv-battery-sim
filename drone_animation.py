"""
FPV Drone — Live Flight Animation
==================================
2D side-view drone animation with real-time telemetry graphs.
Run directly:  python drone_animation.py
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.animation as animation
from matplotlib.patches import FancyArrowPatch
from matplotlib.lines import Line2D

# ── Import sim ───────────────────────────────────────────────────────────────
from fpv_battery_sim import (
    DroneConfig, BatteryConfig, MotorConfig, PropConfig, ESCConfig,
    FlightProfile, DroneSimulator,
)

# ═════════════════════════════════════════════════════════════════════════════
# DRONE SPRITE DRAWING
# ═════════════════════════════════════════════════════════════════════════════

def _draw_drone(ax, x, y, throttle, prop_angle, scale=1.0):
    """
    Draw a side-view quadcopter at (x, y).
    Returns list of artists (cleared each frame).
    """
    artists = []
    arm_len   = 0.12 * scale
    body_w    = 0.10 * scale
    body_h    = 0.03 * scale
    rotor_r   = 0.055 * scale
    prop_w    = 0.10 * scale   # half-blade length

    # ── Body ─────────────────────────────────────────────────────────────────
    body = mpatches.FancyBboxPatch(
        (x - body_w / 2, y - body_h / 2),
        body_w, body_h,
        boxstyle="round,pad=0.005",
        linewidth=1.4, edgecolor="#90caf9", facecolor="#1a237e",
        zorder=6,
    )
    ax.add_patch(body)
    artists.append(body)

    # ── Camera bump ──────────────────────────────────────────────────────────
    cam = mpatches.Circle((x + body_w * 0.38, y - body_h * 0.6),
                          body_h * 0.55, color="#ffb300", zorder=7)
    ax.add_patch(cam)
    artists.append(cam)

    # ── Arms + rotors (2 visible in side view) ───────────────────────────────
    rotor_positions = [x - arm_len, x + arm_len]
    for rx in rotor_positions:
        # Arm
        arm_line = ax.plot([x, rx], [y, y + body_h * 0.1],
                           color="#546e7a", linewidth=2.5 * scale, zorder=5)[0]
        artists.append(arm_line)

        # Rotor ring
        ring = mpatches.Circle((rx, y + body_h * 0.1 + rotor_r * 0.3),
                                rotor_r, linewidth=1.2,
                                edgecolor="#29b6f6", facecolor="none", zorder=6)
        ax.add_patch(ring)
        artists.append(ring)

        # Spinning propeller blades
        ry = y + body_h * 0.1 + rotor_r * 0.3
        spin_color = _throttle_color(throttle)
        for blade_idx in range(2):
            angle = prop_angle + blade_idx * np.pi
            bx_end = rx + prop_w * np.cos(angle)
            by_end = ry + prop_w * 0.18 * np.sin(angle)
            blade = ax.plot([rx, bx_end], [ry, by_end],
                            color=spin_color, linewidth=2.8 * scale,
                            alpha=0.85, zorder=7)[0]
            artists.append(blade)

    # ── Downwash lines ───────────────────────────────────────────────────────
    if throttle > 0.3:
        for rx in rotor_positions:
            ry = y + body_h * 0.1 + rotor_r * 0.3
            n_lines = int(throttle * 5)
            for li in range(n_lines):
                ox = rx + np.random.uniform(-rotor_r * 0.6, rotor_r * 0.6)
                ln = ax.plot([ox, ox + np.random.uniform(-0.015, 0.015)],
                             [ry - rotor_r * 0.5,
                              ry - rotor_r * 0.5 - throttle * 0.06 * scale],
                             color="#29b6f6", linewidth=0.7, alpha=0.35, zorder=4)[0]
                artists.append(ln)

    return artists


def _throttle_color(throttle):
    """Blade colour: green → yellow → red with throttle."""
    if throttle < 0.5:
        return "#69f0ae"
    elif throttle < 0.8:
        return "#ffeb3b"
    else:
        return "#ff5252"


# ═════════════════════════════════════════════════════════════════════════════
# BACKGROUND DRAWING
# ═════════════════════════════════════════════════════════════════════════════

def _draw_background(ax):
    """Static sky + ground scene."""
    ax.set_facecolor("#0d0d1a")

    # Sky gradient via horizontal bands
    sky_colors = ["#0d1b4b", "#0d2a6e", "#0a3a8f", "#0d47a1", "#1565c0"]
    band_h = 1.0 / len(sky_colors)
    for i, col in enumerate(sky_colors):
        ax.axhspan(i * band_h, (i + 1) * band_h,
                   facecolor=col, alpha=0.5, zorder=0)

    # Stars
    rng = np.random.default_rng(42)
    star_x = rng.uniform(0, 1, 60)
    star_y = rng.uniform(0.35, 1.0, 60)
    ax.scatter(star_x, star_y, s=rng.uniform(0.3, 2.5, 60),
               color="white", alpha=0.5, zorder=1)

    # Mountain silhouettes
    mtn_x = [0.0, 0.08, 0.18, 0.30, 0.42, 0.50, 0.60, 0.70, 0.80, 0.88, 1.0]
    mtn_y = [0.10, 0.22, 0.16, 0.28, 0.20, 0.30, 0.18, 0.25, 0.14, 0.22, 0.10]
    ax.fill_between(mtn_x, 0, mtn_y, color="#1a237e", alpha=0.9, zorder=2)
    ax.fill_between(mtn_x, 0,
                    [v * 0.65 for v in mtn_y],
                    color="#111133", alpha=0.9, zorder=3)

    # Ground
    ax.axhspan(0, 0.10, facecolor="#1b5e20", alpha=0.9, zorder=2)
    ax.axhspan(0, 0.04, facecolor="#0a3d0a", alpha=1.0, zorder=3)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")


# ═════════════════════════════════════════════════════════════════════════════
# HUD OVERLAY
# ═════════════════════════════════════════════════════════════════════════════

def _make_hud_texts(ax):
    """Create text objects for HUD. Return dict of Text artists."""
    style = dict(transform=ax.transAxes, fontsize=8.5,
                 fontfamily="monospace", zorder=20)
    hud = {
        "title": ax.text(0.02, 0.97, "", color="#29b6f6",
                         fontweight="bold", fontsize=10, **{k: v for k, v in style.items() if k != "fontsize"},
                         va="top"),
        "time":  ax.text(0.02, 0.91, "", color="#e0e0e0", va="top", **style),
        "volt":  ax.text(0.02, 0.86, "", color="#2196F3", va="top", **style),
        "curr":  ax.text(0.02, 0.81, "", color="#F44336", va="top", **style),
        "soc":   ax.text(0.02, 0.76, "", color="#4CAF50", va="top", **style),
        "throt": ax.text(0.02, 0.71, "", color="#FF9800", va="top", **style),
        "batt_bar": ax.text(0.02, 0.65, "", color="#4CAF50", va="top",
                            fontsize=9, fontfamily="monospace",
                            transform=ax.transAxes, zorder=20),
    }
    return hud


def _update_hud(hud, state, drone_name, total_mah):
    soc_pct = state.soc * 100
    mins = int(state.time_s // 60)
    secs = int(state.time_s % 60)

    bar_len = 16
    filled = int(soc_pct / 100 * bar_len)
    bar = "█" * filled + "░" * (bar_len - filled)
    bar_color = "#4CAF50" if soc_pct > 40 else ("#FF9800" if soc_pct > 20 else "#F44336")

    hud["title"].set_text(f"◈ {drone_name}")
    hud["time"].set_text(f"⏱  {mins:02d}:{secs:02d}")
    hud["volt"].set_text(f"⚡ {state.voltage:.2f} V")
    hud["curr"].set_text(f"⚡ {state.current_a:.1f} A")
    hud["soc"].set_text(f"🔋 {soc_pct:.1f} %")
    hud["throt"].set_text(f"🎮 {state.throttle * 100:.0f} % throttle")
    hud["batt_bar"].set_text(f"[{bar}]")
    hud["batt_bar"].set_color(bar_color)


# ═════════════════════════════════════════════════════════════════════════════
# MAIN ANIMATION
# ═════════════════════════════════════════════════════════════════════════════

def animate_flight(drone: DroneConfig,
                   profile: FlightProfile,
                   use_pybamm: bool = False,
                   speed_multiplier: int = 8,
                   save_gif: bool = False,
                   gif_path: str = "flight_animation.gif"):
    """
    Run simulation then animate results.

    Parameters
    ----------
    speed_multiplier : int
        Animation playback speed vs real time (default 8× = fast enough
        to see the full flight in a reasonable window).
    save_gif : bool
        Save output as GIF (slow, use only for export).
    """
    print("▶  Running simulation…")
    engine = DroneSimulator(drone, dt_s=0.5, use_pybamm=use_pybamm)
    result = engine.run(profile)
    states = result.states
    N = len(states)
    print(f"✔  {N} states  ·  flight time {result.flight_time_str}")

    # Pre-compute arrays
    t_arr   = np.array([s.time_s / 60       for s in states])
    v_arr   = np.array([s.voltage           for s in states])
    I_arr   = np.array([s.current_a         for s in states])
    soc_arr = np.array([s.soc * 100         for s in states])
    th_arr  = np.array([s.throttle          for s in states])

    # Drone Y position: maps throttle to altitude band 0.22 … 0.78
    alt_raw = np.convolve(th_arr, np.ones(30) / 30, mode="same")  # smooth
    alt_arr = 0.22 + (alt_raw - alt_raw.min()) / (np.ptp(alt_raw) + 1e-9) * 0.56

    # X oscillation: slow sinusoid to look like real forward/back motion
    x_arr = 0.5 + 0.22 * np.sin(np.linspace(0, 4 * np.pi, N))

    # ── Figure layout ─────────────────────────────────────────────────────
    fig = plt.figure(figsize=(15, 7), facecolor="#0a0a1a")
    gs  = gridspec.GridSpec(3, 2, figure=fig,
                            left=0.04, right=0.96,
                            top=0.93, bottom=0.08,
                            wspace=0.35, hspace=0.55)

    ax_sky  = fig.add_subplot(gs[:, 0])   # full left — drone view
    ax_volt = fig.add_subplot(gs[0, 1])   # top right  — voltage
    ax_curr = fig.add_subplot(gs[1, 1])   # mid right  — current
    ax_soc  = fig.add_subplot(gs[2, 1])   # bot right  — SoC

    # Static background
    _draw_background(ax_sky)

    # ── Right panel style ──────────────────────────────────────────────────
    graph_axes = [ax_volt, ax_curr, ax_soc]
    graph_colors = ["#2196F3", "#F44336", "#4CAF50"]
    graph_labels = ["Voltage [V]", "Current [A]", "Battery [%]"]
    graph_titles = ["Pack Voltage", "Current Draw", "State of Charge"]
    graph_data   = [v_arr, I_arr, soc_arr]

    for ax, col, lbl, ttl in zip(graph_axes, graph_colors, graph_labels, graph_titles):
        ax.set_facecolor("#0d0d1a")
        ax.tick_params(colors="#888888", labelsize=7)
        ax.set_xlabel("Time [min]", color="#888888", fontsize=7)
        ax.set_ylabel(lbl, color=col, fontsize=7.5)
        ax.set_title(ttl, color="white", fontsize=8.5, fontweight="bold", pad=3)
        for spine in ax.spines.values():
            spine.set_edgecolor("#2a2a4a")
        ax.grid(True, color="#1e1e3a", linewidth=0.5)
        ax.set_xlim(0, t_arr[-1])

    ax_volt.set_ylim(v_arr.min() * 0.97, v_arr.max() * 1.02)
    ax_curr.set_ylim(0, I_arr.max() * 1.1)
    ax_soc.set_ylim(0, 105)

    # Static full trace (faint)
    for ax, data, col in zip(graph_axes, graph_data, graph_colors):
        ax.plot(t_arr, data, color=col, linewidth=0.6, alpha=0.18)

    # Low-SoC warning line
    ax_soc.axhline(20, color="#ff5722", linewidth=0.8, linestyle=":", alpha=0.7)

    # ── Live line objects ──────────────────────────────────────────────────
    live_lines = []
    for ax, col in zip(graph_axes, graph_colors):
        ln, = ax.plot([], [], color=col, linewidth=1.8, zorder=5)
        live_lines.append(ln)

    # Vertical time cursors
    cursors = [ax.axvline(0, color="#ffffff", linewidth=0.9,
                          linestyle="--", alpha=0.5, zorder=6)
               for ax in graph_axes]

    # HUD
    hud = _make_hud_texts(ax_sky)

    # Title
    model_tag = "PyBaMM" if (use_pybamm and result.used_pybamm) else "Empirical"
    fig.suptitle(
        f"FPV Battery Simulator  ·  {drone.name}  ·  "
        f"{profile.name} Profile  ·  [{model_tag}]",
        color="white", fontsize=11, fontweight="bold",
    )

    # Drone sprite state
    prop_angle = [0.0]
    drone_artists = []

    # ── Animation step ─────────────────────────────────────────────────────
    frame_step = max(1, speed_multiplier)
    frame_indices = list(range(0, N, frame_step)) + [N - 1]

    def _init():
        for ln in live_lines:
            ln.set_data([], [])
        return live_lines + cursors

    def _update(fi):
        i = frame_indices[fi]
        state = states[i]
        t_now = t_arr[i]

        # Clear previous drone sprite
        for art in drone_artists:
            art.remove()
        drone_artists.clear()

        # Prop spin speed ∝ throttle
        prop_angle[0] += state.throttle * 0.55
        arts = _draw_drone(ax_sky,
                           x=x_arr[i],
                           y=alt_arr[i],
                           throttle=state.throttle,
                           prop_angle=prop_angle[0],
                           scale=1.0)
        drone_artists.extend(arts)

        # Update graphs
        for ln, data in zip(live_lines, graph_data):
            ln.set_data(t_arr[:i + 1], data[:i + 1])
        for cur in cursors:
            cur.set_xdata([t_now, t_now])

        # HUD
        _update_hud(hud, state, drone.name, drone.battery.capacity_mah)

        return live_lines + cursors + drone_artists

    interval_ms = max(16, int(500 / frame_step))   # ~target 60fps feel
    anim = animation.FuncAnimation(
        fig,
        _update,
        frames=len(frame_indices),
        init_func=_init,
        interval=interval_ms,
        blit=False,
    )

    if save_gif:
        print(f"💾  Saving GIF → {gif_path}  (this may take a minute…)")
        writer = animation.PillowWriter(fps=24)
        anim.save(gif_path, writer=writer, dpi=90)
        print("✔  Saved.")
    else:
        plt.show()

    return anim


# ═════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    # ── Standard 5" freestyle quad ───────────────────────────────────────────
    drone = DroneConfig(
        name='Freestyle 5" 4S',
        weight_grams=380.0,
        num_motors=4,
        battery=BatteryConfig(
            num_cells=4,
            capacity_mah=1500.0,
            internal_resistance_mohm_per_cell=10.0,
            peukert_exponent=1.06,
            cutoff_voltage_per_cell=3.3,
        ),
        motor=MotorConfig(
            kv_rating=2400,
            max_current_amps=35.0,
            motor_efficiency=0.84,
            no_load_current_amps=0.6,
        ),
        prop=PropConfig(diameter_inches=5.1, pitch_inches=4.6, blade_efficiency=0.66),
        esc=ESCConfig(efficiency=0.94, max_current_amps=40.0),
    )

    # Change profile here: FlightProfile.hover() / .aggressive() / .cruising()
    profile = FlightProfile.aggressive()

    animate_flight(
        drone=drone,
        profile=profile,
        use_pybamm=False,   # True if PyBaMM installed
        speed_multiplier=6, # 6× real-time — whole flight plays in ~30s
        save_gif=False,     # True to export GIF
    )
