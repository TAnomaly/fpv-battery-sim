"""
FPV Drone Battery Performance Simulator
========================================
A physics-based simulation tool for LiPo battery performance, flight time
prediction, and power consumption analysis for FPV quadcopters.

Battery physics enhanced by PyBaMM (Python Battery Mathematical Modelling)
electrochemical modelling when available.

Author: FPV Battery Sim
Date:   2026-04-15
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict
import warnings
import time

# ─── PyBaMM optional integration ────────────────────────────────────────────
try:
    import pybamm
    PYBAMM_AVAILABLE = True
except ImportError:
    PYBAMM_AVAILABLE = False
    warnings.warn("PyBaMM not found — using empirical battery model only.")


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1  —  Configuration Data Classes
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class BatteryConfig:
    """LiPo battery specification."""
    num_cells: int = 4                          # e.g. 4S, 6S
    capacity_mah: float = 1500.0                # mAh
    internal_resistance_mohm_per_cell: float = 8.0   # mΩ per cell (typical 5-15 mΩ)
    peukert_exponent: float = 1.05              # 1.0 = ideal, 1.05-1.15 for LiPo
    max_continuous_c_rate: float = 30.0         # max sustained C-rate
    cutoff_voltage_per_cell: float = 3.3        # V — low-voltage cutoff
    nominal_voltage_per_cell: float = 3.7       # V
    full_voltage_per_cell: float = 4.2          # V
    # Temperature
    ambient_temp_c: float = 25.0
    thermal_resistance: float = 2.0             # °C/W — pack insulation factor
    temp_coefficient: float = -0.005            # capacity loss per °C above 25°C

    @property
    def capacity_ah(self) -> float:
        return self.capacity_mah / 1000.0

    @property
    def nominal_voltage(self) -> float:
        return self.num_cells * self.nominal_voltage_per_cell

    @property
    def full_voltage(self) -> float:
        return self.num_cells * self.full_voltage_per_cell

    @property
    def cutoff_voltage(self) -> float:
        return self.num_cells * self.cutoff_voltage_per_cell

    @property
    def internal_resistance_ohm(self) -> float:
        return (self.internal_resistance_mohm_per_cell / 1000.0) * self.num_cells


@dataclass
class MotorConfig:
    """Brushless motor specification."""
    kv_rating: int = 2300                       # RPM/V
    max_current_amps: float = 30.0              # max motor current
    motor_resistance_mohm: float = 60.0         # motor winding resistance mΩ
    motor_efficiency: float = 0.85              # peak motor efficiency
    no_load_current_amps: float = 0.5           # idle current draw


@dataclass
class PropConfig:
    """Propeller specification."""
    diameter_inches: float = 5.0
    pitch_inches: float = 4.3
    # Blade efficiency — function of pitch/diameter ratio
    # Typical: 0.55–0.75 for FPV props
    blade_efficiency: float = 0.65


@dataclass
class ESCConfig:
    """Electronic Speed Controller specification."""
    efficiency: float = 0.93                    # typical 90–95%
    max_current_amps: float = 35.0
    resistance_mohm: float = 3.0               # ESC FET resistance mΩ


@dataclass
class DroneConfig:
    """Complete drone specification."""
    name: str = "FPV Quad"
    weight_grams: float = 350.0                 # all-up weight including battery
    num_motors: int = 4
    battery: BatteryConfig = field(default_factory=BatteryConfig)
    motor: MotorConfig = field(default_factory=MotorConfig)
    prop: PropConfig = field(default_factory=PropConfig)
    esc: ESCConfig = field(default_factory=ESCConfig)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2  —  Battery Physics Model
# ═══════════════════════════════════════════════════════════════════════════════

class LiPoDischargeModel:
    """
    Empirical LiPo discharge curve model.

    Models the OCV (Open Circuit Voltage) vs. State-of-Charge relationship
    using a polynomial fit to typical LiPo discharge curves. Peukert's Law
    adjusts effective capacity at high discharge rates. Internal resistance
    causes voltage sag proportional to current.
    """

    # Physically-measured LiPo OCV data points per cell (no-load, 25 °C).
    # Source: typical LiCoO2/graphite pouch cell — representative of FPV LiPo.
    # SOC = 0 → 3.00 V (fully depleted)  |  SOC = 1 → 4.20 V (fully charged)
    _OCV_SOC = np.array([0.00, 0.05, 0.10, 0.20, 0.30, 0.40,
                         0.50, 0.60, 0.70, 0.80, 0.90, 1.00])
    _OCV_V   = np.array([3.00, 3.10, 3.30, 3.55, 3.62, 3.68,
                         3.75, 3.82, 3.90, 4.00, 4.10, 4.20])

    def __init__(self, config: BatteryConfig):
        self.cfg = config

    def ocv_per_cell(self, soc: float) -> float:
        """Open-circuit voltage per cell at given SOC [0,1] — linear interpolation."""
        return float(np.interp(np.clip(soc, 0.0, 1.0),
                               self._OCV_SOC, self._OCV_V))

    def ocv_pack(self, soc: float) -> float:
        return self.ocv_per_cell(soc) * self.cfg.num_cells

    def effective_capacity_ah(self, c_rate: float) -> float:
        """
        Peukert's Law: C_eff = C_nominal * (1 / c_rate)^(n-1)
        Reduces usable capacity at high discharge rates.
        """
        n = self.cfg.peukert_exponent
        # Peukert factor — apply only when c_rate > 1C to avoid inflation at low rates
        if c_rate <= 1.0:
            return self.cfg.capacity_ah
        factor = (1.0 / c_rate) ** (n - 1.0)
        return self.cfg.capacity_ah * factor

    def terminal_voltage(self, soc: float, current_amps: float,
                         temp_c: float = 25.0) -> float:
        """
        Terminal voltage with IR sag and temperature compensation.

        V_terminal = OCV - I * R_internal  (discharge convention: I > 0)
        Temperature affects both OCV and internal resistance.
        """
        # Temperature correction: OCV shifts ~-1mV/°C on LiPo
        temp_delta = temp_c - 25.0
        temp_ocv_shift = -0.001 * temp_delta * self.cfg.num_cells
        ocv = self.ocv_pack(soc) + temp_ocv_shift

        # Temperature effect on internal resistance: R increases when cold
        temp_r_factor = 1.0 + 0.008 * max(0, 25.0 - temp_c)
        r_int = self.cfg.internal_resistance_ohm * temp_r_factor

        # Voltage sag under load
        v_sag = current_amps * r_int
        return ocv - v_sag

    def pack_temperature(self, ambient_c: float, power_loss_w: float,
                         dt_s: float, current_temp_c: float) -> float:
        """Simple thermal model — steady state approximation."""
        temp_rise = power_loss_w * self.cfg.thermal_resistance
        target_temp = ambient_c + temp_rise
        # First-order thermal lag (tau ≈ 60 s for a small LiPo pack)
        tau = 60.0
        alpha = 1.0 - np.exp(-dt_s / tau)
        return current_temp_c + alpha * (target_temp - current_temp_c)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3  —  PyBaMM Enhanced Battery Model
# ═══════════════════════════════════════════════════════════════════════════════

class PyBAMMBatteryModel:
    """
    Electrochemical battery model using PyBaMM's Single Particle Model (SPM).

    Pre-generates discharge curves at multiple C-rates, then provides an
    OCV(SOC, C-rate) interpolation table used by the main simulation engine.
    This gives more physically-grounded voltage curves than the empirical model.
    """

    C_RATES = [0.5, 1.0, 2.0, 5.0, 10.0, 20.0]

    def __init__(self, capacity_ah: float, num_cells: int):
        self.capacity_ah = capacity_ah
        self.num_cells = num_cells
        self._curves: Dict[float, Tuple[np.ndarray, np.ndarray]] = {}
        self._ready = False
        if PYBAMM_AVAILABLE:
            self._build_curves()

    def _build_curves(self):
        """Run PyBaMM SPM at each C-rate to generate V(SOC) lookup tables."""
        print("  [PyBaMM] Building electrochemical discharge curves...")
        t0 = time.time()

        try:
            # SPM is fast; DFN is more accurate but slower
            model = pybamm.lithium_ion.SPM()
            # Chen2020 parameters: NMC/graphite pouch cell — closest to LiPo
            param = pybamm.ParameterValues("Chen2020")

            for c_rate in self.C_RATES:
                current_a = c_rate * self.capacity_ah  # A

                # Scale nominal cell capacity to match our pack
                # Chen2020 is for a 5 Ah cell; we normalise by C-rate discharge
                experiment = pybamm.Experiment([
                    f"Discharge at {c_rate}C until 3.0V",
                ])

                sim = pybamm.Simulation(model, parameter_values=param,
                                        experiment=experiment)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    sim.solve()

                sol = sim.solution
                # Extract time, voltage, capacity
                t_arr   = sol["Time [s]"].entries
                v_arr   = sol["Terminal voltage [V]"].entries
                cap_arr = sol["Discharge capacity [A.h]"].entries

                # Normalise voltage to our cell count
                # PyBaMM SPM gives single-cell voltage (3.0–4.2 V range)
                # Scale to our pack
                v_cell = v_arr  # single cell
                v_pack = v_cell * self.num_cells

                # Convert to SOC axis (1 → 0)
                q_total = cap_arr[-1] if cap_arr[-1] > 0 else self.capacity_ah
                soc_arr = 1.0 - cap_arr / q_total
                soc_arr = np.clip(soc_arr, 0.0, 1.0)

                # Sort by SOC (ascending) for interpolation
                idx = np.argsort(soc_arr)
                self._curves[c_rate] = (soc_arr[idx], v_pack[idx])

            elapsed = time.time() - t0
            print(f"  [PyBaMM] Done — {len(self.C_RATES)} curves in {elapsed:.1f}s")
            self._ready = True

        except Exception as exc:
            warnings.warn(f"[PyBaMM] Curve generation failed ({exc}). "
                          "Falling back to empirical model.")
            self._ready = False

    def terminal_voltage(self, soc: float, c_rate: float,
                         current_amps: float, r_int_ohm: float) -> Optional[float]:
        """
        Return interpolated terminal voltage from PyBaMM curves.
        Returns None if PyBaMM data is unavailable.
        """
        if not self._ready:
            return None

        # Find the two bounding C-rate curves
        c_rates_avail = sorted(self._curves.keys())
        c_rate_clamped = np.clip(c_rate, c_rates_avail[0], c_rates_avail[-1])

        # Find bounding indices
        idx_lo = max(0, np.searchsorted(c_rates_avail, c_rate_clamped) - 1)
        idx_hi = min(len(c_rates_avail) - 1, idx_lo + 1)

        def interp_curve(cr):
            soc_ax, v_ax = self._curves[cr]
            return float(np.interp(np.clip(soc, 0, 1), soc_ax, v_ax))

        cr_lo = c_rates_avail[idx_lo]
        cr_hi = c_rates_avail[idx_hi]

        if cr_lo == cr_hi:
            v = interp_curve(cr_lo)
        else:
            alpha = (c_rate_clamped - cr_lo) / (cr_hi - cr_lo)
            v = (1 - alpha) * interp_curve(cr_lo) + alpha * interp_curve(cr_hi)

        # Still apply IR sag on top of PyBaMM OCV
        v_terminal = v - current_amps * r_int_ohm
        return v_terminal


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 4  —  Motor & ESC Power Model
# ═══════════════════════════════════════════════════════════════════════════════

class MotorESCModel:
    """
    Models total electrical power consumption from throttle input.

    Power relationship:
      - Thrust ∝ throttle^2  (blade momentum theory)
      - Current ∝ throttle^2 (at constant voltage)
      - Total system current = (motor currents) / ESC_efficiency
    """

    def __init__(self, drone: DroneConfig):
        self.drone = drone
        self._precompute_prop_constant()

    def _precompute_prop_constant(self):
        """
        Estimate maximum system current (A) at full throttle and nominal voltage.

        Uses propeller affinity laws + motor KV to estimate full-throttle power.
        Propeller power  P ∝ (RPM/RPM_max)^3
        RPM_max = KV * V_nominal
        """
        motor = self.drone.motor
        prop  = self.drone.prop
        batt  = self.drone.battery
        esc   = self.drone.esc

        # Propeller coefficient (crude but physically grounded)
        # P_prop [W] = Kt * rho * D^5 * (n/60)^3  — dimensional analysis
        # We estimate max mechanical power per motor from typical FPV data:
        # Use weight-to-power ratio: ~5:1 thrust:weight ratio at hover
        # Hover power: P_hover = Weight * sqrt(Weight / (2 * rho * A_disk))
        rho = 1.225          # kg/m³ air density
        g   = 9.81           # m/s²
        D   = prop.diameter_inches * 0.0254   # metres
        A   = np.pi * (D / 2) ** 2            # disc area per motor

        mass_kg = self.drone.weight_grams / 1000.0
        weight_N = mass_kg * g
        thrust_per_motor_N = weight_N / self.drone.num_motors

        # Ideal hover power per motor (actuator disk theory)
        P_hover_mech = thrust_per_motor_N * np.sqrt(
            thrust_per_motor_N / (2.0 * rho * A)
        )
        P_hover_mech /= prop.blade_efficiency   # account for non-ideal blade

        # Electrical power = mechanical / (motor_eff * ESC_eff)
        P_hover_elec = P_hover_mech / (motor.motor_efficiency * esc.efficiency)

        # At hover (~60% throttle), P ∝ throttle^3 → P_max at 100% throttle
        self.P_max_per_motor = P_hover_elec / (0.60 ** 3)

        # Maximum current from battery at full throttle
        V_nom = batt.nominal_voltage
        self.I_max_total = (self.P_max_per_motor * self.drone.num_motors) / V_nom

    def current_draw(self, throttle: float, battery_voltage: float) -> Tuple[float, float]:
        """
        Compute total battery current and motor power at a given throttle.

        Args:
            throttle:         0.0 – 1.0
            battery_voltage:  actual pack voltage (V)

        Returns:
            (total_current_A, total_power_W)
        """
        motor = self.drone.motor
        esc   = self.drone.esc
        batt  = self.drone.battery

        throttle = np.clip(throttle, 0.0, 1.0)

        # P_max_per_motor already includes motor + ESC losses (derived from
        # actuator-disk hover power divided through both efficiencies).
        # Power scales with throttle^3 (blade-momentum / BET propeller model).
        P_elec_per_motor = self.P_max_per_motor * (throttle ** 3)

        # Current per motor from the DC bus (I = P / V_pack)
        # Motors + ESC sit in parallel across the full pack voltage.
        V = max(battery_voltage, 1.0)
        I_per_motor = P_elec_per_motor / V + motor.no_load_current_amps
        I_per_motor = np.clip(I_per_motor, 0.0, motor.max_current_amps)

        I_total = I_per_motor * self.drone.num_motors
        P_total = I_total * V
        return float(I_total), float(P_total)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 5  —  Flight Profile Definitions
# ═══════════════════════════════════════════════════════════════════════════════

class FlightProfile:
    """Defines throttle time-series for different flight modes."""

    def __init__(self, name: str, throttle_fn, description: str = ""):
        self.name = name
        self.throttle_fn = throttle_fn   # callable: t (seconds) → throttle [0,1]
        self.description = description

    def throttle_at(self, t: float) -> float:
        return float(np.clip(self.throttle_fn(t), 0.0, 1.0))

    # ── Factory methods ──────────────────────────────────────────────────────

    @classmethod
    def hover(cls) -> "FlightProfile":
        """Steady 60% throttle with gentle oscillations."""
        def _fn(t):
            base = 0.60
            # Slight sinusoidal variation simulating altitude adjustments
            return base + 0.04 * np.sin(2 * np.pi * t / 8.0)
        return cls("Hover", _fn,
                   "Stable hover ~60% throttle with minor corrections")

    @classmethod
    def cruise(cls) -> "FlightProfile":
        """Cruising flight with moderate throttle variations."""
        def _fn(t):
            base = 0.62
            wave = 0.08 * np.sin(2 * np.pi * t / 15.0)
            gust = 0.04 * np.sin(2 * np.pi * t / 4.3 + 1.2)
            return base + wave + gust
        return cls("Cruise", _fn,
                   "Forward cruising ~60–70% throttle")

    @classmethod
    def aggressive(cls) -> "FlightProfile":
        """
        Aggressive freestyle profile: bursts to 80–100%, sharp throttle cuts,
        and recovery periods.
        """
        rng = np.random.default_rng(42)

        # Pre-generate event sequence for reproducibility
        SEGMENT_DURATION = 3.0   # seconds per manoeuvre

        def _fn(t):
            seg = int(t / SEGMENT_DURATION)
            phase = (t % SEGMENT_DURATION) / SEGMENT_DURATION

            rng2 = np.random.default_rng(seg * 9973)
            manoeuvre = rng2.choice(["punch", "dive", "cruise", "roll"])

            if manoeuvre == "punch":       # full-throttle punch
                return 0.90 + 0.10 * np.sin(np.pi * phase)
            elif manoeuvre == "dive":      # throttle cut then recovery
                if phase < 0.3:
                    return 0.10
                else:
                    return 0.50 + 0.40 * ((phase - 0.3) / 0.7)
            elif manoeuvre == "roll":      # rapid throttle modulation
                return 0.70 + 0.25 * np.sin(2 * np.pi * phase * 3)
            else:                          # cruising recovery segment
                return 0.55 + 0.10 * np.sin(2 * np.pi * phase)

        return cls("Aggressive", _fn,
                   "Freestyle: full-throttle punches, dives, rolls, recoveries")

    @classmethod
    def racing(cls) -> "FlightProfile":
        """Race track simulation: high sustained throttle with brief cornering dips."""
        def _fn(t):
            # 8-second lap cycle
            lap_phase = (t % 8.0) / 8.0
            # Straight sections: ~85–95% throttle
            # Corner sections: ~60–70% throttle
            if lap_phase < 0.15 or (0.45 < lap_phase < 0.55) or lap_phase > 0.85:
                return 0.65 + 0.05 * np.sin(2 * np.pi * lap_phase * 4)
            else:
                return 0.88 + 0.08 * np.sin(2 * np.pi * lap_phase * 2)
        return cls("Racing", _fn,
                   "Race track: high throttle straights, braking for corners")

    @classmethod
    def custom(cls, time_points: List[float],
               throttle_points: List[float]) -> "FlightProfile":
        """
        User-defined throttle profile via linear interpolation between
        (time, throttle) waypoints.
        """
        t_arr = np.array(time_points)
        th_arr = np.array(throttle_points)
        def _fn(t):
            return float(np.interp(t, t_arr, th_arr,
                                   left=th_arr[0], right=th_arr[-1]))
        return cls("Custom", _fn, "User-defined throttle profile")


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 6  —  Simulation Engine
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class SimulationState:
    """Snapshot of drone state at one time step."""
    time_s: float
    throttle: float
    voltage: float
    current_a: float
    power_w: float
    soc: float                  # State of Charge [0, 1]
    capacity_used_mah: float
    temperature_c: float
    c_rate: float

@dataclass
class SimulationResult:
    """Complete simulation output."""
    drone_name: str
    profile_name: str
    profile_description: str
    flight_time_s: float
    termination_reason: str
    states: List[SimulationState]
    used_pybamm: bool

    # ── Derived statistics (computed post-run) ───────────────────────────────
    avg_current_a: float = 0.0
    avg_power_w: float = 0.0
    avg_voltage: float = 0.0
    peak_current_a: float = 0.0
    peak_power_w: float = 0.0
    capacity_used_mah: float = 0.0
    peak_temperature_c: float = 0.0
    peak_current_duration_s: float = 0.0   # seconds above 80% max current
    avg_c_rate: float = 0.0

    def compute_stats(self):
        arr = self.states
        if not arr:
            return
        self.avg_current_a        = float(np.mean([s.current_a for s in arr]))
        self.avg_power_w          = float(np.mean([s.power_w   for s in arr]))
        self.avg_voltage          = float(np.mean([s.voltage   for s in arr]))
        self.peak_current_a       = float(np.max([s.current_a  for s in arr]))
        self.peak_power_w         = float(np.max([s.power_w    for s in arr]))
        self.capacity_used_mah    = arr[-1].capacity_used_mah
        self.peak_temperature_c   = float(np.max([s.temperature_c for s in arr]))
        self.avg_c_rate           = float(np.mean([s.c_rate for s in arr]))

    @property
    def flight_time_str(self) -> str:
        m = int(self.flight_time_s // 60)
        s = int(self.flight_time_s % 60)
        return f"{m}m {s:02d}s"


class DroneSimulator:
    """
    Main simulation engine.

    Runs a discrete time-step simulation of a drone flight using:
      - LiPo electrochemical model (empirical or PyBaMM-enhanced)
      - Motor/ESC power model
      - Coulomb counting for SOC tracking
      - Low-voltage and over-temperature cutoff detection
    """

    def __init__(self, drone: DroneConfig,
                 dt_s: float = 0.1,
                 use_pybamm: bool = True,
                 max_flight_time_s: float = 3600.0):
        self.drone = drone
        self.dt = dt_s
        self.max_flight_time = max_flight_time_s
        self.use_pybamm = use_pybamm and PYBAMM_AVAILABLE

        # Sub-models
        self.lipo    = LiPoDischargeModel(drone.battery)
        self.powertrain = MotorESCModel(drone)

        # PyBaMM enhanced model
        self._pybamm_model: Optional[PyBAMMBatteryModel] = None
        if self.use_pybamm:
            self._pybamm_model = PyBAMMBatteryModel(
                capacity_ah=drone.battery.capacity_ah,
                num_cells=drone.battery.num_cells,
            )
            if not self._pybamm_model._ready:
                self.use_pybamm = False

    def run(self, profile: FlightProfile) -> SimulationResult:
        """
        Execute the simulation for a given flight profile.
        Returns a SimulationResult with all time-step data.
        """
        batt = self.drone.battery
        states: List[SimulationState] = []

        # ── Initial conditions ───────────────────────────────────────────────
        soc          = 1.0                       # fully charged
        capacity_used_ah = 0.0
        temp_c       = batt.ambient_temp_c
        t            = 0.0
        termination  = "max_time_reached"

        while t <= self.max_flight_time:
            throttle = profile.throttle_at(t)

            # Step 1 — current voltage estimate (use previous SOC)
            if self.use_pybamm and self._pybamm_model is not None:
                capacity_ah_used_sofar = (1.0 - soc) * batt.capacity_ah
                c_rate_instant = (capacity_used_ah / batt.capacity_ah /
                                  max(t / 3600.0, 1e-9))
                v = self._pybamm_model.terminal_voltage(
                    soc, c_rate_instant, 0.0, batt.internal_resistance_ohm
                )
                if v is None:
                    v = self.lipo.terminal_voltage(soc, 0.0, temp_c)
            else:
                v = self.lipo.terminal_voltage(soc, 0.0, temp_c)

            # Step 2 — calculate current and power from powertrain
            I_draw, P_draw = self.powertrain.current_draw(throttle, v)

            # Step 3 — recalculate voltage with actual current (IR sag)
            if self.use_pybamm and self._pybamm_model is not None:
                c_rate = I_draw / batt.capacity_ah
                v = self._pybamm_model.terminal_voltage(
                    soc, c_rate, I_draw, batt.internal_resistance_ohm
                )
                if v is None:
                    v = self.lipo.terminal_voltage(soc, I_draw, temp_c)
            else:
                v = self.lipo.terminal_voltage(soc, I_draw, temp_c)

            # Step 4 — update Coulomb counter
            # Effective capacity from Peukert's law at current C-rate
            c_rate = I_draw / batt.capacity_ah if batt.capacity_ah > 0 else 0
            eff_cap = self.lipo.effective_capacity_ah(c_rate)

            # Temperature capacity correction
            temp_delta = temp_c - 25.0
            temp_cap_factor = 1.0 + batt.temp_coefficient * max(0, temp_delta)
            eff_cap *= temp_cap_factor

            # Capacity consumed this step
            dq = I_draw * (self.dt / 3600.0)          # Ah per step
            capacity_used_ah += dq

            # SOC from effective capacity (accounts for Peukert degradation)
            soc = max(0.0, 1.0 - capacity_used_ah / eff_cap)

            # Step 5 — thermal model
            P_loss = I_draw ** 2 * batt.internal_resistance_ohm
            temp_c = self.lipo.pack_temperature(batt.ambient_temp_c,
                                                P_loss, self.dt, temp_c)

            # Record state
            states.append(SimulationState(
                time_s=t,
                throttle=throttle,
                voltage=v,
                current_a=I_draw,
                power_w=P_draw,
                soc=soc,
                capacity_used_mah=capacity_used_ah * 1000.0,
                temperature_c=temp_c,
                c_rate=c_rate,
            ))

            # Step 6 — termination checks
            if v <= batt.cutoff_voltage:
                termination = "low_voltage_cutoff"
                break
            if soc <= 0.0:
                termination = "battery_depleted"
                break
            if temp_c > 60.0:
                termination = "over_temperature"
                break

            t += self.dt

        result = SimulationResult(
            drone_name       = self.drone.name,
            profile_name     = profile.name,
            profile_description = profile.description,
            flight_time_s    = t,
            termination_reason = termination,
            states           = states,
            used_pybamm      = self.use_pybamm,
        )
        result.compute_stats()
        return result


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 7  —  Visualization & Reports
# ═══════════════════════════════════════════════════════════════════════════════

class SimulationReporter:
    """Generates plots and printed summaries from SimulationResult objects."""

    # Colour palette
    _COLORS = {
        "voltage":    "#2196F3",
        "current":    "#F44336",
        "power":      "#FF9800",
        "soc":        "#4CAF50",
        "throttle":   "#9C27B0",
        "temp":       "#795548",
    }

    @staticmethod
    def print_summary(result: SimulationResult):
        """Print a formatted flight summary to stdout."""
        sep = "─" * 56
        print(f"\n{'═'*56}")
        print(f"  FLIGHT SIMULATION REPORT")
        print(f"  Drone:   {result.drone_name}")
        print(f"  Profile: {result.profile_name}  ({result.profile_description})")
        print(f"  Battery model: {'PyBaMM SPM (electrochemical)' if result.used_pybamm else 'Empirical LiPo'}")
        print(f"{'═'*56}")
        print(f"  Flight time      : {result.flight_time_str}")
        print(f"  Stop reason      : {result.termination_reason.replace('_', ' ')}")
        print(sep)
        print(f"  Avg voltage      : {result.avg_voltage:.2f} V")
        print(f"  Avg current      : {result.avg_current_a:.1f} A")
        print(f"  Avg power        : {result.avg_power_w:.1f} W")
        print(f"  Avg C-rate       : {result.avg_c_rate:.1f} C")
        print(sep)
        print(f"  Peak current     : {result.peak_current_a:.1f} A")
        print(f"  Peak power       : {result.peak_power_w:.1f} W")
        print(f"  Peak temp        : {result.peak_temperature_c:.1f} °C")
        print(sep)
        print(f"  Capacity used    : {result.capacity_used_mah:.0f} mAh")
        print(f"{'═'*56}\n")

    @classmethod
    def plot(cls, result: SimulationResult, show: bool = True,
             save_path: Optional[str] = None):
        """
        Generate a 6-panel dashboard plot:
          1. Voltage vs. Time
          2. Current vs. Time
          3. Power vs. Time
          4. State of Charge vs. Time
          5. Throttle Profile
          6. Battery Temperature
        """
        states = result.states
        t   = np.array([s.time_s / 60.0     for s in states])   # minutes
        v   = np.array([s.voltage           for s in states])
        I   = np.array([s.current_a         for s in states])
        P   = np.array([s.power_w           for s in states])
        soc = np.array([s.soc * 100.0       for s in states])
        th  = np.array([s.throttle * 100.0  for s in states])
        T   = np.array([s.temperature_c     for s in states])

        fig = plt.figure(figsize=(16, 10))
        fig.patch.set_facecolor("#1a1a2e")
        gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.32)

        model_tag = "PyBaMM SPM" if result.used_pybamm else "Empirical"
        fig.suptitle(
            f"FPV Battery Sim  ·  {result.drone_name}  ·  "
            f"{result.profile_name} Profile  ·  "
            f"Flight: {result.flight_time_str}  [{model_tag}]",
            color="white", fontsize=13, fontweight="bold", y=0.98
        )

        axes_specs = [
            (0, 0, "Voltage [V]",     "Pack Voltage",           v,   cls._COLORS["voltage"],  None),
            (0, 1, "Current [A]",     "Total Current Draw",     I,   cls._COLORS["current"],  None),
            (1, 0, "Power [W]",       "Total Power Consumption",P,   cls._COLORS["power"],    None),
            (1, 1, "SoC [%]",         "State of Charge",        soc, cls._COLORS["soc"],      (0, 105)),
            (2, 0, "Throttle [%]",    "Throttle Profile",       th,  cls._COLORS["throttle"], (0, 105)),
            (2, 1, "Temperature [°C]","Pack Temperature",       T,   cls._COLORS["temp"],     None),
        ]

        for row, col, ylabel, title, data, color, ylim in axes_specs:
            ax = fig.add_subplot(gs[row, col])
            ax.set_facecolor("#0d0d1a")
            ax.plot(t, data, color=color, linewidth=1.4, alpha=0.92)
            ax.fill_between(t, data, alpha=0.12, color=color)
            ax.set_xlabel("Time [min]", color="#aaaaaa", fontsize=9)
            ax.set_ylabel(ylabel, color="#aaaaaa", fontsize=9)
            ax.set_title(title, color="white", fontsize=10, fontweight="bold")
            ax.tick_params(colors="#888888", labelsize=8)
            for spine in ax.spines.values():
                spine.set_edgecolor("#333355")
            ax.grid(True, color="#2a2a4a", linewidth=0.5, alpha=0.7)
            if ylim:
                ax.set_ylim(ylim)

            # Annotate averages on key panels
            if title == "Pack Voltage":
                ax.axhline(np.mean(data), color=color, linestyle="--",
                           linewidth=0.8, alpha=0.6, label=f"avg {np.mean(data):.2f}V")
                ax.legend(fontsize=7, facecolor="#1a1a2e", labelcolor="white",
                          edgecolor="#333355")
            if title == "State of Charge":
                ax.axhline(20, color="#ff5722", linestyle=":", linewidth=1.0,
                           alpha=0.8, label="20% warning")
                ax.legend(fontsize=7, facecolor="#1a1a2e", labelcolor="white",
                          edgecolor="#333355")

        plt.tight_layout(rect=[0, 0, 1, 0.96])

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight",
                        facecolor=fig.get_facecolor())
            print(f"  Plot saved → {save_path}")

        if show:
            plt.show()
        else:
            plt.close(fig)

    @classmethod
    def plot_comparison(cls, results: List[SimulationResult],
                        show: bool = True, save_path: Optional[str] = None):
        """
        Overlay multiple simulation results on voltage and SoC plots
        for easy profile comparison.
        """
        profile_colors = ["#2196F3", "#F44336", "#4CAF50", "#FF9800", "#9C27B0"]

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.patch.set_facecolor("#1a1a2e")

        for ax in axes:
            ax.set_facecolor("#0d0d1a")
            ax.tick_params(colors="#888888")
            ax.grid(True, color="#2a2a4a", linewidth=0.5)
            for spine in ax.spines.values():
                spine.set_edgecolor("#333355")

        ax_v, ax_s = axes

        for i, result in enumerate(results):
            color = profile_colors[i % len(profile_colors)]
            states = result.states
            t   = np.array([s.time_s / 60.0  for s in states])
            v   = np.array([s.voltage        for s in states])
            soc = np.array([s.soc * 100.0    for s in states])
            lbl = f"{result.profile_name} ({result.flight_time_str})"
            ax_v.plot(t, v, color=color, linewidth=1.6, label=lbl)
            ax_s.plot(t, soc, color=color, linewidth=1.6, label=lbl)

        for ax, ylabel, title in [
            (ax_v, "Pack Voltage [V]",  "Voltage vs. Time — Profile Comparison"),
            (ax_s, "State of Charge [%]","SoC vs. Time — Profile Comparison"),
        ]:
            ax.set_xlabel("Time [min]", color="#aaaaaa")
            ax.set_ylabel(ylabel, color="#aaaaaa")
            ax.set_title(title, color="white", fontweight="bold")
            ax.legend(facecolor="#1a1a2e", labelcolor="white", edgecolor="#333355",
                      fontsize=8)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight",
                        facecolor=fig.get_facecolor())
            print(f"  Comparison plot saved → {save_path}")

        if show:
            plt.show()
        else:
            plt.close(fig)

    @classmethod
    def plot_discharge_curves(cls, battery: BatteryConfig,
                               c_rates: Optional[List[float]] = None,
                               pybamm_model: Optional["PyBAMMBatteryModel"] = None,
                               show: bool = True, save_path: Optional[str] = None):
        """
        Plot OCV discharge curves at multiple C-rates for the battery,
        comparing empirical model vs. PyBaMM if available.
        """
        if c_rates is None:
            c_rates = [0.5, 1.0, 2.0, 5.0, 10.0, 20.0]

        lipo = LiPoDischargeModel(battery)
        soc_arr = np.linspace(1.0, 0.0, 300)
        colors = plt.cm.plasma(np.linspace(0.15, 0.9, len(c_rates)))

        fig, ax = plt.subplots(figsize=(10, 6))
        fig.patch.set_facecolor("#1a1a2e")
        ax.set_facecolor("#0d0d1a")
        ax.tick_params(colors="#888888")
        ax.grid(True, color="#2a2a4a", linewidth=0.5)
        for spine in ax.spines.values():
            spine.set_edgecolor("#333355")

        for c_rate, color in zip(c_rates, colors):
            # Current at this C-rate
            I = c_rate * battery.capacity_ah
            v_empirical = [lipo.terminal_voltage(s, I) for s in soc_arr]
            ax.plot(soc_arr * 100, v_empirical, color=color,
                    linewidth=1.8, linestyle="-",
                    label=f"{c_rate}C empirical")

            if pybamm_model and pybamm_model._ready and c_rate in pybamm_model._curves:
                soc_pb, v_pb = pybamm_model._curves[c_rate]
                ax.plot(soc_pb * 100, v_pb, color=color,
                        linewidth=2.0, linestyle="--", alpha=0.7,
                        label=f"{c_rate}C PyBaMM")

        ax.axhline(battery.cutoff_voltage, color="#ff5722", linestyle=":",
                   linewidth=1.2, label="Cutoff voltage")
        ax.set_xlabel("State of Charge [%]", color="#aaaaaa")
        ax.set_ylabel("Pack Voltage [V]", color="#aaaaaa")
        ax.set_title(
            f"LiPo Discharge Curves — {battery.num_cells}S {battery.capacity_mah:.0f}mAh",
            color="white", fontweight="bold"
        )
        ax.legend(facecolor="#1a1a2e", labelcolor="white",
                  edgecolor="#333355", fontsize=8, ncol=2)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight",
                        facecolor=fig.get_facecolor())
        if show:
            plt.show()
        else:
            plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 8  —  Quick-Run Helper
# ═══════════════════════════════════════════════════════════════════════════════

def run_standard_suite(drone: DroneConfig,
                       use_pybamm: bool = True,
                       save_plots: bool = True) -> List[SimulationResult]:
    """
    Run all four standard flight profiles and print a comparison table.
    Saves plots to the working directory if save_plots=True.
    """
    print(f"\n{'━'*56}")
    print(f"  FPV Battery Sim — {drone.name}")
    print(f"  Battery: {drone.battery.num_cells}S {drone.battery.capacity_mah:.0f}mAh")
    print(f"  Weight:  {drone.weight_grams}g")
    print(f"  PyBaMM:  {'enabled' if (use_pybamm and PYBAMM_AVAILABLE) else 'disabled / not available'}")
    print(f"{'━'*56}")

    simulator = DroneSimulator(drone, dt_s=0.1, use_pybamm=use_pybamm)
    profiles = [
        FlightProfile.hover(),
        FlightProfile.cruise(),
        FlightProfile.aggressive(),
        FlightProfile.racing(),
    ]

    results = []
    for profile in profiles:
        print(f"\nRunning: {profile.name} ...")
        result = simulator.run(profile)
        results.append(result)
        SimulationReporter.print_summary(result)

        if save_plots:
            fname = (f"{drone.name.replace(' ', '_')}_{profile.name}_"
                     f"{'pybamm' if result.used_pybamm else 'empirical'}.png")
            path = f"/Users/kayra-mac/Desktop/Quadcopter Battery Sim/{fname}"
            SimulationReporter.plot(result, show=False, save_path=path)

    # Comparison plot
    if save_plots:
        cmp_path = (f"/Users/kayra-mac/Desktop/Quadcopter Battery Sim/"
                    f"{drone.name.replace(' ','_')}_comparison.png")
        SimulationReporter.plot_comparison(results, show=False, save_path=cmp_path)

    # Discharge curve plot
    if save_plots:
        pybamm_mdl = simulator._pybamm_model if simulator.use_pybamm else None
        dc_path = (f"/Users/kayra-mac/Desktop/Quadcopter Battery Sim/"
                   f"{drone.name.replace(' ','_')}_discharge_curves.png")
        SimulationReporter.plot_discharge_curves(
            drone.battery, pybamm_model=pybamm_mdl,
            show=False, save_path=dc_path
        )

    # Summary comparison table
    print(f"\n{'─'*70}")
    print(f"  {'Profile':<14} {'Flight Time':<14} {'Avg Power':>10} {'Peak A':>8} {'Cap Used':>10}")
    print(f"{'─'*70}")
    for r in results:
        print(f"  {r.profile_name:<14} {r.flight_time_str:<14} "
              f"{r.avg_power_w:>9.1f}W {r.peak_current_a:>7.1f}A "
              f"{r.capacity_used_mah:>8.0f}mAh")
    print(f"{'─'*70}\n")

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 9  —  Entry Point
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    # ── Example 1: Typical 5" freestyle quad (4S 1500mAh) ───────────────────
    freestyle_quad = DroneConfig(
        name="Freestyle 5\" 4S",
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

    run_standard_suite(freestyle_quad, use_pybamm=True, save_plots=True)

    # ── Example 2: Long-range 7" cruiser (6S 3000mAh) ───────────────────────
    longrange_quad = DroneConfig(
        name="LongRange 7\" 6S",
        weight_grams=580.0,
        num_motors=4,
        battery=BatteryConfig(
            num_cells=6,
            capacity_mah=3000.0,
            internal_resistance_mohm_per_cell=7.0,
            peukert_exponent=1.04,
            cutoff_voltage_per_cell=3.4,
        ),
        motor=MotorConfig(
            kv_rating=1700,
            max_current_amps=28.0,
            motor_efficiency=0.87,
            no_load_current_amps=0.4,
        ),
        prop=PropConfig(diameter_inches=7.0, pitch_inches=3.5, blade_efficiency=0.72),
        esc=ESCConfig(efficiency=0.95, max_current_amps=35.0),
    )

    run_standard_suite(longrange_quad, use_pybamm=False, save_plots=True)

    # ── Example 3: Custom throttle profile demo ──────────────────────────────
    print("\n── Custom Throttle Profile Demo ──")
    custom_profile = FlightProfile.custom(
        time_points  = [0,  10,  20,  40,  60,  90, 120, 150, 180],
        throttle_points=[0.5, 0.8, 0.9, 0.6, 0.7, 0.5, 0.8, 0.4, 0.6],
    )
    sim_custom = DroneSimulator(freestyle_quad, dt_s=0.1, use_pybamm=False)
    result_custom = sim_custom.run(custom_profile)
    SimulationReporter.print_summary(result_custom)
    SimulationReporter.plot(
        result_custom, show=False,
        save_path="/Users/kayra-mac/Desktop/Quadcopter Battery Sim/custom_profile.png"
    )
