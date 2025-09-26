# --- loop_calculator.py -------------------------------------------------------
from dataclasses import dataclass
from typing import List, Tuple
import math

# =========================
# Segments & core utilities
# =========================

@dataclass
class Segment:
    L: float       # length [m]
    D: float       # inner diameter [m]
    eps: float     # roughness [m]
    K: float = 0.0 # minor-loss K (elbows/valves), optional

def area(D: float) -> float:
    return math.pi * D * D / 4.0

def reynolds(mdot: float, D: float, mu: float, A: float) -> float:
    return mdot * D / (mu * A)

def haaland_f(Re: float, rel_rough: float) -> float:
    """Haaland explicit Darcy friction factor."""
    if Re <= 0.0:
        return 1.0
    invs = -1.8 * math.log10((rel_rough / 3.7) ** 1.11 + 6.9 / Re)
    return 1.0 / (invs * invs)  # Darcy f

# Isothermal p^2 drop for one segment at mdot (updates f with Re)
def p2_drop_segment(p_in: float, mdot: float, seg: Segment, T: float, R: float, mu: float):
    A = area(seg.D)
    Re = reynolds(mdot, seg.D, mu, A)
    if 0.0 < Re < 2300.0:
        f = 64.0 / Re
    else:
        f = haaland_f(Re, seg.eps / seg.D)
    L_eff = seg.L + (seg.K * seg.D / f if f > 0.0 else 0.0)
    C = (f * L_eff / seg.D) * (R * T / (A * A))
    # p_in^2 - p_out^2 = C * mdot^2  ->  dp2 = C * mdot^2
    dp2 = C * (mdot ** 2)
    return dp2, f, Re

# Given p_out and mdot, march *backwards* through segments to get p_in
# IMPORTANT: `segs` must be in the order **pump discharge -> ... -> pump suction**
def propagate_loop_p2(p_out: float, mdot: float, segs: List[Segment], T: float, R: float, mu: float):
    p2 = p_out * p_out
    f_last = Re_last = None
    total_dp2 = 0.0
    for seg in segs:
        dp2, f, Re = p2_drop_segment(math.sqrt(p2 + 1e-30), mdot, seg, T, R, mu)
        total_dp2 += dp2
        p2 = max(1e-30, p2 - dp2)        # <-- decrease pressure-squared
        f_last, Re_last = f, Re
    p_in = math.sqrt(p2)
    return p_in, total_dp2, f_last, Re_last

# Loop mass flow that closes pressure-squared balance for given (p_in, p_out)
def mdot_from_loop(p_in: float, p_out: float, segs: List[Segment], T: float, R: float, mu: float):
    target_dp2 = max(0.0, p_out * p_out - p_in * p_in)

    def loop_dp2(mdot: float) -> float:
        _, total_dp2, _, _ = propagate_loop_p2(p_out, mdot, segs, T, R, mu)
        return total_dp2

    # Bracket mdot
    lo, hi = 0.0, 1e-2
    while loop_dp2(hi) < target_dp2:
        hi *= 2.0
        if hi > 1e3:
            raise RuntimeError("mdot bracket too small; increase upper bound.")

    # Bisection
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if loop_dp2(mid) < target_dp2:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)

# Volume-weighted mean pressure for given (p_out, mdot)
# Uses length-average p in each segment: p_avg = (2/3)*(p1^3 - p2^3)/(p1^2 - p2^2)
def mean_pressure_from_profile(p_out: float, mdot: float, segs: List[Segment], T: float, R: float, mu: float):
    Vtot = 0.0 # total volume
    num = 0.0 # numerator for pressure averaging
    p2 = p_out * p_out # start at outlet
    for seg in segs:
        dp2, f, Re = p2_drop_segment(math.sqrt(p2), mdot, seg, T, R, mu)
        p1      = math.sqrt(p2)                                  # upstream (higher)
        p2_end2 = max(1e-30, p2 - dp2)
        p2_end  = math.sqrt(p2_end2)                             # downstream (lower)
        p_avg_seg = (2.0/3.0)*((p1**3 - p2_end**3)/(p1*p1 - p2_end*p2_end))

        #denom = max(1e-30, p1 * p1 - p2_end * p2_end)
        #p_avg_seg = (2.0 / 3.0) * ((p1 ** 3 - p2_end ** 3) / denom)
        Vseg = area(seg.D) * seg.L
        num += Vseg * p_avg_seg
        Vtot += Vseg
        p2 = p2_end2  # move upstream for next segment
    return num / max(Vtot, 1e-30)

# ==================================
# NEW: Pump model with vacuum limiting
# ==================================

def _linear_interp_clamped(x: float, xy: List[Tuple[float, float]]) -> float:
    """x–y linear interpolation with clamping. 'xy' must be sorted by x."""
    if not xy:
        return 0.0
    if x <= xy[0][0]:
        return xy[0][1]
    if x >= xy[-1][0]:
        return xy[-1][1]
    for i in range(len(xy) - 1):
        x0, y0 = xy[i]
        x1, y1 = xy[i + 1]
        if x0 <= x <= x1:
            t = 0.0 if x1 == x0 else (x - x0) / (x1 - x0)
            return y0 + t * (y1 - y0)
    return xy[-1][1]

@dataclass
class PumpCurve:
    """
    Linear overpressure curve + vacuum limit.
      - Overpressure: Q(dp) = Q0 - s*dp   [L/min], dp in bar
      - Vacuum:      Qmax(Pin_abs_bar) from datasheet points (piecewise linear)

    The pump volumetric flow is the MIN of these two; mass flow uses inlet density.
    """
    Q0_Lpm: float
    s_Lpm_per_bar: float
    vac_curve_pts: List[Tuple[float, float]]  # (Pin_bar_abs, Qmax_Lpm), sorted by Pin

    def Q_overpressure(self, dp_bar: float) -> float:
        return max(0.0, self.Q0_Lpm - self.s_Lpm_per_bar * dp_bar)

    def Qmax_vacuum(self, p_in_bar_abs: float) -> float:
        return max(0.0, _linear_interp_clamped(p_in_bar_abs, self.vac_curve_pts))

    def mdot(self, dp_bar: float, p_in_abs_Pa: float, T: float, R: float) -> float:
        # volumetric at inlet conditions (L/min)
        Q_over = self.Q_overpressure(dp_bar)
        Q_vac  = self.Qmax_vacuum(p_in_abs_Pa / 1e5)
        Q_lpm  = min(Q_over, Q_vac)
        # convert to mass flow with inlet density
        Q_m3s  = Q_lpm / 1000.0 / 60.0
        rho_in = p_in_abs_Pa / (R * T)
        return rho_in * Q_m3s

# ======================================
# Closed-loop (asymmetric) and open-case
# ======================================

def solve_closed_loop_asym(p_mean_bar: float, segs: List[Segment], T: float, R: float, mu: float,
                           pump: PumpCurve, dp_max_bar: float = 3.0, tol: float = 1e-4):
    """
    Sealed, isothermal loop with arbitrary series of segments.
    Unknowns: p_in and Δp (p_out = p_in + Δp).
    Constraints:
      (1) Pump mdot(Δp, p_in) == Loop mdot(p_in -> p_out)
      (2) Volume-weighted mean pressure equals p_mean
    """
    p_mean = p_mean_bar * 1e5
    dp_lo, dp_hi = 0.0, dp_max_bar * 1e5
    best = None

    for _ in range(60):
        dp = 0.5 * (dp_lo + dp_hi)

        # Inner solve: find p_in so that mean pressure matches p_mean
        pin_lo = max(1e3, p_mean - 0.99 * dp)  # keep positive, inlet below mean
        pin_hi = p_mean

        def mean_err(p_in: float):
            p_out = p_in + dp
            md = mdot_from_loop(p_in, p_out, segs, T, R, mu)
            p_avg = mean_pressure_from_profile(p_out, md, segs, T, R, mu)
            return p_avg - p_mean, md

        md_mid = 0.0
        for _ in range(50):
            p_in_mid = 0.5 * (pin_lo + pin_hi)
            err, md_mid = mean_err(p_in_mid)
            if err > 0.0:
                # computed mean too high -> lower p_in
                pin_hi = p_in_mid
            else:
                pin_lo = p_in_mid
            if pin_hi - pin_lo < tol * 1e5:
                break

        p_in = 0.5 * (pin_lo + pin_hi)
        p_out = p_in + dp

        md_loop = mdot_from_loop(p_in, p_out, segs, T, R, mu)
        md_pump = pump.mdot(dp / 1e5, p_in, T, R)

        if md_pump > md_loop:
            dp_lo = dp
        else:
            dp_hi = dp

        best = (p_in, p_out, dp, md_loop, md_pump)
        if dp_hi - dp_lo < tol * 1e5:
            break

    p_in, p_out, dp, md, _ = best
    rho_mean = p_mean / (R * T)
    Qavg_L_min = (md / rho_mean) * 60_000.0

    # NEW: inlet/outlet volumetric flows (at local densities)
    rho_in  = p_in /(R*T)
    rho_out = p_out/(R*T)
    Q_in_L_min  = (md / rho_in )*60_000.0
    Q_out_L_min = (md / rho_out)*60_000.0

    return {
        "Pin_bar_abs": p_in / 1e5,
        "Pout_bar_abs": p_out / 1e5,
        "dp_bar": dp / 1e5,
        "mdot_kg_s": md,
        "Qavg_L_min": Qavg_L_min,
        "Q_in_L_min": Q_in_L_min,
        "Q_out_L_min": Q_out_L_min,
    }

def solve_open_case(p_in_bar_abs: float, p_out_bar_abs: float, segs: List[Segment],
                    T: float, R: float, mu: float, pump: PumpCurve):
    """
    Given inlet and outlet absolute pressures (open boundaries):
      - Compute tube-only mdot from the series of segments,
      - Compute pump mdot from the pump model (using inlet density),
      - Return the achievable mdot = min(tube_only, pump_only).
    """
    p_in  = p_in_bar_abs  * 1e5
    p_out = p_out_bar_abs * 1e5
    dp_bar = max(0.0, (p_out - p_in) / 1e5)

    # Tube-only mdot: make sum(dp2_i) = p_out^2 - p_in^2
    md_tube = mdot_from_loop(p_in, p_out, segs, T, R, mu)

    # Pump-only mdot at given dp and inlet pressure
    md_pump = pump.mdot(dp_bar, p_in, T, R)

    md = min(md_tube, md_pump)
    rho_in = p_in / (R * T)
    Q_in_L_min = (md / rho_in) * 60_000.0  # volumetric at the inlet (handy to inspect)
    return {
        "mdot_kg_s": md,
        "Q_in_L_min": Q_in_L_min,
        "tube_only_mdot": md_tube,
        "pump_only_mdot": md_pump,
        "dp_bar": dp_bar,
    }

# =================
# Optional: __main__
# =================

if __name__ == "__main__":
    # Quick smoke test with Xenon at 293 K, asymmetric loop: 3 m of 1/2" + 2 m of 1/4"
    T = 293.0
    M_xe = 0.131293
    R = 8.314 / M_xe
    mu = 2.3e-5

    #segs = [
    #    Segment(L=3.0, D=12.7e-3, eps=0.0),
    #    Segment(L=2.0, D=4.6e-3,  eps=0.0),
    #]
    segs = [
        Segment(L=6.0, D=6.6e-3, eps=0.0)
    ]

    # Pump curve example (50 Hz page): 30 L/min @ 0 bar, 20 L/min @ 3 bar
    pump = PumpCurve(
        Q0_Lpm=30.0,
        s_Lpm_per_bar=(30.0 - 20.0) / 3.0,
        vac_curve_pts=[(0.12, 0.0), (0.20, 5.0), (0.40, 15.0), (1.00, 30.0)],
    )

    print("Closed-loop (mean 1.8 bar abs):")
    print(solve_closed_loop_asym(1.8, segs, T, R, mu, pump))

    #print("\nOpen-case (Pin 0.5 bar abs → Pout 1.5 bar abs):")
    #print(solve_open_case(0.5, 1.5, [Segment(L=0.30, D=4.6e-3, eps=0.0)], T, R, mu, pump))