"""Static PID gains derived from the structure of the vol-targeting loop.

See docs/control_formulation.md for the derivations. In short, in log space:

    plant:   log(portfolio_vol) = log|u| + log(sigma)   -> static gain k = 1
    sensor:  portfolio vol is a rolling std over W bars  -> dead time ~ W / 2
"""
import math


def kp_vol_targeting() -> float:
    """Proportional gain that reproduces exact vol targeting.

    |u| * sigma = target  =>  log|u| = 1 * log(target / sigma), so kp = 1.
    """
    return 1.0


def kd_unbiased_forecast() -> float:
    """Derivative gain for an unbiased vol forecast.

    If the forecast is unbiased the proportional term already sets the right
    exposure, there is no systematic lag for a derivative term to anticipate.
    """
    return 0.0


def sensor_dead_time(port_vol_window: int) -> float:
    """Effective dead time, in bars, of measuring portfolio vol over a rolling window.

    A step in exposure enters the rolling window one bar at a time and is
    fully reflected after `port_vol_window` bars; the standard first-order
    approximation of a moving-average sensor is a dead time of half its length.
    """
    return port_vol_window / 2


def ki_simc(port_vol_window: int, plant_gain: float = 1.0, tau_c: float | None = None) -> float:
    """Integral gain per bar from the SIMC rule (Skogestad, 2003).

    For an integral controller on a plant k * exp(-theta * s):
        ki = 1 / (k * (tau_c + theta))
    tau_c is the desired closed-loop time constant; tau_c = theta is the
    recommended robust default (about 60 degrees of phase margin), which
    gives ki = 1 / (2 * theta) = 1 / port_vol_window for k = 1.
    """
    theta = sensor_dead_time(port_vol_window)
    if tau_c is None:
        tau_c = theta
    if tau_c <= 0 or plant_gain <= 0:
        raise ValueError("tau_c and plant_gain must be positive")
    return 1.0 / (plant_gain * (tau_c + theta))


def integral_limit_for(max_correction: float, ki: float) -> float:
    """Cap on |e_int| so the integral term alone moves exposure by at most `max_correction`x.

    The integral contributes ki * e_int to log exposure, so bounding it by
    log(max_correction) gives |e_int| <= log(max_correction) / ki.
    """
    if max_correction <= 1:
        raise ValueError("max_correction must be greater than 1")
    if ki <= 0:
        raise ValueError("ki must be positive to derive an integral limit")
    return math.log(max_correction) / ki


def derived_gains(port_vol_window: int) -> dict[str, float]:
    """All three static gains for a given portfolio-vol measurement window."""
    return {
        "kp": kp_vol_targeting(),
        "ki": ki_simc(port_vol_window),
        "kd": kd_unbiased_forecast(),
    }
