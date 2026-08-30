"""Legacy expected-count proxy, not a calibrated discovery significance."""
import math
import numpy as np

def weighted_yield(mass, weights, window):
    mass, weights = np.asarray(mass), np.asarray(weights)
    inside = (mass >= window[0]) & (mass < window[1])
    selected = weights[inside]
    return float(selected.sum()), float(np.sqrt(np.square(selected).sum()))

def expected_count_proxy(signal, background, sigma_signal, sigma_background, fractional_systematic=0.30):
    """Source formula; propagated error does not establish p-value calibration."""
    if not all(math.isfinite(x) for x in [signal, background, sigma_signal, sigma_background, fractional_systematic]) or background <= 0:
        return {"Z_proxy": None, "sigma_Z_proxy": None, "status": "undefined: nonpositive background or nonfinite inputs"}
    if signal < 0 or min(sigma_signal, sigma_background, fractional_systematic) < 0:
        raise ValueError("Negative signal or uncertainty is unsupported by this count proxy")
    k = fractional_systematic
    denom = background + (k * background)**2
    z = signal / math.sqrt(denom)
    variance = sigma_signal**2 / denom + signal**2 * (1+2*k*k*background)**2 * sigma_background**2 / (4*denom**3)
    return {"Z_proxy": z, "sigma_Z_proxy": math.sqrt(variance), "status": "approximate expected MC count proxy"}

def summarize_mc(frame, window, systematic, fraction=1.0):
    signal = frame[frame.role == "signal"]
    background = frame[frame.role == "background"]
    s, ds = weighted_yield(signal.mass, signal.weight, window)
    b, db = weighted_yield(background.mass, background.weight, window)
    result = {"S": s, "B": b, "sigma_S_mc": ds, "sigma_B_mc": db}
    if fraction != 1:
        result.update(Z_proxy=None, sigma_Z_proxy=None, status="partial input: yields are unrescaled; significance suppressed")
    else:
        result.update(expected_count_proxy(s,b,ds,db,systematic))
    return result
