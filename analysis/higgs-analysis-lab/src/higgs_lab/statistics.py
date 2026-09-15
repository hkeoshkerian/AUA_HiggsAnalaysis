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


def profile_likelihood_discovery(n_observed, background, sigma_background):
    """One-bin discovery q0 for Poisson data and Gaussian-constrained background.

    The alternative permits a nonnegative signal yield.  This is a local
    counting likelihood for one predefined mass window, not a mass-shape fit.
    """
    n = float(n_observed)
    b0 = float(background)
    sigma = float(sigma_background)
    if not all(math.isfinite(x) for x in (n, b0, sigma)) or n < 0 or b0 <= 0 or sigma < 0:
        return {"profile_q0": None, "profile_Z": None,
                "profile_p_value_one_sided": None,
                "profile_background_hat_null": None,
                "profile_signal_hat": None,
                "profile_status": "undefined: invalid profile-likelihood inputs"}

    def nll(expectation, constrained_background):
        value = expectation - (n * math.log(expectation) if n > 0 else 0.)
        if sigma > 0:
            value += .5*((constrained_background-b0)/sigma)**2
        return value

    if sigma == 0:
        b_null = b0
    else:
        term = sigma*sigma - b0
        b_null = .5*(-term + math.sqrt(term*term + 4*n*sigma*sigma))
    if n <= b0:
        q0 = 0.
        signal_hat = 0.
    else:
        signal_hat = n-b0
        q0 = max(0., 2*(nll(b_null, b_null)-nll(n, b0)))
    z = math.sqrt(q0)
    return {"profile_q0": q0, "profile_Z": z,
            "profile_p_value_one_sided": .5*math.erfc(z/math.sqrt(2)),
            "profile_background_hat_null": b_null,
            "profile_signal_hat": signal_hat,
            "profile_status": "one-bin Poisson likelihood with Gaussian-constrained background"}


def expected_profile_likelihood(signal, background, sigma_background):
    """Asimov discovery significance for the same constrained one-bin model."""
    s, b, sigma = map(float, (signal, background, sigma_background))
    if not all(math.isfinite(x) for x in (s, b, sigma)) or s < 0 or b <= 0 or sigma < 0:
        return None
    if s == 0:
        return 0.
    if sigma == 0:
        return math.sqrt(max(0., 2*((s+b)*math.log1p(s/b)-s)))
    variance = sigma*sigma
    first = (s+b)*math.log((s+b)*(b+variance)/(b*b+(s+b)*variance))
    second = (b*b/variance)*math.log1p(variance*s/(b*(b+variance)))
    return math.sqrt(max(0., 2*(first-second)))

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
        total_background_sigma = math.hypot(db, systematic*b)
        result.update(
            background_constraint_sigma=total_background_sigma,
            expected_profile_Z=expected_profile_likelihood(
                s, b, total_background_sigma))
    return result
