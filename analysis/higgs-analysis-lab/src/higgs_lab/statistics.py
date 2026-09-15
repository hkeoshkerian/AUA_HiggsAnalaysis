"""One-bin profile-likelihood significance utilities."""
import math
import numpy as np

def weighted_yield(mass, weights, window):
    mass, weights = np.asarray(mass), np.asarray(weights)
    inside = (mass >= window[0]) & (mass < window[1])
    selected = weights[inside]
    return float(selected.sum()), float(np.sqrt(np.square(selected).sum()))

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


def _symmetric_stat_error(function, values, errors):
    """First-order numerical propagation of independent statistical errors."""
    variance = 0.
    for index, error in enumerate(errors):
        if not math.isfinite(error) or error <= 0:
            continue
        upper = list(values); lower = list(values)
        upper[index] += error
        lower[index] = max(0., lower[index]-error)
        span = upper[index]-lower[index]
        if span <= 0:
            continue
        z_upper, z_lower = function(*upper), function(*lower)
        if z_upper is None or z_lower is None:
            continue
        derivative = (z_upper-z_lower)/span
        variance += (derivative*error)**2
    return math.sqrt(variance)


def expected_profile_result(signal, background, sigma_signal, sigma_background,
                            fractional_systematic=.30):
    """Asimov profile Z with its propagated weighted-MC statistical error."""
    values = (signal, background, sigma_signal, sigma_background,
              fractional_systematic)
    if (not all(math.isfinite(x) and x >= 0 for x in values)
            or background <= 0):
        return {"expected_profile_Z": None, "sigma_expected_profile_Z": None,
                "background_constraint_sigma": None}

    def evaluate(s, b):
        constraint = math.hypot(sigma_background, fractional_systematic*b)
        return expected_profile_likelihood(s, b, constraint)

    constraint = math.hypot(sigma_background,
                            fractional_systematic*background)
    return {
        "expected_profile_Z": evaluate(signal, background),
        "sigma_expected_profile_Z": _symmetric_stat_error(
            evaluate, (signal, background), (sigma_signal, sigma_background)),
        "background_constraint_sigma": constraint,
    }

def summarize_mc(frame, window, systematic, fraction=1.0):
    signal = frame[frame.role == "signal"]
    background = frame[frame.role == "background"]
    s, ds = weighted_yield(signal.mass, signal.weight, window)
    b, db = weighted_yield(background.mass, background.weight, window)
    result = {"S": s, "B": b, "sigma_S_mc": ds, "sigma_B_mc": db}
    if fraction != 1:
        result.update(expected_profile_Z=None, sigma_expected_profile_Z=None,
                      background_constraint_sigma=None,
                      status="partial input: yields are unrescaled; significance suppressed")
    else:
        result.update(expected_profile_result(s, b, ds, db, systematic))
        result["status"] = "Asimov one-bin profile likelihood"
    return result
