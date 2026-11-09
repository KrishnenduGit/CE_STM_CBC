import numpy
from scipy.integrate import cumulative_trapezoid
from scipy.special import erfc, erfcx

from pop_models import PopModel

def broken_power_law_norm(alpha_1, alpha_2, m_break, m_low, m_high):
    """
    Eq. B11:

        N = m_break [ (1 - (m_low/m_break)^{1-alpha_1}) / (1 - alpha_1)
                     + ((m_high/m_break)^{1-alpha_2} - 1) / (1 - alpha_2) ]
          = m_break [ A(x_low, alpha_1) - A(x_high, alpha_2) ],

    with A(x, alpha) = (1 - x^{1-alpha}) / (1 - alpha) and x = m / m_break.
    The derivatives, with dA/dx = -x^{-alpha}, are

        dN/d alpha_1 =  m_break dA(x_low, alpha_1)/d alpha
        dN/d alpha_2 = -m_break dA(x_high, alpha_2)/d alpha
        dN/d m_low   = -x_low^{-alpha_1}
        dN/d m_high  =  x_high^{-alpha_2}
        dN/d m_break =  N/m_break + x_low^{-alpha_1} x_low - x_high^{-alpha_2} x_high
    """

    def integral(x, alpha):
        """A(x, alpha) and dA/d alpha. The series branches keep alpha -> 1 exact."""
        u = 1.0 - alpha
        log_x = numpy.log(x)
        if abs(u) < 1e-8:
            value = -log_x * (1.0 + 0.5 * u * log_x + (u * log_x) ** 2 / 6.0)
        else:
            value = -numpy.expm1(u * log_x) / u
        if abs(u * log_x) < 1e-4:
            dvalue = log_x ** 2 / 2.0 + u * log_x ** 3 / 3.0 + u ** 2 * log_x ** 4 / 8.0
        else:
            dvalue = (numpy.exp(u * log_x) * log_x - numpy.expm1(u * log_x) / u) / u
        return value, dvalue

    x_low = m_low / m_break
    x_high = m_high / m_break
    a_low, da_low = integral(x_low, alpha_1)
    a_high, da_high = integral(x_high, alpha_2)
    norm = m_break * (a_low - a_high)
    low_edge = x_low ** (-alpha_1)
    high_edge = x_high ** (-alpha_2)
    return norm, {
        "alpha_1": m_break * da_low,
        "alpha_2": -m_break * da_high,
        "m_min": -low_edge,
        "m_max": high_edge,
        "m_break": norm / m_break + low_edge * x_low - high_edge * x_high,
    }


def broken_power_law(mass, alpha_1, alpha_2, m_break, m_low, m_high, derivatives=True):
    """
    Eq. B10:

        p_BP(m) = (1/N) (m/m_break)^{-alpha_1}   m_low   <= m < m_break
                  (1/N) (m/m_break)^{-alpha_2}   m_break <= m <= m_high

    and zero outside [m_low, m_high].  N is Eq. B11.  On segment k,

        d shape/d alpha_k = -ln(m/m_break) shape,
        d shape/d m_break = alpha_k shape / m_break,
        d shape/d m       = -alpha_k shape / m,

    and every parameter derivative of p_BP is d shape/N - shape dN/N^2.
    """
    mass = numpy.asarray(mass, dtype=float)
    low = (mass >= m_low) & (mass < m_break)
    high = (mass >= m_break) & (mass <= m_high)
    shape = numpy.zeros_like(mass)
    shape[low] = (mass[low] / m_break) ** (-alpha_1)
    shape[high] = (mass[high] / m_break) ** (-alpha_2)
    norm, dnorm = broken_power_law_norm(alpha_1, alpha_2, m_break, m_low, m_high)
    if not derivatives:
        return shape / norm, {}

    log_ratio = numpy.zeros_like(mass)
    log_ratio[low | high] = numpy.log(mass[low | high] / m_break)
    slope = numpy.where(low, alpha_1, alpha_2)
    dshape = {
        "alpha_1": numpy.where(low, -log_ratio * shape, 0.0),
        "alpha_2": numpy.where(high, -log_ratio * shape, 0.0),
        "m_break": slope * shape / m_break,
    }
    out = {
        name: dshape.get(name, 0.0) / norm - shape * dnorm[name] / norm ** 2
        for name in dnorm
    }
    out["mass"] = -slope * shape / mass / norm
    return shape / norm, out


def left_truncated_normal(mass, mu, sigma, low, derivatives=True):
    """
    N_lt(m | mu, sigma, low) of Eq. B12: exp(-(m - mu)^2 / 2 sigma^2) / Z, with

        Z = int_low^inf exp(-(m - mu)^2 / 2 sigma^2) dm
          = sigma sqrt(pi/2) erfc((low - mu) / (sigma sqrt 2)).

    The truncation below ``low`` is not applied here. The taper S of Eq. B12
    is zero there, and the support cut in the model removes those events.
    ``erfcx`` keeps Z accurate when ``low`` sits far above ``mu``.

    With e = exp(-(low - mu)^2 / 2 sigma^2):
        dZ/d mu = e,  dZ/d low = -e,  dZ/d sigma = Z/sigma + (low - mu) e / sigma.
    """
    mass = numpy.asarray(mass, dtype=float)
    t = (low - mu) / (sigma * numpy.sqrt(2.0))
    prefactor = sigma * numpy.sqrt(numpy.pi / 2.0)
    norm = prefactor * numpy.exp(-t * t) * erfcx(t) if t > 0.0 else prefactor * erfc(t)
    value = numpy.exp(-0.5 * ((mass - mu) / sigma) ** 2) / norm
    if not derivatives:
        return value, {}

    edge = numpy.exp(-0.5 * ((low - mu) / sigma) ** 2)
    dlog_norm = {
        "mu": edge / norm,
        "sigma": 1.0 / sigma + (low - mu) * edge / (sigma * norm),
        "m_min": -edge / norm,
    }
    return value, {
        "mu": value * ((mass - mu) / sigma ** 2 - dlog_norm["mu"]),
        "sigma": value * ((mass - mu) ** 2 / sigma ** 3 - dlog_norm["sigma"]),
        "m_min": -value * dlog_norm["m_min"],
        "mass": -value * (mass - mu) / sigma ** 2,
    }


def smoothing(mass, m_low, delta_m, derivatives=True):
    """
    Eqs. B13-B14 (Talbot & Thrane 2018):

        S(m) = 0                  m <= m_low
               1 / (1 + f)        m_low < m < m_low + delta_m
               1                  m >= m_low + delta_m

        f = exp(delta_m / x + delta_m / (x - delta_m)),   x = m - m_low.

    With dS/d ln f = -S (1 - S):

        d ln f / d delta_m =  1/x + x / (x - delta_m)^2
        d ln f / d m_low   =  delta_m / x^2 + delta_m / (x - delta_m)^2
        d ln f / d m       = -d ln f / d m_low

    The exponent is clipped at +-500, where S(1 - S) ~ e^-500.
    """
    mass = numpy.asarray(mass, dtype=float)
    window = numpy.ones_like(mass)
    window[mass <= m_low] = 0.0
    ramp = (mass > m_low) & (mass < m_low + delta_m)
    x = mass[ramp] - m_low
    log_f = numpy.clip(delta_m / x + delta_m / (x - delta_m), -500.0, 500.0)
    window[ramp] = 1.0 / (1.0 + numpy.exp(log_f))
    if not derivatives:
        return window, {}

    dwindow_dlog_f = -window[ramp] * (1.0 - window[ramp])
    dlog_f_dlow = delta_m / x ** 2 + delta_m / (x - delta_m) ** 2
    out = {name: numpy.zeros_like(mass) for name in ("delta_m", "m_min", "mass")}
    out["delta_m"][ramp] = dwindow_dlog_f * (1.0 / x + x / (x - delta_m) ** 2)
    out["m_min"][ramp] = dwindow_dlog_f * dlog_f_dlow
    out["mass"][ramp] = -dwindow_dlog_f * dlog_f_dlow
    return window, out


def primary_mass(mass_1, params, derivatives=True):
    """
    Eq. B12 before its normalisation Z_{m1}:

        M(m1) = [ lambda_0 p_BP(m1) + lambda_1 N_lt(m1 | mu_1, sigma_1)
                  + lambda_2 N_lt(m1 | mu_2, sigma_2) ] S(m1),

    with lambda_2 = 1 - lambda_0 - lambda_1 and every N_lt truncated at m_min.
    The derivatives follow from the product rule on the blocks above.
    """
    lambda_0 = params["lambda_0"]
    lambda_1 = params["lambda_1"]
    lambda_2 = 1.0 - lambda_0 - lambda_1
    m_min = params["m_min"]
    p_bp, dp_bp = broken_power_law(
        mass_1, params["alpha_1"], params["alpha_2"], params["m_break"],
        m_min, params["m_max"], derivatives,
    )
    peak_1, dpeak_1 = left_truncated_normal(
        mass_1, params["mu_1"], params["sigma_1"], m_min, derivatives
    )
    peak_2, dpeak_2 = left_truncated_normal(
        mass_1, params["mu_2"], params["sigma_2"], m_min, derivatives
    )
    window, dwindow = smoothing(mass_1, m_min, params["delta_m"], derivatives)
    mix = lambda_0 * p_bp + lambda_1 * peak_1 + lambda_2 * peak_2
    if not derivatives:
        return mix * window, {}

    dmix = {name: lambda_0 * dp_bp[name] for name in ("alpha_1", "alpha_2", "m_break", "m_max")}
    dmix["lambda_0"] = p_bp - peak_2
    dmix["lambda_1"] = peak_1 - peak_2
    for weight, dpeak, index in ((lambda_1, dpeak_1, 1), (lambda_2, dpeak_2, 2)):
        dmix[f"mu_{index}"] = weight * dpeak["mu"]
        dmix[f"sigma_{index}"] = weight * dpeak["sigma"]
    for name in ("m_min", "mass"):
        dmix[name] = lambda_0 * dp_bp[name] + lambda_1 * dpeak_1[name] + lambda_2 * dpeak_2[name]

    out = {name: value * window for name, value in dmix.items()}
    out["delta_m"] = mix * dwindow["delta_m"]
    out["m_min"] = out["m_min"] + mix * dwindow["m_min"]
    out["mass"] = out["mass"] + mix * dwindow["mass"]
    return mix * window, out


def mass_ratio(mass_2, mass_1, beta_q, m_low, delta_m, n_taper=4096, derivatives=True):
    """
    The paper's mass-ratio power law with the taper on m2, conditioned on m1:

        ln p(m2 | m1) = beta_q ln m2 + ln S(m2) - ln C(m1),
        C(m1) = int_{m_low}^{m1} m2^{beta_q} S(m2) dm2.

    This is GWForge's q^beta S / int q^beta S dq times the Jacobian 1/m1.

    Above m_low + delta_m, S = 1 and C is an exact power-law integral. Only
    the taper itself, (m_low, m_low + delta_m], uses quadrature on
    ``n_taper`` nodes.  The moving-edge terms of dC/d delta_m and dC/d m_low
    cancel, because S(m_low) = 0 and S = 1 beyond the taper.  So those
    derivatives are pure interior integrals of m2^beta_q dS.

    Returns ln p (-inf where S(m2) = 0) and its derivatives with respect to
    beta_q, delta_m, m_min, mass_1 (-m1^beta_q S(m1) / C) and mass_2.
    """
    mass_1 = numpy.asarray(mass_1, dtype=float)
    mass_2 = numpy.asarray(mass_2, dtype=float)

    def power_law_integral(lower, upper):
        """int_lower^upper m^beta_q dm and its beta_q derivative; exact at beta_q = -1."""
        exponent = beta_q + 1.0
        log_upper = numpy.log(numpy.asarray(upper, dtype=float))
        log_lower = numpy.log(lower)
        log_ratio = log_upper - log_lower
        if abs(exponent) < 1e-8:
            value = log_ratio * (
                1.0 + 0.5 * exponent * log_ratio + (exponent * log_ratio) ** 2 / 6.0
            )
        else:
            value = lower ** exponent * numpy.expm1(exponent * log_ratio) / exponent
        if abs(exponent) * max(abs(numpy.max(log_upper)), abs(log_lower)) < 1e-6:
            dvalue = (
                (log_upper ** 2 - log_lower ** 2) / 2.0
                + exponent * (log_upper ** 3 - log_lower ** 3) / 3.0
                + exponent ** 2 * (log_upper ** 4 - log_lower ** 4) / 8.0
            )
        else:
            upper_e = numpy.exp(exponent * log_upper)
            lower_e = numpy.exp(exponent * log_lower)
            dvalue = (upper_e * log_upper - lower_e * log_lower) / exponent - (
                upper_e - lower_e
            ) / exponent ** 2
        return value, dvalue

    if not (delta_m > 0.0):
        # No taper: C is analytic, and S(m_low) = 1 gives a lower-limit term.
        norm, dnorm_dbeta = power_law_integral(m_low, mass_1)
        dnorm = {
            "beta_q": dnorm_dbeta,
            "delta_m": numpy.zeros_like(mass_1),
            "m_min": numpy.full_like(mass_1, -(m_low ** beta_q)),
        }
    else:
        taper_edge = m_low + delta_m
        taper = numpy.linspace(m_low, taper_edge, n_taper)
        taper_window, dtaper_window = smoothing(taper, m_low, delta_m, derivatives)
        power = taper ** beta_q
        inside = mass_1 < taper_edge
        bulk, dbulk = power_law_integral(taper_edge, numpy.maximum(mass_1, taper_edge))

        def cumulative(integrand, tail):
            """Taper part of the integral up to m1, plus ``tail`` once m1 is past the taper."""
            running = cumulative_trapezoid(integrand, taper, initial=0.0)
            return numpy.where(inside, numpy.interp(mass_1, taper, running), running[-1] + tail)

        norm = cumulative(power * taper_window, bulk)
        if derivatives:
            dnorm = {
                "beta_q": cumulative(power * numpy.log(taper) * taper_window, dbulk),
                "delta_m": cumulative(power * dtaper_window["delta_m"], 0.0),
                "m_min": cumulative(power * dtaper_window["m_min"], 0.0),
            }

    window_2, dwindow_2 = smoothing(mass_2, m_low, delta_m, derivatives)
    usable = (window_2 > 0.0) & (norm > 0.0)
    log_prob = numpy.full_like(mass_2, -numpy.inf)
    log_prob[usable] = (
        beta_q * numpy.log(mass_2[usable])
        + numpy.log(window_2[usable])
        - numpy.log(norm[usable])
    )
    if not derivatives:
        return log_prob, {}

    with numpy.errstate(divide="ignore", invalid="ignore"):
        window_1, _ = smoothing(mass_1, m_low, delta_m, derivatives=False)
        out = {
            "beta_q": numpy.log(mass_2) - dnorm["beta_q"] / norm,
            "delta_m": dwindow_2["delta_m"] / window_2 - dnorm["delta_m"] / norm,
            "m_min": dwindow_2["m_min"] / window_2 - dnorm["m_min"] / norm,
            "mass_1": -(mass_1 ** beta_q) * window_1 / norm,
            "mass_2": beta_q / mass_2 + dwindow_2["mass"] / window_2,
        }
    return log_prob, out


def _normalisation_grid(m_min, m_max, delta_m, m_break=None, n_taper=2048, n_bulk=32768):
    """
    Integration grid for the primary-mass normalisation, with a node placed on
    every point where the integrand is not smooth: the narrow taper
    (m_min, m_min + delta_m] gets its own dense segment (a plain 3000-point
    linspace steps clean over a delta_m = 0.05 taper, a ~4e-4 error), and the
    grid splits at the m_break slope discontinuity to keep the trapezoid rule
    O(h^2).

    On a fixed node set, ``d/dlambda trapz`` commutes with the parameter
    derivative, so `analytic_score` stays the exact derivative of ``log_prob``
    for every parameter that does not move a node (all except m_min, delta_m,
    m_break, m_max).  Gauss-Legendre would be faster but fails badly for narrow
    peaks (49% error at sigma_1 = 0.05), which the fit does explore; the
    trapezoid degrades gracefully.

    ``m_break`` is optional so the grid can also be built for models without one.
    """
    has_taper = delta_m > 0.0 and m_min + delta_m < m_max
    # Without a taper the density is discontinuous at m_min: `smoothing`
    # returns 0 *at* m_min but 1 immediately above it.  Starting the grid on the
    # zero would chop half of the first trapezoid, which for a power law steep
    # near m_min is a ~6e-4 error in the normalisation.  Nudge off the edge.
    edges = [float(m_min) if has_taper else float(numpy.nextafter(m_min, numpy.inf))]
    if has_taper:
        edges.append(float(m_min + delta_m))
    if m_break is not None and m_min < m_break < m_max:
        edges.append(float(m_break))
    edges.append(float(m_max))
    edges = sorted(set(edges))
    if len(edges) < 2:
        return numpy.array([m_min, m_max], dtype=float)

    # The taper segment (if present) gets its own dense allocation; the rest
    # share n_bulk in proportion to their length.
    bulk_edges = edges[1:] if has_taper else edges
    bulk_span = bulk_edges[-1] - bulk_edges[0]

    def segment(lower, upper, count):
        # Close the segment just below its upper edge.  d(shape)/d(m_break) is
        # *discontinuous* across m_break (it jumps from alpha_1 to alpha_2), so a
        # trapezoid spanning the edge would mix the two branches and cost an
        # O(h) error in d ln Z/d m_break.  The sliver between nextafter(upper)
        # and upper has zero width to machine precision and contributes nothing.
        return numpy.append(
            numpy.linspace(lower, upper, count, endpoint=False),
            numpy.nextafter(upper, -numpy.inf),
        )

    pieces = []
    if has_taper:
        pieces.append(segment(edges[0], edges[1], n_taper))
    for lower, upper in zip(bulk_edges[:-1], bulk_edges[1:]):
        count = max(64, int(round(n_bulk * (upper - lower) / bulk_span)))
        pieces.append(segment(lower, upper, count))
    pieces.append(numpy.array([edges[-1]], dtype=float))
    return numpy.concatenate(pieces)


class BrokenPowerLawPlusTwoPeaksMass(PopModel):
    """
    p(m1, m2) = M(m1) / Z_{m1} * p(m2 | m1),   m_min <= m2 <= m1 <= m_max,

    with M from `primary_mass` (Eq. B12), Z_{m1} = int M dm1 by the trapezoid
    rule on `_normalisation_grid`, and p(m2 | m1) from `mass_ratio`.  Each
    component of Eq. B12 is normalised, so lambda_0, lambda_1 and lambda_2
    are true mixing fractions, and Z_{m1} only corrects for the taper.

    Parameters (fiducial): alpha_1 1.5, alpha_2 5.4, m_break 37.5,
    lambda_0 0.90, lambda_1 0.05, mu_1 10, sigma_1 3, mu_2 35, sigma_2 5,
    beta_q 1.4, m_min 5 (= m_{1,low}), delta_m 4.8, m_max 100 (= m_high).

    Event keys: 'mass_1_source' and either 'mass_2_source' or 'mass_ratio'.
    Outside the support, or with lambda_0 + lambda_1 outside [0, 1], ln p is
    -inf and the score is NaN.
    """

    def __init__(
        self,
        parameter_names = ["alpha_1", "alpha_2", "m_break", "lambda_0", "lambda_1",
                            "mu_1", "sigma_1", "mu_2", "sigma_2",
                            "beta_q", "m_min", "delta_m", "m_max"],
        # Handwritten fiducial values.
        fiducial={
            "alpha_1": 1.5,
            "alpha_2": 5.4,
            "m_break": 37.5,
            "lambda_0": 0.90,
            "lambda_1": 0.05,
            "mu_1": 10.0,
            "sigma_1": 3.0,
            "mu_2": 35.0,
            "sigma_2": 5.0,
            "beta_q": 1.4,
            "m_min": 5.0,
            "delta_m": 4.8,
            "m_max": 100.0,
        },
        phys_event_keys=["mass_1_source", "mass_2_source"],
    ):

        self.parameter_names = parameter_names
        self.fiducial = fiducial
        self.phys_event_keys = phys_event_keys

    def _evaluate(self, events, params, derivatives):
        """
        ln p, the score d ln p / d lambda, and d ln p / d m1 at fixed q.

        Score column for a parameter theta of M:

            d ln M(m1)/d theta - int dM/d theta dm / Z_{m1}

        The trapezoid grid does not depend on theta, so this is the exact
        derivative of ``log_prob``.  For m_max the moving upper limit adds the
        Leibniz term -M(m_max)/Z_{m1}.  The lower-limit term vanishes because
        M(m_min) = 0.  beta_q, delta_m and m_min also pick up their
        `mass_ratio` derivatives.

        At fixed q both masses scale together (the spectral-siren direction),
        so d/dm1|_q = d/dm1 + q d/dm2.  Z_{m1} does not depend on m1.
        """
        mass_1 = numpy.asarray(events["mass_1_source"], dtype=float)
        if "mass_2_source" in events:
            mass_2 = numpy.asarray(events["mass_2_source"], dtype=float)
        else:
            mass_2 = numpy.asarray(events["mass_ratio"], dtype=float) * mass_1
        m_min = params["m_min"]
        m_max = params["m_max"]

        log_prob = numpy.full(len(mass_1), -numpy.inf)
        score = numpy.full((len(mass_1), len(self.parameter_names)), numpy.nan)
        dlogp_dmass_1 = numpy.full(len(mass_1), numpy.nan)
        result = (log_prob, score, dlogp_dmass_1)

        lambda_2 = 1.0 - params["lambda_0"] - params["lambda_1"]
        if min(params["lambda_0"], params["lambda_1"], lambda_2) < 0.0:
            return result
        if not (m_min < params["m_break"] < m_max):
            return result
        valid = (mass_1 >= m_min) & (mass_1 <= m_max) & (mass_2 >= m_min) & (mass_2 <= mass_1)
        if not numpy.any(valid):
            return result
        mass_1 = mass_1[valid]
        mass_2 = mass_2[valid]

        primary, dprimary = primary_mass(mass_1, params, derivatives)
        grid = _normalisation_grid(m_min, m_max, params["delta_m"], params["m_break"])
        primary_grid, dprimary_grid = primary_mass(grid, params, derivatives)
        norm = numpy.trapezoid(primary_grid, grid)
        if not (norm > 0.0):
            return result
        log_ratio, dlog_ratio = mass_ratio(
            mass_2, mass_1, params["beta_q"], m_min, params["delta_m"], derivatives=derivatives
        )

        usable = (primary > 1e-300) & numpy.isfinite(log_ratio)
        rows = numpy.flatnonzero(valid)[usable]
        log_prob[rows] = numpy.log(primary[usable]) - numpy.log(norm) + log_ratio[usable]
        if not derivatives:
            return result

        columns = {
            name: dprimary[name][usable] / primary[usable]
            - numpy.trapezoid(dprimary_grid[name], grid) / norm
            for name in dprimary if name != "mass"
        }
        columns["m_max"] = columns["m_max"] - primary_grid[-1] / norm
        columns["beta_q"] = 0.0
        for name in ("beta_q", "delta_m", "m_min"):
            columns[name] = columns[name] + dlog_ratio[name][usable]
        for index, name in enumerate(self.parameter_names):
            score[rows, index] = columns[name]

        dlogp_dmass_1[rows] = (
            dprimary["mass"] / primary
            + dlog_ratio["mass_1"]
            + mass_2 / mass_1 * dlog_ratio["mass_2"]
        )[usable]
        return result

    def log_prob(self, events, params):
        return self._evaluate(events, params, derivatives=False)[0]

    def analytic_score(self, events, params):
        """Score d ln p / d lambda, shape (N_events, 13); NaN rows outside the support."""
        return self._evaluate(events, params, derivatives=True)[1]

    def dlogp_dmass_1(self, events, params):
        """d ln p / d mass_1_source at fixed mass ratio, shape (N_events,)."""
        return self._evaluate(events, params, derivatives=True)[2]
