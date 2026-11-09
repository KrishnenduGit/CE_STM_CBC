"""
FullPop mass model (GWTC-4 "FullPop", arXiv:2605.27226 App. B.2), coded from
GWForge ``NotchFilterBinnedPairingMassDistribution``, which generated the
catalogue.  One function per equation:

    Eq. B3 (cases)   piecewise_power_law    PL(m)
    Eq. B3 (peaks)   truncated_normal       N_[mmin, mmax](m | mu, sigma)
    Eq. B4           low_pass               l(m | m_edge, eta)
    Eq. B5           high_pass              h(m | m_edge, eta) = 1 - l
    Eqs. B6-B7       notch                  n(m) = 1 - A h(m | m_low) l(m | m_high)
    Eq. B3           mass_function          pi_m(m)
    Eq. B8           pairing                f(q, m2), normalised per branch as in GWForge
    Eq. B2           FullPopMass            pi_m(m1) pi_m(m2) f / Z

Every function returns ``(log value, derivatives)``.  ``derivatives`` maps a
parameter name (plus ``"mass"`` for d/dm) to the derivative of the log value,
and it is empty when called with ``derivatives=False``.  Equations,
derivatives and the GWForge listings are in ``mass_models_equations.pdf``.
"""

import numpy
from scipy.special import ndtr

from pop_models import PopModel


# ===========================================================================
# FullPop (GWTC-4 / GWTC-5 "FullPop", restricted to the BBH region)
# ===========================================================================

# Hyperparameters that generated the CE/STM catalogue
# (pop_models/stellar_mass/injection_sampler_v2/config_pop_v2.ini, FullPop_GWTC4).
FULLPOP_GWTC4_FIDUCIAL = {
    "alpha_1": -4.509282643557759,
    "alpha_dip": -1.6797692040029513,
    "alpha_2": -0.9020346791414204,
    "NSmin": 1.1763665089045245,
    "NSmax": 4.094744137742966,
    "BHmin": 7.7639549426962375,
    "BHmax": 152.05597923492775,
    "UPPERmin": 38.277415427531,
    "UPPERmax": 66.57670527299628,
    "n0": 50.0,
    "n1": 50.0,
    "n2": 50.0,
    "n3": 30.0,
    "n4": 30.0,
    "n5": 10.041072204811995,
    "mix1": 735.473275543081,
    "mu1": 37.811196396328114,
    "sig1": 17.12643064109277,
    "mix2": 211.73332682799375,
    "mu2": 8.897742041719734,
    "sig2": 1.044692632025516,
    "A": 0.091462424745936,
    "A2": 0.8281652204882891,
    "beta_pair_1": 0.9641379639323953,
    "beta_pair_2": 2.160035650137072,
}

# Structural constants of the generator, not hyperparameters.
FULLPOP_SUPPORT = {
    "mbreak": 5.0,       # pairing-function break in m2
    "mmin": 1.0,         # truncation of the Gaussian peaks, and q_min = mmin / mmax
    "mmax": 350.0,       # upper support edge for both components
    "m2_min": 2.06,  # BBH selection m2 > 2.06 (create_injections --max-NS-mass 2.06)
}

# Pinned in the paper's Tab. 4 (sharpness exponents eta_0..eta_4), or with no
# support in the BBH region (NSmin: its high-pass is ~1e-12 above 2.06 Msun).
FULLPOP_FIXED_NAMES = ["NSmin", "n0", "n1", "n2", "n3", "n4"]
# Whole CBC population (BNS + NSBH + BBH): the NS edge is now in the data, so
# NSmin is free and only the sharpness exponents stay pinned.
FULLPOP_SHARPNESS_NAMES = ["n0", "n1", "n2", "n3", "n4"]
# Optional extra pins (``spectral_sirens_forecast.py --fix``): the edge
# positions, so no derivative is taken across an edge.  "alledges" also pins
# the notch edges of the upper mass gap and the BHmax sharpness n5.
FULLPOP_EDGE_SETS = {
    "edges": ["NSmin", "NSmax", "BHmin", "BHmax"],
    "alledges": ["NSmin", "NSmax", "BHmin", "BHmax", "UPPERmin", "UPPERmax", "n5"],
}


def fullpop_fixed_names(population="bbh"):
    return FULLPOP_FIXED_NAMES if population == "bbh" else FULLPOP_SHARPNESS_NAMES

_FULLPOP_PI_M_NAMES = [
    "alpha_1", "alpha_dip", "alpha_2", "NSmin", "NSmax", "BHmin", "BHmax",
    "UPPERmin", "UPPERmax", "n0", "n1", "n2", "n3", "n4", "n5",
    "mix1", "mu1", "sig1", "mix2", "mu2", "sig2", "A", "A2",
]
_FULLPOP_PAIRING_NAMES = ["beta_pair_1", "beta_pair_2"]
FULLPOP_PARAMETER_NAMES = _FULLPOP_PI_M_NAMES + _FULLPOP_PAIRING_NAMES


def fullpop_support(population="bbh", support=None):
    """Support edges; the whole-CBC population extends m2 down to mmin."""
    support = dict(FULLPOP_SUPPORT if support is None else support)
    if population == "all":
        support["m2_min"] = support["mmin"]
    return support


def low_pass(mass, m_edge, eta, derivatives=True):
    """
    Eq. B4:  l(m | m_edge, eta) = [1 + (m / m_edge)^eta]^{-1}.

    In logs, ln l = -softplus(t) with t = eta ln(m / m_edge) and
    softplus(t) = ln(1 + e^t), which is overflow-safe for eta = 50.  With the
    logistic sigma(t) = d softplus/dt:

        d ln l/d m_edge = sigma eta / m_edge
        d ln l/d eta    = -sigma ln(m / m_edge)
        d ln l/d m      = -sigma eta / m
    """
    log_ratio = numpy.log(mass) - numpy.log(m_edge)
    t = eta * log_ratio
    log_l = -numpy.logaddexp(0.0, t)
    if not derivatives:
        return log_l, {}
    sigma = 0.5 * (1.0 + numpy.tanh(0.5 * t))
    return log_l, {
        "m_edge": sigma * eta / m_edge,
        "eta": -sigma * log_ratio,
        "mass": -sigma * eta / mass,
    }


def high_pass(mass, m_edge, eta, derivatives=True):
    """
    Eq. B5:  h(m | m_edge, eta) = 1 - l(m | m_edge, eta) = [1 + (m_edge / m)^eta]^{-1}.

    In logs, ln h = -softplus(t) with t = eta ln(m_edge / m):

        d ln h/d m_edge = -sigma eta / m_edge
        d ln h/d eta    = -sigma ln(m_edge / m)
        d ln h/d m      = sigma eta / m
    """
    log_ratio = numpy.log(m_edge) - numpy.log(mass)
    t = eta * log_ratio
    log_h = -numpy.logaddexp(0.0, t)
    if not derivatives:
        return log_h, {}
    sigma = 0.5 * (1.0 + numpy.tanh(0.5 * t))
    return log_h, {
        "m_edge": -sigma * eta / m_edge,
        "eta": -sigma * log_ratio,
        "mass": sigma * eta / mass,
    }


def notch(mass, m_low, m_high, eta_low, eta_high, depth, derivatives=True):
    """
    Eqs. B6-B7 in the GWForge convention, which generated the catalogue:

        n(m) = 1 - A L(m),   L(m) = h(m | m_low, eta_low) l(m | m_high, eta_high).

    This suppresses the rate between m_low and m_high.  The paper prints
    l(m | m_low) h(m | m_high), and that product vanishes everywhere (see the
    PDF).  n_1 uses (NSmax, BHmin, n1, n2, A) and n_2 uses
    (UPPERmin, UPPERmax, n3, n4, A2).  With d ln n = -(A L / n) d ln L:

        d ln n/d A = -L / n,   d ln n/d theta = -(A L / n) d ln L/d theta.
    """
    log_h, dlog_h = high_pass(mass, m_low, eta_low, derivatives)
    log_l, dlog_l = low_pass(mass, m_high, eta_high, derivatives)
    filters = numpy.exp(log_h + log_l)
    value = 1.0 - depth * filters
    if not derivatives:
        return numpy.log(value), {}
    factor = -depth * filters / value
    return numpy.log(value), {
        "depth": -filters / value,
        "m_low": factor * dlog_h["m_edge"],
        "eta_low": factor * dlog_h["eta"],
        "m_high": factor * dlog_l["m_edge"],
        "eta_high": factor * dlog_l["eta"],
        "mass": factor * (dlog_h["mass"] + dlog_l["mass"]),
    }


def piecewise_power_law(mass, alpha_1, alpha_dip, alpha_2, NSmax, BHmin, derivatives=True):
    """
    The case block of Eq. B3, continuous at NSmax and BHmin:

        PL(m) = m^alpha_1                                            m < NSmax
                m^alpha_dip NSmax^(alpha_1 - alpha_dip)              NSmax <= m < BHmin
                m^alpha_2 NSmax^(alpha_1 - alpha_dip) BHmin^(alpha_dip - alpha_2)   m >= BHmin

    The NSmax and BHmin derivatives come from the continuity constants, and
    d ln PL/d m = alpha(m) / m with alpha(m) the local slope.
    """
    log_m = numpy.log(mass)
    log_NSmax = numpy.log(NSmax)
    log_BHmin = numpy.log(BHmin)
    region_1 = mass < NSmax
    region_3 = mass >= BHmin
    region_2 = ~region_1 & ~region_3
    slope = numpy.where(region_1, alpha_1, numpy.where(region_2, alpha_dip, alpha_2))
    log_pl = (
        slope * log_m
        + numpy.where(region_1, 0.0, (alpha_1 - alpha_dip) * log_NSmax)
        + numpy.where(region_3, (alpha_dip - alpha_2) * log_BHmin, 0.0)
    )
    if not derivatives:
        return log_pl, {}
    return log_pl, {
        "alpha_1": numpy.where(region_1, log_m, log_NSmax),
        "alpha_dip": numpy.where(
            region_1, 0.0, numpy.where(region_2, log_m - log_NSmax, log_BHmin - log_NSmax)
        ),
        "alpha_2": numpy.where(region_3, log_m - log_BHmin, 0.0),
        "NSmax": numpy.where(region_1, 0.0, (alpha_1 - alpha_dip) / NSmax),
        "BHmin": numpy.where(region_3, (alpha_dip - alpha_2) / BHmin, 0.0),
        "mass": slope / mass,
    }


def truncated_normal(mass, mu, sigma, low, high, derivatives=True):
    """
    The peaks of Eq. B3, N_[low, high](m | mu, sigma) = phi(z) / (sigma Delta),
    with z = (m - mu)/sigma and Delta = Phi(z_high) - Phi(z_low).  Returns
    the density itself, not its log, because it enters through
    1 + c_1 N_1 + c_2 N_2:

        dN/d mu    = N [ z/sigma + (phi(z_high) - phi(z_low)) / (sigma Delta) ]
        dN/d sigma = N [ (z^2 - 1)/sigma + (z_high phi(z_high) - z_low phi(z_low)) / (sigma Delta) ]
        dN/d m     = -N z / sigma
    """
    z = (mass - mu) / sigma
    z_low = (low - mu) / sigma
    z_high = (high - mu) / sigma
    phi_low = numpy.exp(-0.5 * z_low ** 2) / numpy.sqrt(2.0 * numpy.pi)
    phi_high = numpy.exp(-0.5 * z_high ** 2) / numpy.sqrt(2.0 * numpy.pi)
    norm = ndtr(z_high) - ndtr(z_low)
    pdf = numpy.exp(-0.5 * z ** 2) / (numpy.sqrt(2.0 * numpy.pi) * sigma * norm)
    if not derivatives:
        return pdf, {}
    dlog_norm_dmu = -(phi_high - phi_low) / (sigma * norm)
    dlog_norm_dsigma = -(z_high * phi_high - z_low * phi_low) / (sigma * norm)
    return pdf, {
        "mu": pdf * (z / sigma - dlog_norm_dmu),
        "sigma": pdf * ((z ** 2 - 1.0) / sigma - dlog_norm_dsigma),
        "mass": -pdf * z / sigma,
    }


def mass_function(mass, params, mmin, mmax, derivatives=True):
    """
    Eq. B3, the universal mass function (unnormalised):

        pi_m(m) = [1 + c_1 N_1(m) + c_2 N_2(m)] n_1(m) n_2(m) h(m | NSmin) l(m | BHmax) PL(m),

    with N_i truncated to [mmin, mmax].  Code names: eta_NSmin..eta_BHmax =
    n0..n5, c_i = mix_i, sigma_i = sig_i, m_UMGmin/max = UPPERmin/max.

    Returns ln pi_m and its derivatives with respect to every name in
    _FULLPOP_PI_M_NAMES, plus "mass".
    """
    p = params
    mass = numpy.asarray(mass, dtype=float)
    log_pl, dlog_pl = piecewise_power_law(
        mass, p["alpha_1"], p["alpha_dip"], p["alpha_2"], p["NSmax"], p["BHmin"], derivatives
    )
    log_h, dlog_h = high_pass(mass, p["NSmin"], p["n0"], derivatives)
    log_l, dlog_l = low_pass(mass, p["BHmax"], p["n5"], derivatives)
    log_n1, dlog_n1 = notch(mass, p["NSmax"], p["BHmin"], p["n1"], p["n2"], p["A"], derivatives)
    log_n2, dlog_n2 = notch(
        mass, p["UPPERmin"], p["UPPERmax"], p["n3"], p["n4"], p["A2"], derivatives
    )
    peak_1, dpeak_1 = truncated_normal(mass, p["mu1"], p["sig1"], mmin, mmax, derivatives)
    peak_2, dpeak_2 = truncated_normal(mass, p["mu2"], p["sig2"], mmin, mmax, derivatives)
    peaks = 1.0 + p["mix1"] * peak_1 + p["mix2"] * peak_2

    log_pi = numpy.log(peaks) + log_pl + log_n1 + log_n2 + log_h + log_l
    if not derivatives:
        return log_pi, {}

    out = {name: dlog_pl[name] for name in ("alpha_1", "alpha_dip", "alpha_2")}
    out["NSmin"], out["n0"] = dlog_h["m_edge"], dlog_h["eta"]
    out["BHmax"], out["n5"] = dlog_l["m_edge"], dlog_l["eta"]
    out["A"], out["n1"], out["n2"] = dlog_n1["depth"], dlog_n1["eta_low"], dlog_n1["eta_high"]
    out["NSmax"] = dlog_pl["NSmax"] + dlog_n1["m_low"]
    out["BHmin"] = dlog_pl["BHmin"] + dlog_n1["m_high"]
    out["A2"], out["n3"], out["n4"] = dlog_n2["depth"], dlog_n2["eta_low"], dlog_n2["eta_high"]
    out["UPPERmin"], out["UPPERmax"] = dlog_n2["m_low"], dlog_n2["m_high"]
    for index, mix, peak, dpeak in ((1, p["mix1"], peak_1, dpeak_1), (2, p["mix2"], peak_2, dpeak_2)):
        out[f"mix{index}"] = peak / peaks
        out[f"mu{index}"] = mix * dpeak["mu"] / peaks
        out[f"sig{index}"] = mix * dpeak["sigma"] / peaks
    out["mass"] = (
        (p["mix1"] * dpeak_1["mass"] + p["mix2"] * dpeak_2["mass"]) / peaks
        + dlog_pl["mass"] + dlog_n1["mass"] + dlog_n2["mass"] + dlog_h["mass"] + dlog_l["mass"]
    )
    return log_pi, out


def pairing(mass_ratio, mass_2, params, support, derivatives=True):
    """
    Eq. B8 as GWForge evaluates it (``NotchFilterBinnedPairingMassDistribution.pairing``):
    a power law in q normalised on [q_low, 1] on each branch,

        f(q) = (beta + 1) q^beta / (1 - q_low^(beta + 1)),

    with (beta, q_low) = (beta_pair_1, mmin/mmax) for m2 < mbreak, else
    (beta_pair_2, mbreak/mmax).  The paper prints a bare q^beta.

        d ln f/d beta = ln q + 1/(beta + 1) + q_low^(beta+1) ln q_low / (1 - q_low^(beta+1))

    on the branch of that beta, and 0 on the other.
    """
    q = numpy.asarray(mass_ratio, dtype=float)
    low_branch = numpy.asarray(mass_2) < support["mbreak"]
    beta = numpy.where(low_branch, params["beta_pair_1"], params["beta_pair_2"])
    q_low = numpy.where(
        low_branch, support["mmin"] / support["mmax"], support["mbreak"] / support["mmax"]
    )
    q_low_power = q_low ** (beta + 1.0)
    log_f = beta * numpy.log(q) + numpy.log(beta + 1.0) - numpy.log1p(-q_low_power)
    if not derivatives:
        return log_f, {}
    dlog_f_dbeta = (
        numpy.log(q) + 1.0 / (beta + 1.0)
        + q_low_power * numpy.log(q_low) / (1.0 - q_low_power)
    )
    return log_f, {
        "beta_pair_1": numpy.where(low_branch, dlog_f_dbeta, 0.0),
        "beta_pair_2": numpy.where(low_branch, 0.0, dlog_f_dbeta),
    }


class FullPopMass(PopModel):
    """
    FullPop mass model (arXiv:2605.27226 App. B.2, Eqs. B2-B8; GWTC-4 FullPop),
    restricted to the BBH part of the catalogue:

        p(m1, m2) = pi_m(m1) pi_m(m2) f(q, m2) / Z(Lambda),
        support   m2_min <= m2 <= m1 <= mmax.

    This is the model that generated the injections, so evaluating it at
    ``FULLPOP_GWTC4_FIDUCIAL`` is the true population (up to the redshift model).

    ``Z`` is a 2D trapezoid over a fixed log-spaced node set on the support
    triangle (inner integral over m2 <= m1 is exact trapezoid on the same nodes).
    ``analytic_score`` subtracts d ln Z / d Lambda, computed on the same grid
    from the closed-form node derivatives, so log_prob and score are mutually
    consistent.  In the centred-score Term-I Fisher this column offset cancels.

    Required event keys: ``mass_1_source``, ``mass_2_source``.
    """

    def __init__(self, fiducial=None, support=None, n_grid=1500, population="bbh"):
        """
        population : "bbh" restricts to m2 >= 2.06 (the BBH file); "all" is the
        whole CBC population on m2 >= mmin (BNS + NSBH + BBH files together).
        """
        if population not in ("bbh", "all"):
            raise ValueError(f"population must be 'bbh' or 'all', not {population!r}")
        self.population = population
        self.fiducial = dict(FULLPOP_GWTC4_FIDUCIAL if fiducial is None else fiducial)
        self.support = fullpop_support(population, support)
        self.parameter_names = list(FULLPOP_PARAMETER_NAMES)
        self.phys_event_keys = ["mass_1_source", "mass_2_source"]
        self._build_grid(n_grid)
        self._cache_key = None

    # -- normalisation grid ------------------------------------------------

    def _build_grid(self, n_grid):
        s = self.support
        nodes = numpy.geomspace(s["m2_min"], s["mmax"], n_grid)
        # resolve the pairing step at mbreak with a node on each side
        nodes = numpy.unique(numpy.concatenate(
            [nodes, [s["mbreak"] * (1 - 1e-9), s["mbreak"] * (1 + 1e-9)]]
        ))
        h = numpy.diff(nodes)
        outer = numpy.zeros(len(nodes))
        outer[:-1] += 0.5 * h
        outer[1:] += 0.5 * h
        # inner trapezoid over nodes 0..i:  interior weights c_j, endpoint h_{i-1}/2
        c = numpy.zeros(len(nodes))
        c[0] = 0.5 * h[0]
        c[1:-1] = 0.5 * (h[:-1] + h[1:])
        inner = numpy.tril(numpy.ones((len(nodes), len(nodes))), k=-1) * c[None, :]
        inner[0, 0] = 0.0
        diag = numpy.arange(1, len(nodes))
        inner[diag, diag] = 0.5 * h
        inner[:, 0] = numpy.where(numpy.arange(len(nodes)) > 0, 0.5 * h[0], 0.0)
        self._nodes = nodes
        self._weights = outer[:, None] * inner          # W_ij, i = m1, j = m2
        mass_1 = numpy.broadcast_to(nodes[:, None], self._weights.shape)
        mass_2 = numpy.broadcast_to(nodes[None, :], self._weights.shape)
        self._grid_mask = self._weights > 0
        self._grid_q = (mass_2 / mass_1)[self._grid_mask]
        self._grid_m2 = mass_2[self._grid_mask]

    def _normalisation(self, params):
        """ln Z and d ln Z / d name (dict) at ``params``; cached on the last call."""
        key = tuple(params[name] for name in self.parameter_names)
        if key == self._cache_key:
            return self._cache_value
        s = self.support
        log_pi, grad_pi = mass_function(self._nodes, params, s["mmin"], s["mmax"])
        log_f, grad_f = pairing(self._grid_q, self._grid_m2, params, s)

        n = len(self._nodes)
        log_integrand = numpy.full((n, n), -numpy.inf)
        log_integrand[self._grid_mask] = log_f
        log_integrand += log_pi[:, None] + log_pi[None, :]
        offset = numpy.max(log_integrand[self._grid_mask])
        integrand = numpy.exp(log_integrand - offset) * self._weights
        total = integrand.sum()
        log_Z = numpy.log(total) + offset

        row = integrand.sum(axis=1) / total   # weight of each node as m1
        col = integrand.sum(axis=0) / total   # weight of each node as m2
        dlog_Z = {}
        for name in _FULLPOP_PI_M_NAMES:
            dlog_Z[name] = float(numpy.sum((row + col) * grad_pi[name]))
        flat = integrand[self._grid_mask] / total
        for name in _FULLPOP_PAIRING_NAMES:
            dlog_Z[name] = float(numpy.sum(flat * grad_f[name]))

        self._cache_key = key
        self._cache_value = (log_Z, dlog_Z)
        return self._cache_value

    # -- PopModel interface --------------------------------------------------

    def _in_support(self, mass_1, mass_2):
        s = self.support
        return (mass_2 >= s["m2_min"]) & (mass_2 <= mass_1) & (mass_1 <= s["mmax"])

    def _evaluate(self, events, params, derivatives):
        """
        Eq. B2:  ln p = ln pi_m(m1) + ln pi_m(m2) + ln f(q, m2) - ln Z.

        Returns ln p, the score and d ln p/d m1 at fixed q (-inf / NaN off the
        support).  The score is d ln pi_m(m1) + d ln pi_m(m2) - d ln Z for a
        parameter of pi_m, and d ln f - d ln Z for beta_pair_1/2.  At fixed q
        both masses scale together (the spectral-siren direction), so
        d ln p/d m1|_q = pi_m'(m1)/pi_m(m1) + q pi_m'(m2)/pi_m(m2).  The
        pairing depends on q alone, apart from its branch step at m2 = mbreak,
        which has measure zero.
        """
        mass_1 = numpy.asarray(events["mass_1_source"], dtype=float)
        mass_2 = numpy.asarray(events["mass_2_source"], dtype=float)
        s = self.support
        log_prob = numpy.full(len(mass_1), -numpy.inf)
        score = numpy.full((len(mass_1), len(self.parameter_names)), numpy.nan)
        dlogp_dmass_1 = numpy.full(len(mass_1), numpy.nan)
        result = (log_prob, score, dlogp_dmass_1)
        valid = self._in_support(mass_1, mass_2)
        if not numpy.any(valid):
            return result
        m1, m2 = mass_1[valid], mass_2[valid]

        log_pi_1, dlog_pi_1 = mass_function(m1, params, s["mmin"], s["mmax"], derivatives)
        log_pi_2, dlog_pi_2 = mass_function(m2, params, s["mmin"], s["mmax"], derivatives)
        log_f, dlog_f = pairing(m2 / m1, m2, params, s, derivatives)
        log_Z, dlog_Z = self._normalisation(params)
        log_prob[valid] = log_pi_1 + log_pi_2 + log_f - log_Z
        if not derivatives:
            return result

        score[valid] = numpy.column_stack([
            (dlog_f[name] if name in dlog_f else dlog_pi_1[name] + dlog_pi_2[name])
            - dlog_Z[name]
            for name in self.parameter_names
        ])
        dlogp_dmass_1[valid] = dlog_pi_1["mass"] + (m2 / m1) * dlog_pi_2["mass"]
        return result

    def log_prob(self, events, params):
        return self._evaluate(events, params, derivatives=False)[0]

    def analytic_score(self, events, params):
        """d ln p / d Lambda, shape (N_events, len(parameter_names)); NaN off support."""
        return self._evaluate(events, params, derivatives=True)[1]

    def dlogp_dmass_1(self, events, params):
        """d ln p / d mass_1_source at fixed mass ratio, shape (N_events,)."""
        return self._evaluate(events, params, derivatives=True)[2]