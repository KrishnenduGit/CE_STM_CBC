import numpy, logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def _log_dVc_dz_over_1plusz(redshift, cosmo=None):
    if cosmo is None:
        from astropy.cosmology import Planck18 as cosmo

    dVc_dz = cosmo.differential_comoving_volume(redshift).value  # Mpc^3/sr (up to constant factors, irrelevant)
    return numpy.log(dVc_dz) - numpy.log1p(redshift)


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------


class PopModel:
    """
    Base class for GW population models.

    Subclasses must define
    ----------------------
    parameter_names  : list[str]
        Ordered list of free hyperparameter names.
    fiducial : dict
        Fiducial (reference) hyperparameter values.

    and implement
    -------------
    log_prob(events, params) -> ndarray(N_events,)
        Log (unnormalised) probability per detected event.
        Return -numpy.inf for events outside the model support.

    Optionally override
    -------------------
    analytic_score(events, params) -> ndarray(N_events, N_params)
        If provided, the population Fisher uses this instead of finite
        differences.  Speeds up computation and eliminates FD noise for
        models with simple analytic gradients.
    """

    parameter_names = []
    fiducial = {}
    phys_event_keys = []  # physical event params this model depends on

    def log_prob(self, events, params):
        raise NotImplementedError

    def parameters_to_vector(self, params):
        """Ordered parameter vector from a dict."""
        return numpy.array([params[n] for n in self.parameter_names], dtype=float)

    def vector_to_parameters(self, vec):
        """Parameter dict from an ordered vector."""
        return {n: float(v) for n, v in zip(self.parameter_names, vec)}

    def __repr__(self):
        return (
            f"{type(self).__name__}("
            + ", ".join(f"{k}={v}" for k, v in self.fiducial.items())
            + ")"
        )


# ---------------------------------------------------------------------------
# Joint model
# ---------------------------------------------------------------------------


class JointModel(PopModel):
    """
    Product of independent models: p(theta | lambda) = prod_k p_k(theta_k | lambda_k).

    The hyperparameter namespace is the union of all constituent models.
    Duplicate names across models raise a ValueError on construction.

    Parameters
    ----------
    *models : PopModel instances
        Independent sub-models to combine.

    Examples
    --------
    >>> joint = JointModel(MadauDickinsonRedshift(), TruncatedPowerLawMass())
    >>> joint.parameter_names
    ['alpha', 'beta', 'z_peak', 'alpha_m', 'beta_q', 'm_min', 'm_max']
    """

    def __init__(self, *models):
        self._models = list(models)
        seen = {}
        for model in self._models:
            for name in model.parameter_names:
                if name in seen:
                    raise ValueError(
                        f"Duplicate hyperparameter '{name}' in "
                        f"{seen[name].__class__.__name__} and "
                        f"{model.__class__.__name__}.  Rename one."
                    )
                seen[name] = model
        self.parameter_names = [
            name for model in self._models for name in model.parameter_names
        ]
        self.fiducial = {
            name: value
            for model in self._models
            for name, value in model.fiducial.items()
        }
        # Union of physical event keys
        # (canonical order: mass_1_source, mass_2_source, redshift)
        _canonical_order = ["mass_1_source", "mass_2_source", "redshift"]
        seen_keys = set()
        for model in self._models:
            for key in getattr(model, "phys_event_keys", []):
                seen_keys.add(key)
        self.phys_event_keys = [key for key in _canonical_order if key in seen_keys]

    def log_prob(self, events, params):
        total = numpy.zeros(len(next(iter(events.values()))))
        for model in self._models:
            sub_params = {name: params[name] for name in model.parameter_names}
            total = total + model.log_prob(events, sub_params)
        return total

    def analytic_score(self, events, params):
        """
        Concatenates scores from sub-models.

        Sub-models with ``analytic_score`` use that; others fall back to
        centred finite differences.
        """
        from pop_fisher_term_I import _numerical_score  # lazy import avoids circular dep

        cols = []
        for model in self._models:
            sub_params = {name: params[name] for name in model.parameter_names}
            if hasattr(model, "analytic_score"):
                cols.append(model.analytic_score(events, sub_params))
            else:
                cols.append(_numerical_score(model, events, sub_params))
        return numpy.hstack(cols)

    def __repr__(self):
        return "JointModel(" + ", ".join(repr(m) for m in self._models) + ")"


# ---------------------------------------------------------------------------
# Redshift models
# ---------------------------------------------------------------------------


class MadauDickinsonRedshift(PopModel):
    """
    BBH merger rate following the Madau-Dickinson star-formation rate (SFR).

    Physical event parameters this model depends on (needed for higher-order
    population Fisher correction terms):
        z = source redshift

    The observed merger rate per unit redshift is

        dN/dz ∝ psi_MD(z) * dV_c/dz(z) / (1+z)

    with the Madau-Dickinson SFR shape

        psi_MD(z) = (1+z)^alpha / [1 + ((1+z)/(1+z_peak))^(alpha+beta)]

    Folding in the 1/(1+z) time-dilation, the full log probability per event is

        ln p(z | alpha, beta, z_peak)
            = alpha ln(1+z) - ln[1 + ((1+z)/(1+z_peak))^(alpha+beta)]
              + ln[dV_c/dz(z)] - ln(1+z)

    The dV_c/dz(z)/(1+z) piece is cosmology-fixed (lambda-independent), so it
    drops out of d ln p / d lambda exactly -- `analytic_score` below omits it
    and only encodes the shape term. It is NOT omitted from `log_prob`,
    however, because `log_prob` is also finite-differenced in z (not just in
    lambda) to build the population-curvature Hessian H_theta used in Fisher
    terms II-V (see `_H_theta`/`_P_theta` in pop_fisher.py). 

    Analytic scores (d ln p / d lambda) are provided (no finite differences
    needed) since the added cosmological term is lambda-independent.

    Parameters
    ----------
    alpha  : low-z power-law slope (fiducial 2.7)
    beta   : high-z fall-off index (fiducial 2.9)
    z_peak : redshift at the SFR peak (fiducial 1.9)

    Required event key
    ------------------
    'redshift' : source redshift array
    """

    def __init__(
        self, 
        parameter_names = ['alpha', 'beta', 'z_peak'],
        fiducial={"alpha": 2.7, "beta": 2.9, "z_peak": 1.9}, phys_event_keys=["redshift"]
    ):

        self.parameter_names = parameter_names
        self.fiducial = fiducial
        self.phys_event_keys = phys_event_keys

    def log_prob(self, events, params):
        redshift = events["redshift"]
        alpha = params["alpha"]
        beta = params["beta"]
        z_peak = params["z_peak"]

        # log psi_MD(z); the 1/(1+z) time dilation lives in the helper below and
        # must NOT be folded in here as well.
        log_shape = alpha * numpy.log1p(redshift) - numpy.log1p(
            ((1.0 + redshift) / (1.0 + z_peak)) ** (alpha + beta)
        )
        return log_shape + _log_dVc_dz_over_1plusz(redshift)

    def analytic_score(self, events, params):
        """
        Analytic score d ln p / d lambda for each event.
        (Look at Koustav's copy for the derivation.)

        Let u = (1+z)/(1+z_peak),  r = u^(alpha+beta).

        d ln p / d alpha  = ln(1+z) - r/(1+r) * ln(u)
        d ln p / d beta   =         - r/(1+r) * ln(u)
        d ln p / d z_peak = (alpha+beta) * r / ((1+r) * (1+z_peak))
        """
        redshift = events["redshift"]
        alpha = params["alpha"]
        beta = params["beta"]
        z_peak = params["z_peak"]

        ratio = (1.0 + redshift) / (1.0 + z_peak)
        r = ratio ** (alpha + beta)
        ln_ratio = numpy.log(ratio)
        frac = r / (1.0 + r)

        d_alpha = numpy.log1p(redshift) - frac * ln_ratio
        d_beta = -frac * ln_ratio
        d_zpeak = (alpha + beta) * frac / (1.0 + z_peak)

        return numpy.column_stack([d_alpha, d_beta, d_zpeak])


class PowerLawRedshift(PopModel):
    """
    BBH merger rate as a power law in (1+z).

    The observed merger rate per unit redshift (Eq. B17 of arXiv:2605.27226):

        dN/dz ∝ (1+z)^kappa * dV_c/dz(z) / (1+z) = (1+z)^(kappa-1) * dV_c/dz

    The lambda-dependent log probability per event, including the
    cosmology-fixed dV_c/dz(z)/(1+z) curvature term (needed for the
    finite-difference theta-Hessian H_theta used in Fisher terms II-V; see
    `MadauDickinsonRedshift`'s docstring for the same rationale), is

        ln p(z | kappa) = kappa ln(1+z) + ln[dV_c/dz(z)] - ln(1+z)

    Analytic score (d ln p / d kappa, unaffected by the added
    lambda-independent cosmological term) provided.

    Parameters
    ----------
    kappa : power-law index in (1+z) (fiducial 2.7)

    Required event key
    ------------------
    'redshift' : source redshift array
    """

    def __init__(self, 
                 parameter_names = ['kappa'],
                 fiducial={"kappa": 2.7}, 
                 phys_event_keys=["redshift"]):

        self.parameter_names = parameter_names
        self.fiducial = fiducial
        self.phys_event_keys = phys_event_keys

    def log_prob(self, events, params):
        redshift = events["redshift"]
        # kappa (not kappa-1): the 1/(1+z) time dilation is already in the helper.
        return params["kappa"] * numpy.log1p(redshift) + _log_dVc_dz_over_1plusz(redshift)

    def analytic_score(self, events, params):
        """d ln p / d kappa = ln(1+z)"""
        return numpy.log1p(events["redshift"]).reshape(-1, 1)


# ---------------------------------------------------------------------------
# Mass models
# ---------------------------------------------------------------------------


class TruncatedPowerLawMass(PopModel):
    """
    Truncated power-law primary mass with a power-law mass ratio.

    Equations B10-B14 of the GWTC-5 populations paper (arXiv:2605.27226).

    Primary mass
    ------------
        p(m1 | alpha_m, m_min, m_max) ∝ m1^{-alpha_m},   m1 ∈ [m_min, m_max]

    Mass ratio conditioned on m1
    ----------------------------
        p(q | beta_q, m_min, m1)      ∝ q^{beta_q},       q ∈ [m_min/m1, 1]

    Change of variables to (m1, m2)
    -------------------------------
    The two factors above give a density in (m1, q), but events are supplied in
    (mass_1_source, mass_2_source) and `_H_theta`/`_P_theta` differentiate with
    respect to
    those.  Since q = m2/m1 at fixed m1, dq = dm2/m1, so

        p(m1, m2) = p(m1) p(q | m1) / m1

    and `log_prob` carries the extra -ln(m1).  Like the dV_c/dz term in the
    redshift models this is lambda-independent -- it cancels from every
    lambda-derivative, so `analytic_score` and Gamma_I are unaffected -- but it
    contributes real m1-curvature to H_theta, and hence to Fisher terms II-V.

    Parameters
    ----------
    alpha_m : primary-mass power-law slope (fiducial 3.5)
    beta_q  : mass-ratio power-law slope   (fiducial 1.4)
    m_min   : minimum component mass [M_sun] (fiducial 5.0)
    m_max   : maximum primary mass   [M_sun] (fiducial 100.0)

    Required event keys
    -------------------
    'mass_1_source', 'mass_2_source'
    ('mass_ratio' = m2/m1 is derived automatically by the catalogue loader)

    """

    def __init__(
        self,
        parameter_names = ['alpha_m', 'beta_q', 'm_min', 'm_max'],
        fiducial={"alpha_m": 3.5, "beta_q": 1.4, "m_min": 5.0, "m_max": 100.0},
        phys_event_keys=["mass_1_source", "mass_2_source"],
    ):

        self.parameter_names = parameter_names
        self.fiducial = fiducial
        self.phys_event_keys = phys_event_keys

    def log_prob(self, events, params):
        mass_1 = events["mass_1_source"]
        mass_ratio = events["mass_ratio"]
        alpha_m = params["alpha_m"]
        beta_q = params["beta_q"]
        m_min = params["m_min"]
        m_max = params["m_max"]
        q_min = m_min / mass_1

        log_prob = numpy.full(len(mass_1), -numpy.inf)
        valid = (
            (mass_1 >= m_min)
            & (mass_1 <= m_max)
            & (mass_ratio >= q_min)
            & (mass_ratio <= 1.0)
        )
        if not numpy.any(valid):
            return log_prob

        log_norm_m1 = _log_pl_norm(alpha_m, m_min, m_max)
        log_norm_q = _log_pl_norm_lower(beta_q, q_min[valid])

        log_prob[valid] = (
            -alpha_m * numpy.log(mass_1[valid])
            + beta_q * numpy.log(mass_ratio[valid])
            - log_norm_m1
            - log_norm_q
            - numpy.log(mass_1[valid])  # dq/dm2 Jacobian, see class docstring
        )
        return log_prob


class PowerLawPlusPeakMass(PopModel):
    """
    Power-law + single Gaussian peak primary mass with a power-law mass ratio.

    Primary mass (PLPP model, Talbot & Thrane 2018)
    ------------------------------------------------
        p(m1) ∝ [(1 - lambda_peak) * m1^{-alpha_m}
                 + lambda_peak * N(m1; mu_m, sigma_m)] * S(m1; m_min, delta_m)

    where S(m1; m_min, delta_m) is the Talbot & Thrane (2018) low-mass
    smoothing window.

    Mass ratio conditioned on m1
    ----------------------------
        p(q | beta_q, m_min, m1) ∝ q^{beta_q},  q ∈ [m_min/m1, 1]

    Parameters
    ----------
    alpha_m     : power-law slope (fiducial 3.5)
    beta_q      : mass-ratio slope (fiducial 1.4)
    m_min       : minimum component mass [M_sun] (fiducial 5.0)
    delta_m     : low-mass smoothing scale [M_sun] (fiducial 4.8)
    m_max       : maximum primary mass [M_sun] (fiducial 87.0)
    lambda_peak : Gaussian fraction in [0, 1] (fiducial 0.04)
    mu_m        : Gaussian peak location [M_sun] (fiducial 33.5)
    sigma_m     : Gaussian width [M_sun] (fiducial 5.7)

    Required event keys
    -------------------
    'mass_1_source', 'mass_2_source'

    Notes
    -----
    The primary-mass normalisation integral is evaluated numerically on a
    2000-point grid at each call.  m_max is degenerate in the Gamma_I
    approximation (see TruncatedPowerLawMass docstring); use fixed_params to
    pin it.
    """

    def __init__(
        self,
        parameter_names = ["alpha_m", "beta_q", "m_min", "delta_m", "m_max",
                           "lambda_peak", "mu_m", "sigma_m"],
        fiducial={
            "alpha_m": 3.5,
            "beta_q": 1.4,
            "m_min": 5.0,
            "delta_m": 4.8,
            "m_max": 87.0,
            "lambda_peak": 0.04,
            "mu_m": 33.5,
            "sigma_m": 5.7,
        },
        phys_event_keys=["mass_1_source", "mass_2_source"],
    ):

        self.parameter_names = parameter_names
        self.fiducial = fiducial
        self.phys_event_keys = phys_event_keys

    def log_prob(self, events, params):
        mass_1 = events["mass_1_source"]
        mass_ratio = events["mass_ratio"]
        alpha_m = params["alpha_m"]
        beta_q = params["beta_q"]
        m_min = params["m_min"]
        delta_m = params["delta_m"]
        m_max = params["m_max"]
        lambda_peak = params["lambda_peak"]
        mu_m = params["mu_m"]
        sigma_m = params["sigma_m"]
        q_min = m_min / mass_1

        log_prob = numpy.full(len(mass_1), -numpy.inf)
        valid = (
            (mass_1 >= m_min)
            & (mass_1 <= m_max)
            & (mass_ratio >= q_min)
            & (mass_ratio <= 1.0)
        )
        if not numpy.any(valid):
            return log_prob

        mass_1_valid = mass_1[valid]
        smooth = _smoothing_window(mass_1_valid, m_min, delta_m)

        pl_shape = mass_1_valid ** (-alpha_m)
        gauss_shape = numpy.exp(-0.5 * ((mass_1_valid - mu_m) / sigma_m) ** 2)
        mix_shape = (1.0 - lambda_peak) * pl_shape + lambda_peak * gauss_shape

        # Numerical normalisation over primary mass
        m1_grid = numpy.linspace(m_min, m_max, 2000)
        sm_grid = _smoothing_window(m1_grid, m_min, delta_m)
        pl_grid = m1_grid ** (-alpha_m)
        ga_grid = numpy.exp(-0.5 * ((m1_grid - mu_m) / sigma_m) ** 2)
        mx_grid = ((1.0 - lambda_peak) * pl_grid + lambda_peak * ga_grid) * sm_grid
        norm_m1 = numpy.trapz(mx_grid, m1_grid)
        if norm_m1 <= 0:
            return log_prob

        log_norm_q = _log_pl_norm_lower(beta_q, q_min[valid])

        log_prob[valid] = (
            numpy.log(mix_shape * smooth)
            + beta_q * numpy.log(mass_ratio[valid])
            - numpy.log(norm_m1)
            - log_norm_q
            - numpy.log(mass_1[valid])  # dq/dm2 Jacobian, see class docstring
        )
        return log_prob


class BrokenPowerLawPlusTwoPeaksMass(PopModel):
    """
    Broken power-law + two Gaussian peaks primary mass model (GWTC-5 Default BBH).

    This is the fiducial mass model used in the GWTC-5 population analysis
    (arXiv:2605.27226, Appendix B, Eqs. B10-B14).

    Primary mass
    ------------
    Mixture of a broken power law and two left-truncated Gaussian peaks,
    with the Talbot & Thrane (2018) low-mass smoothing applied globally:

        pi(m1) ∝ [ lambda_0 * p_BP(m1 | alpha_1, alpha_2, m_break, m_min, m_max)
                  + lambda_1 * N_lt(m1 | mu_1, sigma_1, low=m_min)
                  + (1 - lambda_0 - lambda_1) * N_lt(m1 | mu_2, sigma_2, low=m_min)
                 ] * S(m1 | m_min, delta_m)

    where the broken power law is (Eq. B10):

        p_BP(m1) ∝ (m1/m_break)^{-alpha_1}   for m_min <= m1 < m_break
                   (m1/m_break)^{-alpha_2}   for m_break <= m1 < m_max

    with the analytic normalisation constant (Eq. B11):

        N = m_break * [ (1-(m_min/m_break)^{1-alpha_1}) / (1-alpha_1)
                       + ((m_max/m_break)^{1-alpha_2} - 1) / (1-alpha_2) ]

    N_lt(m1 | mu, sigma, low=m_min) is a normal distribution truncated from
    below at m_min (un-normalised here; the global normalisation is absorbed
    into the numerical integral over the full distribution).

    S(m1 | m_min, delta_m) is the Talbot & Thrane (2018) smoothing window.

    Mass ratio conditioned on m1
    ----------------------------
        p(q | beta_q, m_min, m1) ∝ q^{beta_q},  q ∈ [m_min/m1, 1]

    Parameters
    ----------
    alpha_1   : low-mass power-law slope (fiducial 1.5)
    alpha_2   : high-mass power-law slope (fiducial 5.4)
    m_break   : break mass between the two power-law segments [M_sun] (fiducial 37.5)
    lambda_0  : weight on the broken power law in [0,1] (fiducial 0.90)
    lambda_1  : weight on first Gaussian peak in [0, 1-lambda_0] (fiducial 0.05)
                Second Gaussian has weight (1 - lambda_0 - lambda_1).
    mu_1      : first Gaussian peak location [M_sun] (fiducial 10.0)
    sigma_1   : first Gaussian width [M_sun] (fiducial 3.0)
    mu_2      : second Gaussian peak location [M_sun] (fiducial 35.0)
    sigma_2   : second Gaussian width [M_sun] (fiducial 5.0)
    beta_q    : mass-ratio power-law slope (fiducial 1.4)
    m_min     : minimum component mass [M_sun] (fiducial 5.0)
    delta_m   : low-mass smoothing scale [M_sun] (fiducial 4.8)
    m_max     : maximum primary mass [M_sun] (fiducial 100.0)

    Required event keys
    -------------------
    'mass_1_source', 'mass_2_source'

    Notes
    -----
    lambda_0 + lambda_1 must lie in [0, 1].  Values outside this range
    produce -inf for all events.

    m_max is degenerate in the Gamma_I approximation; pin it via fixed_params.
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

    def log_prob(self, events, params):
        mass_1 = events["mass_1_source"]
        mass_ratio = events["mass_ratio"]
        alpha_1 = params["alpha_1"]
        alpha_2 = params["alpha_2"]
        m_break = params["m_break"]
        lambda_0 = params["lambda_0"]
        lambda_1 = params["lambda_1"]
        mu_1 = params["mu_1"]
        sigma_1 = params["sigma_1"]
        mu_2 = params["mu_2"]
        sigma_2 = params["sigma_2"]
        beta_q = params["beta_q"]
        m_min = params["m_min"]
        delta_m = params["delta_m"]
        m_max = params["m_max"]
        lambda_2 = 1.0 - lambda_0 - lambda_1

        log_prob = numpy.full(len(mass_1), -numpy.inf)

        # Sanity check on mixture weights
        if lambda_0 < 0.0 or lambda_1 < 0.0 or lambda_2 < 0.0:
            return log_prob
        if m_break <= m_min or m_break >= m_max:
            return log_prob

        q_min = m_min / mass_1
        valid = (
            (mass_1 >= m_min)
            & (mass_1 <= m_max)
            & (mass_ratio >= q_min)
            & (mass_ratio <= 1.0)
        )
        if not numpy.any(valid):
            return log_prob

        mass_1_valid = mass_1[valid]
        smooth = _smoothing_window(mass_1_valid, m_min, delta_m)

        # Broken power law (un-normalised, normalised jointly below)
        bp_shape = _broken_power_law_shape(
            mass_1_valid, alpha_1, alpha_2, m_break, m_min, m_max
        )

        # Left-truncated Gaussians (un-normalised at m_min; global norm handles this)
        gauss_1 = numpy.exp(-0.5 * ((mass_1_valid - mu_1) / sigma_1) ** 2)
        gauss_2 = numpy.exp(-0.5 * ((mass_1_valid - mu_2) / sigma_2) ** 2)

        mix_shape = lambda_0 * bp_shape + lambda_1 * gauss_1 + lambda_2 * gauss_2

        # Numerical normalisation of the full mixture * smoothing over [m_min, m_max]
        n_grid = 3000
        m1_grid = numpy.linspace(m_min, m_max, n_grid)
        sm_grid = _smoothing_window(m1_grid, m_min, delta_m)
        bp_grid = _broken_power_law_shape(
            m1_grid, alpha_1, alpha_2, m_break, m_min, m_max
        )
        g1_grid = numpy.exp(-0.5 * ((m1_grid - mu_1) / sigma_1) ** 2)
        g2_grid = numpy.exp(-0.5 * ((m1_grid - mu_2) / sigma_2) ** 2)
        mx_grid = (
            lambda_0 * bp_grid + lambda_1 * g1_grid + lambda_2 * g2_grid
        ) * sm_grid
        norm_m1 = numpy.trapz(mx_grid, m1_grid)
        if norm_m1 <= 0:
            return log_prob

        log_norm_q = _log_pl_norm_lower(beta_q, q_min[valid])

        log_prob[valid] = (
            numpy.log(numpy.maximum(mix_shape * smooth, 1e-300))
            + beta_q * numpy.log(mass_ratio[valid])
            - numpy.log(norm_m1)
            - log_norm_q
            - numpy.log(mass_1[valid])  # dq/dm2 Jacobian, see class docstring
        )
        return log_prob


# names and dragons and aliases for backwards compatibility
PowerLawPlusTwoPeaksMass = BrokenPowerLawPlusTwoPeaksMass


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _broken_power_law_shape(mass_1, alpha_1, alpha_2, m_break, m_min, m_max):
    """
    Un-normalised broken power-law shape (Eq. B10 of arXiv:2605.27226).

        p_BP ∝ (m1/m_break)^{-alpha_1}   for m_min <= m1 < m_break
               (m1/m_break)^{-alpha_2}   for m_break <= m1 <= m_max

    Vectorised over mass_1.  Events outside [m_min, m_max] return 0.
    """
    mass_1 = numpy.asarray(mass_1, dtype=float)
    shape = numpy.zeros_like(mass_1)
    low = (mass_1 >= m_min) & (mass_1 < m_break)
    high = (mass_1 >= m_break) & (mass_1 <= m_max)
    if numpy.any(low):
        shape[low] = (mass_1[low] / m_break) ** (-alpha_1)
    if numpy.any(high):
        shape[high] = (mass_1[high] / m_break) ** (-alpha_2)
    return shape


def _log_pl_norm(alpha, x_min, x_max):
    """
    log integral_{x_min}^{x_max} x^{-alpha} dx.

    Handles alpha == 1 via log(x_max/x_min).
    """
    if abs(alpha - 1.0) < 1e-10:
        return numpy.log(numpy.log(x_max / x_min))
    value = (x_min ** (1.0 - alpha) - x_max ** (1.0 - alpha)) / (alpha - 1.0)
    return numpy.log(value) if value > 0 else -numpy.inf


def _log_pl_norm_lower(beta, q_min):
    """
    log integral_{q_min}^{1} q^{beta} dq,  broadcast over array q_min.

    Handles beta == -1 exactly.
    """
    q_min = numpy.asarray(q_min, dtype=float)
    if abs(beta + 1.0) < 1e-10:
        return -numpy.log(q_min)
    value = (1.0 - q_min ** (beta + 1.0)) / (beta + 1.0)
    tiny = numpy.finfo(float).tiny
    return numpy.log(numpy.where(value > 0, value, tiny))


def _smoothing_window(mass, m_min, delta_m):
    """
    Low-mass smoothing window S(m; m_min, delta_m) from Talbot & Thrane 2018.

    Smoothly transitions from 0 at m = m_min to 1 at m = m_min + delta_m.
    Identically 0 for m <= m_min; identically 1 for m >= m_min + delta_m.
    """
    mass = numpy.asarray(mass, dtype=float)
    out = numpy.ones_like(mass)
    out[mass <= m_min] = 0.0

    in_ramp = (mass > m_min) & (mass < m_min + delta_m)
    if numpy.any(in_ramp):
        x = mass[in_ramp] - m_min
        d = delta_m
        exponent = numpy.clip(d / x + d / (x - d), -500.0, 500.0)
        out[in_ramp] = 1.0 / (1.0 + numpy.exp(exponent))

    return out
