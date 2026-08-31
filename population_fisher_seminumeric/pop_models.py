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

    analytic_score(events, params) -> ndarray(N_events, N_params)
        Closed-form score d ln p / d lambda per event; required by the
        population Fisher.
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
    >>> joint = JointModel(BrokenPowerLawPlusTwoPeaksMass(), MadauDickinsonRedshift())
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
        """Concatenate the analytic scores of the sub-models (all must have one)."""
        cols = []
        for model in self._models:
            sub_params = {name: params[name] for name in model.parameter_names}
            cols.append(model.analytic_score(events, sub_params))
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
    terms II-V (see `_H_theta`/`_P_theta` in pop_fisher_higher_order.py). 

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


class BrokenPowerLawPlusTwoPeaksMass(PopModel):
    """
    Broken power-law + two Gaussian peaks primary mass model (GWTC-5 Default BBH).

    Implements Eqs. B10-B14 of arXiv:2605.27226, Appendix B.4.

    Primary mass (Eq. B12)
    ----------------------
        pi(m1) ∝ [ lambda_0 p_BP(m1 | alpha_1, alpha_2, m_break, m_min, m_max)
                  + lambda_1 N_lt(m1 | mu_1, sigma_1, low=m_min)
                  + (1 - lambda_0 - lambda_1) N_lt(m1 | mu_2, sigma_2, low=m_min)
                 ] S(m1 | m_min, delta_m)

    **All three components are individually normalised**, so lambda_0, lambda_1
    and (1 - lambda_0 - lambda_1) are genuine mixing fractions -- which is what
    the Dirichlet prior Dir(alpha=(1,1,1)) of the paper's Tab. 5 presumes.  The
    broken power law (Eq. B10) carries the analytic constant N of Eq. B11,

        N = m_break [ (1 - (m_min/m_break)^{1-alpha_1}) / (1 - alpha_1)
                     + ((m_max/m_break)^{1-alpha_2} - 1) / (1 - alpha_2) ],

    and N_lt is a *left-truncated normal distribution*, normalised over
    [m_min, inf).  The remaining overall proportionality is fixed by one
    numerical integral, which therefore corrects only for the taper S -- exactly
    as ``gwpopulation``'s ``BaseSmoothedMassDistribution.norm_p_m1`` does.

    S is the Planck taper of Eqs. B13-B14 (`_smoothing_window`).

    Mass ratio conditioned on m1
    ----------------------------
    The paper applies the same low-mass taper to the secondary mass, so

        p(m2 | m1) = m2^{beta_q} S(m2 | m_min, delta_m) / C(m1),
        C(m1) = int_{m_min}^{m1} m2^{beta_q} S(m2 | m_min, delta_m) dm2.

    ``C`` has no closed form for the Planck taper.  It is evaluated as an
    analytic power-law integral above m_min + delta_m plus one dense quadrature
    across the taper itself (`_mass_ratio_norm`), which is exact to roundoff in
    the bulk and costs O(n_taper + N).

    With S(m2) == 1 this reduces to the hard-truncated form
    ``C = m1^{beta_q+1} (1 - q_min^{beta_q+1}) / (beta_q + 1)``.

    Parameters
    ----------
    alpha_1   : low-mass power-law slope (fiducial 1.5)
    alpha_2   : high-mass power-law slope (fiducial 5.4)
    m_break   : break mass between the two power-law segments [M_sun] (fiducial 37.5)
    lambda_0  : mixing fraction of the broken power law (fiducial 0.90)
    lambda_1  : mixing fraction of the first Gaussian peak (fiducial 0.05)
                Second peak carries (1 - lambda_0 - lambda_1).
    mu_1      : first Gaussian peak location [M_sun] (fiducial 10.0)
    sigma_1   : first Gaussian width [M_sun] (fiducial 3.0)
    mu_2      : second Gaussian peak location [M_sun] (fiducial 35.0)
    sigma_2   : second Gaussian width [M_sun] (fiducial 5.0)
    beta_q    : mass-ratio power-law slope (fiducial 1.4)
    m_min     : minimum component mass, lower edge of the taper [M_sun] (fiducial 5.0)
    delta_m   : low-mass taper width [M_sun] (fiducial 4.8)
    m_max     : maximum primary mass [M_sun] (fiducial 100.0)

    Required event keys
    -------------------
    'mass_1_source' and either 'mass_2_source' or 'mass_ratio'.

    Notes
    -----
    lambda_0 + lambda_1 must lie in [0, 1]; outside that every event gets -inf.

    Unlike the previous un-normalised implementation, ``m_max`` is **not**
    degenerate here: it enters the density through N (Eq. B11), whose effect is
    proportional to the broken-power-law share of each event, so the centred
    score no longer vanishes identically.  Whether it is well enough constrained
    to leave free is a separate, catalogue-dependent question.
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

    # -- internals ---------------------------------------------------------

    @staticmethod
    def _secondary_mass(events):
        """``mass_2_source``, falling back to ``mass_ratio * mass_1_source``."""
        if "mass_2_source" in events:
            return numpy.asarray(events["mass_2_source"], dtype=float)
        return numpy.asarray(events["mass_ratio"], dtype=float) * numpy.asarray(
            events["mass_1_source"], dtype=float
        )

    def _primary_numerator(self, mass_1, params, derivatives=True):
        """
        Un-normalised primary-mass density of Eq. B12 (the bracket times S), and
        its derivative with respect to every hyperparameter that appears in it.

        Returns ``(numer, dnumer)`` where ``dnumer`` is a dict keyed by
        hyperparameter name (empty when ``derivatives`` is False).  Callers
        evaluate this once on the events and once on the normalisation grid, so
        both halves of the score share one code path.
        """
        alpha_1 = params["alpha_1"]
        alpha_2 = params["alpha_2"]
        m_break = params["m_break"]
        lambda_0 = params["lambda_0"]
        lambda_1 = params["lambda_1"]
        mu_1 = params["mu_1"]
        sigma_1 = params["sigma_1"]
        mu_2 = params["mu_2"]
        sigma_2 = params["sigma_2"]
        m_min = params["m_min"]
        delta_m = params["delta_m"]
        m_max = params["m_max"]
        lambda_2 = 1.0 - lambda_0 - lambda_1

        mass_1 = numpy.asarray(mass_1, dtype=float)

        bp = _broken_power_law_shape(mass_1, alpha_1, alpha_2, m_break, m_min, m_max)
        norm_bp = _broken_power_law_norm(alpha_1, alpha_2, m_break, m_min, m_max)
        gauss_1 = numpy.exp(-0.5 * ((mass_1 - mu_1) / sigma_1) ** 2)
        gauss_2 = numpy.exp(-0.5 * ((mass_1 - mu_2) / sigma_2) ** 2)
        norm_1 = _left_truncated_normal_norm(mu_1, sigma_1, m_min)
        norm_2 = _left_truncated_normal_norm(mu_2, sigma_2, m_min)
        smooth = _smoothing_window(mass_1, m_min, delta_m)

        mix = (
            lambda_0 * bp / norm_bp
            + lambda_1 * gauss_1 / norm_1
            + lambda_2 * gauss_2 / norm_2
        )
        numer = mix * smooth
        if not derivatives:
            return numer, {}

        dnorm_bp = _dbroken_power_law_norm(alpha_1, alpha_2, m_break, m_min, m_max)
        dnorm_1 = _dleft_truncated_normal_norm(mu_1, sigma_1, m_min)
        dnorm_2 = _dleft_truncated_normal_norm(mu_2, sigma_2, m_min)

        low = (mass_1 >= m_min) & (mass_1 < m_break)
        high = (mass_1 >= m_break) & (mass_1 <= m_max)
        log_ratio = numpy.zeros_like(mass_1)
        inside = low | high
        log_ratio[inside] = numpy.log(mass_1[inside] / m_break)

        # d(shape)/d(parameter) for the broken power law itself
        dbp_dalpha_1 = numpy.where(low, -log_ratio * bp, 0.0)
        dbp_dalpha_2 = numpy.where(high, -log_ratio * bp, 0.0)
        dbp_dm_break = numpy.where(low, alpha_1 * bp / m_break, 0.0) + numpy.where(
            high, alpha_2 * bp / m_break, 0.0
        )

        def bp_column(dshape, dnorm_key):
            """Quotient rule for lambda_0 * bp / N."""
            return (
                lambda_0
                * (dshape / norm_bp - bp * dnorm_bp[dnorm_key] / norm_bp ** 2)
                * smooth
            )

        def peak_column(weight, gauss, dgauss, norm, dnorm_value):
            """Quotient rule for lambda_i * g_i / Z_i."""
            return weight * (dgauss / norm - gauss * dnorm_value / norm ** 2) * smooth

        dnumer = {
            "alpha_1": bp_column(dbp_dalpha_1, "alpha_1"),
            "alpha_2": bp_column(dbp_dalpha_2, "alpha_2"),
            "m_break": bp_column(dbp_dm_break, "m_break"),
            "lambda_0": (bp / norm_bp - gauss_2 / norm_2) * smooth,
            "lambda_1": (gauss_1 / norm_1 - gauss_2 / norm_2) * smooth,
            "mu_1": peak_column(
                lambda_1, gauss_1, gauss_1 * (mass_1 - mu_1) / sigma_1 ** 2,
                norm_1, dnorm_1["mu"],
            ),
            "sigma_1": peak_column(
                lambda_1, gauss_1, gauss_1 * (mass_1 - mu_1) ** 2 / sigma_1 ** 3,
                norm_1, dnorm_1["sigma"],
            ),
            "mu_2": peak_column(
                lambda_2, gauss_2, gauss_2 * (mass_1 - mu_2) / sigma_2 ** 2,
                norm_2, dnorm_2["mu"],
            ),
            "sigma_2": peak_column(
                lambda_2, gauss_2, gauss_2 * (mass_1 - mu_2) ** 2 / sigma_2 ** 3,
                norm_2, dnorm_2["sigma"],
            ),
            "delta_m": mix * _dsmoothing_window(mass_1, m_min, delta_m, "delta_m"),
            # m_max acts only through N (Eq. B11)
            "m_max": -lambda_0 * bp * dnorm_bp["m_max"] / norm_bp ** 2 * smooth,
        }
        # m_min moves every component normalisation *and* the taper
        dnumer["m_min"] = (
            -lambda_0 * bp * dnorm_bp["m_min"] / norm_bp ** 2
            - lambda_1 * gauss_1 * dnorm_1["m_min"] / norm_1 ** 2
            - lambda_2 * gauss_2 * dnorm_2["m_min"] / norm_2 ** 2
        ) * smooth + mix * _dsmoothing_window(mass_1, m_min, delta_m, "m_min")

        return numer, dnumer

    # -- public API --------------------------------------------------------

    def log_prob(self, events, params):
        mass_1 = numpy.asarray(events["mass_1_source"], dtype=float)
        mass_2 = self._secondary_mass(events)
        beta_q = params["beta_q"]
        m_min = params["m_min"]
        delta_m = params["delta_m"]
        m_max = params["m_max"]

        log_prob = numpy.full(len(mass_1), -numpy.inf)

        if params["lambda_0"] < 0.0 or params["lambda_1"] < 0.0:
            return log_prob
        if 1.0 - params["lambda_0"] - params["lambda_1"] < 0.0:
            return log_prob
        if params["m_break"] <= m_min or params["m_break"] >= m_max:
            return log_prob

        valid = (
            (mass_1 >= m_min)
            & (mass_1 <= m_max)
            & (mass_2 >= m_min)
            & (mass_2 <= mass_1)
        )
        if not numpy.any(valid):
            return log_prob

        mass_1_valid = mass_1[valid]
        mass_2_valid = mass_2[valid]

        numer, _ = self._primary_numerator(mass_1_valid, params, derivatives=False)
        grid = _normalisation_grid(m_min, m_max, delta_m, params["m_break"])
        numer_grid, _ = self._primary_numerator(grid, params, derivatives=False)
        norm_m1 = numpy.trapz(numer_grid, grid)
        if not (norm_m1 > 0.0):
            return log_prob

        norm_q, _ = _mass_ratio_norm(
            mass_1_valid, beta_q, m_min, delta_m, derivatives=False
        )
        smooth_2 = _smoothing_window(mass_2_valid, m_min, delta_m)

        usable = (numer > 1e-300) & (smooth_2 > 0.0) & (norm_q > 0.0)
        rows = numpy.flatnonzero(valid)[usable]
        log_prob[rows] = (
            numpy.log(numer[usable])
            - numpy.log(norm_m1)
            + beta_q * numpy.log(mass_2_valid[usable])
            + numpy.log(smooth_2[usable])
            - numpy.log(norm_q[usable])
        )
        return log_prob

    def analytic_score(self, events, params):
        """
        Analytic score d ln p / d lambda, shape (N_events, 13).

        Every column is a closed-form derivative of ``log_prob`` *as coded*.  The
        two normalisations are integrals on grids that do not depend on the
        parameter being differentiated, and for those
        ``d/dlambda trapz(f, grid) == trapz(df/dlambda, grid)`` holds exactly, so
        this is the exact derivative rather than an approximation of it.

        ``m_min`` and ``m_max`` do move the integration endpoints.  There the
        continuum Leibniz rule is used: the upper boundary contributes
        ``+ numer(m_max)`` to dZ/d m_max, while both lower-boundary terms vanish
        because the taper makes the integrands vanish at m_min.  This is *not*
        the derivative of the discretised integral, and it is the correct one --
        the finite-difference alternative converges to it only as the grid is
        refined.

        Rows outside the support get NaN; ``compute_pop_fisher`` drops them.
        """
        mass_1 = numpy.asarray(events["mass_1_source"], dtype=float)
        mass_2 = self._secondary_mass(events)
        beta_q = params["beta_q"]
        m_min = params["m_min"]
        delta_m = params["delta_m"]
        m_max = params["m_max"]

        score = numpy.full((len(mass_1), len(self.parameter_names)), numpy.nan)

        if params["lambda_0"] < 0.0 or params["lambda_1"] < 0.0:
            return score
        if 1.0 - params["lambda_0"] - params["lambda_1"] < 0.0:
            return score
        if params["m_break"] <= m_min or params["m_break"] >= m_max:
            return score

        valid = (
            (mass_1 >= m_min)
            & (mass_1 <= m_max)
            & (mass_2 >= m_min)
            & (mass_2 <= mass_1)
        )
        if not numpy.any(valid):
            return score

        mass_1_valid = mass_1[valid]
        mass_2_valid = mass_2[valid]

        numer, dnumer = self._primary_numerator(mass_1_valid, params)
        grid = _normalisation_grid(m_min, m_max, delta_m, params["m_break"])
        numer_grid, dnumer_grid = self._primary_numerator(grid, params)
        norm_m1 = numpy.trapz(numer_grid, grid)
        if not (norm_m1 > 0.0):
            return score

        norm_q, dnorm_q = _mass_ratio_norm(mass_1_valid, beta_q, m_min, delta_m)
        smooth_2 = _smoothing_window(mass_2_valid, m_min, delta_m)

        usable = (numer > 1e-300) & (smooth_2 > 0.0) & (norm_q > 0.0)
        if not numpy.any(usable):
            return score
        rows = numpy.flatnonzero(valid)[usable]

        numer_u = numer[usable]
        norm_q_u = norm_q[usable]
        smooth_2_u = smooth_2[usable]
        mass_2_u = mass_2_valid[usable]

        shape_names = (
            "alpha_1", "alpha_2", "m_break", "lambda_0", "lambda_1",
            "mu_1", "sigma_1", "mu_2", "sigma_2", "delta_m", "m_min", "m_max",
        )
        columns = {
            name: dnumer[name][usable] / numer_u
            - numpy.trapz(dnumer_grid[name], grid) / norm_m1
            for name in shape_names
        }

        # Leibniz term for the moving upper endpoint of the m1 normalisation.
        columns["m_max"] = columns["m_max"] - numer_grid[-1] / norm_m1

        # Mass-ratio sector: p(m2|m1) = m2^beta_q S(m2) / C(m1).
        columns["beta_q"] = (
            numpy.log(mass_2_u) - dnorm_q["beta_q"][usable] / norm_q_u
        )
        for name in ("delta_m", "m_min"):
            columns[name] = (
                columns[name]
                + _dsmoothing_window(mass_2_u, m_min, delta_m, name) / smooth_2_u
                - dnorm_q[name][usable] / norm_q_u
            )

        for index, name in enumerate(self.parameter_names):
            score[rows, index] = columns[name]
        return score

    def dlogp_dmass_1(self, events, params):
        """
        d ln p / d mass_1_source at **fixed mass ratio**, shape (N_events,).

        Fixed q is the direction in which both component masses scale as
        1/(1+z), which is what the spectral-siren H0 chain needs.  Written out,

            d ln p/d m1 |_q = numer'(m1)/numer(m1) + beta_q/m1
                              + q S'(m2)/S(m2) - m1^{beta_q} S(m1) / C(m1),

        the last term being dC/dm1 from the fundamental theorem of calculus.
        The primary-mass normalisation does not depend on m1 and drops out.
        """
        mass_1 = numpy.asarray(events["mass_1_source"], dtype=float)
        mass_2 = self._secondary_mass(events)
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

        out = numpy.full(len(mass_1), numpy.nan)
        valid = (
            (mass_1 >= m_min)
            & (mass_1 <= m_max)
            & (mass_2 >= m_min)
            & (mass_2 <= mass_1)
        )
        if not numpy.any(valid):
            return out

        m1 = mass_1[valid]
        m2 = mass_2[valid]
        mass_ratio = m2 / m1

        bp = _broken_power_law_shape(m1, alpha_1, alpha_2, m_break, m_min, m_max)
        norm_bp = _broken_power_law_norm(alpha_1, alpha_2, m_break, m_min, m_max)
        gauss_1 = numpy.exp(-0.5 * ((m1 - mu_1) / sigma_1) ** 2)
        gauss_2 = numpy.exp(-0.5 * ((m1 - mu_2) / sigma_2) ** 2)
        norm_1 = _left_truncated_normal_norm(mu_1, sigma_1, m_min)
        norm_2 = _left_truncated_normal_norm(mu_2, sigma_2, m_min)
        smooth_1 = _smoothing_window(m1, m_min, delta_m)
        smooth_2 = _smoothing_window(m2, m_min, delta_m)

        mix = (
            lambda_0 * bp / norm_bp
            + lambda_1 * gauss_1 / norm_1
            + lambda_2 * gauss_2 / norm_2
        )
        low = (m1 >= m_min) & (m1 < m_break)
        dbp_dm1 = numpy.where(low, -alpha_1 * bp / m1, -alpha_2 * bp / m1)
        dmix_dm1 = (
            lambda_0 * dbp_dm1 / norm_bp
            - lambda_1 * gauss_1 * (m1 - mu_1) / sigma_1 ** 2 / norm_1
            - lambda_2 * gauss_2 * (m1 - mu_2) / sigma_2 ** 2 / norm_2
        )
        dsmooth_1_dm1 = _dsmoothing_window(m1, m_min, delta_m, "mass")
        numer = mix * smooth_1

        norm_q, _ = _mass_ratio_norm(m1, beta_q, m_min, delta_m, derivatives=False)

        usable = (numer > 1e-300) & (smooth_2 > 0.0) & (norm_q > 0.0)
        rows = numpy.flatnonzero(valid)[usable]
        out[rows] = (
            (dmix_dm1[usable] * smooth_1[usable] + mix[usable] * dsmooth_1_dm1[usable])
            / numer[usable]
            + beta_q / m1[usable]
            + mass_ratio[usable]
            * _dsmoothing_window(m2, m_min, delta_m, "mass")[usable]
            / smooth_2[usable]
            - m1[usable] ** beta_q * smooth_1[usable] / norm_q[usable]
        )
        return out



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


# ---------------------------------------------------------------------------
# GWTC-5 Eq. B10-B14 building blocks
# ---------------------------------------------------------------------------


def _pl_integral_term(x, alpha):
    """
    A(x, alpha) = (1 - x^{1-alpha}) / (1 - alpha),  i.e.  int_x^1 t^{-alpha} dt.

    Written with ``expm1`` so the alpha -> 1 limit (A -> -ln x) is exact rather
    than 0/0.
    """
    u = 1.0 - alpha
    log_x = numpy.log(x)
    if abs(u) < 1e-8:
        # -(exp(u L) - 1)/u expanded; the u=0 value is -L.
        return -log_x * (1.0 + 0.5 * u * log_x + (u * log_x) ** 2 / 6.0)
    return -numpy.expm1(u * log_x) / u


def _dpl_integral_term_dalpha(x, alpha):
    """
    dA/dalpha for ``_pl_integral_term``.

        dA/dalpha = (1/u) [ x^u ln x - (x^u - 1)/u ],   u = 1 - alpha

    which suffers catastrophic cancellation as u -> 0, so switch to the series
    L^2/2 + u L^3/3 + u^2 L^4/8 there (L = ln x).
    """
    u = 1.0 - alpha
    log_x = numpy.log(x)
    if abs(u * log_x) < 1e-4:
        return log_x ** 2 / 2.0 + u * log_x ** 3 / 3.0 + u ** 2 * log_x ** 4 / 8.0
    x_u = numpy.exp(u * log_x)
    return (x_u * log_x - numpy.expm1(u * log_x) / u) / u


def _broken_power_law_norm(alpha_1, alpha_2, m_break, m_min, m_max):
    """
    Normalisation constant N of the broken power law, Eq. B11 of arXiv:2605.27226:

        N = m_break [ (1 - (m_min/m_break)^{1-alpha_1}) / (1 - alpha_1)
                     + ((m_max/m_break)^{1-alpha_2} - 1) / (1 - alpha_2) ]

    so that ``_broken_power_law_shape / N`` integrates to 1 over [m_min, m_max].
    In terms of ``A(x, a) = int_x^1 t^{-a} dt`` this is
    ``N = m_break [A(x_low, alpha_1) - A(x_high, alpha_2)]``.
    """
    x_low = m_min / m_break
    x_high = m_max / m_break
    return m_break * (
        _pl_integral_term(x_low, alpha_1) - _pl_integral_term(x_high, alpha_2)
    )


def _dbroken_power_law_norm(alpha_1, alpha_2, m_break, m_min, m_max):
    """
    Closed-form derivatives of `_broken_power_law_norm` (Eq. B11).

    With x_low = m_min/m_break, x_high = m_max/m_break and dA/dx = -x^{-alpha}:

        dN/d alpha_1 =  m_break dA(x_low, alpha_1)/d alpha
        dN/d alpha_2 = -m_break dA(x_high, alpha_2)/d alpha
        dN/d m_min   = -x_low^{-alpha_1}
        dN/d m_max   =  x_high^{-alpha_2}
        dN/d m_break =  N/m_break + x_low^{-alpha_1} m_min/m_break
                                  - x_high^{-alpha_2} m_max/m_break

    Returns
    -------
    dict with keys 'alpha_1', 'alpha_2', 'm_break', 'm_min', 'm_max'
    """
    x_low = m_min / m_break
    x_high = m_max / m_break
    low_edge = x_low ** (-alpha_1)
    high_edge = x_high ** (-alpha_2)
    norm = _broken_power_law_norm(alpha_1, alpha_2, m_break, m_min, m_max)
    return {
        "alpha_1": m_break * _dpl_integral_term_dalpha(x_low, alpha_1),
        "alpha_2": -m_break * _dpl_integral_term_dalpha(x_high, alpha_2),
        "m_min": -low_edge,
        "m_max": high_edge,
        "m_break": norm / m_break
        + low_edge * m_min / m_break
        - high_edge * m_max / m_break,
    }


def _left_truncated_normal_norm(mu, sigma, m_min):
    """
    Z = int_{m_min}^{inf} exp(-(m - mu)^2 / (2 sigma^2)) dm
      = sigma sqrt(pi/2) erfc(t),        t = (m_min - mu) / (sigma sqrt(2))

    so that ``exp(-(m-mu)^2/(2 sigma^2)) / Z`` is the left-truncated normal
    ``N_lt(m | mu, sigma, low=m_min)`` of Eq. B12.

    The ``erfcx`` branch keeps full relative accuracy when the truncation point
    sits far above the mean (t >> 0), where ``erfc`` itself underflows.
    """
    from scipy.special import erfc, erfcx

    t = (m_min - mu) / (sigma * numpy.sqrt(2.0))
    prefactor = sigma * numpy.sqrt(numpy.pi / 2.0)
    if t > 0.0:
        return prefactor * numpy.exp(-t * t) * erfcx(t)
    return prefactor * erfc(t)


def _dleft_truncated_normal_norm(mu, sigma, m_min):
    """
    Closed-form derivatives of `_left_truncated_normal_norm`.

    Differentiating under the integral (and by Leibniz for the lower limit),
    with e = exp(-(m_min - mu)^2 / (2 sigma^2)):

        dZ/d mu    =  e
        dZ/d m_min = -e
        dZ/d sigma =  Z/sigma + (m_min - mu) e / sigma

    Returns
    -------
    dict with keys 'mu', 'sigma', 'm_min'
    """
    edge = numpy.exp(-0.5 * ((m_min - mu) / sigma) ** 2)
    norm = _left_truncated_normal_norm(mu, sigma, m_min)
    return {
        "mu": edge,
        "m_min": -edge,
        "sigma": norm / sigma + (m_min - mu) * edge / sigma,
    }


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
    m_break, m_max).  The normalisation is event-independent and cancels in the
    centred Term-I Fisher anyway.  Gauss-Legendre would be faster and more
    accurate here but fails badly for narrow peaks (49% error at sigma_1 =
    0.05), which the fit does explore; the trapezoid degrades gracefully.

    ``m_break`` is optional so the grid can also be built for models without one.
    """
    has_taper = delta_m > 0.0 and m_min + delta_m < m_max
    # Without a taper the density is discontinuous at m_min: `_smoothing_window`
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


def _power_law_segment(lower, upper, beta):
    """
    int_lower^upper m^beta dm, written with ``expm1`` so beta -> -1 is exact.
    ``upper`` may be an array; ``lower`` is scalar.
    """
    exponent = beta + 1.0
    log_ratio = numpy.log(numpy.asarray(upper, dtype=float) / lower)
    if abs(exponent) < 1e-8:
        return log_ratio * (
            1.0 + 0.5 * exponent * log_ratio + (exponent * log_ratio) ** 2 / 6.0
        )
    return lower ** exponent * numpy.expm1(exponent * log_ratio) / exponent


def _dpower_law_segment_dbeta(lower, upper, beta):
    """
    d/dbeta of `_power_law_segment`:

        [u^e ln u - l^e ln l] / e  -  (u^e - l^e) / e^2,   e = beta + 1

    which cancels catastrophically as e -> 0, hence the series branch.
    """
    exponent = beta + 1.0
    log_upper = numpy.log(numpy.asarray(upper, dtype=float))
    log_lower = numpy.log(lower)
    if abs(exponent) * max(abs(numpy.max(log_upper)), abs(log_lower)) < 1e-6:
        return (
            (log_upper ** 2 - log_lower ** 2) / 2.0
            + exponent * (log_upper ** 3 - log_lower ** 3) / 3.0
            + exponent ** 2 * (log_upper ** 4 - log_lower ** 4) / 8.0
        )
    upper_e = numpy.exp(exponent * log_upper)
    lower_e = numpy.exp(exponent * log_lower)
    return (upper_e * log_upper - lower_e * log_lower) / exponent - (
        upper_e - lower_e
    ) / exponent ** 2


def _mass_ratio_norm(mass_1, beta_q, m_min, delta_m, n_taper=4096, derivatives=True):
    """
    C(m1) = int_{m_min}^{m1} m2^{beta_q} S(m2 | m_min, delta_m) dm2, and its
    derivatives with respect to beta_q, delta_m and m_min.

    The Planck taper has no closed-form integral, but it only acts on
    (m_min, m_min + delta_m].  Above that S == 1 and the integral is the exact
    power law `_power_law_segment`, so only the narrow taper interval needs
    quadrature, keeping C accurate to roundoff in the bulk (~1e-10, against
    ~1e-5 if a cumulative integral were interpolated across the whole range).

    For m1 >= m_min + delta_m the moving-endpoint terms cancel exactly:
    d/d delta_m of ``T(b) + int_b^{m1}`` picks up ``+b^{beta_q} S(b)`` from the
    taper piece and ``-b^{beta_q}`` from the power-law piece, and S(b) == 1.
    So the delta_m and m_min derivatives are pure interior integrals.

    Returns
    -------
    norm : ndarray, shape of ``mass_1``
    derivatives : dict with keys 'beta_q', 'delta_m', 'm_min' (empty if not asked)
    """
    from scipy.integrate import cumulative_trapezoid

    mass_1 = numpy.asarray(mass_1, dtype=float)

    if not (delta_m > 0.0):
        # No taper: the whole integral is analytic and S(m_min) = 1, so the
        # lower limit does contribute a Leibniz term.
        norm = _power_law_segment(m_min, mass_1, beta_q)
        if not derivatives:
            return norm, {}
        return norm, {
            "beta_q": _dpower_law_segment_dbeta(m_min, mass_1, beta_q),
            "delta_m": numpy.zeros_like(mass_1),
            "m_min": numpy.full_like(mass_1, -(m_min ** beta_q)),
        }

    taper_edge = m_min + delta_m
    taper_grid = numpy.linspace(m_min, taper_edge, n_taper)
    smooth = _smoothing_window(taper_grid, m_min, delta_m)
    power = taper_grid ** beta_q

    cumulative = cumulative_trapezoid(power * smooth, taper_grid, initial=0.0)
    above = numpy.maximum(mass_1, taper_edge)
    norm = numpy.where(
        mass_1 < taper_edge,
        numpy.interp(mass_1, taper_grid, cumulative),
        cumulative[-1] + _power_law_segment(taper_edge, above, beta_q),
    )
    if not derivatives:
        return norm, {}

    cumulative_beta = cumulative_trapezoid(
        power * numpy.log(taper_grid) * smooth, taper_grid, initial=0.0
    )
    out = {
        "beta_q": numpy.where(
            mass_1 < taper_edge,
            numpy.interp(mass_1, taper_grid, cumulative_beta),
            cumulative_beta[-1]
            + _dpower_law_segment_dbeta(taper_edge, above, beta_q),
        )
    }
    # Pure interior integrals; numpy.interp clamps to the full value above the
    # taper, which is exactly what the cancellation above requires.
    for name in ("delta_m", "m_min"):
        cumulative_name = cumulative_trapezoid(
            power * _dsmoothing_window(taper_grid, m_min, delta_m, name),
            taper_grid,
            initial=0.0,
        )
        out[name] = numpy.interp(mass_1, taper_grid, cumulative_name)
    return norm, out


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


def _dsmoothing_window(mass, m_min, delta_m, wrt):
    """
    Derivative of the Talbot & Thrane window `_smoothing_window`.

    With x = m - m_min,  f = delta_m/x + delta_m/(x - delta_m),
    S = 1 / (1 + exp(f)),  so that  dS/df = -S (1 - S):

        df/d(delta_m) =  1/x + x/(x - delta_m)^2
        df/d(m_min)   =  delta_m/x^2 + delta_m/(x - delta_m)^2
        df/d(mass)    = -df/d(m_min)

    Parameters
    ----------
    wrt : {'delta_m', 'm_min', 'mass'}

    Notes
    -----
    Identically zero outside the ramp.  There `_smoothing_window` returns exactly
    0 or 1 and S approaches those limits with all derivatives vanishing
    exponentially, so this also matches the ``numpy.clip(..., -500, 500)``
    saturation in that function (where S(1-S) ~ e^-500).
    """
    mass = numpy.asarray(mass, dtype=float)
    out = numpy.zeros_like(mass)

    in_ramp = (mass > m_min) & (mass < m_min + delta_m)
    if not numpy.any(in_ramp):
        return out

    x = mass[in_ramp] - m_min
    d = delta_m
    exponent = numpy.clip(d / x + d / (x - d), -500.0, 500.0)
    window = 1.0 / (1.0 + numpy.exp(exponent))

    if wrt == "delta_m":
        dexponent = 1.0 / x + x / (x - d) ** 2
    elif wrt == "m_min":
        dexponent = d / x ** 2 + d / (x - d) ** 2
    elif wrt == "mass":
        dexponent = -(d / x ** 2 + d / (x - d) ** 2)
    else:
        raise ValueError(
            f"unknown wrt={wrt!r}; expected 'delta_m', 'm_min' or 'mass'"
        )

    out[in_ramp] = -window * (1.0 - window) * dexponent
    return out
