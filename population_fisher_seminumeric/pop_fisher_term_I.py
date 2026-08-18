import logging, numpy
from dataclasses import dataclass, field

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# ---------------------------------------------------------------------------
# Score computation
# ---------------------------------------------------------------------------
def _numerical_score(model, events, parameters, h_rel=1e-4, h_abs=1e-8):
    """
    Score d ln p(theta_k | lambda) / d lambda  via centred finite differences.

    The step for parameter i is  h_i = max(h_rel * |lambda_i|, h_abs).

    Parameters
    ----------
    model      : PopModel
    events     : dict of per-event arrays
    parameters : dict – hyperparameters at which to evaluate the score
    h_rel      : float – relative step size (default 1e-4)
    h_abs      : float – absolute minimum step (default 1e-8)

    Returns
    -------
    score : ndarray, shape (N_events, N_params)
        Events outside the model support (or with numerical failures) have
        NaN in the corresponding row; they are excluded before computing
        the Fisher.
    """
    lambda_vec = model.parameters_to_vector(parameters)
    n_parameters = len(lambda_vec)
    n_events = len(next(iter(events.values())))

    score = numpy.zeros((n_events, n_parameters))

    for idx, lambda_i in enumerate(lambda_vec):
        step = max(h_rel * abs(lambda_i), h_abs)

        lam_plus = lambda_vec.copy()
        lam_plus[idx] += step
        lam_minus = lambda_vec.copy()
        lam_minus[idx] -= step

        lp = model.log_prob(events, model.vector_to_parameters(lam_plus))
        lm = model.log_prob(events, model.vector_to_parameters(lam_minus))

        finite = numpy.isfinite(lp) & numpy.isfinite(lm)
        score[finite, idx] = (lp[finite] - lm[finite]) / (2.0 * step)
        score[~finite, idx] = numpy.nan

    return score

# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class PopFisherResult:
    """
    Population Fisher matrix result.

    Attributes
    ----------
    parameter_names : list[str]   – free (varied) hyperparameter names
    fixed_parameters : dict        – parameters held fixed
    fiducial_parameters : dict        – fiducial values at which Fisher was evaluated
    fisher : ndarray     – Gamma_lambda, shape (n_free, n_free)
    covariance : ndarray     – Gamma_lambda^{-1}, shape (n_free, n_free)
    sigma : ndarray     – 1-sigma marginal uncertainties sqrt(diag(cov))
    condition_number: float       – condition number of the Fisher matrix
    n_events : int         – events with finite scores (used in Fisher sum)
    n_support : int         – events inside model support (before finite-diff)
    n_total : int         – total injected events
    score_matrix : ndarray     – raw (uncentred) scores, (n_support, n_free)
    """

    parameter_names: list
    fixed_parameters: dict
    fiducial_parameters: dict
    fisher: numpy.ndarray
    covariance: numpy.ndarray
    sigma: numpy.ndarray
    condition_number: float
    n_events: int
    n_support: int
    n_total: int
    score_matrix: numpy.ndarray = field(repr=False)

    def summary(self):
        """Summary of the Fisher results."""
        lines = [
            "-" * 100,
            "Population Fisher summary (Gair+2022 Gamma_I approximation)",
            "-" * 100,
            f"  Events used    : {self.n_events} / {self.n_total} injected",
            f"  In-support     : {self.n_support} / {self.n_total}",
            f"  Condition #    : {self.condition_number:.3g}",
        ]
        if self.fixed_parameters:
            lines.append(
                "  Fixed params   : "
                + ", ".join(f"{k}={v}" for k, v in self.fixed_parameters.items())
            )
        lines += [
            "",
            f'  {"Parameter":>14s}   {"Fiducial":>10s}   '
            f'{"sigma":>10s}   {"sigma/|fid|":>11s}',
            "  " + "-" * 54,
        ]
        for name in self.parameter_names:
            fiducial_value = self.fiducial_parameters[name]
            idx = self.parameter_names.index(name)
            sigma_value = self.sigma[idx]
            rel = (
                sigma_value / abs(fiducial_value)
                if abs(fiducial_value) > 1e-30
                else float("nan")
            )
            lines.append(
                f"  {name:>14s}   {fiducial_value:>10.4g}   "
                f"{sigma_value:>10.4g}   {rel:>11.3g}"
            )
        lines.append("-" * 100)
        return "\n".join(lines)

    def correlation_matrix(self):
        """Correlation matrix derived from the covariance."""
        diag_sigma = numpy.sqrt(numpy.diag(self.covariance))
        return self.covariance / numpy.outer(diag_sigma, diag_sigma)

# ---------------------------------------------------------------------------
# Main function
# ---------------------------------------------------------------------------

def compute_pop_fisher(
    model,
    events,
    params=None,
    fixed_parameters=None,
    n_total=None,
    h_rel=1e-4,
    h_abs=1e-8,
    rcond=1e-12,
):
    """
    Compute the population Fisher matrix (Gair et al. 2022, arXiv:2205.07893).

    Evaluates the dominant Gamma_I term from Eq. 21:

        Gamma_lambda^{ij} = sum_{k detected} d_k^i * d_k^j

    where  d_k = score_k - mean(score)  is the centred score vector.

    The mean subtraction is the Monte-Carlo estimator of d ln P_det / d lambda
    at the fiducial parameters, which accounts for selection effects.

    Parameters
    ----------
    model : PopModel
        Instantiated population model.
    events : dict
        Per-event source parameter arrays (from ``load_events``).
    params : dict, optional
        Hyperparameters at which to evaluate the Fisher.
        Defaults to ``model.fiducial``.
    fixed_parameters : dict, optional
        Parameters to pin at fixed values (excluded from the Fisher).
        Useful for degenerate parameters (e.g. m_max in TruncatedPowerLawMass;
        see that class's docstring for explanation).
    n_total : int, optional
        Total number of injected events; used in reporting only.
    h_rel : float
        Relative finite-difference step (default 1e-4).
    h_abs : float
        Absolute minimum finite-difference step (default 1e-8).
    rcond : float
        Singular-value threshold for pseudo-inverse fallback (default 1e-12).

    Returns
    -------
    PopFisherResult
        Contains the Fisher matrix, covariance, 1-sigma uncertainties,
        condition number, and diagnostics.

    Notes
    -----
    **Gamma_I approximation** (small measurement error limit)
        Valid when per-event measurement uncertainties are much smaller than
        the width of the population distribution; see Gair+2022 Sec. 2.1 for
        the formal validity criterion.  For next-generation detectors (ET,
        CE) this is an excellent approximation for most sources.

    **Degenerate parameters**
        Hard cutoff parameters (m_min, m_max) have event-independent scores
        within the power-law interior, so their centred scores are identically
        zero.  The Fisher row/column for such a parameter is zero, causing
        singularity.  Pass them via ``fixed_parameters`` to exclude them.

    **Score computation**
        If ``model.analytic_score`` is defined it is used; otherwise centred
        finite differences with relative step ``h_rel`` are used.
    """
    if params is None:
        params = dict(model.fiducial)
    if fixed_parameters is None:
        fixed_parameters = {}

    # Build the effective parameter dict and free parameter list
    effective_params = {**params, **fixed_parameters}
    free_names = [name for name in model.parameter_names if name not in fixed_parameters]

    if not free_names:
        raise ValueError("All parameters are fixed; nothing to compute.")

    # Temporarily wrap the model to expose only the free parameters
    free_model = _FreeParamWrapper(model, free_names, fixed_parameters)

    # Fiducial values restricted to free parameters
    free_fiducial = {name: effective_params[name] for name in free_names}

    logging.info(
        f"Computing population Fisher: model={type(model).__name__}, "
        f"free_params={free_names}, "
        f"fixed_parameters={list(fixed_parameters)}, "
        f"N_events={len(next(iter(events.values())))}"
    )

    # Compute scores
    if hasattr(free_model, "analytic_score"):
        score_raw = free_model.analytic_score(events, free_fiducial)
        logging.info("Using analytic score.")
    else:
        score_raw = _numerical_score(free_model, events, free_fiducial, h_rel, h_abs)

    # Drop events with any non-finite score (outside support or numerical failure)
    finite_mask = numpy.all(numpy.isfinite(score_raw), axis=1)
    n_support = int(finite_mask.sum())
    n_dropped = len(score_raw) - n_support

    if n_dropped > 0:
        logging.warning(
            f"{n_dropped} events excluded (outside model support or "
            f"non-finite score).")
    if n_support == 0:
        raise ValueError(
            "No events have finite scores.  Check that model parameters "
            "are consistent with the event data (e.g. m_min < min(mass_1_source)).")

    score = score_raw[finite_mask]

    # Centre scores: mean approximates d ln P_det / d lambda
    mean_score = score.mean(axis=0)
    score_centred = score - mean_score

    logging.info(
        f"Mean score (= d ln P_det / d lambda): "
        + ", ".join(f"{name}={value:.4g}" for name, value in zip(free_names, mean_score)))

    # Fisher matrix = S^T S  (sum over events of centred outer products)
    fisher = score_centred.T @ score_centred

    # Condition number
    eigenvalues = numpy.linalg.eigvalsh(fisher)
    condition = (
        float(eigenvalues[-1] / eigenvalues[0]) if eigenvalues[0] > 0 else numpy.inf
    )

    if not numpy.isfinite(condition) or condition > 1e14:
        logging.warning(
            f"Fisher matrix is ill-conditioned (kappa = {condition:.3g}).  "
            f"Some parameters may be degenerate in the Gamma_I approximation.  "
            f"Consider passing them via fixed_parameters."
        )

    # Invert Fisher to get covariance
    try:
        if numpy.isfinite(condition) and condition < 1.0 / rcond:
            covariance = numpy.linalg.inv(fisher)
        else:
            logging.warning(
                f"Using pseudo-inverse (rcond={rcond}).  "
                f"Degenerate eigenvalues will give sigma = inf or 0."
            )
            covariance = numpy.linalg.pinv(fisher, rcond=rcond)
    except numpy.linalg.LinAlgError:
        logging.warning("numpy.linalg.inv failed; using pseudo-inverse.")
        covariance = numpy.linalg.pinv(fisher, rcond=rcond)

    sigma = numpy.sqrt(numpy.diag(covariance))

    return PopFisherResult(
        parameter_names=free_names,
        fixed_parameters=dict(fixed_parameters),
        fiducial_parameters={**free_fiducial, **fixed_parameters},
        fisher=fisher,
        covariance=covariance,
        sigma=sigma,
        condition_number=condition,
        n_events=n_support,
        n_support=n_support,
        n_total=n_total if n_total is not None else n_support,
        score_matrix=score_raw,
    )

# ---------------------------------------------------------------------------
# Internal helper: wrapper that exposes only free parameters to the Fisher
# ---------------------------------------------------------------------------

# Completely Claudio doing some shit. Didn't look into it too much.
class _FreeParamWrapper:
    """Wraps a PopModel, fixing certain parameters and forwarding calls."""

    def __init__(self, model, free_names, fixed_parameters):
        self._model = model
        self.parameter_names = free_names
        self.fiducial = {name: model.fiducial[name] for name in free_names}
        self._fixed = fixed_parameters

        # Dynamically attach analytic_score only when the underlying model has it,
        # so that hasattr(wrapper, 'analytic_score') correctly reflects availability.
        if hasattr(model, "analytic_score"):
            all_names = model.parameter_names
            free_idx = [all_names.index(name) for name in free_names]

            def analytic_score(
                events, params, _m=model, _fp=fixed_parameters, _idx=free_idx
            ):
                full_params = {**params, **_fp}
                all_scores = _m.analytic_score(events, full_params)
                return all_scores[:, _idx]

            self.analytic_score = analytic_score

    def log_prob(self, events, params):
        return self._model.log_prob(events, {**params, **self._fixed})

    def parameters_to_vector(self, params):
        return numpy.array([params[name] for name in self.parameter_names], dtype=float)

    def vector_to_parameters(self, vec):
        return {name: float(value) for name, value in zip(self.parameter_names, vec)}
