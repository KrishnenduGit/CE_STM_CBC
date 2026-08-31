"""
Completely written by Claude. Again didn't pay too much attention to the details, but it seems to work.
----------------------------------------------------------------------------
Population (hyperparameter) Fisher matrix, Gair et al. 2022,
`arXiv:2205.07893 <https://arxiv.org/abs/2205.07893>`_ Eq. 21.

This module assembles the full five-term Fisher from the two halves that
actually do the work:

    ``load``                     H5 I/O and the source-frame basis rotation
    ``pop_fisher_term_I``        Gamma_I,   the dominant score term
    ``pop_fisher_higher_order``  Gamma_II .. Gamma_V, the measurement-error
                                 corrections

Gamma_I alone is what matters in the large-N, small-measurement-error limit, so
it is worth reaching for ``pop_fisher_term_I.compute_pop_fisher`` directly when
that approximation is adequate.  Use ``compute_pop_fisher_full`` here when the
per-event uncertainties are not negligible compared with the population width.

Typical use::

    from load import load_population_catalogue
    from pop_models import MadauDickinsonRedshift
    from pop_fisher import compute_pop_fisher_full

    catalogue = load_population_catalogue(path, snr_threshold=30.0)
    result = compute_pop_fisher_full(MadauDickinsonRedshift(), catalogue)
    print(result.summary())
"""

import logging, numpy
from dataclasses import dataclass

from load import (
    POPULATION_KEYS,
    PopulationCatalogue,
    load_population_catalogue,
)
from pop_fisher_term_I import (
    PopFisherResult,
    _FreeParamWrapper,
    compute_pop_fisher,
)
from pop_fisher_higher_order import (
    CorrectionTerms,
    compute_correction_terms,
    compute_det_score,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

__all__ = [
    "PopFisherResult",
    "PopFisherFullResult",
    "PopulationCatalogue",
    "CorrectionTerms",
    "load_population_catalogue",
    "compute_pop_fisher",
    "compute_pop_fisher_full",
    "compute_correction_terms",
    "compute_det_score",
]


# ---------------------------------------------------------------------------
# Result container (full five-term Fisher)
# ---------------------------------------------------------------------------


@dataclass
class PopFisherFullResult:
    """
    Population Fisher matrix result including all Gair+2022 terms.

    Attributes
    ----------
    parameter_names      : list[str]
    fixed_parameters     : dict
    fiducial_parameters  : dict
    fisher           : (n_free, n_free)  total Fisher = I + II + III + IV + V
    fisher_I         : (n_free, n_free)  Term I  (dominant score term)
    fisher_II        : (n_free, n_free)  Term II (log-det correction)
    fisher_III       : (n_free, n_free)  Term III (Tr correction)
    fisher_IV        : (n_free, n_free)  Term IV (cross term; 0 if no det_score)
    fisher_V         : (n_free, n_free)  Term V  (prior-score variance)
    covariance       : (n_free, n_free)
    sigma            : (n_free,)
    condition_number : float
    n_events         : int
    n_det            : int
    n_total          : int
    """

    parameter_names: list
    fixed_parameters: dict
    fiducial_parameters: dict
    fisher: numpy.ndarray
    fisher_I: numpy.ndarray
    fisher_II: numpy.ndarray
    fisher_III: numpy.ndarray
    fisher_IV: numpy.ndarray
    fisher_V: numpy.ndarray
    covariance: numpy.ndarray
    sigma: numpy.ndarray
    condition_number: float
    n_events: int
    n_det: int
    n_total: int

    def summary(self):
        lines = [
            "=" * 70,
            "Population Fisher (Gair+2022 Eq. 21 – all terms)",
            "=" * 70,
            f"  Events used    : {self.n_events} / {self.n_total} injected",
            f"  Detected       : {self.n_det} / {self.n_total} "
            f"(P_det = {self.n_det/self.n_total:.4f})",
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
            f'{"sigma_full":>12s}   {"sigma_I":>10s}',
            "  " + "-" * 56,
        ]
        # Per-parameter 1-sigma from full and Term-I-only inverses
        try:
            cov_I = numpy.linalg.inv(self.fisher_I)
            sig_I = numpy.sqrt(numpy.diag(cov_I))
        except numpy.linalg.LinAlgError:
            sig_I = numpy.full(len(self.parameter_names), numpy.nan)

        for idx, name in enumerate(self.parameter_names):
            fiducial_value = self.fiducial_parameters[name]
            sigma_full = self.sigma[idx]
            sigma_term_I = sig_I[idx]
            lines.append(
                f"  {name:>14s}   {fiducial_value:>10.4g}   "
                f"{sigma_full:>12.4g}   {sigma_term_I:>10.4g}"
            )
        lines += [
            "",
            "  Relative contributions (diag of each term / diag of total Fisher):",
            f'  {"Parameter":>14s}   '
            f'{"I":>11s}   {"II":>11s}   {"III":>11s}   {"IV":>11s}   {"V":>11s}',
            "  " + "-" * 80,
        ]
        diag_total = numpy.diag(self.fisher)
        for idx, name in enumerate(self.parameter_names):
            d = diag_total[idx]
            if abs(d) < 1e-30:
                continue
            lines.append(
                f"  {name:>14s}"
                + "".join(
                    f"   {numpy.diag(f)[idx]/d:>11.3e}"
                    for f in [
                        self.fisher_I,
                        self.fisher_II,
                        self.fisher_III,
                        self.fisher_IV,
                        self.fisher_V,
                    ]
                )
            )
        lines.append("=" * 70)
        return "\n".join(lines)

    def correlation_matrix(self):
        diag_sigma = numpy.sqrt(numpy.diag(self.covariance))
        return self.covariance / numpy.outer(diag_sigma, diag_sigma)


# ---------------------------------------------------------------------------
# Main function: avengers assemble!
# ---------------------------------------------------------------------------

def compute_pop_fisher_full(
    model,
    events,
    fim_phys=None,
    snr_threshold=None,
    params=None,
    fixed_parameters=None,
    n_total=None,
    n_det=None,
    h_lam_terms=3e-2,
    h_theta=1e-3,
    det_score=None,
    p_det_per_event=None,
    snr=None,
    sigma_rho=1.0,
    p_det_min=1e-3,
    rcond=1e-12,
):
    """
    Compute the full population Fisher matrix (Gair+2022 Eq. 21).

      Gamma_I  : centred-score Fisher (dominant term) -- ``pop_fisher_term_I``
      Gamma_II : log-det correction        \\
      Gamma_III: trace correction           |  ``pop_fisher_higher_order``
      Gamma_IV : detection cross term       |
      Gamma_V  : prior-score variance      /

    Parameters
    ----------
    model : PopModel
    events : PopulationCatalogue or dict
        A ``PopulationCatalogue`` (from ``load.load_population_catalogue``) is
        the intended input: it carries the per-event Fisher, the SNR array and
        all three event counts, so ``fim_phys``, ``snr``, ``snr_threshold``,
        ``n_total`` and ``n_det`` are all filled in from it.  A plain dict of
        per-event arrays is still accepted, in which case ``fim_phys`` and
        ``snr_threshold`` are required.
    fim_phys : (N, 3, 3) ndarray, optional
        Per-event Fisher in the ``load.POPULATION_KEYS`` basis.
    snr_threshold : float, optional
        Detection threshold used by the erfc detection model.
    params : dict, optional
        Fiducial hyperparameters (default: ``model.fiducial``).
    fixed_parameters : dict, optional
        Hyperparameters pinned at fixed values and excluded from the Fisher.
    n_total : int, optional
        Total injections (P_det denominator; reporting only).
    n_det : int, optional
        Injections above threshold (P_det numerator; reporting only).
    h_lam_terms : float
        Relative FD step in lambda-space for the Terms II-V scalars
        (default 3e-2); see
        ``pop_fisher_higher_order._hessians_correction_terms_fd``.
    h_theta : float
        Relative FD step in theta-space (default 1e-3).
    det_score : (N, n_phys) ndarray, optional
        Detection gradient d ln p_det / d theta.  Needed only for Term IV; if
        neither this nor ``snr`` is available, Term IV is zero.
    p_det_per_event : (N,) ndarray, optional
        Per-event detection probability, the Term III importance weight.
        Derived from ``snr`` when not supplied.
    snr : (N,) ndarray, optional
        Per-event optimal SNR; supplies both of the above via the erfc model.
    sigma_rho : float
        SNR uncertainty width in the erfc model (default 1.0).
    p_det_min : float
        Events with p_det below this get zero detection score (default 1e-3).
    rcond : float
        Pseudo-inverse threshold for an ill-conditioned Fisher.

    Returns
    -------
    PopFisherFullResult
    """
    # ------------------------------------------------------------------
    # Unpack a catalogue, if that is what we were handed
    # ------------------------------------------------------------------
    if isinstance(events, PopulationCatalogue):
        catalogue = events
        events = catalogue.events
        fim_phys = catalogue.fisher if fim_phys is None else fim_phys
        snr = catalogue.snr if snr is None else snr
        snr_threshold = (
            catalogue.snr_threshold if snr_threshold is None else snr_threshold
        )
        n_total = catalogue.number_total if n_total is None else n_total
        n_det = catalogue.number_above_threshold if n_det is None else n_det

    if fim_phys is None:
        raise ValueError(
            "No per-event Fisher supplied.  Pass a PopulationCatalogue, or "
            "give fim_phys explicitly."
        )
    if snr_threshold is None:
        raise ValueError(
            "snr_threshold is required (it sets the erfc detection model used "
            "by Terms III and IV)."
        )

    if params is None:
        params = dict(model.fiducial)
    if fixed_parameters is None:
        fixed_parameters = {}

    effective_params = {**params, **fixed_parameters}
    free_names = [
        name for name in model.parameter_names if name not in fixed_parameters
    ]
    if not free_names:
        raise ValueError("All parameters are fixed; nothing to compute.")

    free_model = _FreeParamWrapper(model, free_names, fixed_parameters)
    free_fiducial = {name: effective_params[name] for name in free_names}
    params_vec = numpy.array([free_fiducial[name] for name in free_names], dtype=float)

    # ------------------------------------------------------------------
    # Which physical parameters does this model actually depend on?
    # ------------------------------------------------------------------
    phys_keys_all = getattr(model, "phys_event_keys", []) or [
        key for key in POPULATION_KEYS if key in events
    ]
    phys_keys = [key for key in phys_keys_all if key in events]
    if not phys_keys:
        raise ValueError(
            "Cannot determine physical event keys for the correction terms.  "
            f"Set model.phys_event_keys or ensure {POPULATION_KEYS} are in events."
        )

    n_events_catalogue = len(events[phys_keys[0]])
    if n_det is None:
        n_det = n_events_catalogue
    if n_total is None:
        n_total = n_events_catalogue
    logging.info(f"P_det(lambda) = {n_det}/{n_total} = {n_det/n_total:.4f}")

    # ------------------------------------------------------------------
    # Term I
    # ------------------------------------------------------------------
    logging.info("Computing Term I (score Fisher)...")
    result_I = compute_pop_fisher(
        model,
        events,
        params=effective_params,
        fixed_parameters=fixed_parameters,
        n_total=n_total,
        rcond=rcond,
    )
    fisher_I = result_I.fisher

    # Restrict everything to the events Term I could actually use
    finite_mask = numpy.all(numpy.isfinite(result_I.score_matrix), axis=1)
    logging.info(f"Events with finite scores: {int(finite_mask.sum())}")

    def _restrict(array):
        return None if array is None else numpy.asarray(array)[finite_mask]

    # ------------------------------------------------------------------
    # Terms II-V
    # ------------------------------------------------------------------
    corrections = compute_correction_terms(
        free_model,
        {key: value[finite_mask] for key, value in events.items()},
        _restrict(fim_phys),
        free_names,
        params_vec,
        phys_keys,
        snr=_restrict(snr),
        snr_threshold=snr_threshold,
        det_score=_restrict(det_score),
        p_det_per_event=_restrict(p_det_per_event),
        sigma_rho=sigma_rho,
        p_det_min=p_det_min,
        h_lam=h_lam_terms,
        h_theta=h_theta,
    )

    fisher_total = fisher_I + corrections.total

    # ------------------------------------------------------------------
    # Invert for covariance and uncertainties
    # ------------------------------------------------------------------
    eigenvalues = numpy.linalg.eigvalsh(fisher_total)
    condition = (
        float(eigenvalues[-1] / eigenvalues[0]) if eigenvalues[0] > 0 else numpy.inf
    )
    if not numpy.isfinite(condition) or condition > 1e14:
        logging.warning(
            f"Total Fisher matrix ill-conditioned (kappa={condition:.3g}). "
            f"Using pseudo-inverse."
        )
    try:
        if numpy.isfinite(condition) and condition < 1.0 / rcond:
            covariance = numpy.linalg.inv(fisher_total)
        else:
            covariance = numpy.linalg.pinv(fisher_total, rcond=rcond)
    except numpy.linalg.LinAlgError:
        covariance = numpy.linalg.pinv(fisher_total, rcond=rcond)

    sigma = numpy.sqrt(numpy.diag(covariance))

    logging.info(
        "Full Fisher computed. Relative term sizes (diagonal): "
        + ", ".join(
            f"{name}: I={numpy.diag(fisher_I)[i]:.3g} "
            f"II={numpy.diag(corrections.fisher_II)[i]:.3g} "
            f"III={numpy.diag(corrections.fisher_III)[i]:.3g} "
            f"V={numpy.diag(corrections.fisher_V)[i]:.3g}"
            for i, name in enumerate(free_names)
        )
    )

    return PopFisherFullResult(
        parameter_names=free_names,
        fixed_parameters=dict(fixed_parameters),
        fiducial_parameters={**free_fiducial, **fixed_parameters},
        fisher=fisher_total,
        fisher_I=fisher_I,
        fisher_II=corrections.fisher_II,
        fisher_III=corrections.fisher_III,
        fisher_IV=corrections.fisher_IV,
        fisher_V=corrections.fisher_V,
        covariance=covariance,
        sigma=sigma,
        condition_number=condition,
        n_events=result_I.n_events,
        n_det=n_det,
        n_total=n_total,
    )
