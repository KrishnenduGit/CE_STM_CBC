#!/opt/anaconda3/envs/igwn-py311/bin/python
"""
Now include cosmology parameters (H0, Om, w0) as free hyperparameters also, because why not 🤪

Spectral sirens: the detector measures redshifted masses, so a feature at a fixed
source-frame mass (here the sharp ~9.2 Msun peak) appears at a detector-frame mass
that grows with z.  Combined with the measured d_L, that pins the d_L-z relation and
hence (H0, Om, w0), the last being the dark-energy equation of state
(w0 = -1 recovers flat LCDM).

The formal point that makes this work: we promote (H0, Om, w0) to hyperparameters and
build the population density in the *detector* frame, over (m1_det, m2_det, d_L).
P_det depends only on those, so it carries no cosmology, the per-event Fisher is not
a function of the hyperparameters, and the centred-score identity
d ln beta / d Lambda = E_det[d ln p / d Lambda] holds exactly.  Push events to the
source frame at a trial cosmology instead and both of those break.

What this script reports:
  * the Term-I Fisher over the free population parameters + (H0, Om, w0), all
    marginalised jointly;
  * the validity parameter ||H_k F_k^-1|| of the Term-I expansion, which is NOT
    small in the mu_1 peak (~0.3), so Term I is optimistic here;
  * a bracket on sigma(H0) between Term I and a measurement-jittered refit.
"""

import argparse
import logging
import os

import numpy

import cosmology
from cosmology import FIDUCIAL_COSMOLOGY
from load import (
    jacobian_detector_masses,
    load_detector_frame_covariance,
    load_detector_frame_fisher,
    load_injected_masses,
    load_population_catalogue,
)
from pop_fisher import compute_pop_fisher
from bgp import BrokenPowerLawPlusTwoPeaksMass
from fullpop import (
    FULLPOP_GWTC4_FIDUCIAL,
    FULLPOP_PARAMETER_NAMES,
    FULLPOP_EDGE_SETS,
    FULLPOP_SHARPNESS_NAMES,
    FullPopMass,
    fullpop_fixed_names,
    fullpop_support,
)
from pop_models import PopModel
from redshift import MadauDickinsonRedshift
from forecast_mass_redshift import (
    MASS_FIXED_NAMES,
    MD_FIDUCIAL,
    fit_bgp_to_samples,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")

COSMO_NAMES = ["H0", "Om", "w0"]

# Per-event Fisher files in ~/projects/data_dump: network_<pop>_<NETWORK_FILES[network]>.h5
NETWORK_FILES = {
    "CE4020ETD": "CE40km_1p5MW_Aplus_coat_7.0hz_CE20km_1p5MW_Aplus_coat_7.0hz_ETD_5.0hz",
    "CE4020LIA": "CE40km_1p5MW_Aplus_coat_7.0hz_CE20km_1p5MW_Aplus_coat_7.0hz_LIA+_10.0hz",
}
POPULATION_FILES = {"bns": "bns", "bhns": "nsbh", "bbh": "bbh"}  # CLI name -> file tag

MASS_MODEL_LABELS = {
    "bgp": "BGP (broken power law + 2 peaks, fitted to the injections)",
    "fullpop": "FullPop (the generating model, at the injected hyperparameters)",
}

# Prior-like box for the FullPop ML fit (Tab. 4 of arXiv:2605.27226, widened
# where the generator's value sits near an edge).
FULLPOP_FIT_BOUNDS = {
    "alpha_1": (-10.0, 2.0), "alpha_dip": (-5.0, 2.0), "alpha_2": (-5.0, 2.0),
    "NSmax": (1.4, 5.0), "BHmin": (5.0, 9.0), "BHmax": (60.0, 300.0),  # NSmax: paper Tab. 4 prior
    "n5": (1.0, 30.0), "UPPERmin": (30.0, 60.0), "UPPERmax": (45.0, 150.0),
    "mix1": (0.0, 5000.0), "mu1": (17.0, 60.0), "sig1": (2.0, 40.0),
    "mix2": (0.0, 5000.0), "mu2": (6.0, 12.0), "sig2": (0.1, 5.0),
    "A": (0.0, 0.999), "A2": (0.0, 0.999),
    "beta_pair_1": (-0.99, 10.0), "beta_pair_2": (-0.99, 10.0),
    "NSmin": (1.0, 1.4),
    "n0": (2.0, 300.0), "n1": (2.0, 300.0), "n2": (2.0, 300.0),
    "n3": (2.0, 300.0), "n4": (2.0, 300.0),
}

# FullPop options threaded through the functions below:
#   population      "bbh" (BBH file only, m2 >= 2.06) or "all" (BNS + NSBH + BBH
#                   files together, the whole CBC population on m >= 1);
#   free_sharpness  also free the edge sharpness exponents n0..n4 (used for the
#                   whole-CBC jitter bracket, where the measured NS masses are
#                   far broader than the eta = 50 edges).


# ===========================================================================
# Multi-file loading (a population split across BNS / NSBH / BBH files)
# ===========================================================================


def _as_list(file_paths):
    return [file_paths] if isinstance(file_paths, str) else list(file_paths)


def load_catalogues(file_paths, snr_threshold):
    """Detected events of several catalogue files, concatenated (source frame)."""
    import types

    catalogues = [
        load_population_catalogue(f, snr_threshold=snr_threshold, with_fisher=False)
        for f in _as_list(file_paths)
    ]
    keys = set.intersection(*[set(c.events) for c in catalogues])
    return types.SimpleNamespace(
        events={k: numpy.concatenate([c.events[k] for c in catalogues]) for k in keys},
        number_detected=sum(c.number_detected for c in catalogues),
        number_total=sum(c.number_total for c in catalogues),
    )


def load_covariances(file_paths, snr_threshold, measured_basis=False):
    """
    ``load_detector_frame_covariance`` over several files, concatenated.

    ``measured_basis`` returns the covariance in (Mc_det, eta, d_L), the basis
    gwfast measures in, mapped back with the forward Jacobian.  (The loader
    rotates it to masses with the inverse Jacobian, which blows up near equal
    mass where d eta / d m vanishes.)  The round trip reproduces the stored
    block to 6e-3 at worst and 1.4e-4 at the 99.9th percentile (LIA+ BNS).
    """
    parts = [load_detector_frame_covariance(f, snr_threshold) for f in _as_list(file_paths)]
    keys = set.intersection(*[set(p[1]) for p in parts])
    covariance = numpy.concatenate([p[0] for p in parts])
    events = {k: numpy.concatenate([p[1][k] for p in parts]) for k in keys}
    if measured_basis:
        jacobian = jacobian_detector_masses(
            events["mass_1_det"], events["mass_2_det"], events["luminosity_distance"]
        )
        covariance = numpy.einsum("nia,nab,njb->nij", jacobian, covariance, jacobian)
    return (
        covariance,
        events,
        numpy.concatenate([p[2] for p in parts]),
        sum(p[3] for p in parts),
        sum(p[4] for p in parts),
        sum(p[5] for p in parts),
    )


def load_masses(file_paths):
    parts = [load_injected_masses(f) for f in _as_list(file_paths)]
    return numpy.concatenate([p[0] for p in parts]), numpy.concatenate([p[1] for p in parts])


# ===========================================================================
# Mass-model switch (BGP fitted to the injections, or FullPop = the truth)
# ===========================================================================


def make_mass_model(kind, fiducial, population="bbh"):
    if kind == "bgp":
        return BrokenPowerLawPlusTwoPeaksMass(fiducial=fiducial)
    if kind == "fullpop":
        return FullPopMass(fiducial=fiducial, population=population)
    raise ValueError(f"unknown mass model {kind!r}")


def mass_fixed_names(kind, population="bbh", free_sharpness=False, extra=()):
    """Pinned mass hyperparameters; ``extra`` adds more (e.g. FULLPOP_EDGE_SETS)."""
    if kind == "bgp":
        return MASS_FIXED_NAMES
    fixed = fullpop_fixed_names(population)
    if free_sharpness:
        fixed = [n for n in fixed if n not in FULLPOP_SHARPNESS_NAMES]
    return list(dict.fromkeys([*fixed, *extra]))


def feature_windows(kind, fiducial, population="bbh"):
    """
    Source-frame m1 windows around the sharp features that carry the H0
    information, for the Term-I validity diagnostic.  The first entry is the
    feature quoted in the summary.  Logistic edges m_e with sharpness eta have
    width ~ m_e / eta, so their windows are m_e +/- 2 m_e / eta.
    """
    f = fiducial
    if kind == "bgp":
        spec = [("mu_1 peak", f["mu_1"], 2.0 * f["sigma_1"]),
                ("mu_2 peak", f["mu_2"], 2.0 * f["sigma_2"])]
    elif population == "all":
        spec = [("NSmin edge", f["NSmin"], 2.0 * f["NSmin"] / f["n0"]),
                ("NSmax edge", f["NSmax"], 2.0 * f["NSmax"] / f["n1"]),
                ("mu2 peak", f["mu2"], 2.0 * f["sig2"]),
                ("BHmin edge", f["BHmin"], 2.0 * f["BHmin"] / f["n2"]),
                ("UMGmin edge", f["UPPERmin"], 2.0 * f["UPPERmin"] / f["n3"])]
    else:
        spec = [("mu2 peak", f["mu2"], 2.0 * f["sig2"]),
                ("BHmin edge", f["BHmin"], 2.0 * f["BHmin"] / f["n2"]),
                ("UMGmin edge", f["UPPERmin"], 2.0 * f["UPPERmin"] / f["n3"]),
                ("UMGmax edge", f["UPPERmax"], 2.0 * f["UPPERmax"] / f["n4"])]
    return [(label, float(c), float(c - w), float(c + w)) for label, c, w in spec]


def fullpop_free_names(population="bbh", free_sharpness=False):
    fixed = mass_fixed_names("fullpop", population, free_sharpness)
    return [n for n in FULLPOP_PARAMETER_NAMES if n not in fixed]


def fit_fullpop_to_samples(mass_1, mass_2, start=None, max_iterations=100,
                           tolerance=0.05, population="bbh", free_sharpness=False):
    """
    Maximum-likelihood FullPop hyperparameters (free set) for source-frame masses,
    by Fisher scoring: Newton steps with the empirical information
    F = sum_k (s_k - s_bar)(s_k - s_bar)^T as the Hessian, a backtracking line
    search, and steps clipped to ``FULLPOP_FIT_BOUNDS``.

    Goodness-of-fit stop: the Newton decrement  g^T F^-1 g / 2  is the
    log-likelihood still to be gained, *including* parameter correlations.  The
    fit stops once it drops below ``tolerance`` (default 0.05, negligible next to
    the ~P/2 statistical scatter of ln L between a fit and the truth), or after
    ``max_iterations``.  (L-BFGS-B with a diagonal sigma scaling needed >300
    iterations here because the 24 parameters are strongly correlated.)

    Returns ``(fiducial, sigma_at_start, result)``; events outside the support
    are dropped.  ``result`` has ``nit, nfev, remaining_gain, good_enough,
    log_l_gain, success, message``.
    """
    import types

    start = dict(FULLPOP_GWTC4_FIDUCIAL if start is None else start)
    model = FullPopMass(fiducial=start, population=population)
    keep = model._in_support(mass_1, mass_2)
    events = {"mass_1_source": mass_1[keep], "mass_2_source": mass_2[keep]}
    names = fullpop_free_names(population, free_sharpness)
    columns = [model.parameter_names.index(n) for n in names]
    lower = numpy.array([FULLPOP_FIT_BOUNDS[n][0] for n in names])
    upper = numpy.array([FULLPOP_FIT_BOUNDS[n][1] for n in names])

    def unpack(theta):
        return {**start, **dict(zip(names, theta))}

    def log_likelihood(theta):
        log_prob = model.log_prob(events, unpack(theta))
        return float(log_prob.sum()) if numpy.all(numpy.isfinite(log_prob)) else -numpy.inf

    def gradient_and_information(theta):
        score = model.analytic_score(events, unpack(theta))[:, columns]
        centred = score - score.mean(axis=0)
        return score.sum(axis=0), numpy.einsum("ni,nj->ij", centred, centred)

    theta = numpy.array([start[n] for n in names], dtype=float)
    log_l_start = log_l = log_likelihood(theta)
    n_evaluations = 1
    sigma_at_start = None
    remaining = numpy.inf
    iteration = 0
    stalled = False
    damping = 0.0
    for iteration in range(1, max_iterations + 1):
        gradient, information = gradient_and_information(theta)
        n_evaluations += 1
        if sigma_at_start is None:
            sigma_at_start = numpy.sqrt(numpy.diag(numpy.linalg.pinv(information)))
        # Parameters pinned at a bound with the gradient pushing outward drop out.
        active = ~(((theta <= lower) & (gradient < 0)) | ((theta >= upper) & (gradient > 0)))
        sub = numpy.ix_(active, active)
        info_active, grad_active = information[sub], gradient[active]
        # Newton decrement via a pseudo-inverse: parameters that have become
        # irrelevant (e.g. the gap edges once A2 -> 0) make F singular.
        remaining = 0.5 * float(
            grad_active @ numpy.linalg.lstsq(info_active, grad_active, rcond=1e-12)[0]
        )
        if remaining < tolerance:
            break
        # Levenberg-Marquardt: (F + lambda diag F) step = g.  lambda -> 0 is
        # Fisher scoring; large lambda is scaled gradient ascent, which always
        # ascends for a small enough step unless we sit on a kink.
        scale = numpy.diag(info_active).copy()
        scale[scale <= 0] = 1.0
        improved = False
        for _ in range(20):
            step = numpy.zeros_like(theta)
            step[active] = numpy.linalg.solve(
                info_active + (damping + 1e-10) * numpy.diag(scale), grad_active
            )
            trial = numpy.clip(theta + step, lower, upper)
            trial_log_l = log_likelihood(trial)
            n_evaluations += 1
            if trial_log_l > log_l:
                improved = True
                damping = 0.0 if damping < 1e-4 else damping / 10.0
                break
            damping = max(1e-3, 10.0 * damping)
        if not improved:
            stalled = True
            break
        theta, log_l = trial, trial_log_l

    # A stall is accepted only when the quadratic estimate of what is left is
    # modest: the power-law kinks (NSmax, BHmin) and eta = 50 edges make the
    # decrement over-state the true remaining gain by a few units (checked
    # against 300 L-BFGS iterations, same ln L to 0.2), not by thousands.
    good_enough = remaining < 0.5 or (stalled and remaining < 10.0)
    at_bound = [n for n, t, lo, hi in zip(names, theta, lower, upper)
                if t <= lo or t >= hi]
    if at_bound:
        logging.warning(f"  FullPop refit ended on a fit bound for {at_bound}: "
                        "the Fisher there is not at a likelihood maximum")
    result = types.SimpleNamespace(
        nit=iteration, nfev=n_evaluations, remaining_gain=remaining,
        good_enough=good_enough, success=good_enough,
        log_l_gain=log_l - log_l_start,
        message=("no further ascent" if stalled else
                 "converged" if good_enough else f"stopped after {iteration} iterations"),
    )
    logging.info(
        f"  FullPop refit: {iteration} Fisher-scoring iterations, {n_evaluations} "
        f"evaluations, quadratic estimate of ln L left {remaining:.3f}, "
        f"{result.message} ({'good enough' if good_enough else 'NOT converged'}), "
        f"ln L gain over start {result.log_l_gain:.1f}"
    )
    return unpack(theta), dict(zip(names, sigma_at_start)), result


def fit_mass_model(kind, mass_1, mass_2, population="bbh", free_sharpness=False):
    """Fiducial for ``kind`` refitted to (possibly jittered) source-frame masses."""
    if kind == "bgp":
        m_min_upper = 0.999 * min(mass_1.min(), mass_2.min())
        m_max = 1.001 * mass_1.max()
        fiducial, _ = fit_bgp_to_samples(mass_1, mass_2, m_min_upper, m_max)
        return fiducial
    fiducial, _, result = fit_fullpop_to_samples(
        mass_1, mass_2, population=population, free_sharpness=free_sharpness
    )
    if not result.good_enough:
        logging.warning(f"FullPop refit stopped early: {result.message}")
    return fiducial


# ===========================================================================
# Observable-frame population model with cosmology
# ===========================================================================


class SpectralSirenModel(PopModel):
    """
    Source-frame population model observed in the detector frame, with the
    cosmology ``(H0, Omega_m)`` as extra hyperparameters (see module docstring).

    The per-event ``events`` dict is in **detector-frame observables**:
    ``mass_1_det``, ``mass_2_det``, ``luminosity_distance`` [Gpc], plus the
    frame-invariant ``mass_ratio``.

    Parameters
    ----------
    mass_model : PopModel
        Source-frame primary-mass + mass-ratio model (BGP here).
    redshift_model : MadauDickinsonRedshift
        Supplies the MD *shape* parameters; its Planck18 dV_c/dz term is NOT
        used (the measure is rebuilt at the trial cosmology).
    cosmo_fiducial : dict
        ``{"H0": ..., "Om": ...}``.
    """

    def __init__(self, mass_model, redshift_model, cosmo_fiducial):
        self.mass_model = mass_model
        self.redshift_model = redshift_model
        self.parameter_names = (
            list(mass_model.parameter_names)
            + list(redshift_model.parameter_names)
            + list(COSMO_NAMES)
        )
        self.fiducial = {
            **mass_model.fiducial,
            **redshift_model.fiducial,
            **{name: cosmo_fiducial[name] for name in COSMO_NAMES},
        }
        self.phys_event_keys = [
            "mass_1_det", "mass_2_det", "luminosity_distance"
        ]

    def log_prob(self, events, params):
        H0 = params["H0"]
        Om = params["Om"]
        w0 = params["w0"]

        distance = events["luminosity_distance"]
        redshift = cosmology.redshift_of_distance(distance, H0, Om, w0)
        one_plus_z = 1.0 + redshift
        mass_1_det = events["mass_1_det"]
        mass_2_det = events["mass_2_det"]
        mass_1_source = mass_1_det / one_plus_z
        mass_2_source = mass_2_det / one_plus_z

        mass_events = {
            "mass_1_source": mass_1_source,
            "mass_2_source": mass_2_source,
            "mass_ratio": mass_2_det / mass_1_det,
        }
        mass_params = {n: params[n] for n in self.mass_model.parameter_names}
        mass_log_prob = self.mass_model.log_prob(mass_events, mass_params)

        # Madau-Dickinson rate *shape* psi(z) (no dV_c/dz here -- see below).
        alpha = params["alpha"]
        beta = params["beta"]
        z_peak = params["z_peak"]
        rate_shape = alpha * numpy.log1p(redshift) - numpy.log1p(
            ((1.0 + redshift) / (1.0 + z_peak)) ** (alpha + beta)
        )

        differential_volume = cosmology.differential_comoving_volume(
            redshift, H0, Om, w0
        )
        ddL = cosmology.ddL_dz(redshift, H0, Om, w0)
        measure = (
            numpy.log(differential_volume)
            - 3.0 * numpy.log(one_plus_z)
            - numpy.log(ddL)
        )

        return mass_log_prob + rate_shape + measure

    def analytic_score(self, events, params):
        """
        Score ``d ln p / d Lambda``, shape ``(N_events, 19)``, column-aligned with
        ``parameter_names``.  Mass and (alpha, beta, z_peak) columns delegate to
        the sub-models' analytic scores; ``H0`` is closed form via
        dz/dH0 = d_L / (H0 dd_L/dz) plus d(measure)/dH0|_z = -2/H0, chained
        through m1_source = m1_det/(1+z), the MD shape, and the measure.

        ``Om`` and ``w0`` are one central finite difference each, by design:
        d D_C/d Om has no closed-form integral, so an "analytic" version would
        just be another quadrature (checked against an independent quadrature in
        ``validate_analytic_scores.py``).  The 1e-4 relative step is deliberate —
        the error floor is the per-cosmology CubicSpline rebuild, not truncation,
        so a smaller step is *worse*.
        """
        H0 = params["H0"]
        Om = params["Om"]
        w0 = params["w0"]

        distance = events["luminosity_distance"]
        redshift = cosmology.redshift_of_distance(distance, H0, Om, w0)
        one_plus_z = 1.0 + redshift
        mass_1_source = events["mass_1_det"] / one_plus_z

        mass_events = {
            "mass_1_source": mass_1_source,
            "mass_2_source": events["mass_2_det"] / one_plus_z,
            "mass_ratio": events["mass_2_det"] / events["mass_1_det"],
        }
        mass_params = {n: params[n] for n in self.mass_model.parameter_names}
        mass_score = self.mass_model.analytic_score(mass_events, mass_params)

        redshift_params = {n: params[n] for n in self.redshift_model.parameter_names}
        redshift_score = self.redshift_model.analytic_score(
            {"redshift": redshift}, redshift_params
        )

        # ---- H0: chain through z(d_L) ------------------------------------
        alpha = params["alpha"]
        beta = params["beta"]
        z_peak = params["z_peak"]
        ratio = one_plus_z / (1.0 + z_peak)
        rate = ratio ** (alpha + beta)
        drate_shape_dz = (
            alpha / one_plus_z
            - (alpha + beta) * rate / ((1.0 + rate) * one_plus_z)
        )

        comoving, hubble_distance, efunc, defunc_dz = cosmology.background_terms(
            redshift, H0, Om, w0
        )
        ddL = cosmology.ddL_dz(redshift, H0, Om, w0)
        # d/dz of  dd_L/dz = D_C + (1+z) D_H / E
        d2dL_dz2 = (
            2.0 * hubble_distance / efunc
            - one_plus_z * hubble_distance * defunc_dz / efunc ** 2
        )
        # measure = ln(dV_c/dz) - 3 ln(1+z) - ln(dd_L/dz),  dV_c/dz ~ D_C^2 / E
        dmeasure_dz = (
            2.0 * hubble_distance / (efunc * comoving)
            - defunc_dz / efunc
            - 3.0 / one_plus_z
            - d2dL_dz2 / ddL
        )
        dmeasure_dH0 = -2.0 / H0
        dz_dH0 = distance / (H0 * ddL)

        dlogp_dmass_1 = self.mass_model.dlogp_dmass_1(mass_events, mass_params)
        score_H0 = (
            dlogp_dmass_1 * (-mass_1_source / one_plus_z)
            + drate_shape_dz
            + dmeasure_dz
        ) * dz_dH0 + dmeasure_dH0

        # ---- Om, w0: single-parameter central finite differences -----------
        log_prob_centre = self.log_prob(events, params)

        def finite_difference(name, value):
            step = max(1e-4 * abs(value), 1e-8)
            log_prob_plus = self.log_prob(events, {**params, name: value + step})
            log_prob_minus = self.log_prob(events, {**params, name: value - step})
            finite = (
                numpy.isfinite(log_prob_plus) & numpy.isfinite(log_prob_minus)
                & numpy.isfinite(log_prob_centre)
            )
            forward = (log_prob_plus[finite] - log_prob_centre[finite]) / step
            backward = (log_prob_centre[finite] - log_prob_minus[finite]) / step
            central = 0.5 * (forward + backward)
            # A stencil that straddles a step in the density (FullPop's pairing
            # switches branch at m2 = mbreak) returns jump/step, i.e. a spurious,
            # arbitrarily large score.  For a smooth density the one-sided
            # differences agree to O(step); where they disagree wildly keep the
            # smaller one, which is the derivative on the side without the step.
            # The analytic H0 score ignores the step in the same way.
            straddles = numpy.abs(forward - backward) > 1.0 + 0.5 * numpy.minimum(
                numpy.abs(forward), numpy.abs(backward)
            )
            one_sided = numpy.where(
                numpy.abs(forward) < numpy.abs(backward), forward, backward
            )
            column = numpy.full(len(redshift), numpy.nan)
            column[finite] = numpy.where(straddles, one_sided, central)
            return column

        score_Om = finite_difference("Om", Om)
        score_w0 = finite_difference("w0", w0)

        return numpy.column_stack(
            [mass_score, redshift_score, score_H0, score_Om, score_w0]
        )


# ===========================================================================
# Driver
# ===========================================================================


def detector_frame_events(catalogue, cosmo):
    """Detector-frame observables at the fiducial cosmology from a catalogue."""
    mass_1_source = catalogue.events["mass_1_source"]
    mass_2_source = catalogue.events["mass_2_source"]
    redshift = catalogue.events["redshift"]
    one_plus_z = 1.0 + redshift
    return {
        "mass_1_det": mass_1_source * one_plus_z,
        "mass_2_det": mass_2_source * one_plus_z,
        "luminosity_distance": cosmology.luminosity_distance(
            redshift, cosmo["H0"], cosmo["Om"], cosmo["w0"]
        ),
        "mass_ratio": mass_2_source / mass_1_source,
    }


def validate_frame(catalogue, events, cosmo):
    """Round-trip checks: the fiducial detector frame must recover the source frame."""
    reconstructed = cosmology.detector_to_source_frame(
        events["mass_1_det"], events["mass_2_det"],
        events["luminosity_distance"], cosmo["H0"], cosmo["Om"], cosmo["w0"],
    )
    mass_error = numpy.abs(reconstructed[0] - catalogue.events["mass_1_source"]).max()
    z_error = numpy.abs(reconstructed[2] - catalogue.events["redshift"]).max()
    logging.info(
        f"  Frame round-trip: max|dm1_src|={mass_error:.2e}, max|dz|={z_error:.2e}"
    )
    return mass_error < 1e-5 and z_error < 1e-5


def compute_forecast(file_path, snr_thresholds, mass_fiducial, kind="bgp", population="bbh",
                     extra_fixed=()):
    """
    Term-I spectral-siren Fisher at each SNR threshold, with (H0, Om, w0) free
    alongside the population.  ``extra_fixed`` pins more mass hyperparameters at
    the fiducial (no derivative is taken).

    A parameter whose score is the same on every event (e.g. alpha_2 for a
    BNS-only sample: no event above BHmin) has an all-zero centred-score Fisher
    row.  It is pinned; dropping a zero row leaves every other sigma unchanged,
    and avoids the pseudo-inverse.  Those names are stored on
    the result as ``unconstrained``.

    Returns ``{snr: PopFisherResult}``.
    """
    mass_fixed = {
        name: mass_fiducial[name]
        for name in mass_fixed_names(kind, population, extra=extra_fixed)
    }
    mass_model = make_mass_model(kind, mass_fiducial, population)
    redshift_model = MadauDickinsonRedshift(fiducial=MD_FIDUCIAL)
    model = SpectralSirenModel(mass_model, redshift_model, FIDUCIAL_COSMOLOGY)

    results = {}
    for snr in snr_thresholds:
        catalogue = load_catalogues(file_path, snr)
        events = detector_frame_events(catalogue, FIDUCIAL_COSMOLOGY)
        logging.info(f"SNR>={snr}: {catalogue.number_detected} events")
        validate_frame(catalogue, events, FIDUCIAL_COSMOLOGY)
        score = model.analytic_score(events, model.fiducial)
        score = score[numpy.all(numpy.isfinite(score), axis=1)]
        spread = numpy.ptp(score, axis=0)
        scale = numpy.max(numpy.abs(score), axis=0)
        unconstrained = [
            name for name, s, a in zip(model.parameter_names, spread, scale)
            if name in mass_fiducial and name not in mass_fixed and s <= 1e-10 * a
        ]
        if unconstrained:
            logging.info(f"  same score on every event, pinned: {', '.join(unconstrained)}")
        result = compute_pop_fisher(
            model, events, n_total=catalogue.number_total,
            fixed_parameters={**mass_fixed, **{n: mass_fiducial[n] for n in unconstrained}},
        )
        result.unconstrained = unconstrained
        results[snr] = result
        index = result.parameter_names.index("H0")
        logging.info(
            f"  SNR>={snr}: condition # = {result.condition_number:.3g}, "
            f"sigma(H0) = {result.sigma[index]:.4g} km/s/Mpc "
            f"({100 * result.sigma[index] / FIDUCIAL_COSMOLOGY['H0']:.3f}%)"
        )
    return results


def measurement_degraded_forecast(
    file_path, snr_thresholds, n_realisations=3, seed=0, kind="bgp",
    clean_fiducial=None, population="bbh", free_sharpness=False,
):
    """
    Pessimistic edge of the measurement-error bracket:
    sigma_TermI <= sigma_true <= sigma_jittered.

    Each event's (Mc_det, eta, d_L) is jittered by a draw from its own marginal
    measurement covariance (eta reflected at 0.25, eta <= 0 dropped) and mapped
    to (m1_det, m2_det, d_L); the mass model is re-fitted to the jittered masses
    so model and data stay consistent, and Term I is recomputed.  Treating each
    noisy point estimate as truth discards the per-event uncertainties a real
    hierarchical analysis would deconvolve with, hence a genuine upper edge.

    Returns ``{snr: [sigma(H0) per realisation]}`` plus the jittered fiducials.
    """
    rng = numpy.random.default_rng(seed)
    out = {}
    fiducials = {}

    for snr in snr_thresholds:
        covariance, events, _, n_det, _, n_total = load_covariances(
            file_path, snr, measured_basis=True
        )
        total = events["mass_1_det"] + events["mass_2_det"]
        eta = events["mass_1_det"] * events["mass_2_det"] / total ** 2
        stacked = numpy.stack(
            [total * eta ** 0.6, eta, events["luminosity_distance"]], axis=1
        )
        # Cholesky per event; a handful of events have marginally non-PD 3x3 blocks
        # from the stored inversion, so fall back to the diagonal for those.
        try:
            chol = numpy.linalg.cholesky(covariance)
        except numpy.linalg.LinAlgError:
            chol = None
        if chol is None:
            chol = numpy.zeros_like(covariance)
            for index in range(len(covariance)):
                try:
                    chol[index] = numpy.linalg.cholesky(covariance[index])
                except numpy.linalg.LinAlgError:
                    chol[index] = numpy.diag(
                        numpy.sqrt(numpy.abs(numpy.diag(covariance[index])))
                    )

        # Draw every realisation's noise here, in the original RNG order, so the
        # jitter is identical to the serial version; refit + Fisher run in parallel.
        tasks = []
        for realisation in range(n_realisations):
            noise = numpy.einsum(
                "nij,nj->ni", chol, rng.standard_normal(stacked.shape)
            )
            tasks.append(dict(
                jittered=_measured_to_masses(stacked + noise), kind=kind, snr=snr,
                n_total=n_total, realisation=realisation, population=population,
                free_sharpness=free_sharpness, clean_fiducial=clean_fiducial,
            ))
        out[snr] = []
        workers = max(1, min(n_realisations, (os.cpu_count() or 2) // 2))
        if workers > 1:
            from concurrent.futures import ProcessPoolExecutor

            with ProcessPoolExecutor(max_workers=workers) as pool:
                results = list(pool.map(_jitter_realisation, tasks))
        else:
            results = [_jitter_realisation(task) for task in tasks]
        for task, (sigma_h0, jittered_fiducial) in zip(tasks, results):
            out[snr].append(sigma_h0)
            fiducials[(snr, task["realisation"])] = jittered_fiducial

    return out, fiducials


def _measured_to_masses(measured):
    """
    (Mc_det, eta, d_L) rows -> (m1_det, m2_det, d_L).  eta above the physical
    maximum 0.25 is reflected back below it; eta <= 0 gives NaN masses, which
    ``_jitter_realisation`` drops with the other unphysical draws.
    """
    chirp_mass, eta, distance = measured.T
    eta = numpy.where(eta > 0.25, 0.5 - eta, eta)
    eta = numpy.where(eta > 0.0, eta, numpy.nan)
    total = chirp_mass / eta ** 0.6
    root = numpy.sqrt(1.0 - 4.0 * eta)
    return numpy.stack([0.5 * total * (1.0 + root), 0.5 * total * (1.0 - root), distance], axis=1)


def _jitter_realisation(task):
    """
    One measurement-error realisation: source frame at the fiducial cosmology,
    support cut, mass-model refit, Term I.  Module-level so a process pool can
    run realisations in parallel.
    """
    population, free_sharpness = task["population"], task["free_sharpness"]
    kind, snr, realisation = task["kind"], task["snr"], task["realisation"]
    clean_fiducial = task["clean_fiducial"]
    jittered = task["jittered"]

    mass_1_det = numpy.maximum(jittered[:, 0], jittered[:, 1])
    mass_2_det = numpy.minimum(jittered[:, 0], jittered[:, 1])
    distance = jittered[:, 2]
    keep = (mass_2_det > 0.0) & (distance > 0.0)  # NaN (eta <= 0) fails too
    mass_1_det, mass_2_det, distance = mass_1_det[keep], mass_2_det[keep], distance[keep]
    redshift = cosmology.redshift_of_distance(
        distance, FIDUCIAL_COSMOLOGY["H0"], FIDUCIAL_COSMOLOGY["Om"],
        FIDUCIAL_COSMOLOGY["w0"],
    )
    one_plus_z = 1.0 + redshift
    mass_1_source = mass_1_det / one_plus_z
    mass_2_source = mass_2_det / one_plus_z

    if kind == "fullpop":
        # The generator's hard support edges are model constants: drop jittered
        # events that leave the support.
        support = fullpop_support(population)
        inside = (mass_2_source >= support["m2_min"]) & (mass_1_source <= support["mmax"])
        mass_1_det, mass_2_det, distance = (
            mass_1_det[inside], mass_2_det[inside], distance[inside]
        )
        mass_1_source, mass_2_source = mass_1_source[inside], mass_2_source[inside]
    jittered_fiducial = fit_mass_model(
        kind, mass_1_source, mass_2_source, population, free_sharpness
    )

    model = SpectralSirenModel(
        make_mass_model(kind, jittered_fiducial, population),
        MadauDickinsonRedshift(fiducial=MD_FIDUCIAL),
        FIDUCIAL_COSMOLOGY,
    )
    jittered_events = {
        "mass_1_det": mass_1_det,
        "mass_2_det": mass_2_det,
        "luminosity_distance": distance,
        "mass_ratio": mass_2_det / mass_1_det,
    }
    mass_fixed = {
        n: jittered_fiducial[n] for n in mass_fixed_names(kind, population, free_sharpness)
    }
    result = compute_pop_fisher(
        model, jittered_events, fixed_parameters=mass_fixed, n_total=task["n_total"]
    )
    sigma_h0 = float(result.sigma[result.parameter_names.index("H0")])

    width_name = "sigma_1" if kind == "bgp" else "sig2"
    logging.info(
        f"  [jitter] SNR>={snr} realisation {realisation}: "
        f"{width_name} {jittered_fiducial[width_name]:.4f} "
        f"(clean {clean_fiducial[width_name] if clean_fiducial else float('nan'):.3f}), "
        f"sigma(H0) {sigma_h0:.4g}"
    )
    return sigma_h0, jittered_fiducial


def detector_det_score(events, snr, snr_threshold, sigma_rho=1.0, p_det_min=1e-3):
    """
    Used only by ``compute_full_forecast``, which ``main`` does not call -- see the
    note there.

    Detection gradient ``d ln p_det / d theta`` in the detector-frame basis
    ``(mass_1_det, mass_2_det, luminosity_distance)`` for the erfc detection
    model and the inspiral SNR scaling ``rho ~ Mc_det^{5/6} / d_L``.

    Detector-frame analogue of ``pop_fisher_higher_order.compute_det_score``;
    written here because that function hard-codes the source-frame keys and the
    ``(1+z)`` scaling.  Uses ``erfcx`` for numerical stability (see that
    function's docstring).

    Returns
    -------
    (N, 3) ndarray, aligned with (mass_1_det, mass_2_det, luminosity_distance).
    """
    from scipy.special import erfc, erfcx

    mass_1 = events["mass_1_det"]
    mass_2 = events["mass_2_det"]
    distance = events["luminosity_distance"]
    rho = numpy.asarray(snr, dtype=float)

    x = (snr_threshold - rho) / (numpy.sqrt(2.0) * sigma_rho)
    # d ln p_det / d rho = 2 / (sqrt(2 pi) sigma_rho erfcx(x))
    dlnp_drho = 2.0 / (numpy.sqrt(2.0 * numpy.pi) * sigma_rho * erfcx(x))

    total = mass_1 + mass_2
    # d rho / d theta = rho * d ln rho / d theta
    drho_dm1 = rho * (5.0 / 6.0) * (0.6 / mass_1 - 0.2 / total)
    drho_dm2 = rho * (5.0 / 6.0) * (0.6 / mass_2 - 0.2 / total)
    drho_ddL = -rho / distance

    score = numpy.column_stack([
        dlnp_drho * drho_dm1,
        dlnp_drho * drho_dm2,
        dlnp_drho * drho_ddL,
    ])

    # Zero the gradient for events far below threshold (would not be detected).
    p_det = 0.5 * erfc(x)
    score[p_det < p_det_min] = 0.0
    return score


def measurement_error_diagnostic(file_path, snr_threshold, mass_fiducial, kind="bgp",
                                 population="bbh"):
    """
    Validity parameter of the Gair+2022 small-measurement-error expansion:
    Terms II-V are O(||H_k F_k^-1||), so Term I is trustworthy only where that
    is small.  Two constraints matter: F_k^-1 must be the *marginal*
    (m1_det, m2_det, d_L) covariance, not 1/F_aa (the diagonal understates the
    ratio by ~7 orders of magnitude); and the ratio is reported per mass peak,
    since the sharp mu_1 peak carries most of the H0 information while the
    global median is dominated by near-zero-curvature continuum events.

    Uses the linear mass-basis covariance; for near-equal-mass NS events that
    overstates sigma(m1, m2) (see ``load_covariances``), so the NS-edge ratios
    are upper bounds.

    Returns a dict keyed by observable direction (per-direction |H_aa| C_aa),
    plus ``"_matrix"`` (spectral radius rho(H C)) and ``"_peaks"`` entries.
    """
    from pop_fisher_higher_order import _H_theta

    covariance, events, snr, n_det, _, _ = load_covariances(file_path, snr_threshold)
    mass_model = make_mass_model(kind, mass_fiducial, population)
    redshift_model = MadauDickinsonRedshift(fiducial=MD_FIDUCIAL)
    model = SpectralSirenModel(mass_model, redshift_model, FIDUCIAL_COSMOLOGY)
    phys_keys = list(model.phys_event_keys)

    hessian = _H_theta(model, events, dict(model.fiducial), phys_keys, 1e-3)
    finite = numpy.all(numpy.isfinite(hessian.reshape(len(snr), -1)), axis=1)

    # Feature windows in source-frame m1 (peaks +/- 2 sigma, edges +/- 2 m/eta).
    peaks = {}
    for label, centre, lo, hi in feature_windows(kind, mass_fiducial, population):
        inside = (events["mass_1_source"] > lo) & (events["mass_1_source"] < hi)
        peaks[label] = {
            "mask": finite & inside, "centre": centre, "lo": lo, "hi": hi,
        }

    def summarise(ratio, mask):
        selected = ratio[mask]
        if selected.size == 0:
            return {"median": float("nan"), "p99": float("nan"), "fraction_above_1": float("nan")}
        return {
            "median": float(numpy.median(selected)),
            "p99": float(numpy.percentile(selected, 99)),
            "fraction_above_1": float((selected > 1.0).mean()),
        }

    diagnostics = {}
    for axis, key in enumerate(phys_keys):
        ratio = numpy.abs(hessian[:, axis, axis]) * covariance[:, axis, axis]
        entry = {"all": summarise(ratio, finite)}
        for label, peak in peaks.items():
            entry[label] = summarise(ratio, peak["mask"])
        entry["sigma_median"] = float(numpy.median(numpy.sqrt(covariance[finite, axis, axis])))
        if key == "luminosity_distance":
            entry["sigma_dL_over_dL_median"] = float(
                numpy.median(
                    numpy.sqrt(covariance[finite, axis, axis])
                    / events["luminosity_distance"][finite]
                )
            )
        diagnostics[key] = entry

    # True matrix expansion parameter: spectral radius of H_k C_k.
    spectral_radius = numpy.full(len(snr), numpy.nan)
    where = numpy.where(finite)[0]
    products = numpy.einsum("nij,njk->nik", hessian[where], covariance[where])
    spectral_radius[where] = numpy.max(
        numpy.abs(numpy.linalg.eigvals(products)), axis=1
    )
    matrix_entry = {"all": summarise(spectral_radius, finite)}
    for label, peak in peaks.items():
        matrix_entry[label] = summarise(spectral_radius, peak["mask"])
    diagnostics["_matrix"] = matrix_entry
    diagnostics["_peaks"] = {
        label: {k: v for k, v in peak.items() if k != "mask"}
        for label, peak in peaks.items()
    }
    diagnostics["_peaks"]["counts"] = {
        label: int(peak["mask"].sum()) for label, peak in peaks.items()
    }
    diagnostics["_n_events"] = int(finite.sum())

    first = next(iter(peaks))
    logging.info(f"[diagnostic] SNR>={snr_threshold}, {n_det} events:")
    for key in phys_keys:
        entry = diagnostics[key]
        logging.info(
            f"  |H C| {key:>20s}: all median={entry['all']['median']:.2e}, "
            f"{first} median={entry[first]['median']:.2e} "
            f"(frac>1 {entry[first]['fraction_above_1']:.3f}), "
            f"sigma median={entry['sigma_median']:.4g}"
        )
    logging.info(
        f"  rho(H C) matrix      : all median={matrix_entry['all']['median']:.2e}, "
        f"{first} median={matrix_entry[first]['median']:.2e} "
        f"(frac>1 {matrix_entry[first]['fraction_above_1']:.3f})"
    )
    return diagnostics


def compute_full_forecast(file_path, snr_thresholds, mass_fiducial, kind="bgp"):
    """
    Full 5-term spectral-siren Fisher, folding in the per-event measurement error
    via the detector-frame per-event Fisher.

    NOT CALLED BY ``main``, deliberately.  Terms II-V here go through nested
    finite differences, and for this sharp ``sigma_1 ~ 0.9 Msun`` peak the true
    correction sits below the FD noise floor: the raw sum returns spurious
    non-positive-definite matrices.  ``measurement_degraded_forecast`` brackets the
    same physics robustly instead.  Kept because an exact (automatic-
    differentiation) derivative implementation would make this path usable, and
    then it -- not the bracket -- would be the right answer.

    Drives ``compute_pop_fisher_full`` with the detector-frame per-event Fisher,
    an explicit detector-frame ``det_score`` and ``p_det`` (so the source-frame
    ``compute_det_score`` is never called), and a temporary override of
    ``pop_fisher_higher_order.POPULATION_KEYS`` so that ``_marginalise_event_fisher``
    treats the detector-frame keys as the full (un-marginalised) basis.
    """
    import pop_fisher_higher_order as higher_order
    from scipy.special import erfc

    from pop_fisher import compute_pop_fisher_full

    mass_fixed = {name: mass_fiducial[name] for name in mass_fixed_names(kind)}
    mass_model = make_mass_model(kind, mass_fiducial)
    redshift_model = MadauDickinsonRedshift(fiducial=MD_FIDUCIAL)
    model = SpectralSirenModel(mass_model, redshift_model, FIDUCIAL_COSMOLOGY)

    results = {}

    saved_keys = higher_order.POPULATION_KEYS
    higher_order.POPULATION_KEYS = list(model.phys_event_keys)
    try:
        for snr in snr_thresholds:
            (fisher_det, events, snr_array, n_det, n_above,
             n_total) = load_detector_frame_fisher(file_path, snr)
            det_score = detector_det_score(events, snr_array, snr)
            p_det = 0.5 * erfc((snr - snr_array) / numpy.sqrt(2.0))
            logging.info(f"[full] SNR>={snr}: {n_det} events, P_det={n_above/n_total:.4f}")
            result = compute_pop_fisher_full(
                model, events, fim_phys=fisher_det, snr=snr_array,
                snr_threshold=snr, fixed_parameters=mass_fixed,
                n_total=n_total, n_det=n_above,
                det_score=det_score, p_det_per_event=p_det,
            )
            results[snr] = result
            index = result.parameter_names.index("H0")
            logging.info(
                f"  [full] SNR>={snr}: sigma(H0) = {result.sigma[index]:.4g} "
                f"({100 * result.sigma[index] / FIDUCIAL_COSMOLOGY['H0']:.3f}%)"
            )
    finally:
        higher_order.POPULATION_KEYS = saved_keys
    return results


# ===========================================================================
# Summary
# ===========================================================================


def write_summary(results, mass_fiducial, path, diagnostics=None,
                  jitter=None, kind="bgp", fit_check=None):
    lines = [
        "Spectral-siren cosmology forecast",
        f"Mass model: {MASS_MODEL_LABELS[kind]}",
        "Free parameters: "
        + ", ".join(next(iter(results.values())).parameter_names),
        "Pinned, same score on every event: "
        + (", ".join(next(iter(results.values())).unconstrained) or "none"),
        "",
        "Catalogue: one year of observation (R0 = 130 Gpc^-3 yr^-1 for all CBCs,",
        "BBH share 23.7; z <= 10, Planck18), so every sigma below scales as",
        "1/sqrt(T_obs).",
        "",
    ]
    if fit_check is not None:
        lines += fit_check + [""]
    for snr in sorted(results):
        joint = results[snr]
        ih0 = joint.parameter_names.index("H0")
        iom = joint.parameter_names.index("Om")
        iw0 = joint.parameter_names.index("w0")
        correlations = joint.correlation_matrix()
        H0 = FIDUCIAL_COSMOLOGY["H0"]
        Om = FIDUCIAL_COSMOLOGY["Om"]
        w0 = FIDUCIAL_COSMOLOGY["w0"]
        lines += [
            f"===== SNR >= {snr:g}  ({joint.n_events} events) =====",
            f"  joint (H0, Om, w0), {len(joint.parameter_names)} free parameters:",
            f"    sigma(H0) = {joint.sigma[ih0]:.4g} km/s/Mpc   "
            f"({100 * joint.sigma[ih0] / H0:.3f}% of H0={H0:.2f})",
            f"    sigma(Om) = {joint.sigma[iom]:.4g}            "
            f"({100 * joint.sigma[iom] / Om:.2f}% of Om={Om:.4f})",
            f"    sigma(w0) = {joint.sigma[iw0]:.4g}            "
            f"({100 * joint.sigma[iw0] / abs(w0):.2f}% of |w0|={abs(w0):.2f})",
            f"    corr(H0, Om) = {correlations[ih0, iom]:+.3f}   "
            f"corr(H0, w0) = {correlations[ih0, iw0]:+.3f}   "
            f"corr(Om, w0) = {correlations[iom, iw0]:+.3f}",
            f"  condition # (joint) = {joint.condition_number:.3g}",
            "",
        ]
        # Largest |corr(H0, population parameter)| -- what the ruler is tied to.
        population = [i for i, n in enumerate(joint.parameter_names)
                      if n not in COSMO_NAMES]
        ranked = sorted(population, key=lambda i: -abs(correlations[ih0, i]))[:4]
        lines.append(
            "  strongest corr(H0, population): "
            + ", ".join(f"{joint.parameter_names[i]} {correlations[ih0, i]:+.2f}"
                        for i in ranked)
        )
        lines.append("")

    # --- measurement-error validity ---------------------------------------
    if diagnostics is not None:
        lines += [
            "-" * 70,
            "Per-event measurement-error impact",
            "-" * 70,
            "Term I is the leading term of an expansion in ||H_k F_k^-1||, where",
            "H_k is the population log-density Hessian and F_k the per-event",
            "measurement Fisher.  F_k^-1 here is the *marginal* covariance over",
            "(m1_det, m2_det, d_L): the population density does not depend on the",
            "eight nuisance directions (inclination, polarisation, sky, spins,",
            "t_c, phi_c), so H_k vanishes there and the product picks out exactly",
            "that sub-block.",
            "",
            "Columns are source-frame m1 windows around the sharp features that",
            "anchor the spectral-siren ruler (peaks +/- 2 sigma, logistic edges",
            "+/- 2 m/eta); frac>1 refers to the first feature.",
            "",
        ]
        for snr in sorted(diagnostics):
            entry = diagnostics[snr]
            peaks = entry["_peaks"]
            labels = [k for k in peaks if k != "counts"]
            first = labels[0]
            lines.append(
                f"===== SNR >= {snr:g}  ({entry['_n_events']} events with finite H) ====="
            )
            for label in labels:
                lines.append(
                    f"  window {label:<12s} [{peaks[label]['lo']:.2f}, "
                    f"{peaks[label]['hi']:.2f}]  ({peaks['counts'][label]} events)"
                )
            header = f"  {'|H C| direction':>22s}  {'all':>10s}"
            for label in labels:
                header += f"  {label:>12s}"
            header += f"  {'frac>1':>7s}  {'sigma (marg)':>13s}"
            lines.append(header)
            for key in ("mass_1_det", "mass_2_det", "luminosity_distance", "_matrix"):
                value = entry[key]
                name = "rho(H C) [matrix]" if key == "_matrix" else key
                row = f"  {name:>22s}  {value['all']['median']:10.2e}"
                for label in labels:
                    row += f"  {value[label]['median']:12.2e}"
                row += f"  {value[first]['fraction_above_1']:7.3f}"
                if key != "_matrix":
                    row += f"  {value['sigma_median']:13.4g}"
                lines.append(row)
            lines.append(
                f"  marginal sigma(dL)/dL median = "
                f"{entry['luminosity_distance']['sigma_dL_over_dL_median']:.4f}"
            )
            lines.append("")
        lines += [
            "Where the expansion parameter is O(0.1) or more, Term I is NOT exact:",
            "measurement error on the individual masses is comparable to the width",
            "of the feature that anchors the ruler.  The correction can only loosen",
            "sigma(H0), never tighten it.",
            "",
        ]

    # --- measurement-error bracket ----------------------------------------
    if jitter is not None:
        lines += [
            "-" * 70,
            "Measurement-error bracket on sigma(H0)",
            "-" * 70,
            "Lower edge: Term I on the true simulated parameters (zero measurement",
            "error).  Upper edge: each event jittered by a draw from its own",
            "marginal measurement covariance, the mass model re-fitted to the",
            "jittered masses so model and data stay consistent, Term I recomputed.",
            "That treats each noisy point estimate as the truth and so discards the",
            "per-event uncertainties a real hierarchical analysis deconvolves with:",
            "",
            "    sigma(Term I)  <=  sigma(true)  <=  sigma(jittered)",
            "",
        ]
        for snr in sorted(jitter):
            joint = results[snr]
            clean_joint = joint.sigma[joint.parameter_names.index("H0")]
            jitter_joint = numpy.array(jitter[snr])
            lines += [
                f"===== SNR >= {snr:g}  ({len(jitter_joint)} noise realisations) =====",
                f"  joint (H0, Om, w0) : sigma(H0) in [{clean_joint:.3g}, "
                f"{jitter_joint.mean():.3g}]  (x{jitter_joint.mean() / clean_joint:.2f}, "
                f"realisation scatter {jitter_joint.std():.3g}; "
                f"each: {', '.join(f'{v:.3g}' for v in jitter_joint)})",
                "",
            ]
        lines += ["Quote the range, not the Term-I number alone.", ""]

    text = "\n".join(lines)
    with open(path, "w") as handle:
        handle.write(text + "\n")
    logging.info(f"Wrote {path}")
    print(text)


def fullpop_fit_check(mass_1, mass_2, mass_fiducial, population="bbh"):
    """
    Two sanity checks on the FullPop fiducial against the injected masses:
    the ML refit must land within ~Fisher errors of the generator's values, and
    the mean log-likelihood of the injections shows how much better FullPop
    describes them than the BGP fit.
    """
    fitted, sigma, result = fit_fullpop_to_samples(
        mass_1, mass_2, start=mass_fiducial, population=population
    )
    model = FullPopMass(fiducial=mass_fiducial, population=population)
    keep = model._in_support(mass_1, mass_2)
    events = {"mass_1_source": mass_1[keep], "mass_2_source": mass_2[keep],
              "mass_ratio": mass_2[keep] / mass_1[keep]}
    log_l_truth = float(model.log_prob(events, mass_fiducial).sum())
    log_l_fit = float(model.log_prob(events, fitted).sum())
    pulls = {n: (fitted[n] - mass_fiducial[n]) / sigma[n] for n in sigma}
    label = "BBHs" if population == "bbh" else "CBCs (BNS + NSBH + BBH)"
    lines = [
        f"FullPop check on all {int(keep.sum())} injected {label} (source frame):",
        f"  ML refit {'converged' if result.success else 'NOT converged: ' + str(result.message)};"
        f" ln L(fit) - ln L(truth) = {log_l_fit - log_l_truth:.2f}"
        f" (expected ~ {len(sigma) / 2:.1f} for {len(sigma)} free parameters)",
        f"  max |pull| (fit - truth)/sigma_Fisher = "
        f"{max(abs(v) for v in pulls.values()):.2f}  ("
        + ", ".join(f"{n} {v:+.1f}" for n, v in pulls.items()) + ")",
    ]
    if population == "bbh":
        m_min_upper = 0.999 * min(mass_1.min(), mass_2.min())
        bgp_fiducial, _ = fit_bgp_to_samples(mass_1, mass_2, m_min_upper, 1.001 * mass_1.max())
        log_l_bgp = float(
            BrokenPowerLawPlusTwoPeaksMass(fiducial=bgp_fiducial).log_prob(events, bgp_fiducial).sum()
        )
        lines.append(f"  ln L(FullPop truth) - ln L(BGP best fit) = {log_l_truth - log_l_bgp:.1f}")
    else:
        # The generator's proposal grid (0.035 Msun) softens the eta = 50 NS
        # edge slightly; a refit with free sharpness measures by how much.
        loose, loose_sigma, _ = fit_fullpop_to_samples(
            mass_1, mass_2, start=mass_fiducial, population=population, free_sharpness=True
        )
        gain = float(model.log_prob(events, loose).sum()) - log_l_fit
        lines.append(
            "  refit with sharpness free: "
            + ", ".join(f"{n} {loose[n]:.1f}+-{loose_sigma[n]:.1f} (truth {mass_fiducial[n]:g})"
                        for n in FULLPOP_SHARPNESS_NAMES)
            + f";  NSmin {loose['NSmin']:.4f};  ln L gain {gain:.1f}"
        )
    for line in lines:
        logging.info(line)
    return lines


def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__.split("Run with")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    for name in POPULATION_FILES:
        parser.add_argument(f"--{name}", action="store_true",
                            help=f"Include the {name.upper()} catalogue (at least one of "
                                 "--bns, --bhns, --bbh).")
    parser.add_argument("--network", choices=sorted(NETWORK_FILES), default="CE4020LIA")
    parser.add_argument("--data-directory", default=os.path.expanduser("~/projects/data_dump"))
    parser.add_argument("--snr-thresholds", type=float, nargs="+", default=[10.0, 20.0])
    parser.add_argument(
        "--mass-model", choices=["bgp", "fullpop"], default="bgp",
        help="bgp: BGP fitted to the injections (misspecified); fullpop: the "
             "generating FullPop model at the injected hyperparameters.",
    )
    parser.add_argument("--output-directory", default="spectral_sirens_output",
                        help="Every run writes here, under its own file names.")
    parser.add_argument("--fix", choices=sorted(FULLPOP_EDGE_SETS), default=None,
                        help="FullPop: also pin these edge parameters at truth "
                             "(edges: NSmin, NSmax, BHmin, BHmax; alledges: also "
                             "UPPERmin, UPPERmax, n5).")
    parser.add_argument("--term-one-only", action="store_true",
                        help="Skip the fit check and the measurement-error steps.")
    parser.add_argument("--seed", type=int, default=250114)
    parser.add_argument(
        "--n-realisations", type=int, default=3,
        help="Noise realisations for the measurement-error bracket (Stage 2).",
    )
    args = parser.parse_args(argv)
    args.populations = [name for name in POPULATION_FILES if getattr(args, name)]
    if not args.populations:
        parser.error("give at least one of --bns, --bhns, --bbh")
    return args


def main(argv=None):
    args = parse_arguments(argv)
    kind = args.mass_model
    fisher_files = [
        os.path.join(args.data_directory,
                     f"network_{POPULATION_FILES[name]}_{NETWORK_FILES[args.network]}.h5")
        for name in args.populations
    ]
    # FullPop support: "bbh" (m2 >= 2.06) for the BBH file alone, else the whole
    # CBC range.  For Term I a sub-sample of the "all" support is exact (a
    # truncated support only shifts ln p by a constant, which the centred score
    # removes); the fit check and the jitter refits need the truncation itself.
    population = "bbh" if args.populations == ["bbh"] else "all"
    if kind == "bgp" and args.populations != ["bbh"]:
        raise SystemExit("--mass-model bgp takes --bbh only (BGP has no NS part)")
    if args.populations not in (["bbh"], list(POPULATION_FILES)) and not args.term_one_only:
        raise SystemExit("a partial NS sample needs --term-one-only")
    if args.fix and kind != "fullpop":
        raise SystemExit("--fix needs --mass-model fullpop")
    extra_fixed = FULLPOP_EDGE_SETS[args.fix] if args.fix else ()
    # One directory; the run is named by its file stem, e.g.
    # fullpop_CE4020LIA_bns+bhns_fix_alledges_snr10.npz.
    stem = "_".join(
        part for part in (kind, args.network, "+".join(args.populations),
                          args.fix and f"fix_{args.fix}") if part
    )
    os.makedirs(args.output_directory, exist_ok=True)

    mass_1, mass_2 = load_masses(fisher_files)
    fit_check = None
    if kind == "bgp":
        # Effective BGP fiducial from the injected masses (support edges from range).
        mass_fiducial = fit_mass_model("bgp", mass_1, mass_2)
        logging.info(
            "Effective BGP fiducial: "
            + ", ".join(f"{k}={mass_fiducial[k]:.4g}" for k in
                        ["mu_1", "sigma_1", "mu_2", "sigma_2", "alpha_1", "alpha_2"])
        )
    else:
        # The generator's own hyperparameters: no fit, no misspecification.
        mass_fiducial = dict(FULLPOP_GWTC4_FIDUCIAL)
        if not args.term_one_only:
            fit_check = fullpop_fit_check(mass_1, mass_2, mass_fiducial, population)

    results = compute_forecast(
        fisher_files, args.snr_thresholds, mass_fiducial, kind, population, extra_fixed
    )

    # Measure the validity parameter of the Term-I expansion around the sharp
    # features, then bracket sigma(H0) from the pessimistic side.
    diagnostics = jitter = None
    if not args.term_one_only:
        diagnostics = {
            snr: measurement_error_diagnostic(
                fisher_files, snr, mass_fiducial, kind, population
            )
            for snr in args.snr_thresholds
        }
        # Whole CBC: the jitter refit frees the edge sharpness too, since the
        # measured NS masses are far broader than the eta = 50 edges.
        jitter, _ = measurement_degraded_forecast(
            fisher_files, args.snr_thresholds,
            n_realisations=args.n_realisations, seed=args.seed, kind=kind,
            clean_fiducial=mass_fiducial, population=population,
            free_sharpness=population == "all",
        )

    for snr, result in results.items():
        extra = {} if jitter is None else {"jitter_sigma_H0": numpy.array(jitter[snr])}
        numpy.savez(
            os.path.join(args.output_directory, f"{stem}_snr{snr:g}.npz"),
            covariance=result.covariance,
            parameter_names=numpy.array(result.parameter_names),
            fiducial=numpy.array(
                [result.fiducial_parameters[n] for n in result.parameter_names]
            ),
            snr_threshold=snr,
            n_events=result.n_events,
            unconstrained=numpy.array(result.unconstrained, dtype=str),
            **extra,
        )

    write_summary(
        results, mass_fiducial, os.path.join(args.output_directory, f"{stem}_summary.txt"),
        diagnostics, jitter, kind=kind, fit_check=fit_check,
    )
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
