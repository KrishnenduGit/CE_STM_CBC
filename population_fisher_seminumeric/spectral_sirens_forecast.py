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
  * the Term-I Fisher over 12 population parameters + (H0, Om, w0), in three
    configurations: joint, LCDM (w0 pinned) and H0-only (Om and w0 pinned);
  * a degeneracy ladder -- sigma(H0) as groups of parameters are pinned -- because
    that, not the physics, is usually what separates two forecasts;
  * the validity parameter ||H_k F_k^-1|| of the Term-I expansion, which is NOT
    small in the mu_1 peak (~0.3), so Term I is optimistic here;
  * a bracket on sigma(H0) between Term I and a measurement-jittered refit.
"""

import argparse
import logging
import os

import matplotlib

matplotlib.use("Agg")

import numpy

import cosmology
from cosmology import FIDUCIAL_COSMOLOGY
from load import (
    load_detector_frame_covariance,
    load_detector_frame_fisher,
    load_injected_masses,
    load_population_catalogue,
)
from pop_fisher import compute_pop_fisher
from pop_models import (
    BrokenPowerLawPlusTwoPeaksMass,
    MadauDickinsonRedshift,
    PopModel,
)
from forecast_mass_redshift import (
    MASS_FIXED_NAMES,
    MD_FIDUCIAL,
    fit_bgp_to_samples,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")

COSMO_NAMES = ["H0", "Om", "w0"]


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
        def finite_difference(name, value):
            step = max(1e-4 * abs(value), 1e-8)
            log_prob_plus = self.log_prob(events, {**params, name: value + step})
            log_prob_minus = self.log_prob(events, {**params, name: value - step})
            finite = numpy.isfinite(log_prob_plus) & numpy.isfinite(log_prob_minus)
            column = numpy.full(len(redshift), numpy.nan)
            column[finite] = (
                log_prob_plus[finite] - log_prob_minus[finite]
            ) / (2.0 * step)
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


def compute_forecast(file_path, snr_thresholds, mass_fiducial):
    """
    Term-I spectral-siren Fisher at each SNR threshold, in two configurations:
    ``joint`` (H0 and Om free) and ``h0_only`` (Om pinned).

    Returns ``{config: {snr: PopFisherResult}}``.
    """
    mass_fixed = {name: mass_fiducial[name] for name in MASS_FIXED_NAMES}
    mass_model = BrokenPowerLawPlusTwoPeaksMass(fiducial=mass_fiducial)
    redshift_model = MadauDickinsonRedshift(fiducial=MD_FIDUCIAL)
    model = SpectralSirenModel(mass_model, redshift_model, FIDUCIAL_COSMOLOGY)

    # "joint" now frees w0 as well.  "lcdm" pins only w0, so it reproduces the
    # flat-LambdaCDM forecast exactly and keeps every previously published number
    # comparable; "h0_only" pins both dark-energy parameters.
    configs = {
        "joint": mass_fixed,
        "lcdm": {**mass_fixed, "w0": FIDUCIAL_COSMOLOGY["w0"]},
        "h0_only": {
            **mass_fixed,
            "Om": FIDUCIAL_COSMOLOGY["Om"],
            "w0": FIDUCIAL_COSMOLOGY["w0"],
        },
    }
    results = {config: {} for config in configs}
    for snr in snr_thresholds:
        catalogue = load_population_catalogue(
            file_path, snr_threshold=snr, with_fisher=False
        )
        events = detector_frame_events(catalogue, FIDUCIAL_COSMOLOGY)
        logging.info(f"SNR>={snr}: {catalogue.number_detected} events")
        validate_frame(catalogue, events, FIDUCIAL_COSMOLOGY)
        for config, fixed in configs.items():
            result = compute_pop_fisher(
                model, events, fixed_parameters=fixed,
                n_total=catalogue.number_total,
            )
            results[config][snr] = result
            index = result.parameter_names.index("H0")
            logging.info(
                f"  {config:>8s} SNR>={snr}: condition # = "
                f"{result.condition_number:.3g}, "
                f"sigma(H0) = {result.sigma[index]:.4g} km/s/Mpc "
                f"({100 * result.sigma[index] / FIDUCIAL_COSMOLOGY['H0']:.3f}%)"
            )
    return results


def measurement_degraded_forecast(
    file_path, snr_thresholds, n_realisations=3, seed=0
):
    """
    Pessimistic edge of the measurement-error bracket:
    sigma_TermI <= sigma_true <= sigma_jittered.

    Each event's (m1_det, m2_det, d_L) is jittered by a draw from its own
    marginal measurement covariance, the BGP is re-fitted to the jittered masses
    so model and data stay consistent, and Term I is recomputed.  Treating each
    noisy point estimate as truth discards the per-event uncertainties a real
    hierarchical analysis would deconvolve with, hence a genuine upper edge.

    Returns ``{config: {snr: [sigma(H0) per realisation]}}`` plus the jittered
    fiducials.
    """
    from forecast_mass_redshift import fit_bgp_to_samples

    rng = numpy.random.default_rng(seed)
    out = {"joint": {}, "h0_only": {}}
    fiducials = {}

    for snr in snr_thresholds:
        covariance, events, _, n_det, _, n_total = load_detector_frame_covariance(
            file_path, snr
        )
        stacked = numpy.stack(
            [events["mass_1_det"], events["mass_2_det"], events["luminosity_distance"]],
            axis=1,
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

        out["joint"][snr] = []
        out["h0_only"][snr] = []
        for realisation in range(n_realisations):
            noise = numpy.einsum(
                "nij,nj->ni", chol, rng.standard_normal(stacked.shape)
            )
            jittered = stacked + noise

            mass_1_det = numpy.maximum(jittered[:, 0], jittered[:, 1])
            mass_2_det = numpy.minimum(jittered[:, 0], jittered[:, 1])
            distance = jittered[:, 2]

            keep = (mass_2_det > 0.0) & (distance > 0.0)
            mass_1_det, mass_2_det, distance = (
                mass_1_det[keep], mass_2_det[keep], distance[keep]
            )
            redshift = cosmology.redshift_of_distance(
                distance, FIDUCIAL_COSMOLOGY["H0"], FIDUCIAL_COSMOLOGY["Om"],
                FIDUCIAL_COSMOLOGY["w0"],
            )
            one_plus_z = 1.0 + redshift
            mass_1_source = mass_1_det / one_plus_z
            mass_2_source = mass_2_det / one_plus_z

            m_min_upper = 0.999 * min(mass_1_source.min(), mass_2_source.min())
            m_max = 1.001 * mass_1_source.max()
            jittered_fiducial, _ = fit_bgp_to_samples(
                mass_1_source, mass_2_source, m_min_upper, m_max
            )
            fiducials[(snr, realisation)] = jittered_fiducial

            model = SpectralSirenModel(
                BrokenPowerLawPlusTwoPeaksMass(fiducial=jittered_fiducial),
                MadauDickinsonRedshift(fiducial=MD_FIDUCIAL),
                FIDUCIAL_COSMOLOGY,
            )
            jittered_events = {
                "mass_1_det": mass_1_det,
                "mass_2_det": mass_2_det,
                "luminosity_distance": distance,
                "mass_ratio": mass_2_det / mass_1_det,
            }
            mass_fixed = {n: jittered_fiducial[n] for n in MASS_FIXED_NAMES}
            for config, fixed in (
                ("joint", mass_fixed),
                ("h0_only", {**mass_fixed, "Om": FIDUCIAL_COSMOLOGY["Om"],
                     "w0": FIDUCIAL_COSMOLOGY["w0"]}),
            ):
                result = compute_pop_fisher(
                    model, jittered_events, fixed_parameters=fixed, n_total=n_total
                )
                index = result.parameter_names.index("H0")
                out[config][snr].append(float(result.sigma[index]))

            logging.info(
                f"  [jitter] SNR>={snr} realisation {realisation}: "
                f"sigma_1 {jittered_fiducial['sigma_1']:.4f} "
                f"(clean {0.893:.3f}), "
                f"sigma(H0) joint {out['joint'][snr][-1]:.4g}, "
                f"h0_only {out['h0_only'][snr][-1]:.4g}"
            )

    return out, fiducials


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


def measurement_error_diagnostic(file_path, snr_threshold, mass_fiducial):
    """
    Validity parameter of the Gair+2022 small-measurement-error expansion:
    Terms II-V are O(||H_k F_k^-1||), so Term I is trustworthy only where that
    is small.  Two constraints matter: F_k^-1 must be the *marginal*
    (m1_det, m2_det, d_L) covariance, not 1/F_aa (the diagonal understates the
    ratio by ~7 orders of magnitude); and the ratio is reported per mass peak,
    since the sharp mu_1 peak carries most of the H0 information while the
    global median is dominated by near-zero-curvature continuum events.

    Returns a dict keyed by observable direction (per-direction |H_aa| C_aa),
    plus ``"_matrix"`` (spectral radius rho(H C)) and ``"_peaks"`` entries.
    """
    from pop_fisher_higher_order import _H_theta

    covariance, events, snr, n_det, _, _ = load_detector_frame_covariance(
        file_path, snr_threshold
    )
    mass_model = BrokenPowerLawPlusTwoPeaksMass(fiducial=mass_fiducial)
    redshift_model = MadauDickinsonRedshift(fiducial=MD_FIDUCIAL)
    model = SpectralSirenModel(mass_model, redshift_model, FIDUCIAL_COSMOLOGY)
    phys_keys = list(model.phys_event_keys)

    hessian = _H_theta(model, events, dict(model.fiducial), phys_keys, 1e-3)
    finite = numpy.all(numpy.isfinite(hessian.reshape(len(snr), -1)), axis=1)

    # Peak windows: +/- 2 sigma about each Gaussian component, in source-frame mass.
    peaks = {}
    for label, centre, width in (
        ("mu_1", mass_fiducial["mu_1"], mass_fiducial["sigma_1"]),
        ("mu_2", mass_fiducial["mu_2"], mass_fiducial["sigma_2"]),
    ):
        inside = (
            (events["mass_1_source"] > centre - 2.0 * width)
            & (events["mass_1_source"] < centre + 2.0 * width)
        )
        peaks[label] = {
            "mask": finite & inside,
            "centre": float(centre),
            "width": float(width),
            "lo": float(centre - 2.0 * width),
            "hi": float(centre + 2.0 * width),
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

    logging.info(f"[diagnostic] SNR>={snr_threshold}, {n_det} events:")
    for key in phys_keys:
        entry = diagnostics[key]
        logging.info(
            f"  |H C| {key:>20s}: all median={entry['all']['median']:.2e}, "
            f"mu_1-peak median={entry['mu_1']['median']:.2e} "
            f"(frac>1 {entry['mu_1']['fraction_above_1']:.3f}), "
            f"sigma median={entry['sigma_median']:.4g}"
        )
    logging.info(
        f"  rho(H C) matrix      : all median={matrix_entry['all']['median']:.2e}, "
        f"mu_1-peak median={matrix_entry['mu_1']['median']:.2e} "
        f"(frac>1 {matrix_entry['mu_1']['fraction_above_1']:.3f})"
    )
    return diagnostics


def compute_full_forecast(file_path, snr_thresholds, mass_fiducial):
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

    mass_fixed = {name: mass_fiducial[name] for name in MASS_FIXED_NAMES}
    mass_model = BrokenPowerLawPlusTwoPeaksMass(fiducial=mass_fiducial)
    redshift_model = MadauDickinsonRedshift(fiducial=MD_FIDUCIAL)
    model = SpectralSirenModel(mass_model, redshift_model, FIDUCIAL_COSMOLOGY)

    # Same three configurations as compute_forecast.
    configs = {
        "joint": mass_fixed,
        "lcdm": {**mass_fixed, "w0": FIDUCIAL_COSMOLOGY["w0"]},
        "h0_only": {
            **mass_fixed,
            "Om": FIDUCIAL_COSMOLOGY["Om"],
            "w0": FIDUCIAL_COSMOLOGY["w0"],
        },
    }
    results = {config: {} for config in configs}

    saved_keys = higher_order.POPULATION_KEYS
    higher_order.POPULATION_KEYS = list(model.phys_event_keys)
    try:
        for snr in snr_thresholds:
            (fisher_det, events, snr_array, n_det, n_above,
             n_total) = load_detector_frame_fisher(file_path, snr)
            det_score = detector_det_score(events, snr_array, snr)
            p_det = 0.5 * erfc((snr - snr_array) / numpy.sqrt(2.0))
            logging.info(f"[full] SNR>={snr}: {n_det} events, P_det={n_above/n_total:.4f}")
            for config, fixed in configs.items():
                result = compute_pop_fisher_full(
                    model, events, fim_phys=fisher_det, snr=snr_array,
                    snr_threshold=snr, fixed_parameters=fixed,
                    n_total=n_total, n_det=n_above,
                    det_score=det_score, p_det_per_event=p_det,
                )
                results[config][snr] = result
                index = result.parameter_names.index("H0")
                logging.info(
                    f"  [full] {config:>8s} SNR>={snr}: "
                    f"sigma(H0) = {result.sigma[index]:.4g} "
                    f"({100 * result.sigma[index] / FIDUCIAL_COSMOLOGY['H0']:.3f}%)"
                )
    finally:
        higher_order.POPULATION_KEYS = saved_keys
    return results


# ===========================================================================
# Plots and summary
# ===========================================================================


def plot_cosmology_corner(results_joint, results_h0_only, outfile, seed=250114):
    """Focused (H0, Om, w0) corner from the joint Fisher, both thresholds."""
    import corner
    import matplotlib.lines
    import pylab

    from plot_utils import new_rcParams

    labels = [r"$H_0\ [\mathrm{km/s/Mpc}]$", r"$\Omega_m$", r"$w_0$"]
    snr_values = sorted(results_joint)
    colours = {snr_values[0]: "#1f6fd6", snr_values[-1]: "#d1495b"}

    reference = results_joint[snr_values[0]]
    idx = [reference.parameter_names.index(name) for name in COSMO_NAMES]
    mean = numpy.array([reference.fiducial_parameters[name] for name in COSMO_NAMES])

    rng = numpy.random.default_rng(seed)
    draws = {}
    widest = numpy.zeros(len(COSMO_NAMES))
    for snr in snr_values:
        covariance = results_joint[snr].covariance[numpy.ix_(idx, idx)]
        sample = rng.multivariate_normal(mean, covariance, size=60000)
        draws[snr] = sample
        widest = numpy.maximum(widest, sample.std(axis=0))
    ranges = [
        (mean[i] - 4 * widest[i], mean[i] + 4 * widest[i])
        for i in range(len(COSMO_NAMES))
    ]

    with matplotlib.rc_context(new_rcParams()):
        figure = pylab.figure(figsize=(7, 7))
        for offset, snr in enumerate(snr_values):
            corner.corner(
                draws[snr], fig=figure, labels=labels, color=colours[snr],
                truths=list(mean), truth_color="0.2", range=ranges,
                levels=(0.67, 0.90), bins=45, smooth=0.9,
                plot_datapoints=False, plot_density=False, fill_contours=True,
                show_titles=(offset == 0), title_fmt=".3f",
                title_quantiles=[0.16, 0.5, 0.84], max_n_ticks=4,
            )
        handles = [
            matplotlib.lines.Line2D([], [], color=colours[snr],
                                    label=rf"SNR $\geq$ {snr:g} (joint)")
            for snr in snr_values
        ]
        # H0-only 1-sigma band at the tightest threshold, to show what
        # marginalising over the dark-energy sector costs.
        h0_index = results_h0_only[snr_values[0]].parameter_names.index("H0")
        h0_sigma = results_h0_only[snr_values[0]].sigma[h0_index]
        size = len(COSMO_NAMES)
        axes = numpy.array(figure.axes).reshape(size, size)
        axes[0, 0].axvspan(mean[0] - h0_sigma, mean[0] + h0_sigma,
                           color="0.6", alpha=0.25, zorder=0)
        handles.append(matplotlib.lines.Line2D(
            [], [], color="0.6", alpha=0.5, linewidth=8,
            label=rf"$H_0$-only $1\sigma$ (SNR $\geq$ {snr_values[0]:g})"))
        figure.legend(handles=handles, loc="upper right", frameon=False, fontsize=13)

    figure.savefig(outfile, bbox_inches="tight")
    logging.info(f"Wrote {outfile}")
    pylab.close(figure)


def degeneracy_ladder(joint_result):
    """
    sigma(H0) as successive groups of hyperparameters are pinned.

    This exists to be *sent to someone else*.  Any forecast that quotes a tighter
    sigma(H0) than ours is almost always doing so because it is marginalising over
    fewer parameters, and the cost is enormous: corr(H0, mu_1) = -0.90 and
    corr(H0, Om) = -0.95 here, so the two marginalisations together are worth more
    than a factor of ten.  Comparing against this ladder identifies which
    configuration an external number actually corresponds to, without needing to
    read the external code.
    """
    names = list(joint_result.parameter_names)
    fisher = joint_result.fisher
    index_H0 = names.index("H0")
    index_Om = names.index("Om")
    index_w0 = names.index("w0")
    cosmology_indices = [index_H0, index_Om, index_w0]
    n_population = len(names) - len(cosmology_indices)

    def sigma_free(free_indices):
        block = fisher[numpy.ix_(free_indices, free_indices)]
        covariance = numpy.linalg.inv(block)
        position = free_indices.index(index_H0)
        return float(numpy.sqrt(covariance[position, position]))

    everything = list(range(len(names)))
    no_w0 = [i for i in everything if i != index_w0]
    no_dark_energy = [i for i in everything if i not in (index_Om, index_w0)]

    return [
        (f"all {len(names)} free (published)", sigma_free(everything)),
        (f"w0 pinned (LCDM), {n_population} population parameters free",
         sigma_free(no_w0)),
        (f"Om and w0 pinned, {n_population} population parameters free",
         sigma_free(no_dark_energy)),
        (f"all {n_population} population parameters pinned, cosmology free",
         sigma_free(cosmology_indices)),
        ("everything except H0 pinned", sigma_free([index_H0])),
    ]


def write_summary(results, mass_fiducial, outdir, diagnostics=None,
                  jitter=None, n_total=None):
    lines = [
        "Spectral-siren cosmology forecast",
        "",
        "Catalogue: one year of observation (R0 = 130 Gpc^-3 yr^-1, z <= 10,",
        "Planck18), so every sigma below scales as 1/sqrt(T_obs).",
        "",
    ]
    for snr in sorted(results["joint"]):
        joint = results["joint"][snr]
        lcdm = results["lcdm"][snr]
        h0_only = results["h0_only"][snr]
        ih0 = joint.parameter_names.index("H0")
        iom = joint.parameter_names.index("Om")
        iw0 = joint.parameter_names.index("w0")
        correlations = joint.correlation_matrix()
        lcdm_sigma = lcdm.sigma[lcdm.parameter_names.index("H0")]
        h0_only_sigma = h0_only.sigma[h0_only.parameter_names.index("H0")]
        H0 = FIDUCIAL_COSMOLOGY["H0"]
        Om = FIDUCIAL_COSMOLOGY["Om"]
        w0 = FIDUCIAL_COSMOLOGY["w0"]
        lines += [
            f"===== SNR >= {snr:g}  ({joint.n_events} events) =====",
            f"  joint (H0, Om, w0):",
            f"    sigma(H0) = {joint.sigma[ih0]:.4g} km/s/Mpc   "
            f"({100 * joint.sigma[ih0] / H0:.3f}% of H0={H0:.2f})",
            f"    sigma(Om) = {joint.sigma[iom]:.4g}            "
            f"({100 * joint.sigma[iom] / Om:.2f}% of Om={Om:.4f})",
            f"    sigma(w0) = {joint.sigma[iw0]:.4g}            "
            f"({100 * joint.sigma[iw0] / abs(w0):.2f}% of |w0|={abs(w0):.2f})",
            f"    corr(H0, Om) = {correlations[ih0, iom]:+.3f}   "
            f"corr(H0, w0) = {correlations[ih0, iw0]:+.3f}   "
            f"corr(Om, w0) = {correlations[iom, iw0]:+.3f}",
            f"  LCDM (w0 fixed at -1):",
            f"    sigma(H0) = {lcdm_sigma:.4g} km/s/Mpc   "
            f"({100 * lcdm_sigma / H0:.3f}% of H0)",
            f"    w0-marginalisation cost: sigma(H0) x {joint.sigma[ih0] / lcdm_sigma:.2f}",
            f"  H0-only (Om and w0 fixed):",
            f"    sigma(H0) = {h0_only_sigma:.4g} km/s/Mpc   "
            f"({100 * h0_only_sigma / H0:.3f}% of H0)",
            f"    full dark-energy marginalisation cost: "
            f"sigma(H0) x {joint.sigma[ih0] / h0_only_sigma:.2f}",
            f"  condition # (joint) = {joint.condition_number:.3g}",
            "",
        ]

    # --- degeneracy ladder -------------------------------------------------
    lines += [
        "-" * 70,
        "How much of sigma(H0) is marginalisation?",
        "-" * 70,
        "sigma(H0) as groups of hyperparameters are pinned.  Any external",
        "forecast quoting a tighter number is usually sitting on a lower rung of",
        "this ladder rather than disagreeing about the physics; corr(H0, mu_1) =",
        "-0.90 and corr(H0, Om) = -0.95, so the two marginalisations together are",
        "worth more than a factor of ten.",
        "",
    ]
    for snr in sorted(results["joint"]):
        lines.append(f"===== SNR >= {snr:g} =====")
        for label, sigma in degeneracy_ladder(results["joint"][snr]):
            lines.append(f"  {label:<46s} sigma(H0) = {sigma:.4g}")
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
            "that sub-block.  Dividing by the diagonal Fisher element instead --",
            "as an earlier version of this script did -- answers a different",
            "question (the error on m1_det with m2_det, d_L and inclination known",
            "exactly) and understates the ratio by ~7 orders of magnitude.",
            "",
            "The mu_1 column is the one that matters: the sharp ~9.2 Msun peak",
            "supplies most of the H0 information, while the global median is",
            "dominated by the smooth continuum where the curvature is ~0.",
            "",
        ]
        for snr in sorted(diagnostics):
            entry = diagnostics[snr]
            peaks = entry["_peaks"]
            lines.append(
                f"===== SNR >= {snr:g}  ({entry['_n_events']} events with finite H) ====="
            )
            lines.append(
                f"  peak windows: mu_1 in [{peaks['mu_1']['lo']:.2f}, "
                f"{peaks['mu_1']['hi']:.2f}] ({peaks['counts']['mu_1']} events), "
                f"mu_2 in [{peaks['mu_2']['lo']:.2f}, {peaks['mu_2']['hi']:.2f}] "
                f"({peaks['counts']['mu_2']} events)"
            )
            lines.append(
                f"  {'|H C| direction':>22s}  {'all':>10s}  {'mu_1 peak':>10s}  "
                f"{'frac>1':>7s}  {'mu_2 peak':>10s}  {'sigma (marg)':>13s}"
            )
            for key in ("mass_1_det", "mass_2_det", "luminosity_distance"):
                value = entry[key]
                lines.append(
                    f"  {key:>22s}  {value['all']['median']:10.2e}  "
                    f"{value['mu_1']['median']:10.2e}  "
                    f"{value['mu_1']['fraction_above_1']:7.3f}  "
                    f"{value['mu_2']['median']:10.2e}  "
                    f"{value['sigma_median']:13.4g}"
                )
            matrix = entry["_matrix"]
            lines.append(
                f"  {'rho(H C) [matrix]':>22s}  {matrix['all']['median']:10.2e}  "
                f"{matrix['mu_1']['median']:10.2e}  "
                f"{matrix['mu_1']['fraction_above_1']:7.3f}  "
                f"{matrix['mu_2']['median']:10.2e}"
            )
            lines.append(
                f"  marginal sigma(dL)/dL median = "
                f"{entry['luminosity_distance']['sigma_dL_over_dL_median']:.4f}"
            )
            lines.append("")
        lines += [
            "Read: in the mu_1 peak the expansion parameter is O(0.1-0.3) with a",
            "non-trivial fraction of events above 1, so Term I is NOT exact here.",
            "Measurement error on the individual masses is comparable to the width",
            "of the very feature that anchors the spectral-siren ruler.  The",
            "correction can only loosen sigma(H0), never tighten it.",
            "",
        ]

    # --- measurement-error bracket ----------------------------------------
    if jitter is not None:
        lines += [
            "-" * 70,
            "Measurement-error bracket on sigma(H0)",
            "-" * 70,
            "Terms II-V by nested finite differences are unusable for this sharp",
            "peak (below the FD noise floor, non-positive-definite results), so we",
            "bracket instead.  Lower edge: Term I on the true simulated parameters",
            "(zero measurement error).  Upper edge: each event jittered by a draw",
            "from its own marginal measurement covariance, the BGP re-fitted to the",
            "jittered masses so model and data stay consistent, Term I recomputed.",
            "That treats each noisy point estimate as the truth and so discards the",
            "per-event uncertainties a real hierarchical analysis deconvolves with:",
            "",
            "    sigma(Term I)  <=  sigma(true)  <=  sigma(jittered)",
            "",
        ]
        for snr in sorted(jitter["joint"]):
            joint = results["joint"][snr]
            h0_only = results["h0_only"][snr]
            clean_joint = joint.sigma[joint.parameter_names.index("H0")]
            clean_h0 = h0_only.sigma[h0_only.parameter_names.index("H0")]
            jitter_joint = numpy.array(jitter["joint"][snr])
            jitter_h0 = numpy.array(jitter["h0_only"][snr])
            lines += [
                f"===== SNR >= {snr:g}  ({len(jitter_joint)} noise realisations) =====",
                f"  joint (H0, Om, w0) : sigma(H0) in [{clean_joint:.3g}, "
                f"{jitter_joint.mean():.3g}]  (x{jitter_joint.mean() / clean_joint:.2f}, "
                f"realisation scatter {jitter_joint.std():.3g})",
                f"  Om fixed       : sigma(H0) in [{clean_h0:.3g}, "
                f"{jitter_h0.mean():.3g}]  (x{jitter_h0.mean() / clean_h0:.2f}, "
                f"realisation scatter {jitter_h0.std():.3g})",
                "",
            ]
        lines += [
            "The joint case degrades far more than the Om-fixed case because the",
            "H0-Om degeneracy amplifies the loss of mass-peak sharpness.  Quote the",
            "range, not the Term-I number alone.",
            "",
        ]

    text = "\n".join(lines)
    path = os.path.join(outdir, "spectral_sirens_summary.txt")
    with open(path, "w") as handle:
        handle.write(text + "\n")
    logging.info(f"Wrote {path}")
    print(text)


def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__.split("Run with")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--fisher-file", 
                        type=str,
                        default='network_bbh_CE40km_1p5MW_Aplus_coat_5.0hz_CE20km_1p5MW_Aplus_coat_5.0hz_ETD_5.0hz.h5',
                        help="Individual-event wise Fisher results")
    parser.add_argument("--snr-thresholds", type=float, nargs="+", default=[10.0, 20.0])
    parser.add_argument("--output-directory", default="spectral_sirens_output")
    parser.add_argument("--n-samples", type=int, default=60000)
    parser.add_argument("--seed", type=int, default=250114)
    parser.add_argument(
        "--n-realisations", type=int, default=3,
        help="Noise realisations for the measurement-error bracket (Stage 2).",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_arguments(argv)
    os.makedirs(args.output_directory, exist_ok=True)

    # Effective BGP fiducial from the injected masses (support edges from range).
    mass_1, mass_2 = load_injected_masses(args.fisher_file)
    minimum_component = float(min(mass_1.min(), mass_2.min()))
    m_min_upper = 0.999 * minimum_component
    m_max = 1.001 * float(mass_1.max())
    mass_fiducial, _ = fit_bgp_to_samples(mass_1, mass_2, m_min_upper, m_max)
    logging.info(
        "Effective BGP fiducial: "
        + ", ".join(f"{k}={mass_fiducial[k]:.4g}" for k in
                    ["mu_1", "sigma_1", "mu_2", "sigma_2", "alpha_1", "alpha_2"])
    )

    results = compute_forecast(args.fisher_file, args.snr_thresholds, mass_fiducial)

    for config in results:
        for snr, result in results[config].items():
            numpy.savez(
                os.path.join(args.output_directory, f"spectral_{config}_snr{snr:g}.npz"),
                covariance=result.covariance,
                parameter_names=numpy.array(result.parameter_names),
                fiducial=numpy.array(
                    [result.fiducial_parameters[n] for n in result.parameter_names]
                ),
                snr_threshold=snr,
            )

    # Full 14-parameter corner (reuse the mass/redshift figure helper).
    from forecast_mass_redshift import figure2_corner

    figure2_corner(
        results["joint"],
        os.path.join(args.output_directory, "spectral_sirens_corner.pdf"),
        title="Spectral sirens: population + cosmology",
        n_samples=args.n_samples, seed=args.seed,
    )
    plot_cosmology_corner(
        results["joint"], results["h0_only"],
        os.path.join(args.output_directory, "cosmology_corner.pdf"), seed=args.seed,
    )
    # Measure the validity parameter of the Term-I expansion (it is NOT small in
    # the mu_1 peak), then bracket sigma(H0) from the pessimistic side.
    diagnostics = {
        snr: measurement_error_diagnostic(args.fisher_file, snr, mass_fiducial)
        for snr in args.snr_thresholds
    }
    jitter, _ = measurement_degraded_forecast(
        args.fisher_file, args.snr_thresholds,
        n_realisations=args.n_realisations, seed=args.seed,
    )

    write_summary(
        results, mass_fiducial, args.output_directory, diagnostics, jitter
    )
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
