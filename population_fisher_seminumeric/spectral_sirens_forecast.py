#!/opt/anaconda3/envs/igwn-py311/bin/python
"""
Now include cosmology parameters (H0, Om) as free hyperparameters also, because why not 🤪
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

COSMO_NAMES = ["H0", "Om"]


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

        distance = events["luminosity_distance"]
        redshift = cosmology.redshift_of_distance(distance, H0, Om)
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

        differential_volume = cosmology.differential_comoving_volume(redshift, H0, Om)
        ddL = cosmology.ddL_dz(redshift, H0, Om)
        measure = (
            numpy.log(differential_volume)
            - 3.0 * numpy.log(one_plus_z)
            - numpy.log(ddL)
        )

        return mass_log_prob + rate_shape + measure


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
            redshift, cosmo["H0"], cosmo["Om"]
        ),
        "mass_ratio": mass_2_source / mass_1_source,
    }


def validate_frame(catalogue, events, cosmo):
    """Round-trip checks: the fiducial detector frame must recover the source frame."""
    reconstructed = cosmology.detector_to_source_frame(
        events["mass_1_det"], events["mass_2_det"],
        events["luminosity_distance"], cosmo["H0"], cosmo["Om"],
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

    configs = {
        "joint": mass_fixed,
        "h0_only": {**mass_fixed, "Om": FIDUCIAL_COSMOLOGY["Om"]},
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


def detector_det_score(events, snr, snr_threshold, sigma_rho=1.0, p_det_min=1e-3):
    """
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
    Bound the impact of per-event measurement error on the spectral-siren Fisher
    **without** the numerically-unstable nested-FD correction terms.

    The Gair+2022 Term I is the leading term of a small-measurement-error
    expansion whose validity parameter is, per event and per observable
    direction, ``|H_theta / FIM|`` -- the ratio of the population log-density
    curvature to the per-event measurement Fisher.  Terms II-V are O(H/FIM), so
    when this ratio is << 1 the corrections (and hence the measurement-error
    degradation of the forecast) are negligible and Term I is essentially exact.

    Returns a dict of the per-direction ratios and the implied fractional
    correction bound; also logs a human-readable summary.  This replaces trying
    to *evaluate* the corrections by nested finite differences, which for the
    sharp effective mass peak sits below the FD noise floor and produces spurious
    non-positive-definite Fishers.
    """
    from pop_fisher_higher_order import _H_theta

    fisher_det, events, snr, n_det, _, _ = load_detector_frame_fisher(
        file_path, snr_threshold
    )
    mass_model = BrokenPowerLawPlusTwoPeaksMass(fiducial=mass_fiducial)
    redshift_model = MadauDickinsonRedshift(fiducial=MD_FIDUCIAL)
    model = SpectralSirenModel(mass_model, redshift_model, FIDUCIAL_COSMOLOGY)
    phys_keys = list(model.phys_event_keys)

    hessian = _H_theta(model, events, dict(model.fiducial), phys_keys, 1e-3)
    diagnostics = {}
    for axis, key in enumerate(phys_keys):
        fim = fisher_det[:, axis, axis]
        curvature = numpy.abs(hessian[:, axis, axis])
        ratio = curvature / fim
        sigma = 1.0 / numpy.sqrt(numpy.abs(fim))
        diagnostics[key] = {
            "ratio_median": float(numpy.median(ratio)),
            "ratio_p99": float(numpy.percentile(ratio, 99)),
            "fraction_above_1": float((ratio > 1.0).mean()),
        }
        if key == "luminosity_distance":
            diagnostics[key]["sigma_dL_over_dL_median"] = float(
                numpy.median(sigma / events["luminosity_distance"])
            )

    logging.info(f"[diagnostic] SNR>={snr_threshold}, {n_det} events:")
    for key, value in diagnostics.items():
        extra = (
            f", sigma(dL)/dL median={value['sigma_dL_over_dL_median']:.3f}"
            if "sigma_dL_over_dL_median" in value else ""
        )
        logging.info(
            f"  |H_theta/FIM| {key:>20s}: median={value['ratio_median']:.2e}, "
            f"99th pct={value['ratio_p99']:.2e}, frac>1={value['fraction_above_1']:.4f}{extra}"
        )
    return diagnostics


def compute_full_forecast(file_path, snr_thresholds, mass_fiducial):
    """
    **Stage 2** -- full 5-term spectral-siren Fisher, folding in the per-event
    measurement error via the detector-frame per-event Fisher.

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

    configs = {
        "joint": mass_fixed,
        "h0_only": {**mass_fixed, "Om": FIDUCIAL_COSMOLOGY["Om"]},
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
    """Focused (H0, Om) corner from the joint Fisher, both thresholds."""
    import corner
    import matplotlib.lines
    import pylab

    from plot_utils import new_rcParams

    labels = [r"$H_0\ [\mathrm{km/s/Mpc}]$", r"$\Omega_m$"]
    snr_values = sorted(results_joint)
    colours = {snr_values[0]: "#1f6fd6", snr_values[-1]: "#d1495b"}

    reference = results_joint[snr_values[0]]
    idx = [reference.parameter_names.index(name) for name in COSMO_NAMES]
    mean = numpy.array([reference.fiducial_parameters[name] for name in COSMO_NAMES])

    rng = numpy.random.default_rng(seed)
    draws = {}
    widest = numpy.zeros(2)
    for snr in snr_values:
        covariance = results_joint[snr].covariance[numpy.ix_(idx, idx)]
        sample = rng.multivariate_normal(mean, covariance, size=60000)
        draws[snr] = sample
        widest = numpy.maximum(widest, sample.std(axis=0))
    ranges = [(mean[i] - 4 * widest[i], mean[i] + 4 * widest[i]) for i in range(2)]

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
        # H0-only 1-sigma band at the tightest threshold, to show the Om cost.
        h0_index = results_h0_only[snr_values[0]].parameter_names.index("H0")
        h0_sigma = results_h0_only[snr_values[0]].sigma[h0_index]
        axes = numpy.array(figure.axes).reshape(2, 2)
        axes[0, 0].axvspan(mean[0] - h0_sigma, mean[0] + h0_sigma,
                           color="0.6", alpha=0.25, zorder=0)
        handles.append(matplotlib.lines.Line2D(
            [], [], color="0.6", alpha=0.5, linewidth=8,
            label=rf"$H_0$-only $1\sigma$ (SNR $\geq$ {snr_values[0]:g})"))
        figure.legend(handles=handles, loc="upper right", frameon=False, fontsize=13)

    figure.savefig(outfile, bbox_inches="tight")
    logging.info(f"Wrote {outfile}")
    pylab.close(figure)


def write_summary(results, mass_fiducial, outdir, diagnostics=None):
    lines = ["Spectral-siren cosmology forecast", ""]
    for snr in sorted(results["joint"]):
        joint = results["joint"][snr]
        h0_only = results["h0_only"][snr]
        ih0 = joint.parameter_names.index("H0")
        iom = joint.parameter_names.index("Om")
        correlation = joint.correlation_matrix()[ih0, iom]
        h0_only_sigma = h0_only.sigma[h0_only.parameter_names.index("H0")]
        H0 = FIDUCIAL_COSMOLOGY["H0"]
        Om = FIDUCIAL_COSMOLOGY["Om"]
        lines += [
            f"===== SNR >= {snr:g}  ({joint.n_events} events) =====",
            f"  joint (H0, Om):",
            f"    sigma(H0) = {joint.sigma[ih0]:.4g} km/s/Mpc   "
            f"({100 * joint.sigma[ih0] / H0:.3f}% of H0={H0})",
            f"    sigma(Om) = {joint.sigma[iom]:.4g}            "
            f"({100 * joint.sigma[iom] / Om:.2f}% of Om={Om})",
            f"    corr(H0, Om) = {correlation:+.3f}",
            f"  H0-only (Om fixed):",
            f"    sigma(H0) = {h0_only_sigma:.4g} km/s/Mpc   "
            f"({100 * h0_only_sigma / H0:.3f}% of H0)",
            f"    Om-marginalisation cost: sigma(H0) x {joint.sigma[ih0] / h0_only_sigma:.2f}",
            f"  condition # (joint) = {joint.condition_number:.3g}",
            "",
        ]

    if diagnostics is not None:
        lines += [
            "-" * 70,
            "Per-event measurement-error impact (Stage 2)",
            "-" * 70,
            "The Term-I result above is the leading term of a small-measurement-",
            "error expansion; Terms II-V are O(|H_theta/FIM|) per event.  Below,",
            "|H_theta/FIM| is the ratio of population curvature to per-event",
            "measurement Fisher in each observable direction: << 1 means the",
            "measurement error negligibly degrades the forecast (full ~ Term I).",
            "",
        ]
        for snr in sorted(diagnostics):
            lines.append(f"===== SNR >= {snr:g} =====")
            for key, value in diagnostics[snr].items():
                extra = (
                    f"  [sigma(dL)/dL median = {value['sigma_dL_over_dL_median']:.3f}]"
                    if "sigma_dL_over_dL_median" in value else ""
                )
                lines.append(
                    f"  |H_theta/FIM| {key:>20s}: median {value['ratio_median']:.2e}, "
                    f"99th {value['ratio_p99']:.2e}, frac>1 {value['fraction_above_1']:.4f}{extra}"
                )
            distance = diagnostics[snr]["luminosity_distance"]
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
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_arguments(argv)
    os.makedirs(args.output_directory, exist_ok=True)

    # Effective BGP fiducial from the injected masses (support edges from range).
    mass_1, mass_2 = load_injected_masses(args.fisher_file)
    minimum_component = float(min(mass_1.min(), mass_2.min()))
    m_min = 0.99 * minimum_component
    m_max = 1.001 * float(mass_1.max())
    mass_fiducial, _ = fit_bgp_to_samples(mass_1, mass_2, m_min, m_max)
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
    # Stage 2: bound the per-event measurement-error impact (the corrections are
    # provably sub-percent here, so we report the bound rather than the unstable
    # nested-FD value).
    diagnostics = {
            snr: measurement_error_diagnostic(args.fisher_file, snr, mass_fiducial)
        for snr in args.snr_thresholds
    }

    write_summary(results, mass_fiducial, args.output_directory, diagnostics)
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
