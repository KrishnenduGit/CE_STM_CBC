#!/opt/anaconda3/envs/igwn-py311/bin/python
"""
Models
------
* Mass: ``pop_models.BrokenPowerLawPlusTwoPeaksMass`` (BGP).  The catalogue was
  drawn from a much richer "fullpop" model (NS + BH + dip + upper gap + peaks),
  so BGP is an *effective* fit -- exactly as the effective Madau-Dickinson (MD)
  form is for the redshift sector.  The fiducial is obtained by a maximum-
  likelihood fit of BGP to the *injected* component masses.
* Redshift: ``pop_models.MadauDickinsonRedshift``, at the effective MD
  parameters of the injected redshifts.
* Joint: ``pop_models.JointModel(BGP, MD)``.

Fixed vs free
-------------
The support-discontinuous BGP parameters are pinned: ``m_min, m_max`` from the
injected component-mass *range* (they bracket every detected event), and
``m_break, delta_m`` at their injection-fit values.  The nine smooth shape
parameters (``alpha_1, alpha_2, lambda_0, lambda_1, mu_1, sigma_1, mu_2,
sigma_2, beta_q``) are free.

Fisher
------
Term I (score) only -- ``pop_fisher.compute_pop_fisher``.  At SNR>=10 selection
is weak (P_det ~ 0.96), so Term I dominates, and it needs only the event source
parameters (``with_fisher=False``), not the per-event measurement covariance.
The broadening is shown by evaluating at SNR>10 and SNR>20.
"""

import argparse
import logging
import os

import matplotlib

matplotlib.use("Agg")  # no X server required

import numpy
import pandas

from load import load_injected_masses, load_population_catalogue
from pop_fisher import compute_pop_fisher
from pop_models import (
    BrokenPowerLawPlusTwoPeaksMass,
    JointModel,
    MadauDickinsonRedshift,
    _normalisation_grid,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")

# Effective MD parameters from the maximum-likelihood fit to the injected
# redshifts (alpha + beta = kappa = 5.364).
MD_FIDUCIAL = {"alpha": 1.881, "beta": 3.483, "z_peak": 1.818}

# BGP parameters pinned in the Fisher (the discontinuous / support ones).
MASS_FIXED_NAMES = ["m_min", "m_max", "m_break", "delta_m"]
# BGP smooth shape parameters, free in the fit and in the Fisher.
MASS_FREE_NAMES = [
    "alpha_1", "alpha_2", "lambda_0", "lambda_1",
    "mu_1", "sigma_1", "mu_2", "sigma_2", "beta_q",
]

# ===========================================================================
# Fiducial: fit BGP to the injected masses
# ===========================================================================


def fit_bgp_to_samples(mass_1, mass_2, m_min_upper, m_max):
    """
    Maximum-likelihood BGP parameters for a set of injected component masses.

    Parameters
    ----------
    mass_1, mass_2 : (N,) ndarray
        Injected source-frame component masses.
    m_min_upper : float
        Upper bound for the fitted ``m_min``; must be below ``mass_2.min()``.
    m_max : float
        Fixed upper support edge.

    Returns
    -------
    fiducial : dict
        Full 13-parameter BGP fiducial (free fit + fixed edges).
    result : scipy OptimizeResult
    """
    from scipy.optimize import minimize

    events = {
        "mass_1_source": mass_1,
        "mass_2_source": mass_2,
        "mass_ratio": mass_2 / mass_1,
    }
    model = BrokenPowerLawPlusTwoPeaksMass()

    def unpack(vector):
        (alpha_1, alpha_2, m_break, w0, w1, mu_1, sigma_1,
         mu_2, sigma_2, beta_q, delta_m, m_min) = vector
        weights = numpy.exp(numpy.array([w0, w1, 0.0]))
        weights /= weights.sum()
        return {
            "alpha_1": alpha_1, "alpha_2": alpha_2, "m_break": m_break,
            "lambda_0": weights[0], "lambda_1": weights[1],
            "mu_1": mu_1, "sigma_1": sigma_1, "mu_2": mu_2, "sigma_2": sigma_2,
            "beta_q": beta_q, "m_min": m_min, "delta_m": delta_m, "m_max": m_max,
        }

    def negative_log_likelihood(vector):
        log_prob = model.log_prob(events, unpack(vector))
        if not numpy.all(numpy.isfinite(log_prob)):
            return 1e12
        return -float(numpy.mean(log_prob))

    m_min_lower = 0.5 * m_min_upper
    initial_guess = [
        2.0, 4.0, 30.0, 1.5, -0.5, 9.0, 1.5, 35.0, 8.0, 1.4, 1.0,
        0.98 * m_min_upper,
    ]
    bounds = [
        (-10.0, 10.0),                       # alpha_1
        (-10.0, 10.0),                       # alpha_2
        (1.2 * m_min_upper, 0.9 * m_max),    # m_break
        (-15.0, 15.0),                       # w0
        (-15.0, 15.0),                       # w1
        (m_min_lower, m_max),                # mu_1
        (0.3, 40.0),                         # sigma_1
        (m_min_lower, m_max),                # mu_2
        (0.3, 40.0),                         # sigma_2
        (-4.0, 8.0),                         # beta_q
        (1e-3, 20.0),                        # delta_m
        (m_min_lower, m_min_upper),          # m_min
    ]
    result = minimize(
        negative_log_likelihood,
        initial_guess,
        method="L-BFGS-B", # dual_annealing
        bounds=bounds,
        options=dict(maxiter=3000, ftol=1e-12),
    )
    return unpack(result.x), result


def primary_mass_density(mass_grid, fiducial):
    model = BrokenPowerLawPlusTwoPeaksMass()
    density, _ = model._primary_numerator(mass_grid, fiducial, derivatives=False)
    grid = _normalisation_grid(
        fiducial["m_min"], fiducial["m_max"], fiducial["delta_m"],
        fiducial["m_break"],
    )
    reference, _ = model._primary_numerator(grid, fiducial, derivatives=False)
    norm = numpy.trapz(reference, grid)
    return density / norm if norm > 0 else density


# ===========================================================================
# Fisher
# ===========================================================================


def compute_forecast(file_path, snr_thresholds, mass_fiducial):
    """
    Term-I population Fisher for the mass, redshift and joint models at each SNR
    threshold.  Returns ``{model_key: {snr: PopFisherResult}}``.
    """
    mass_fixed = {name: mass_fiducial[name] for name in MASS_FIXED_NAMES}

    bgp = BrokenPowerLawPlusTwoPeaksMass(fiducial=mass_fiducial)
    md = MadauDickinsonRedshift(fiducial=MD_FIDUCIAL)
    joint = JointModel(bgp, md)

    specs = {
        "mass": (bgp, mass_fixed),
        "redshift": (md, {}),
        "joint": (joint, mass_fixed),
    }
    results = {key: {} for key in specs}
    for snr in snr_thresholds:
        catalogue = load_population_catalogue(
            file_path, snr_threshold=snr, with_fisher=False
        )
        logging.info(
            f"SNR>={snr}: {catalogue.number_detected} analysis events, "
            f"P_det={catalogue.p_det:.4f}"
        )
        for key, (model, fixed) in specs.items():
            result = compute_pop_fisher(
                model,
                catalogue.events,
                fixed_parameters=fixed,
                n_total=catalogue.number_total,
            )
            results[key][snr] = result
            logging.info(
                f"  {key:>9s} SNR>={snr}: condition # = "
                f"{result.condition_number:.3g}, {len(result.parameter_names)} params"
            )
    return results


# ===========================================================================
# Plots
# ===========================================================================


def _samples(result, size, seed):
    """Fisher-posterior samples N(fiducial, covariance) as an (size, P) array."""
    rng = numpy.random.default_rng(seed)
    mean = numpy.array([result.fiducial_parameters[n] for n in result.parameter_names])
    return rng.multivariate_normal(mean, result.covariance, size=size), mean


def figure2_corner(results_by_snr, outfile, title, n_samples=60000, seed=250114):
    """
    Figure-2-style corner: 67% and 90% contours, SNR>10 (dark) vs SNR>20 (light)
    overlaid, injected (fiducial) values as dashed truth lines.

    Uses ``plot_utils.plot_corner`` for the first threshold (it already draws the
    nested credible levels and resolves the latex labels) and overlays the
    second threshold with a matched-range ``corner.corner`` call.
    """
    import corner
    import matplotlib.lines
    import pylab

    from plot_utils import CORNER_KWARGS, GWlatex_labels, plot_corner

    snr_values = sorted(results_by_snr)
    reference = results_by_snr[snr_values[0]]
    names = reference.parameter_names
    labels = [GWlatex_labels.get(name, name) for name in names]

    samples = {}
    mean = None
    widest = numpy.zeros(len(names))
    for snr in snr_values:
        draws, mean = _samples(results_by_snr[snr], n_samples, seed)
        samples[snr] = draws
        widest = numpy.maximum(widest, draws.std(axis=0))
    # Common axis range so the two overlays line up (corner ranges per-call).
    ranges = [(mean[i] - 4.0 * widest[i], mean[i] + 4.0 * widest[i])
              for i in range(len(names))]

    colours = {snr_values[0]: "#1f6fd6", snr_values[-1]: "#d1495b"}

    base_kwargs = {
        **CORNER_KWARGS,
        "levels": (0.67, 0.90),
        "truths": list(mean),
        "truth_color": "0.2",
        "range": ranges,
        "plot_datapoints": False,
        "color": colours[snr_values[0]],
    }
    figure = plot_corner(
        pandas.DataFrame(samples[snr_values[0]], columns=names),
        corner_kwargs=dict(base_kwargs),
        title=title,
    )
    for snr in snr_values[1:]:
        corner.corner(
            samples[snr],
            fig=figure,
            labels=labels,
            color=colours.get(snr, "0.5"),
            levels=(0.67, 0.90),
            range=ranges,
            bins=CORNER_KWARGS["bins"],
            smooth=CORNER_KWARGS["smooth"],
            plot_datapoints=False,
            plot_density=False,
            fill_contours=True,
            show_titles=False,
            max_n_ticks=CORNER_KWARGS["max_n_ticks"],
        )

    figure.legend(
        handles=[
            matplotlib.lines.Line2D([], [], color=colours[snr_values[0]],
                                    label=rf"SNR $\geq$ {snr_values[0]:g}"),
            matplotlib.lines.Line2D([], [], color=colours[snr_values[-1]],
                                    label=rf"SNR $\geq$ {snr_values[-1]:g}"),
        ],
        loc="upper right",
        frameon=False,
        fontsize=16,
    )
    figure.savefig(outfile, bbox_inches="tight")
    logging.info(f"Wrote {outfile}")
    pylab.close(figure)
    return figure


def plot_mass_fit_check(mass_1, mass_fiducial, outfile):
    """Fitted BGP p(m1) over the injected primary-mass histogram."""
    import pylab

    from plot_utils import new_rcParams

    m_min, m_max = mass_fiducial["m_min"], mass_fiducial["m_max"]
    grid = numpy.linspace(m_min, min(m_max, 120.0), 2000)
    density = primary_mass_density(grid, mass_fiducial)

    with matplotlib.rc_context(new_rcParams(width="page")):
        figure, axes = pylab.subplots()
        axes.hist(
            mass_1[mass_1 <= 120.0], bins=120, density=True, histtype="stepfilled",
            color="0.8", edgecolor="0.5", label="injected $m_1$",
        )
        axes.plot(grid, density, color="#d1495b", linewidth=2.2,
                  label="fitted BGP $p(m_1)$")
        axes.set_xlabel(r"$m_1\ [M_\odot]$")
        axes.set_ylabel(r"$p(m_1)$")
        axes.set_xlim(0.0, 120.0)
        axes.legend(frameon=False)

    figure.savefig(outfile, bbox_inches="tight")
    logging.info(f"Wrote {outfile}")
    pylab.close(figure)


def write_summary(results, mass_fiducial, outdir):
    """Text table of 1-sigma per parameter, both thresholds, all three models."""
    lines = ["Joint mass+redshift population-Fisher forecast (Term I)", ""]
    lines.append("Effective BGP fiducial (fit to injected masses):")
    for name in MASS_FREE_NAMES + MASS_FIXED_NAMES:
        lines.append(f"    {name:>10s} = {mass_fiducial[name]:.5g}")
    lines.append("")
    for key in ("mass", "redshift", "joint"):
        for snr in sorted(results[key]):
            lines.append(f"===== {key} model, SNR >= {snr:g} =====")
            lines.append(results[key][snr].summary())
            lines.append("")
    text = "\n".join(lines)
    path = os.path.join(outdir, "forecast_summary.txt")
    with open(path, "w") as handle:
        handle.write(text + "\n")
    logging.info(f"Wrote {path}")
    print(text)


# ===========================================================================
# Entry point
# ===========================================================================


def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--fisher-file",
                        type=str, 
                        default='network_bbh_CE40km_1p5MW_Aplus_coat_5.0hz_CE20km_1p5MW_Aplus_coat_5.0hz_ETD_5.0hz.h5',
                        help="Individual-event wise Fisher results")
    parser.add_argument("--snr-thresholds", 
                        type=float, 
                        nargs="+", 
                        default=[10.0, 20.0], 
                        help="SNR thresholds for the population Fisher")
    parser.add_argument("--output-directory", 
                        default="forecast_mass_redshift_output",
                        help="directory to write the forecast results")
    parser.add_argument("--n-samples", 
                        type=int,
                        default=60000,
                        help="samples drawn from each Fisher posterior for the corners")
    parser.add_argument("--seed", 
                        type=int, 
                        default=250114)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_arguments(argv)
    os.makedirs(args.output_directory, exist_ok=True)

    # 1. Fiducial from the injected masses; support edges from the data range.
    mass_1, mass_2 = load_injected_masses(args.fisher_file)
    minimum_component = float(min(mass_1.min(), mass_2.min()))
    m_min_upper = 0.999 * minimum_component
    m_max = 1.001 * float(mass_1.max())       # bracket the heaviest primary
    logging.info(
        f"Injected masses: {len(mass_1)} injections, "
        f"m_min fitted below {m_min_upper:.3f}, m_max set to {m_max:.3f}"
    )
    mass_fiducial, fit_result = fit_bgp_to_samples(
        mass_1, mass_2, m_min_upper, m_max
    )
    logging.info(f"Fitted m_min = {mass_fiducial['m_min']:.4f}")
    logging.info("Effective BGP fiducial: " + ", ".join(
        f"{name}={mass_fiducial[name]:.4g}" for name in MASS_FREE_NAMES
    ))
    logging.info(f"BGP fit success={fit_result.success}, "
                 f"mean ln L={-fit_result.fun:.4f}")

    plot_mass_fit_check(
        mass_1, mass_fiducial, os.path.join(args.output_directory, "mass_fit_check.pdf")
    )

    # 2. Term-I Fisher for mass, redshift, joint at each threshold.
    results = compute_forecast(args.fisher_file, args.snr_thresholds, mass_fiducial)

    # Cache the covariances.
    for key in results:
        for snr, result in results[key].items():
            numpy.savez(
                os.path.join(args.output_directory, f"forecast_{key}_snr{snr:g}.npz"),
                covariance=result.covariance,
                parameter_names=numpy.array(result.parameter_names),
                fiducial=numpy.array(
                    [result.fiducial_parameters[n] for n in result.parameter_names]
                ),
                snr_threshold=snr,
            )

    # 3. Figure-2-style corners + fit check + summary.
    titles = {
        "joint": "Joint mass + redshift",
        "mass": "Mass (BGP)",
        "redshift": "Redshift (Madau-Dickinson)",
    }
    for key in ("joint", "mass", "redshift"):
        figure2_corner(
            results[key],
            os.path.join(args.output_directory, f"{key}_corner.pdf"),
            title=titles[key],
            n_samples=args.n_samples,
            seed=args.seed,
        )

    write_summary(results, mass_fiducial, args.output_directory)
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
