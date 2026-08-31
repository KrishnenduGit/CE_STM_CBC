#!/opt/anaconda3/envs/igwn-py311/bin/python
"""
GWTC-5 (LVK, O4b default) vs ET+CE Term-I forecast: per-parameter uncertainty
comparison for the nine shared mass hyperparameters of the broken power law +
two peaks model.

LVK side: sigma = (q84 - q16)/2 of the hyperparameter posterior in the bilby
result file (247 events).  ET+CE side: sqrt(diag(covariance)) of the Term-I
population Fisher from ``forecast_mass_redshift.py`` (SNR >= 10 by default).

Output is a two-panel forest figure -- fractional uncertainty sigma/|centre|
per parameter (each side normalised by its own central value), and the
improvement factor sigma_LVK / sigma_ET+CE -- plus a text table.

Only mass parameters are compared because the redshift models are different
"""

import argparse
import logging
import os

import matplotlib

matplotlib.use("Agg")

import h5py
import numpy

from plot_utils import GWlatex_labels, new_rcParams

logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")

GWTC5_DEFAULT = (
    "/Users/kchandra/projects/data_dump/"
    "o4b_default_mass_TwoPeakBrokenPowerLawSmoothedMassDistribution_"
    "redshift_PowerLawRedshift_magnitude_iid_spin_magnitude_gaussian_"
    "tilt_iid_spin_orientation_result.hdf5"
)

# forecast name -> GWTC-5 posterior column (both models are BPL + two peaks).
NAME_MAPPING = {
    "alpha_1": "alpha_1",
    "alpha_2": "alpha_2",
    "lambda_0": "lam_0",
    "lambda_1": "lam_1",
    "mu_1": "mpp_1",
    "sigma_1": "sigpp_1",
    "mu_2": "mpp_2",
    "sigma_2": "sigpp_2",
    "beta_q": "beta",
}

COLOUR_FORECAST = "#1f6fd6"
COLOUR_LVK = "#d1495b"


def load_lvk(path):
    """Median and sigma = (q84 - q16)/2 per mapped GWTC-5 hyperparameter."""
    out = {}
    with h5py.File(path, "r") as handle:
        posterior = handle["posterior"]
        for name, column in NAME_MAPPING.items():
            samples = posterior[column][:]
            q16, q50, q84 = numpy.percentile(samples, [16, 50, 84])
            out[name] = {"centre": q50, "sigma": 0.5 * (q84 - q16),
                         "n_samples": len(samples)}
    return out


def load_forecast(path):
    """Fiducial and sigma = sqrt(diag(cov)) from a forecast npz cache."""
    data = numpy.load(path, allow_pickle=False)
    names = [str(name) for name in data["parameter_names"]]
    sigma = numpy.sqrt(numpy.diag(data["covariance"]))
    fiducial = data["fiducial"]
    return {
        name: {"centre": fiducial[i], "sigma": sigma[i]}
        for i, name in enumerate(names)
    }, float(data["snr_threshold"])


def comparison_table(lvk, forecast):
    """One row per parameter: centres, sigmas, fractional sigmas, improvement."""
    rows = []
    for name in NAME_MAPPING:
        lvk_entry = lvk[name]
        forecast_entry = forecast[name]
        rows.append({
            "name": name,
            "lvk_centre": lvk_entry["centre"],
            "lvk_sigma": lvk_entry["sigma"],
            "lvk_frac": lvk_entry["sigma"] / abs(lvk_entry["centre"]),
            "fc_centre": forecast_entry["centre"],
            "fc_sigma": forecast_entry["sigma"],
            "fc_frac": forecast_entry["sigma"] / abs(forecast_entry["centre"]),
            "improvement": lvk_entry["sigma"] / forecast_entry["sigma"],
        })
    return rows


def plot_comparison(rows, snr_threshold, outfile):
    """Two-panel forest: fractional uncertainty (dumbbell) + improvement factor."""
    import pylab

    n_rows = len(rows)
    positions = numpy.arange(n_rows)[::-1]  # first parameter on top
    labels = [GWlatex_labels.get(row["name"], row["name"]) for row in rows]

    with matplotlib.rc_context(new_rcParams(width="page", aspect_ratio=1.5)):
        figure, (left, right) = pylab.subplots(
            1, 2, sharey=True, gridspec_kw={"width_ratios": [1.6, 1.0], "wspace": 0.06},
        )

        for row, y in zip(rows, positions):
            left.plot([row["fc_frac"], row["lvk_frac"]], [y, y],
                      color="0.75", linewidth=1.4, zorder=1)
        left.scatter([row["lvk_frac"] for row in rows], positions,
                     s=55, color=COLOUR_LVK, zorder=2, label="LVK (GWTC-5, 247 events)")
        left.scatter([row["fc_frac"] for row in rows], positions,
                     s=55, color=COLOUR_FORECAST, zorder=3,
                     label=rf"ET+CE forecast (SNR $\geq$ {snr_threshold:g})")
        left.set_xscale("log")
        left.set_xlabel(r"fractional uncertainty $\sigma_\theta / |\theta|$")
        left.set_yticks(positions)
        left.set_yticklabels(labels)
        left.legend(frameon=False, loc="upper left", fontsize=9)

        improvements = [row["improvement"] for row in rows]
        right.scatter(improvements, positions, s=55, color="0.25", zorder=3)
        for value, y in zip(improvements, positions):
            right.annotate(f"{value:.0f}", (value, y), textcoords="offset points",
                           xytext=(0, 7), ha="center", fontsize=8, color="0.35")
        right.set_xscale("log")
        right.set_xlim(1.0, 10 ** numpy.ceil(numpy.log10(max(improvements))))
        right.axvline(1.0, color="0.8", linewidth=1.0, zorder=1)
        right.set_xlabel(
            r"improvement $\sigma_\theta^{\rm LVK} / \sigma_\theta^{\rm ET+CE}$")

        for axes in (left, right):
            axes.grid(axis="x", color="0.9", linewidth=0.6, zorder=0)
            axes.set_ylim(-0.6, n_rows - 0.4)
            axes.tick_params(axis="y", length=0)

    figure.savefig(outfile, bbox_inches="tight")
    logging.info(f"Wrote {outfile}")
    pylab.close(figure)


def write_table(rows, snr_threshold, n_samples, outfile):
    lines = [
        "GWTC-5 (LVK O4b default) vs ET+CE Term-I forecast, mass hyperparameters",
        "",
        f"LVK: 247 events, {n_samples} posterior samples; sigma = (q84 - q16)/2.",
        f"ET+CE: Term-I population Fisher, SNR >= {snr_threshold:g}; "
        "sigma = sqrt(diag(cov)).",
        "",
        f"{'parameter':>10s} {'LVK centre':>11s} {'LVK sigma':>10s} "
        f"{'fc centre':>10s} {'fc sigma':>10s} {'improvement':>12s}",
        "-" * 68,
    ]
    for row in rows:
        lines.append(
            f"{row['name']:>10s} {row['lvk_centre']:>11.4g} {row['lvk_sigma']:>10.3g} "
            f"{row['fc_centre']:>10.4g} {row['fc_sigma']:>10.3g} "
            f"{row['improvement']:>12.1f}"
        )
    lines += [
        "",
        "Caveats: only the nine shared mass-shape parameters are compared -- the",
        "redshift models differ (GWTC-5 power-law lamb vs forecast Madau-Dickinson).",
        "The forecast pins m_min, m_max, m_break, delta_m while GWTC-5 samples its",
        "analogues, so the forecast is slightly optimistic.  Central values differ",
        "between the two sides; each fractional sigma is normalised to its own.",
    ]
    text = "\n".join(lines)
    with open(outfile, "w") as handle:
        handle.write(text + "\n")
    logging.info(f"Wrote {outfile}")
    print(text)


def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--gwtc5-file", default=GWTC5_DEFAULT,
                        help="GWTC-5 bilby result HDF5 with a posterior/ group")
    parser.add_argument(
        "--forecast-file",
        default="forecast_mass_redshift_output/forecast_mass_snr10.npz",
        help="Mass-model covariance cache from forecast_mass_redshift.py")
    parser.add_argument("--output-directory", default="forecast_mass_redshift_output")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_arguments(argv)
    os.makedirs(args.output_directory, exist_ok=True)

    lvk = load_lvk(args.gwtc5_file)
    forecast, snr_threshold = load_forecast(args.forecast_file)
    rows = comparison_table(lvk, forecast)

    plot_comparison(
        rows, snr_threshold,
        os.path.join(args.output_directory, "gwtc5_forecast_comparison.pdf"))
    write_table(
        rows, snr_threshold, lvk["alpha_1"]["n_samples"],
        os.path.join(args.output_directory, "gwtc5_forecast_comparison.txt"))
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
