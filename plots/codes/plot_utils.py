import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

import numpy, scipy.stats, matplotlib
import pylab, logging, corner

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

GWlatex_labels = {
    "detection_efficiency": r"$\epsilon(z)$",
    "log_likelihood": r"$\ln{\mathcal{L}}$",
    "luminosity_distance": r"$d_{L} [\mathrm{Gpc}]$",
    "geocent_time": r"$t_{c} [\mathrm{s}]$",
    "dec": r"$\delta [\mathrm{rad}]$",
    "ra": r"$\alpha [\mathrm{rad}]$",
    "a_1": r"$a_{1}$",
    "a_2": r"$a_{2}$",
    "phi_jl": r"$\phi_{JL} [\mathrm{rad}]$",
    "phase": r"$\phi [\mathrm{rad}]$",
    "psi": r"$\Psi [\mathrm{rad}]$",
    "iota": r"$\iota [\mathrm{rad}]$",
    "tilt_1": r"$\theta_{1} [\mathrm{rad}]$",
    "tilt_2": r"$\theta_{2} [\mathrm{rad}]$",
    "phi_12": r"$\phi_{12} [\mathrm{rad}]$",
    "mass_2": r"$m_{2} [M_{\odot}]$",
    "mass_1": r"$m_{1} [M_{\odot}]$",
    "total_mass": r"$M [M_{\odot}]$",
    "chirp_mass": r"$\mathcal{M} [M_{\odot}]$",
    "spin_1x": r"$S_{1x}$",
    "spin_1y": r"$S_{1y}$",
    "spin_1z": r"$S_{1z}$",
    "spin_2x": r"$S_{2x}$",
    "spin_2y": r"$S_{2y}$",
    "spin_2z": r"$S_{2z}$",
    "chi_p": r"$\chi_{\mathrm{p}}$",
    "chi_eff": r"$\chi_{\mathrm{eff}}$",
    "mass_ratio": r"$q$",
    "symmetric_mass_ratio": r"$\eta$",
    "inverted_mass_ratio": r"$1/q$",
    "cos_tilt_1": r"$\cos{\theta_{1}}$",
    "cos_tilt_2": r"$\cos{\theta_{2}}$",
    "redshift": r"$z$",
    "mass_1_source": r"$m_{1}^{\mathrm{source}} [M_{\odot}]$",
    "mass_2_source": r"$m_{2}^{\mathrm{source}} [M_{\odot}]$",
    "chirp_mass_source": r"$\mathcal{M}^{\mathrm{source}} [M_{\odot}]$",
    "total_mass_source": r"$M^{\mathrm{source}} [M_{\odot}]$",
    "cos_iota": r"$\cos{\iota}$",
    "theta_jn": r"$\theta_{JN} [\mathrm{rad}]$",
    "cos_theta_jn": r"$\cos{\theta_{JN}}$",
    "lambda_1": r"$\lambda_{1}$",
    "lambda_2": r"$\lambda_{2}$",
    "lambda_tilde": r"$\tilde{\lambda}$",
    "delta_lambda_tilde": r"$\delta\tilde{\lambda}$",
    "matched_filter_snr": r"$\rho_{\mathrm{MF}}$",
    "optimal_snr": r"$\rho_{\mathrm{opt}}$",
}

GWlatex_labels.update(
    {
        "snr": r"$\rho$",
        "sky_area_90": r"$\Delta\Omega_{90}\ [\mathrm{deg}^2]$",
        "delta_chirp_mass": r"$\Delta\mathcal{M}\ [M_\odot]$",
        "delta_chirp_mass_fractional": r"$\Delta\mathcal{M}/\mathcal{M}$",
        "delta_luminosity_distance": r"$\Delta d_L\ [\mathrm{Gpc}]$",
        "delta_luminosity_distance_fractional": r"$\Delta d_L / d_L$",
        "delta_symmetric_mass_ratio": r"$\Delta\eta$",
        "delta_theta_jn": r"$\Delta\iota\ [\mathrm{rad}]$",
        "delta_ra": r"$\Delta\alpha\ [\mathrm{rad}]$",
        "delta_dec": r"$\Delta\delta\ [\mathrm{rad}]$",
        "delta_psi": r"$\Delta\Psi\ [\mathrm{rad}]$",
        "delta_geocent_time": r"$\Delta t_c\ [\mathrm{s}]$",
        "delta_phase": r"$\Delta\phi\ [\mathrm{rad}]$",
        "delta_chi_1": r"$\Delta\chi_1$",
        "delta_chi_2": r"$\Delta\chi_2$",
        "delta_lambda_tilde": r"$\Delta\tilde{\Lambda}$",
        "delta_delta_lambda_tilde": r"$\Delta\delta\tilde{\Lambda}$",
    }
)

# Population (hyper)parameters -- see pop_models.py.  Without these, plot_corner
# falls back to the raw dict key.
GWlatex_labels.update(
    {
        # Redshift models
        "alpha": r"$\alpha$",
        "beta": r"$\beta$",
        "z_peak": r"$z_{\mathrm{p}}$",
        "kappa": r"$\kappa$",
        # Mass models
        "alpha_m": r"$\alpha_{m}$",
        "beta_q": r"$\beta_{q}$",
        "m_min": r"$m_{\mathrm{min}} [M_{\odot}]$",
        "m_max": r"$m_{\mathrm{max}} [M_{\odot}]$",
        "delta_m": r"$\delta_{m} [M_{\odot}]$",
        "lambda_peak": r"$\lambda_{\mathrm{peak}}$",
        "mu_m": r"$\mu_{m} [M_{\odot}]$",
        "sigma_m": r"$\sigma_{m} [M_{\odot}]$",
        # Broken power law + two peaks
        "alpha_1": r"$\alpha_{1}$",
        "alpha_2": r"$\alpha_{2}$",
        "m_break": r"$m_{\mathrm{break}} [M_{\odot}]$",
        "lambda_0": r"$\lambda_{0}$",
        "mu_1": r"$\mu_{1} [M_{\odot}]$",
        "sigma_1": r"$\sigma_{1} [M_{\odot}]$",
        "mu_2": r"$\mu_{2} [M_{\odot}]$",
        "sigma_2": r"$\sigma_{2} [M_{\odot}]$",
    }
)

# Time-delay hyperparameters -- see time_delay_model.py.  The SFR parameters
# carry an explicit subscript because they are the *star-formation-rate* Madau-
# Dickinson parameters, not the effective merger-rate ones above, and the two
# are easy to confuse in a corner plot.
GWlatex_labels.update(
    {
        "tau_min": r"$\tau_{\mathrm{min}} [\mathrm{Gyr}]$",
        "tau_max": r"$\tau_{\mathrm{max}} [\mathrm{Gyr}]$",
        "d": r"$d$",
        "mu_tau": r"$\mu_{\tau} [\mathrm{Gyr}]$",
        "sigma_tau": r"$\sigma_{\tau} [\mathrm{Gyr}]$",
        "t_ln": r"$t_{\ln} [\mathrm{Gyr}]$",
        "sigma_ln": r"$\sigma_{\ln}$",
        "gamma_sfr": r"$\gamma_{\mathrm{SFR}}$",
        "kappa_sfr": r"$\kappa_{\mathrm{SFR}}$",
        "z_peak_sfr": r"$z_{\mathrm{p,SFR}}$",
    }
)


def create_custom_colormap(colors):
    """Create a custom colormap from a list of colors.

    Parameters
    ----------
    colors : list
        List of colors (as RGB tuples or hex strings).

    Returns
    -------
    LinearSegmentedColormap
        The resulting colormap.

    Examples
    --------
    >>> colors = ["blue", "white", "red"]
    >>> cmap = create_custom_colormap(colors)
    >>> pylab.imshow([np.linspace(0, 1, 256)], aspect='auto, cmap=cmap)
    >>> pylab.axis('off')
    >>> pylab.show()
    """
    return LinearSegmentedColormap.from_list("blue_white_red", colors, N=256)


PHI = 1.618033988749895


def new_rcParams(width="column", aspect_ratio=PHI):

    scale_factor = 2

    if width == "column":
        fig_width_pt = scale_factor * 246.0
        # we shouldn't need to adjust this manually,
        # and the same value should work for both widths
        # but it just doesn't, somehow this pixel-optimized value does
        fs = scale_factor * 7.96

    elif width == "page":
        fig_width_pt = scale_factor * 510.0
        fs = scale_factor * 9

    inches_per_pt = 1.0 / 72.27

    fig_width = fig_width_pt * inches_per_pt
    fig_height = fig_width / aspect_ratio

    figsize = (fig_width, fig_height)

    new_params = {}
    new_params["figure.figsize"] = figsize
    new_params["font.size"] = fs
    new_params["text.usetex"] = False
    new_params["text.latex.preamble"] = r"\usepackage{amsmath}\usepackage{amssymb}"
    new_params["axes.labelsize"] = "medium"
    new_params["font.family"] = "stixgeneral"
    # new_params['font.serif'] = 'Computer Modern'
    new_params["mathtext.fontset"] = "stix"
    new_params["xtick.direction"] = "in"
    new_params["ytick.direction"] = "in"
    new_params["xtick.minor.visible"] = True
    new_params["ytick.minor.visible"] = True
    new_params["legend.fontsize"] = "medium"
    new_params["legend.handlelength"] = 1.5

    new_params["grid.linestyle"] = "--"
    new_params["grid.color"] = "#bbbbbb"
    new_params["axes.linewidth"] = 1.0

    new_params["savefig.bbox"] = "tight"
    new_params["savefig.dpi"] = 300
    new_params["savefig.format"] = "pdf"

    return new_params

    # all other font sizes should be relative: if fs is 10, then
    # xx-small =  5.79
    # x-small =  6.94
    # small =  8.33
    # medium = 10.0
    # large = 12.0
    # x-large = 14.4
    # xx-large = 17.28


import re, sys, os

CE_RE = re.compile(
    r"^CE(?P<km>\d+)km_"
    r"(?P<power_int>\d+)p(?P<power_dec>\d+)MW_"
    r"(?P<design>Aplus|aLIGO)_coat$"
)

DESIGN_MAP = {
    "Aplus": "A+",
    "aLIGO": "aLIGO",
}

DETECTOR_MAP = {
    "ETD": "ET",
    "ET2L": "ET2L",
    "LIA+": "LIA+",
    "LHA+": "LHA+",
    "LLA+": "LLA+",
    "LIAsharp": "LIA#",
    "LHAsharp": "LHA#",
    "LLAsharp": "LLA#",
}


def parse_detector(token):
    token = token.strip()

    ce_match = CE_RE.match(token)
    if ce_match:
        km = ce_match.group("km")
        power = f"{ce_match.group('power_int')}.{ce_match.group('power_dec')}MW"
        design = DESIGN_MAP[ce_match.group("design")]

        name = f"CE{km}"

        # Matches your example:
        # CE40 gets the design label, CE20 does not
        if km == "40":
            return f"{name} ({power} {design})"
        else:
            return f"{name} ({power})"

    return DETECTOR_MAP.get(token, token)


def parse_combo(combo):
    tokens = [t.strip() for t in combo.split(",") if t.strip()]

    ce_parts = []
    other_parts = []

    for token in tokens:
        parsed = parse_detector(token)

        if parsed.startswith("CE"):
            ce_parts.append(parsed)
        else:
            other_parts.append(parsed)

    output_parts = []

    if ce_parts:
        output_parts.append(" ".join(ce_parts))

    output_parts.extend(other_parts)

    return ", ".join(output_parts)


CORNER_KWARGS = dict(
    bins=50,
    smooth=0.99,
    plot_datapoints=True,
    label_kwargs=dict(fontsize=16),
    show_titles=True,
    title_kwargs=dict(fontsize=16),
    plot_density=False,
    title_quantiles=[0.16, 0.5, 0.84],
    levels=(1 - numpy.exp(-0.5), 1 - numpy.exp(-2), 1 - numpy.exp(-9 / 2.0)),
    fill_contours=True,
    max_n_ticks=3,
    title_fmt=".3f",
)


def plot_corner(
    posterior_samples, parameters=None, corner_kwargs=None, save=False, title=None):

    pylab.rcParams.update(new_rcParams())
    if corner_kwargs is None:
        corner_kwargs = CORNER_KWARGS
    if parameters is None:
        parameters = list(posterior_samples.columns)
    samples, labels = [], []

    for key in parameters:
        samples.append(posterior_samples[key].values)
        try:
            labels.append(GWlatex_labels[key])
        except KeyError:
            labels.append(key)
    figure = pylab.figure(figsize=(3 * len(parameters), 3 * len(parameters)))
    corner_kwargs["fig"] = figure
    corner.corner(numpy.asarray(samples).T, labels=labels, **corner_kwargs)
    if title is None and "snr" in posterior_samples:
        # Per-event posteriors carry an 'snr' column; hyperparameter posteriors
        # do not, and then there is nothing sensible to auto-title with.
        title = rf"$\rho = {posterior_samples.iloc[0]['snr']:.2f}$"
    if title is not None:
        # Above y=1 so it clears the per-parameter titles corner puts on the
        # top-left diagonal panel; savefig(bbox_inches='tight') keeps it in.
        figure.suptitle(title, fontsize=16, y=1.02)
    if save:
        figure.savefig(f"corner_plot_{parameters}.pdf", dpi=100, bbox_inches="tight")
    return figure


def plot_redshift_distribution(
    model,
    hyperparameter_samples,
    redshift_grid=None,
    credible_interval=0.9,
    color="#ca2000",
    title=None,
    max_curves=2000,
    save=False,
):
    """
    Merger-rate density p(z) implied by a set of hyperparameter samples.

    Each sample is turned into a curve p(z) ∝ exp(model.log_prob), normalised
    to unit integral over ``redshift_grid``.  The pointwise median is drawn as
    a solid line and the central ``credible_interval`` as a shaded band.

    Parameters
    ----------
    model : pop_models.PopModel
        A redshift model, i.e. one whose only physical event parameter is
        'redshift'.
    hyperparameter_samples : pandas.DataFrame
        Columns are hyperparameter names, rows are samples (e.g. from a Fisher
        covariance).
    redshift_grid : ndarray, optional
        Defaults to ``numpy.linspace(1e-2, 10, 400)``.
    credible_interval : float, optional
        Width of the shaded band (default 0.9, i.e. the 5th-95th percentiles).
    color : str, optional
        Colour of the median curve; the band is the same colour, lightened.
    title : str, optional
    max_curves : int, optional
        Cap on the number of samples evaluated.  ``log_prob`` calls astropy's
        ``differential_comoving_volume`` once per sample, so the cost is linear
        in the number of curves; 2000 is a couple of seconds.
    save : bool, optional
        Write ``redshift_distribution.pdf`` alongside returning the figure.

    Returns
    -------
    matplotlib figure
    """
    physical_keys = list(getattr(model, "phys_event_keys", []) or [])
    if physical_keys != ["redshift"]:
        raise ValueError(
            f"plot_redshift_distribution needs a redshift-only model; "
            f"{type(model).__name__} depends on {physical_keys}."
        )

    if redshift_grid is None:
        redshift_grid = numpy.linspace(1e-2, 10.0, 400)

    parameter_names = list(hyperparameter_samples.columns)
    values = numpy.asarray(hyperparameter_samples[parameter_names])
    if len(values) > max_curves:
        logging.info(
            f"plot_redshift_distribution: using the first {max_curves} of "
            f"{len(values)} samples."
        )
        values = values[:max_curves]

    events = {"redshift": redshift_grid}
    curves = numpy.empty((len(values), len(redshift_grid)))
    for index, row in enumerate(values):
        log_probability = model.log_prob(
            events, dict(zip(parameter_names, row))
        )
        # Subtract the max before exponentiating; the normalisation below
        # removes the resulting constant anyway.
        probability = numpy.exp(log_probability - numpy.max(log_probability))
        curves[index] = probability / numpy.trapz(probability, redshift_grid)

    usable = numpy.all(numpy.isfinite(curves), axis=1)
    if not usable.all():
        logging.warning(
            f"plot_redshift_distribution: dropping {int((~usable).sum())} / "
            f"{len(curves)} samples with a non-finite p(z) "
            f"(outside the model support)."
        )
    curves = curves[usable]

    lower_percentile = 100.0 * (1.0 - credible_interval) / 2.0
    upper_percentile = 100.0 - lower_percentile
    median = numpy.median(curves, axis=0)
    lower, upper = numpy.percentile(
        curves, [lower_percentile, upper_percentile], axis=0
    )

    with matplotlib.rc_context(new_rcParams()):
        figure, axes = pylab.subplots()
        axes.fill_between(
            redshift_grid,
            lower,
            upper,
            color=color,
            alpha=0.2,
            linewidth=0,
            label=f"{100 * credible_interval:.0f}% credible region",
        )
        axes.plot(redshift_grid, median, color=color, linewidth=2, label="median")
        axes.set_xlabel(GWlatex_labels["redshift"])
        axes.set_ylabel(r"$p(z)$")
        axes.set_xlim(redshift_grid.min(), redshift_grid.max())
        axes.set_ylim(bottom=0.0)
        axes.legend(frameon=False)
        if title is not None:
            axes.set_title(title)

    if save:
        figure.savefig("redshift_distribution.pdf", bbox_inches="tight")
    return figure
