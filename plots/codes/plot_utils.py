import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

import numpy, scipy.stats, matplotlib
import pylab, logging, corner

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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
    "optimal_snr": r"$\rho_{\mathrm{opt}}$",}

GWlatex_labels.update({
    'snr': r'$\rho$',
    'sky_area_90': r'$\Delta\Omega_{90}\ [\mathrm{deg}^2]$',
    'delta_chirp_mass': r'$\Delta\mathcal{M}\ [M_\odot]$',
    'delta_chirp_mass_fractional': r'$\Delta\mathcal{M}/\mathcal{M}$',
    'delta_luminosity_distance': r'$\Delta d_L\ [\mathrm{Gpc}]$',
    'delta_luminosity_distance_fractional': r'$\Delta d_L / d_L$',
    'delta_symmetric_mass_ratio': r'$\Delta\eta$',
    'delta_theta_jn': r'$\Delta\iota\ [\mathrm{rad}]$',
    'delta_ra': r'$\Delta\alpha\ [\mathrm{rad}]$',
    'delta_dec': r'$\Delta\delta\ [\mathrm{rad}]$',
    'delta_psi': r'$\Delta\Psi\ [\mathrm{rad}]$',
    'delta_geocent_time': r'$\Delta t_c\ [\mathrm{s}]$',
    'delta_phase': r'$\Delta\phi\ [\mathrm{rad}]$',
    'delta_chi_1': r'$\Delta\chi_1$',
    'delta_chi_2': r'$\Delta\chi_2$',
    'delta_lambda_tilde': r'$\Delta\tilde{\Lambda}$',
    'delta_delta_lambda_tilde': r'$\Delta\delta\tilde{\Lambda}$',
})

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
def new_rcParams(width='column', aspect_ratio=PHI):
    
    scale_factor = 2

    if width == 'column':
        fig_width_pt = scale_factor*246.0
        # we shouldn't need to adjust this manually,
        # and the same value should work for both widths
        # but it just doesn't, somehow this pixel-optimized value does
        fs = scale_factor*7.96
        
    elif width == 'page':
        fig_width_pt = scale_factor*510.0
        fs = scale_factor*9
    
    inches_per_pt = 1.0/72.27

    fig_width = fig_width_pt*inches_per_pt
    fig_height = fig_width/aspect_ratio

    figsize = (fig_width, fig_height)

    new_params = {}
    new_params['figure.figsize'] = figsize
    new_params['font.size'] = fs
    new_params['text.usetex'] = False
    new_params['text.latex.preamble'] = r'\usepackage{amsmath}\usepackage{amssymb}'
    new_params['axes.labelsize'] = 'medium'
    new_params['font.family'] = 'stixgeneral'
    #new_params['font.serif'] = 'Computer Modern'
    new_params['mathtext.fontset'] = 'stix'
    new_params['xtick.direction'] = 'in'
    new_params['ytick.direction'] = 'in'
    new_params['xtick.minor.visible'] = True
    new_params['ytick.minor.visible'] = True
    new_params['legend.fontsize'] = 'medium'
    new_params['legend.handlelength'] = 1.5

    new_params['grid.linestyle'] = '--'
    new_params['grid.color'] = '#bbbbbb'
    new_params['axes.linewidth'] = 1.0

    new_params['savefig.bbox'] = 'tight'
    new_params['savefig.dpi'] = 300
    new_params['savefig.format'] = 'pdf'

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
    levels=(1 - numpy.exp(-0.5), 1 - numpy.exp(-2), 1 - numpy.exp(-9 / 2.)),
    fill_contours=True,
    max_n_ticks=3,
    title_fmt=".3f",
)

def plot_corner(posterior_samples, 
                parameters = None, 
                corner_kwargs=None, 
                save = False, 
                title=None):                
    if corner_kwargs is None:
        corner_kwargs = CORNER_KWARGS
    if parameters is None:
        parameters = list(posterior_samples.columns)
    samples, labels = [], []
    
    for key in parameters:
        samples.append(posterior_samples[key].values)
        labels.append(GWlatex_labels[key])
    figure = pylab.figure(figsize=(3* len(parameters), 3*len(parameters)))
    corner_kwargs['fig'] = figure
    corner.corner(numpy.asarray(samples).T, labels=labels, **corner_kwargs)
    if title is None:
        title = rf"$\rho = {posterior_samples.iloc[0]['snr']:.2f}$"
    figure.suptitle(title, fontsize=16)
    if save:
        figure.savefig(f'corner_plot_{parameters}.pdf', dpi=100, bbox_inches='tight')
    return figure