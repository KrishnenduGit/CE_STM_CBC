import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

import numpy, scipy.stats, matplotlib
import pylab

GWlatex_labels = {
    "log_likelihood": r"$\ln{\mathcal{L}}$",
    "luminosity_distance": r"$d_{L} [\mathrm{Mpc}]$",
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
    "matched_filter_snr": r"$\rho_{\mathrm{MF}}$",
    "optimal_snr": r"$\rho_{\mathrm{opt}}$",}


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
    >>> plt.imshow([np.linspace(0, 1, 256)], aspect='auto, cmap=cmap)
    >>> plt.axis('off')
    >>> plt.show()
    """
    return LinearSegmentedColormap.from_list("blue_white_red", colors, N=256)


def set_plot_style():
    """
    Set the plotting style for matplotlib

    Returns
    -------
    None
    """
    sns.set_context("talk")
    sns.set_theme(font_scale=1.2)
    sns.set_style("whitegrid")
    plt.rcParams.update(
        {
            "text.usetex": False,
            "font.family": "stixgeneral",
            "mathtext.fontset": "stix",
            "axes.grid": True,
            "grid.linestyle": ":",
            "grid.color": "#bbbbbb",
            "axes.linewidth": 1,
            "legend.frameon": False,
            "lines.linewidth": 2,
        }
    )


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