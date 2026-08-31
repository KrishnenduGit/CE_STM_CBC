"""
redshift_forecast.py  -  Population Fisher for a Madau-Dickinson redshift model
(Gair+2022).

Runs the population Fisher for a Madau-Dickinson redshift model on a gwfast
catalogue and turns the resulting hyperparameter covariance into two figures:

  * a corner plot of the Fisher-approximated hyperparameter posterior, and
  * the implied merger-rate density p(z), with a 90% credible band.
"""
import matplotlib
matplotlib.use('Agg')  # no X server required
import logging, warnings
import numpy, pandas, argparse

logging.basicConfig(level=logging.WARNING, format='%(levelname)s - %(message)s')
warnings.filterwarnings('ignore', category=RuntimeWarning)

from load import load_population_catalogue
from plot_utils import plot_corner, plot_redshift_distribution
from pop_models import MadauDickinsonRedshift
from pop_fisher import compute_pop_fisher_full

# ===========================================================================
# Runners
# ===========================================================================

def run_all_terms(model, file_path, snr_threshold, fixed_params=None):
    print('\n' + '=' * 72)
    print('  All five terms  (Gair+2022 Eq. 21)')
    print('=' * 72)

    catalogue = load_population_catalogue(file_path, snr_threshold=snr_threshold)

    print(f'  Analysis sample : {catalogue.number_detected} events  '
          f'(is_detected & snr >= {snr_threshold})')
    print(f'  P_det numerator : {catalogue.number_above_threshold} events  '
          f'(snr >= {snr_threshold})')
    print(f'  Total injections: {catalogue.number_total}')
    print(f'  P_det           : {catalogue.number_above_threshold}/'
          f'{catalogue.number_total} = {catalogue.p_det:.4f}')

    result = compute_pop_fisher_full(
        model, catalogue,
        fixed_parameters=fixed_params or {},
    )
    print(result.summary())
    return result


# ===========================================================================
# Plotting
# ===========================================================================

def draw_hyperparameter_samples(result, number_of_samples=1e4, rng=250114):
    """
    Fisher-approximated hyperparameter posterior samples.
    """
    numpy.random.seed(rng)
    mean = [result.fiducial_parameters[name] for name in result.parameter_names]
    samples = numpy.random.multivariate_normal(
        mean=mean,
        cov=result.covariance,
        size=int(number_of_samples),
    )
    return pandas.DataFrame(samples, columns=result.parameter_names)


def make_plots(model, result, snr_threshold, number_of_samples=1e4):
    """Corner plot of the hyperparameters and the implied p(z) band."""
    label = rf'{type(model).__name__} for SNR $\geq$ {snr_threshold:g}'
    posterior_samples = draw_hyperparameter_samples(
        result, number_of_samples=number_of_samples
    )

    corner_figure = plot_corner(posterior_samples, save=False, title=label)
    corner_name = f'corner_{type(model).__name__}_snr{snr_threshold:g}.pdf'
    corner_figure.savefig(corner_name, bbox_inches='tight')
    print(f'  Wrote {corner_name}')

    redshift_figure = plot_redshift_distribution(
        model, posterior_samples, title=label
    )
    redshift_name = f'pz_{type(model).__name__}_snr{snr_threshold:g}.pdf'
    redshift_figure.savefig(redshift_name, bbox_inches='tight')
    print(f'  Wrote {redshift_name}')

    return corner_figure, redshift_figure


# ===========================================================================
# Entry point
# ===========================================================================

args = argparse.ArgumentParser(
    description='Run the population Fisher for a Madau-Dickinson redshift model')

args.add_argument('--fisher-file', 
                  type=str, 
                  default='network_bbh_CE40km_1p5MW_Aplus_coat_5.0hz_CE20km_1p5MW_Aplus_coat_5.0hz_ETD_5.0hz.h5',
                  help='Path to the gwfast catalogue HDF5 file.  If not given, '
                       'the default CE40km+CE20km file is used.')
args.add_argument('--snr-thresholds', type=float, nargs='+', default=[10, 20, 30, 50],
                  help='SNR thresholds for the analysis sample.  Default: '
                       '10, 20, 30, 50.')
args = args.parse_args()
# Effective Madau-Dickinson parameters of the injected population, from a
# maximum-likelihood fit of the MD form to the catalogue's 34166 injected
# redshifts.
#
# alpha + beta = kappa
# Fit gives (gamma, kappa, z_peak) = (1.881, 5.364, 1.818)
model = MadauDickinsonRedshift(
    fiducial={'alpha': 1.881, 'beta': 3.483, 'z_peak': 1.818}
)
fixed = {}

for snr_threshold in args.snr_thresholds:
    result = run_all_terms(
        model,
        args.fisher_file,
        fixed_params=fixed,
        snr_threshold=snr_threshold,
    )

    corner_figure, redshift_figure = make_plots(model, result, snr_threshold)
