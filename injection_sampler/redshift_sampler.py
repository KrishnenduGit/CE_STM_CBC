#!/ligo/home/ligo.org/shiksha.pandey/.conda/envs/gwfast311/bin/python
from GWForge.population.redshift import Redshift
from configparser import ConfigParser
from lal import YRJUL_SI
import json
from gwpopulation.models.redshift import MadauDickinsonRedshift
import numpy as np
import bilby
import argparse
from pycbc.population.population_models import coalescence_rate, norm_redshift_distribution


def generate_redshift_samples(config_file, n_samples=None, duration=None):
    """
    Generates redshift samples using the Madau-Dickinson model.
    
    Args:
        config_file (str): path to the .ini configuration file.
        n_samples (int): number of samples to generate.
        
    Returns:
        tuple: (z_samples, p_z_samples) as numpy arrays.
    """

    config = ConfigParser()
    config.read(config_file)

    # Load Config Parameters
    redshift_model = config.get('Redshift', 'redshift-model')
    maximum_redshift = config.getfloat('Redshift', 'maximum-redshift')
    local_merger_rate_density = config.getfloat('Redshift', 'local-merger-rate-density')
    gps_start_time = config.getfloat('Redshift', 'gps-start-time')
    
    # This fallback option for analysis_time is given because Redshift needs to 
    # be called with an analysis time argument. If n_samples is provided 
    # to the function, this duration will not be used

    analysis_time = config.getfloat('Redshift', 'duration', fallback=YRJUL_SI)
    cosmology = config.get('Redshift', 'cosmology', fallback='Planck18')
    
    redshift_parameters = config.get('Redshift', 'redshift-parameters', 
                                     fallback='{"gamma": 2.7, "kappa": 5.6, "z_peak": 1.9}')
    redshift_parameters = json.loads(redshift_parameters.replace("'", "\""))
    
    time_delay_model = config.get('Redshift', 'time-delay-model', fallback='inverse')
    H0 = config.getfloat('Redshift', 'H0', fallback=70)
    Om0 = config.getfloat('Redshift', 'Om0', fallback=0.3)
    Ode0 = config.getfloat('Redshift', 'Ode0', fallback=0.7)
    Tcmb0 = config.getfloat('Redshift', 'Tcmb0', fallback=2.735)

    print(f"Using redshift model: {redshift_model}")

    # Create Redshift object
    z = Redshift(redshift_model=redshift_model,
            local_merger_rate_density=local_merger_rate_density,
            maximum_redshift=maximum_redshift, 
            gps_start_time=gps_start_time,
            analysis_time=analysis_time,
            cosmology = cosmology,
            parameters = redshift_parameters,
            time_delay_model = time_delay_model,
            H0=H0, Om0=Om0, Ode0=Ode0)

    # Calculate Rate
    rate = z.coalescence_rate()
    
    # Use gwpopulation for the grid
    if redshift_model == 'madaudickinson':
        logging.info('Generating samples assumming Madau-Dickinson Model')
        from gwpopulation.models.redshift import MadauDickinsonRedshift
        model = MadauDickinsonRedshift(z_max=self.maximum_redshift)
    elif redshift_model == 'powerlaw':
        logging.info('Generating samples Power-Law Model')
        from gwpopulation.models.redshift import PowerLawRedshift
        model = PowerLawRedshift(z_max=self.maximum_redshift)
    else:
        raise ValueError('Redshift model {} is not implemented in GWPopulation')
    
    xx = model.zs
    prob = rate(xx)

    # Create Prior and Sample
    prior = bilby.core.prior.Interped(xx=xx, yy=prob, minimum=0., maximum=maximum_redshift, name='redshift')

    if n_samples:
        if duration:
            raise ValueError('num_samples provided. Duration has to be None')
        else:
            num_samples = n_samples
    elif duration:
        average_time_interval = z.average_time_between_signals()
        logging.info('Average time interval between signals = {:.2f}'.format(average_time_interval))
        num_samples = int(analysis_time / average_time_interval)
    else:
        raise ValueError('Either n_samples or duration must be provided')
    
    z_samples = prior.sample(num_samples)
    p_z_samples = prior.probability_density(z_samples)
    
    return z_samples, p_z_samples

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file", required=True, help="path to the .ini configuration file")
    parser.add_argument("--n-samples", type=int, default=None, 
                        help="number of samples to generate. If None, n_samples is calculated from duration")
    parser.add_argument("--duration", type=float, default=None,
                        help="Duration used to calcuated n_samples")
    parser.add_argument("--output", default="md_z_samples.dat", help="output filename")
    args = parser.parse_args()

    # Run Generation
    z_vals, p_z_vals = generate_md_redshift_samples(args.config_file, 
                                                    n_samples=args.n_samples, 
                                                    duration=args.duration)

    # Save to File
    print(f"\nSaving {len(z_vals)} samples to {args.output}...")
    np.savetxt(args.output, np.c_[z_vals, p_z_vals])
    print("Done.")
