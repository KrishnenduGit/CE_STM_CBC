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


def generate_md_redshift_samples(config_file, n_samples):
    """
    Generates redshift samples using the Madau-Dickinson model.
    
    Args:
        config_file (str): path to the .ini configuration file.
        n_samples (int): number of samples to generate.
        
    Returns:
        tuple: (z_samples, p_z_samples) as numpy arrays.
    """
    print(f"n_samples = {n_samples}")

    config = ConfigParser()
    config.read(config_file)

    # Load Config Parameters
    redshift_model = config.get('Redshift', 'redshift-model')
    maximum_redshift = config.getfloat('Redshift', 'maximum-redshift')
    local_merger_rate_density = config.getfloat('Redshift', 'local-merger-rate-density')
    gps_start_time = config.getfloat('Redshift', 'gps-start-time')
    analysis_time = config.getfloat('Redshift', 'duration', fallback=YRJUL_SI)
    cosmology = config.get('Redshift', 'cosmology', fallback='Planck18')
    
    redshift_parameters = config.get('Redshift', 'redshift-parameters', fallback='{"gamma": 2.7, "kappa": 5.6, "z_peak": 1.9}')
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
    model = MadauDickinsonRedshift(z_max=z.maximum_redshift)
    xx = model.zs
    prob = rate(xx)

    # Create Prior and Sample
    prior = bilby.core.prior.Interped(xx=xx, yy=prob, minimum=0., maximum=maximum_redshift, name='redshift')
    z_samples = prior.sample(n_samples)
    p_z_samples = prior.probability_density(z_samples)
    
    return z_samples, p_z_samples

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file", required=True, help="path to the .ini configuration file")
    parser.add_argument("--n-samples", type=int, default=10000, help="number of samples to generate")
    parser.add_argument("--output", default="md_z_samples.dat", help="output filename")
    args = parser.parse_args()

    # Run Generation
    z_vals, p_z_vals = generate_md_redshift_samples(args.config_file, n_samples=args.n_samples)

    # Save to File
    print(f"\nSaving {len(z_vals)} samples to {args.output}...")
    np.savetxt(args.output, np.c_[z_vals, p_z_vals])
    print("Done.")
