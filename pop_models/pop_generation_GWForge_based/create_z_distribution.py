from GWForge.population.redshift import Redshift
from configparser import ConfigParser
from lal import YRJUL_SI
import json
from gwpopulation.models.redshift import MadauDickinsonRedshift
import numpy as np
import bilby
from pycbc.population.population_models import coalescence_rate, norm_redshift_distribution

n_samples = 10000
print(f"n_samples = {n_samples}")

config_file = 'uniform_z_bbh2.ini'
config = ConfigParser()
config.read(config_file)

redshift_model = config.get('Redshift', 'redshift-model')
maximum_redshift = config.getfloat('Redshift', 'maximum-redshift')

if redshift_model.lower() != 'uniform':
    print("Not using uniform redshift model")
    local_merger_rate_density = config.getfloat('Redshift', 'local-merger-rate-density')
    gps_start_time = config.getfloat('Redshift', 'gps-start-time')
    analysis_time = config.getfloat('Redshift', 'duration', fallback=YRJUL_SI)
    cosmology = config.get('Redshift', 'cosmology', fallback='Planck18')
    redshift_parameters = config.get('Redshift', 'redshift-parameters', fallback='{"gamma": 2.7, "kappa": 5.6, "z_peak": 1.9}')
    # Use json.loads to parse the JSON string
    redshift_parameters = json.loads(redshift_parameters.replace("'", "\""))
    time_delay_model = config.get('Redshift', 'time-delay-model', fallback='inverse')
    H0 = config.getfloat('Redshift', 'H0', fallback=70)
    Om0 = config.getfloat('Redshift', 'Om0', fallback=0.3)
    Ode0 = config.getfloat('Redshift', 'Ode0', fallback=0.7)
    Tcmb0 = config.getfloat('Redshift', 'Tcmb0', fallback=2.735)

    # Create Redshift object and generate samples
    z = Redshift(redshift_model=redshift_model,
            local_merger_rate_density=local_merger_rate_density,
            maximum_redshift=maximum_redshift, gps_start_time=gps_start_time,
            analysis_time=analysis_time,
            cosmology = cosmology,
            parameters = redshift_parameters,
            time_delay_model = time_delay_model,
            H0=H0, Om0=Om0, Ode0=Ode0)

    rate = z.coalescence_rate()
    model = MadauDickinsonRedshift(z_max=z.maximum_redshift)
    xx = model.zs
    prob = rate(xx)
else:
    print("Using uniform model for redshift")
    
    def uniform_z(z, min_z=0., max_z=maximum_redshift):
        prior = bilby.core.prior.Uniform(minimum=min_z, maximum=max_z)
        return(prior.prob(z))

    mr_uniform = coalescence_rate(uniform_z)
    xx = np.linspace(0., maximum_redshift, n_samples)
    prob = norm_redshift_distribution(xx, mr_uniform)

prior = bilby.core.prior.Interped(xx=xx, yy=prob, minimum=0.,maximum=maximum_redshift, name='redshift')
z_samples = prior.sample(n_samples)
p_z_samples = prior.probability_density(z_samples)
np.savetxt("2_uniform_z_samples_prob_z_zmax_10.dat", np.c_[z_samples, p_z_samples])
