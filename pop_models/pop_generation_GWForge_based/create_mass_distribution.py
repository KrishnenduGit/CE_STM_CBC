from GWForge.population.mass import Mass
from configparser import ConfigParser
import json
import numpy as np
import bilby
from GWForge import utils

n_samples = 10000
print(f"n_samples = {n_samples}")

samples = {}

config_file = 'bbh1.ini'
config = ConfigParser()
config.read(config_file)

config.optionxform = utils.custom_optionxform
mass_model = config.get('Mass', 'mass-model')
parameters = json.loads(config.get('Mass', 'mass-parameters').replace("'", "\""))
m = Mass(mass_model=mass_model, 
         number_of_samples=n_samples, 
         parameters=parameters)

if 'powerlawpeak' in m.mass_model:
    from gwpopulation.models.mass import SinglePeakSmoothedMassDistribution
    model = SinglePeakSmoothedMassDistribution(normalization_shape=(1000, 1000))
elif 'multipeak' in m.mass_model:
    from gwpopulation.models.mass import MultiPeakSmoothedMassDistribution
    model = MultiPeakSmoothedMassDistribution(normalization_shape=(1000, 1000))
elif 'brokenpowerlaw' in m.mass_model:
    from gwpopulation.models.mass import BrokenPowerLawSmoothedMassDistribution
    model = BrokenPowerLawSmoothedMassDistribution(normalization_shape=(1000, 1000))

mass1, mass_ratio = model.m1s, model.qs

# Create dictionaries for supported parameters
mass_parameters = {param: m.parameters[param] for param in m.parameters if param not in ('beta')}
mass_ratio_parameters = {param:m.parameters[param] for param in m.parameters if param in ('beta', 'mmin', 'delta_m')}

prob_mass_1 = model.p_m1({'mass_1': mass1}, **mass_parameters)
prob_mass_ratio = model.p_q({'mass_ratio': mass_ratio, 'mass_1' : mass1}, **mass_ratio_parameters)

primary_mass_prior = bilby.core.prior.Interped(mass1, prob_mass_1,
                                               minimum=np.min(mass1),
                                               maximum=np.max(mass1),
                                               name='mass_1_source')

mass_ratio_prior = bilby.core.prior.Interped(mass_ratio, prob_mass_ratio,
                                             minimum=np.min(mass_ratio),
                                             maximum=np.max(mass_ratio),
                                             name='mass_ratio')
m1_samples = primary_mass_prior.sample(n_samples)
p_m1_samples = primary_mass_prior.probability_density(m1_samples)
q_samples = mass_ratio_prior.sample(n_samples)
p_q_samples = mass_ratio_prior.probability_density(q_samples)
np.savetxt("m1_pm1_q_pq.dat", np.c_[m1_samples, p_m1_samples, q_samples, p_q_samples])
