#!/usr/bin/env python
import configparser, argparse, logging, json, h5py, numpy
from GWForge.conversion import *
from GWForge import utils
from GWForge.population.mass import Mass
from GWForge.population.spin import Spin
from GWForge.population.redshift import Redshift
from GWForge.population.extrinsic import Extrinsic
from GWForge import conversion
from lal import YRJUL_SI

# Read configuration from ini file
config = configparser.ConfigParser()
parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)

# Parse command-line arguments
parser.add_argument('--config-file', 
                    required=True,
                    help='Configuration ini file')
parser.add_argument('--output-file', 
                    required=True, 
                    help='HDF5 file to store injection information')
parser.add_argument('--save-config',
		    action='store_true',
		    help='If provided, saves settings from config file in the group "config" in the output HDF5 file (uses configparser)')
parser.add_argument('--source-type', 
                    type=str.lower,
                    choices=['bbh', 'bns', 'nsbh', 'pbh', 'imbhb', 'imbbh'],
                    default='bbh',
                    help='Signal source type')
parser.add_argument('--reference-frequency', 
                    type=float,
                    default=5.,
                    help='Reference frequency (in Hz)')
parser.add_argument('--num-samples',
                    type=int,
                    default=10000,
                    help='number of samples')
parser.add_argument('--z-prob-file', 
                    help='path to dat file containing z probabilities with columns [z, p_z]')
parser.add_argument('--mass-prob-file',
                    help='path to dat file containing m1 and q probabilities with columns [m1, pm1, q, pq]')
# Add optional log level argument
parser.add_argument('--log-level',
                    choices=['debug', 'info', 'warning', 'error', 'critical'],
                    default='error',
                    help='Set logging verbosity level')

opts = parser.parse_args()

# --- Force reset of all previous logging configs ---
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

# Set up logging configuration based on user input
logging.basicConfig(level=getattr(logging, opts.log_level.upper()),
                    format='%(asctime)s %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')

# Read configuration file
config.read(opts.config_file)
config.optionxform = utils.custom_optionxform

# Initialise samples dictionary
samples = {}

# Generate redshift samples
logging.info('Generating redshift samples')

z, p_z = np.loadtxt(opts.z_prob_file, unpack=True)
maximum_redshift = config.getfloat('Redshift', 'maximum-redshift')
z_prior = bilby.core.prior.Interped(xx=z, yy=p_z, minimum=0.,maximum=maximum_redshift, name='redshift')

redshift_samples = {'redshift': z_prior.sample(opts.num_samples)}
for key in redshift_samples.keys():
    samples[key] = redshift_samples[key]
samples['luminosity_distance'] = redshift_to_luminosity_distance(samples['redshift'])

# Generate mass samples
logging.info('Generating mass samples')

if opts.mass_prob_file:
    m1, p_m1, q, p_q = np.loadtxt(opts.mass_prob_file, unpack=True)
    m1_prior = bilby.core.prior.Interped(xx=m1, yy=p_m1, name='mass_1_source')
    q_prior = bilby.core.prior.Interped(xx=q, yy=p_q, name='mass_ratio')
    samples['mass_1_source'] = m1_prior.sample(opts.num_samples)
    samples['mass_ratio'] = q_prior.sample(opts.num_samples)
    mass_samples = conversion.generate_mass_parameters(samples, source=True)
else:
    mass_model = config.get('Mass', 'mass-model')
    parameters = json.loads(config.get('Mass', 'mass-parameters').replace("'", "\""))

    # Create Mass object and generate samples
    m = Mass(mass_model=mass_model, 
             number_of_samples=len(samples['redshift']), 
             parameters=parameters)

    mass_samples = m.sample()

for key in mass_samples.keys():
    samples[key] = mass_samples[key]

# Generate spin samples
logging.info('Generating spin samples')
spin_model = config.get('Spin', 'spin-model')
try:
    parameters = json.loads(config.get('Spin', 'spin-parameters').replace("'", "\""))
except:
    logging.warning('spin_parameters not provided. Assuming default values')
    parameters = {}

# Create Spin object and generate samples    
s = Spin(spin_model = spin_model,
         number_of_samples=len(samples['redshift']), 
         parameters=parameters)

spin_samples = s.sample()
for key in spin_samples.keys():
    samples[key] = spin_samples[key]

# Generate extrinsic parameter samples
logging.info('Generating extrinsic parameters')
if 'Extrinsic' in config:
    if config.has_option('Extrinsic', 'extrinsic-prior-file'):
        extrinsic_prior_file = config.get('Extrinsic', 'extrinsic-prior-file')
    else:
        logging.warning("No 'extrinsic_prior_file' provided. Using default priors")
        extrinsic_prior_file = None
else:
    logging.warning("No 'extrinsic_prior_file' provided. Using default priors")
    extrinsic_prior_file = None

# Create Extrinsic object and generate samples    
e = Extrinsic(number_of_samples = len(samples['redshift']), 
              prior_file = extrinsic_prior_file)

extrinsic_samples = e.sample()
for key in extrinsic_samples.keys():
    samples[key] = extrinsic_samples[key]

# Handle additional calculations for specific source types (e.g., BNS, NSBH)
if opts.source_type == 'bns' or opts.source_type == 'nsbh':
    try:
        eos_file = config.get('EOS', 'eos_file')
        logging.info('Trying {} file to estimate tidal deformability parameter of components'.format(eos_file))
        samples['lambda_2'] = get_lambda(file=eos_file, source_mass=samples['mass_2_source'])
        if opts.source_type == 'bns':
            samples['lambda_1'] = get_lambda(file=eos_file, source_mass=samples['mass_1_source'])
    except:
        logging.warning('Failed to estimate tidal parameters.')


# Generate all possible parameters of the specified source type
logging.info('Generating all possible parameters of {}'.format(opts.source_type))
samples = generate_detector_frame_parameters(samples)
samples['reference_frequency'] = numpy.ones_like(samples['mass_1']) * opts.reference_frequency
samples = generate_spin_parameters(samples)
logging.info('Saving data in {}'.format(opts.output_file))

with h5py.File(opts.output_file,'w') as f:
    for key in samples.keys():
        utils.hdf_append(f, key, samples[key])

    if opts.save_config:
        # Save configuration settings in a "config" group
        config_group = f.create_group("config")
        for section in config.sections():
            section_group = config_group.create_group(section)
            for key, value in config[section].items():
                section_group.attrs[key] = value
logging.info('Done!')
