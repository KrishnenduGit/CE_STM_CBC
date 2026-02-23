#!/ligo/home/ligo.org/shiksha.pandey/.conda/envs/gwfast311/bin/python
"""
Create injection samples for GW analysis.

Uses:
- Mass sampler that follows LVK FullPop - returns samples directly
- Madau-Dickinson redshift sampler - returns samples directly  
- SFHo Lambda calculator - computes tidal deformability
"""
import configparser
import argparse
import logging
import json
import h5py
import numpy as np
import bilby
import sys
import os

from GWForge.conversion import redshift_to_luminosity_distance, generate_detector_frame_parameters, generate_spin_parameters
from GWForge import utils, conversion
from GWForge.population.spin import Spin
from GWForge.population.extrinsic import Extrinsic

# Add local directory for custom modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import custom samplers
from mass_sampler import generate_mass_ratio_samples
from redshift_sampler import generate_md_redshift_samples
from generate_tidal_params import LambdaCalculator, add_lambdas

# Argument Parsing
parser = argparse.ArgumentParser(
    description=__doc__,
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)

parser.add_argument('--config-file', required=True, help='Configuration ini file')
parser.add_argument('--output-file', required=True, help='HDF5 file to store injection information')
parser.add_argument('--save-config', action='store_true', help='Save config settings in output HDF5')
parser.add_argument('--source-type', type=str.lower, required=True, choices=['bbh', 'bns', 'nsbh'], help='Signal source type')
parser.add_argument('--num-samples', type=int, default=10000, help='Number of samples')
parser.add_argument('--reference-frequency', type=float, default=20., help='Reference frequency (Hz)')
parser.add_argument('--pdb-file', help='Path to LVK FullPop H5 file for mass sampling')
parser.add_argument('--eos-dir', help='Path to CompOSE EOS directory for Lambda (required for BNS/NSBH)')
parser.add_argument('--z-config-file', help='Config file for redshift (defaults to --config-file)')

parser.add_argument('--seed', type=int, default=None,
                    help='Random seed for reproducibility (default: auto-generated)')

# Logging
parser.add_argument('--log-level',
                    choices=['debug', 'info', 'warning', 'error', 'critical'],
                    default='info',
                    help='Logging verbosity')

opts = parser.parse_args()

for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

logging.basicConfig(
    level=getattr(logging, opts.log_level.upper()),
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# Set random seed
if opts.seed is None:
    opts.seed = np.random.randint(0, 2**31)
    
np.random.seed(opts.seed)
logging.info(f"Random seed: {opts.seed}")

# Load config
config = configparser.ConfigParser()
config.optionxform = utils.custom_optionxform
config.read(opts.config_file)

samples = {}

#####################################################################################################
# REDSHIFT SAMPLING
#####################################################################################################

logging.info('Generating redshift samples (Madau-Dickinson)...')

z_config = opts.z_config_file if opts.z_config_file else opts.config_file
z_samples, p_z_samples = generate_md_redshift_samples(z_config, n_samples=opts.num_samples)

# Sanity check on number of samples
if len(z_samples) != opts.num_samples:
    raise ValueError(
        f"Redshift sampler returned {len(z_samples)} samples, expected {opts.num_samples}"
    )

samples["redshift"] = z_samples
samples["p_z"] = p_z_samples
samples["luminosity_distance"] = redshift_to_luminosity_distance(samples["redshift"])

logging.info(f"  z range: [{samples['redshift'].min():.4f}, {samples['redshift'].max():.4f}]")


#####################################################################################################
# MASS SAMPLING
#####################################################################################################

logging.info(f'Generating mass samples, {opts.source_type.upper()})...')

pdb_file = opts.pdb_file
if pdb_file is None:
    pdb_file = config.get('Mass', 'pdb-file', fallback=None)
if pdb_file is None:
    raise ValueError("--pdb-file or [Mass] pdb-file in config is required")

m1, p_m1, q, p_q_given_m1 = generate_mass_ratio_samples(
    n_samples=opts.num_samples,
    pdb_file=pdb_file,
    source_type=opts.source_type
)

# Sanity check on number of samples
n_actual = min(len(m1), opts.num_samples)
if len(m1) < opts.num_samples:
    logging.warning(f"Mass sampler returned {len(m1)} samples (requested {opts.num_samples})")

samples['mass_1_source'] = m1[:n_actual]
samples['mass_ratio'] = q[:n_actual]
samples['mass_2_source'] = m1[:n_actual] * q[:n_actual]
samples['p_m1'] = p_m1[:n_actual]
samples['p_q_given_m1'] = p_q_given_m1[:n_actual]

# Truncate redshift arrays to match n_actual
for k in ["redshift", "p_z", "luminosity_distance"]:
    samples[k] = samples[k][:n_actual]

logging.info(f"  m1 range: [{samples['mass_1_source'].min():.2f}, {samples['mass_1_source'].max():.2f}] Msun")
logging.info(f"  m2 range: [{samples['mass_2_source'].min():.2f}, {samples['mass_2_source'].max():.2f}] Msun")

# Derived mass parameters
mass_samples = conversion.generate_mass_parameters(samples, source=True)
for key in mass_samples:
    samples[key] = mass_samples[key]

#####################################################################################################
# SPIN SAMPLING
#####################################################################################################
logging.info('Generating spin samples...')

spin_model = config.get('Spin', 'spin-model')
try:
    spin_params = json.loads(config.get('Spin', 'spin-parameters').replace("'", "\""))
except Exception:
    logging.warning('  spin-parameters not provided, using defaults')
    spin_params = {}

n_actual = len(samples['mass_1_source'])
s = Spin(spin_model=spin_model, number_of_samples=n_actual, parameters=spin_params)
spin_samples = s.sample()

for key in spin_samples:
    samples[key] = spin_samples[key]

#####################################################################################################
# EXTRINSIC PARAMETERS
#####################################################################################################
logging.info('Generating extrinsic parameters...')

extrinsic_prior_file = None
if 'Extrinsic' in config and config.has_option('Extrinsic', 'extrinsic-prior-file'):
    extrinsic_prior_file = config.get('Extrinsic', 'extrinsic-prior-file')

e = Extrinsic(number_of_samples=n_actual, prior_file=extrinsic_prior_file)
extrinsic_samples = e.sample()

for key in extrinsic_samples:
    samples[key] = extrinsic_samples[key]

#####################################################################################################
# TIDAL DEFORMABILITY
#####################################################################################################

samples["lambda_1"] = np.zeros(n_actual)
samples["lambda_2"] = np.zeros(n_actual)

if opts.source_type in ['bns', 'nsbh']:
    logging.info('Calculating tidal deformability (Lambda)...')
    
    eos_dir = opts.eos_dir
    if eos_dir is None:
        eos_dir = config.get('EOS', 'eos-dir', fallback=None)
    
    if eos_dir is None:
        logging.error(f"--eos-dir required for {opts.source_type}")
        sys.exit(1)
    
    try:
        calc = LambdaCalculator(eos_dir, method="interp")
        logging.info(f"  EOS max mass: {calc.eos_max_mass:.4f} Msun")
        
        mapping = {'lambda_2': 'mass_2_source'}
        if opts.source_type == 'bns':
            mapping['lambda_1'] = 'mass_1_source'
        
        samples = add_lambdas(samples, calc, mapping)
        
        if opts.source_type == 'nsbh':
            samples['lambda_1'] = np.zeros(n_actual)
        
        logging.info(f"  lambda_2 range: [{samples['lambda_2'].min():.1f}, {samples['lambda_2'].max():.1f}]")
        if opts.source_type == 'bns':
            logging.info(f"  lambda_1 range: [{samples['lambda_1'].min():.1f}, {samples['lambda_1'].max():.1f}]")
            
    except Exception as ex:
        logging.error(f'Lambda calculation failed: {ex}')
        sys.exit(1)

#####################################################################################################
# DETECTOR FRAME CONVERSIONS and DERIVED SPIN QUANTITIES
#####################################################################################################
logging.info('Generating detector-frame parameters...')

samples = generate_detector_frame_parameters(samples)
samples['reference_frequency'] = np.ones(n_actual) * opts.reference_frequency

samples = generate_spin_parameters(samples)
is_aligned = "aligned" in spin_model.lower()

if is_aligned:
    n = len(samples["mass_1"])
    for k in ["chi_1_in_plane", "chi_2_in_plane", "spin_1x", "spin_1y", "spin_2x", "spin_2y"]:
        samples[k] = np.zeros(n)

#####################################################################################################
# SAVE TO HDF5
#####################################################################################################
logging.info(f'Saving {n_actual} samples to {opts.output_file}...')

with h5py.File(opts.output_file, 'w') as f:
    for key in samples:
        utils.hdf_append(f, key, samples[key])
    
    # Metadata for reproducibility
    f.attrs['seed'] = opts.seed
    f.attrs['source_type'] = opts.source_type
    f.attrs['num_samples'] = n_actual
    f.attrs['reference_frequency'] = opts.reference_frequency
    f.attrs['pdb_file'] = pdb_file
    if opts.eos_dir:
        f.attrs['eos_dir'] = opts.eos_dir
    f.attrs['numpy_version'] = np.__version__
    f.attrs['bilby_version'] = bilby.__version__
    
    if opts.save_config:
        config_group = f.create_group("config")
        for section in config.sections():
            section_group = config_group.create_group(section)
            for key, value in config[section].items():
                section_group.attrs[key] = value

logging.info('Done!')
