#!/ligo/home/ligo.org/shiksha.pandey/.conda/envs/gwfast311/bin/python
"""
Create injection samples for GW analysis.

Uses:
- Mass sampler that follows LVK FullPop - returns samples directly
- Madau-Dickinson redshift sampler - returns samples directly  
- SFHo Lambda calculator - computes tidal deformability
"""
from configparser import ConfigParser, NoOptionError
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
from GWForge.population.mass import Mass
from GWForge.population.spin import Spin
from GWForge.population.extrinsic import Extrinsic

# Add local directory for custom modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import custom samplers
from redshift_sampler import generate_redshift_samples
from generate_tidal_params import LambdaCalculator, add_lambdas

# Argument Parsing
parser = argparse.ArgumentParser(
    description=__doc__,
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)

parser.add_argument('--config-file', required=True, 
                    help='Configuration ini file')
parser.add_argument('--out-dir', required=True,
                    help='Directory where output files will be saved')
parser.add_argument('--out-file-prefix', default='pop',
                    help='Prefix used for naming the file. \
                        Other details like cbc_type and models will be added based on config')
parser.add_argument('--out-file-suffix', default='v0',
                    help='Suffix used for naming the file.')
parser.add_argument('--save-config', action='store_true', 
                    help='Save config settings in output HDF5')
parser.add_argument('--num-samples', type=int, default=None, 
                    help='Number of samples. If None, the number is calculated \
                          from duration provided in the Redshift section of the config file')
parser.add_argument('--reference-frequency', type=float, default=5., 
                    help='Reference frequency (Hz)')
parser.add_argument('--max-NS-mass', type=float, default=2.8,
                    help='Maximum mass to categorize a component as neutron star. \
                        Any component above this will be categorized as black hole.')
parser.add_argument('--eos-dir', 
                    help='Path to CompOSE EOS directory for Lambda (required for BNS/NSBH)')
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
config = ConfigParser()
config.optionxform = utils.custom_optionxform
config.read(opts.config_file)

samples = {}

#####################################################################################################
# REDSHIFT SAMPLING
#####################################################################################################

if opts.num_samples:
    num_samples = opts.num_samples
    logging.info(f'Num samples = {opts.num_samples} provided by the user')

    logging.info('Generating redshift samples')
else:
    try:
        duration = float(config.get('Redshift', 'duration'))
    except NoOptionError:
        logging.error('Must provide [Redshift] [duration] in config if num_samples is None')
        raise NoOptionError('duration', 'Redshift')
    num_samples=None
    
z_samples, p_z_samples, tc_samples = generate_redshift_samples(opts.config_file, n_samples=num_samples)
num_samples = len(z_samples)

samples["redshift"] = z_samples
samples["luminosity_distance"] = redshift_to_luminosity_distance(samples["redshift"])
samples['geocent_time'] = tc_samples

logging.info(f"z range: [{samples['redshift'].min():.4f}, {samples['redshift'].max():.4f}]")


#####################################################################################################
# MASS SAMPLING
#####################################################################################################

logging.info(f'Generating mass samples...')

logging.info('Generating mass samples')
mass_model = config.get('Mass', 'mass-model')
mass_parameters = json.loads(config.get('Mass', 'mass-parameters').replace("'", "\""))

m = Mass(mass_model=mass_model, 
         number_of_samples=num_samples, 
         parameters=mass_parameters, 
         full_pop_sampler='importance_m1_m2')
m_samples = m.sample(oversample_factor=10)
# Check that all m1 > m2
if np.all(m_samples['mass_1_source'] >= m_samples['mass_2_source']):
    pass
else:
    raise ValueError('Samples with m1 < m2 found. Something is wrong! Aborting!')

samples.update(m_samples)

logging.info(f"  m1 range: [{samples['mass_1_source'].min():.2f}, {samples['mass_1_source'].max():.2f}] Msun")
logging.info(f"  m2 range: [{samples['mass_2_source'].min():.2f}, {samples['mass_2_source'].max():.2f}] Msun")

#####################################################################################################
# EXTRINSIC PARAMETERS
#####################################################################################################
logging.info('Generating extrinsic parameters...')

extrinsic_prior_file = None
if 'Extrinsic' in config and config.has_option('Extrinsic', 'extrinsic-prior-file'):
    extrinsic_prior_file = config.get('Extrinsic', 'extrinsic-prior-file')

e = Extrinsic(number_of_samples=num_samples, prior_file=extrinsic_prior_file)
extrinsic_samples = e.sample()

for key in extrinsic_samples:
    samples[key] = extrinsic_samples[key]

###################################################################################################
# Divide samples into cbc types for spin and tidal samples
###################################################################################################

logging.info(f'Dividing {num_samples} samples into BNS, NSBH, BBH')
bns_samples = {}
nsbh_samples = {}
bbh_samples = {}

mask_bns = (samples['mass_1_source'] <= opts.max_NS_mass)
mask_nsbh = ((samples['mass_1_source'] > opts.max_NS_mass) & (samples['mass_2_source'] <= opts.max_NS_mass))
mask_bbh = (samples['mass_2_source'] > opts.max_NS_mass)

for cat_dict, cat_mask in zip([bns_samples, nsbh_samples, bbh_samples], 
                              [mask_bns, mask_nsbh, mask_bbh]):
    for key in samples.keys():
        cat_dict[key] = samples[key][cat_mask]

n_samples_cbc = {'BBH':len(bbh_samples['mass_1_source']),
                 'NSBH':len(nsbh_samples['mass_1_source']),
                 'BNS':len(bns_samples['mass_1_source'])}
logging.info(f'BBH samples = {n_samples_cbc["BBH"]}')
logging.info(f'NSBH samples = {n_samples_cbc["NSBH"]}')
logging.info(f'BNS samples = {n_samples_cbc["BNS"]}')

#####################################################################################################
# SPIN SAMPLING
#####################################################################################################

for cbc_type, cbc_dict in zip(['BBH', 'NSBH', 'BNS'], [bbh_samples, nsbh_samples, bns_samples]):
    logging.info(f'Generating spin samples for {cbc_type}...')

    spin_model = config.get(f'Spin-{cbc_type}', 'spin-model')
    try:
        spin_params = json.loads(config.get(f'Spin-{cbc_type}', 
                                            'spin-parameters').replace("'", "\""))
    except Exception:
        logging.warning('spin-parameters not provided, using defaults')
        spin_params = {}
    if spin_model.lower() == 'custom':
        from spin_sampler import sample_gwpop_gaussian_component
        a_1, a_2, cos_tilt_1, cos_tilt_2 = sample_gwpop_gaussian_component(n_samples_cbc[cbc_type], spin_params)
        phi_12 = bilby.gw.prior.Uniform(name="phi_12", minimum=0, maximum=2 * np.pi, boundary="periodic")
        phi_jl = bilby.gw.prior.Uniform(name="phi_jl", minimum=0, maximum=2 * np.pi, boundary="periodic")

        spin_samples = {}
        spin_samples['a_1'] = a_1
        spin_samples['a_2'] = a_2
        spin_samples['tilt_1'] = np.arccos(cos_tilt_1)
        spin_samples['tilt_2'] = np.arccos(cos_tilt_2)
        spin_samples["phi_12"] = phi_12.sample(n_samples_cbc[cbc_type])
        spin_samples["phi_jl"] = phi_jl.sample(n_samples_cbc[cbc_type])
    else:
        s = Spin(spin_model=spin_model, number_of_samples=n_samples_cbc[cbc_type], parameters=spin_params)
        spin_samples = s.sample()
    cbc_dict.update(spin_samples)

#####################################################################################################
# TIDAL DEFORMABILITY
#####################################################################################################

logging.info(f'Calculating tidal deformability (Lambda)...')

for cbc_dict in [bbh_samples, nsbh_samples, bns_samples]:
    cbc_dict["lambda_1"] = np.zeros(len(cbc_dict['mass_1_source']))
    cbc_dict["lambda_2"] = np.zeros(len(cbc_dict['mass_1_source']))

eos_dir = opts.eos_dir
if eos_dir is None:
    eos_dir = config.get('EOS', 'eos-dir', fallback=None)

if eos_dir is None and n_samples_cbc['BNS']!=0:
    logging.error(f"--eos-dir required for BNS")
    sys.exit(1)

if eos_dir is None and n_samples_cbc['NSBH']!=0:
    logging.error(f"--eos-dir required for NSBH")
    sys.exit(1)

try:
    calc = LambdaCalculator(eos_dir, method="interp")
    logging.info(f"  EOS max mass: {calc.eos_max_mass:.4f} Msun")

    logging.info('Calculating lambdas for BNS')
    mapping_bns = {'lambda_1': 'mass_1_source',
                   'lambda_2': 'mass_2_source'}
    bns_samples = add_lambdas(bns_samples, calc, mapping_bns)
    logging.info(f"  lambda_1 range: [{bns_samples['lambda_1'].min():.1f}, {bns_samples['lambda_1'].max():.1f}]")
    logging.info(f"  lambda_2 range: [{bns_samples['lambda_2'].min():.1f}, {bns_samples['lambda_2'].max():.1f}]")
    
    logging.info('Calculating lambdas for NSBH')
    mapping_nsbh = {'lambda_2': 'mass_2_source'}
    nsbh_samples = add_lambdas(nsbh_samples, calc, mapping_nsbh)
    nsbh_samples['lambda_1'] = np.zeros(n_samples_cbc['NSBH'])
    logging.info(f"  lambda_2 range: [{nsbh_samples['lambda_2'].min():.1f}, {nsbh_samples['lambda_2'].max():.1f}]")
        
except Exception as ex:
    logging.error(f'Lambda calculation failed: {ex}')
    sys.exit(1)

#####################################################################################################
# DETECTOR FRAME CONVERSIONS and DERIVED SPIN QUANTITIES
#####################################################################################################
def gen_frame_spins(samples_dict, cbc_type):
    logging.info(f'Generating detector-frame parameters for {cbc_type}...')
    samples_dict = generate_detector_frame_parameters(samples_dict)
    samples_dict['reference_frequency'] = np.ones(n_samples_cbc[cbc_type]) * opts.reference_frequency

    samples_dict = generate_spin_parameters(samples_dict)
    spin_model = utils.remove_special_characters(config.get(f'Spin-{cbc_type}', 'spin-model').lower())
    is_aligned = "aligned" in spin_model
    is_nonspinning = "nonspinning" in spin_model

    if is_aligned or is_nonspinning:
        for key in ["chi_1_in_plane", "chi_2_in_plane", "spin_1x", "spin_1y", "spin_2x", "spin_2y"]:
            samples_dict[key] = np.zeros(n_samples_cbc[cbc_type])
    return(samples_dict)

bbh_samples = gen_frame_spins(bbh_samples, 'BBH')
nsbh_samples = gen_frame_spins(nsbh_samples, 'NSBH')
bns_samples = gen_frame_spins(bns_samples, 'BNS')

#####################################################################################################
# SAVE TO HDF5
#####################################################################################################
file_naming_strs = {'madaudickinson':'MD',
                    'fullpopgwtc4':'FP4',
                    'isotropicbilby':'IB',
                    'gaussiannonspinning':'GNS',
                    'nonspinning':'NS',
                    'custom':'custom',
                    'alignedgaussianuniform':'AGU'}

def get_model_str(param_section, param_option):
    param_model = utils.remove_special_characters(config.get(param_section, param_option).lower())
    param_str = file_naming_strs[param_model]
    return(param_str)

spin_strs = {cbc_type:get_model_str(f'Spin-{cbc_type}', 'spin-model') for cbc_type in ['BBH', 'NSBH', 'BNS']}
m_str = get_model_str('Mass', 'mass-model')
z_str = get_model_str('Redshift', 'redshift-model')
lmr = config.get('Redshift', 'local-merger-rate-density')
n_samples_strs = {}
for cbc_type in ['BBH', 'NSBH', 'BNS']:
    if n_samples_cbc[cbc_type]//1000 == 0:
        n_samples_strs[cbc_type] = str(n_samples_cbc[cbc_type])
    else:
        n_samples_strs[cbc_type] = str(n_samples_cbc[cbc_type]//1000)+'K'

out_file_names = {}

for cbc_type in ['BBH', 'NSBH', 'BNS']:
    out_file_names[cbc_type] = '_'.join([opts.out_file_prefix,
                                         'm', m_str,
                                         'spin', spin_strs[cbc_type],
                                         'fref', str(opts.reference_frequency),
                                         'z', z_str, 'lmr', lmr,
                                         cbc_type, n_samples_strs[cbc_type], 'samples', 
                                         opts.out_file_suffix]) + '.h5'

for cbc_type, cbc_dict in zip(['BBH', 'NSBH', 'BNS'], [bbh_samples, nsbh_samples, bns_samples]):
    logging.info(f'Saving {n_samples_cbc[cbc_type]} {cbc_type} samples to {out_file_names[cbc_type]}...')
    os.makedirs(opts.out_dir, exist_ok=True)
    out_file_path = os.path.join(opts.out_dir, out_file_names[cbc_type])
    with h5py.File(out_file_path, 'w') as f:
        for key in cbc_dict:
            utils.hdf_append(f, key, cbc_dict[key])
        
        # Metadata for reproducibility
        f.attrs['seed'] = opts.seed
        f.attrs['source_type'] = cbc_type
        f.attrs['num_samples'] = n_samples_cbc[cbc_type]
        f.attrs['reference_frequency'] = opts.reference_frequency
        f.attrs['config_file'] = opts.config_file
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
