import h5py, json, numpy, pandas, bilby
from plot_utils import *

def load_event_parameters(file_path, 
                          event_index, 
                          number_of_samples, 
                          rng = 250114,
                          check_covariance = False,
                          tides = True,
                          enforce_physicality = True):

    # Generate a posterior_samples file similar to Bilby
    # And in a format Koustav likes! 
    
    numpy.random.seed(rng)
    f = h5py.File(file_path, 'r')
    snr = f['snr'][event_index]
    parameter_map = {'chirp_mass': 'Mc', 'symmetric_mass_ratio': 'eta',
                       'luminosity_distance': 'dL',
                       'ra': 'ra',
                       'dec': 'dec',
                       'theta_jn': 'thetaJN',
                       'psi': 'psi',
                       'geocent_time': 'tcoal',
                       'phase': 'Phicoal',
                       'chi_1': 'chi1z',
                       'chi_2': 'chi2z',}
    if tides:
        parameter_map.update({'lambda_1': 'Lambda1', 'lambda_2': 'Lambda2'})
    with h5py.File(file_path, 'r') as f:
        event_parameters = {}
        for param, h5_key in parameter_map.items():
            event_parameters[param] = f['event_parameters'][h5_key][event_index]
        covariance = f['covariance'][:, :, event_index]
    if numpy.any(numpy.isnan(covariance)):
                raise ValueError(f"Covariance matrix for event index {event_index} contains NaN values.")

    if check_covariance:
        # checks if the covariance matrix is positive definite
        cov = (covariance + covariance.T) / 2  # Ensure symmetry
        eigen_values, eigen_vectors = numpy.linalg.eigh(cov)
        if numpy.any(eigen_values <= 0):
            logging.warning(f"Covariance matrix for event index {event_index} is not positive definite, \n"
                            f'Clipping {numpy.sum(eigen_values <= 0)} non-positive eigenvalues')

            eigen_values = numpy.clip(eigen_values, a_min=1e-30, a_max=None)
            covariance = eigen_vectors @ numpy.diag(eigen_values) @ eigen_vectors.T
    
    samples = numpy.random.multivariate_normal(mean=list(event_parameters.values()), cov=covariance, size=int(number_of_samples))
    posterior_samples = pandas.DataFrame(samples, columns=list(event_parameters.keys()))
    posterior_samples['snr'] = snr
    if enforce_physicality:
        mask = (posterior_samples['chirp_mass'] > 0) & \
                (posterior_samples['symmetric_mass_ratio'] > 0) & \
                (posterior_samples['symmetric_mass_ratio'] <= 0.25) & \
                (posterior_samples['luminosity_distance'] > 0) &\
                (posterior_samples['chi_1'] > -0.99) & \
                (posterior_samples['chi_1'] < 0.99) & \
                (posterior_samples['chi_2'] > -0.99) & \
                (posterior_samples['chi_2'] < 0.99)
        posterior_samples = posterior_samples[mask]

    if tides:
        posterior_samples = bilby.gw.conversion.generate_all_bns_parameters(posterior_samples)
    else:
        posterior_samples = bilby.gw.conversion.generate_all_bbh_parameters(posterior_samples)
    return posterior_samples

