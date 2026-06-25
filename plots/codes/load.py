import h5py, numpy, pandas, bilby
from math import sqrt
from plot_utils import *

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Which index corresponds to which parameter in the covariance matrix
PARAM_IDX = {
    'chirp_mass': 0,
    'symmetric_mass_ratio': 1,
    'luminosity_distance': 2,
    'ra': 3,
    'dec': 4,
    'theta_jn': 5,
    'psi': 6,
    'geocent_time': 7,
    'phase': 8,
    'chi_1': 9,
    'chi_2': 10,
    'lambda_tilde': 11,
    'delta_lambda_tilde': 12,
}

# I hate GWFAST names! I will name them in a more bilby way which is easy enough for stupid Koustav to remember
H5_KEY_MAP = {
    'chirp_mass': 'Mc',
    'symmetric_mass_ratio': 'eta',
    'luminosity_distance': 'dL',
    'ra': 'ra',
    'dec': 'dec',
    'theta_jn': 'thetaJN',
    'psi': 'psi',
    'geocent_time': 'tcoal',
    'phase': 'Phicoal',
    'chi_1': 'chi1z',
    'chi_2': 'chi2z',
    'lambda_1': 'Lambda1',
    'lambda_2': 'Lambda2',
    'redshift': 'z',
}

# Parameters whose marginal sigma comes directly from the covariance diagonal.
# These are exactly the parameters in PARAM_IDX (the Fisher basis).
_SIGMA_PARAMS = list(PARAM_IDX)

# Parameters for which a fractional uncertainty makes physical sense
# (i.e. the parameter is strictly positive and lives in H5_KEY_MAP).
_FRACTIONAL_PARAMS = [
    'chirp_mass', 
    'luminosity_distance',
    'symmetric_mass_ratio',
]

def _make_sigma(param):
    """compute_fn: sqrt(cov[param, param])"""
    return lambda raw, cov: sqrt(cov[param])

def _make_relative_difference(param):
    """compute_fn: sqrt(cov[param, param]) / raw[param]"""
    return lambda raw, cov: sqrt(cov[param]) / raw[param]

# delta_X        -> absolute uncertainty from Fisher diagonal
DERIVED = {
    f'delta_{p}': ([], [p], _make_sigma(p))
    for p in _SIGMA_PARAMS
}

# delta_X_fractional -> relative uncertainty; needs X loaded from h5 as well
DERIVED.update({
    f'delta_{p}_fractional': ([p], [p], _make_relative_difference(p))
    for p in _FRACTIONAL_PARAMS
})

ALL_PARAMETERS = list(H5_KEY_MAP) + list(DERIVED)

class FisherResults:
    '''
    I probably did an overkill here. But meh!
    '''
    def __init__(self, file_path):
        self.file_path = file_path
        self._read_metadata()

    def _read_metadata(self):
        with h5py.File(self.file_path, 'r') as f:
            self.snr_threshold = float(f.attrs['snr_threshold'])
            self.total_number_of_events = int(f.attrs['total_events'])
            self.detected_events = int(f.attrs['detected_events'])
            self.detectors = parse_combo(f.attrs['detectors'])
        logging.info('Loaded Fisher file:\n'
            f'{self.file_path}\n'
            f' which has detectors : {self.detectors}\n'
            f' If I put an SNR threshold of: {self.snr_threshold}\n'
            f' then I detected : {self.detected_events} / {self.total_number_of_events} events')

    @property
    def metadata(self):
        return {'snr_threshold': self.snr_threshold,
                'total_number_of_events': self.total_number_of_events,
                'detected_events': self.detected_events,
                'detectors': self.detectors}

    def load(self, parameters=None):
        """
        Load per-event data from the file.

        Parameters
        ----------
        parameters : list of str, optional
            Mix of raw parameters (e.g. 'chirp_mass') and derived quantities
            (e.g. 'delta_luminosity_distance_fractional').
            'snr' and 'sky_area_90' are always included.
            Defaults to all available parameters.

        Returns
        -------
        dict with keys:
            'metadata' : dict
            'events' : list of per-event dicts
        """
        parameters = parameters or ALL_PARAMETERS
        self._validate(parameters)

        requested_raw = [p for p in parameters if p in H5_KEY_MAP]
        requested_derived = [p for p in parameters if p in DERIVED]

        # Collect side-dependencies introduced by derived quantities
        h5_side_dependencies  = set()
        cov_side_dependencies = set()
        for d in requested_derived:
            h5_dependencies, cov_dependencies, _ = DERIVED[d]
            h5_side_dependencies.update(h5_dependencies)
            cov_side_dependencies.update(cov_dependencies)

        h5_to_load = set(requested_raw) | h5_side_dependencies
        cov_to_load = {p: PARAM_IDX[p] for p in cov_side_dependencies if p in PARAM_IDX}

        raw_arrays, cov_diag, snr, sky_area_90, is_detected = self._read_arrays(h5_to_load, cov_to_load)

        events = self._build_events(snr, sky_area_90, is_detected, raw_arrays, cov_diag, requested_raw, requested_derived)

        return {'metadata': self.metadata, 'events': events}

    def load_event_parameters(self,
                              event_index,
                              number_of_samples,
                              rng=250114,
                              check_covariance=False,
                              tides=True,
                              enforce_physicality=True,
                              npool=10):
        """
        Draw posterior samples for a single event via the Fisher covariance.
 
        Parameters
        ----------
        event_index : int
        number_of_samples : int
        rng : int, optional
        check_covariance : bool, optional
        tides : bool, optional
        enforce_physicality : bool, optional
        npool : int, optional
 
        Returns
        -------
        pandas.DataFrame of posterior samples

        Note: If enforce_physicality is True, the number of returned samples may be less than number_of_samples.
        """
        numpy.random.seed(rng)
 
        parameter_map = {
            'chirp_mass': 'Mc',
            'symmetric_mass_ratio': 'eta',
            'luminosity_distance': 'dL',
            'ra': 'ra',
            'dec': 'dec',
            'theta_jn': 'thetaJN',
            'psi': 'psi',
            'geocent_time': 'tcoal',
            'phase': 'Phicoal',
            'chi_1': 'chi1z',
            'chi_2': 'chi2z'}
 
        with h5py.File(self.file_path, 'r') as f:
            snr = f['snr'][event_index]
            event_parameters = {
                param: f['event_parameters'][h5_key][event_index]
                for param, h5_key in parameter_map.items()
            }
            if tides:
                lambda_1_true = f['event_parameters']['Lambda1'][event_index]
                lambda_2_true = f['event_parameters']['Lambda2'][event_index]
            covariance = f['covariance'][:, :, event_index]
 
        if numpy.any(numpy.isnan(covariance)):
            raise ValueError(f"Covariance for event {event_index} contains NaN values.")
 
        if tides:
            event_parameters['lambda_tilde'], event_parameters['delta_lambda_tilde'] = \
                self._to_tidal_fisher_basis(lambda_1_true, lambda_2_true,
                                            event_parameters['chirp_mass'],
                                            event_parameters['symmetric_mass_ratio'])
 
        covariance = self._fix_covariance(covariance, event_index, check_covariance)
 
        samples = numpy.random.multivariate_normal(mean=list(event_parameters.values()),
                                                   cov=covariance,
                                                   size=int(number_of_samples))
        posterior_samples = pandas.DataFrame(samples, columns=list(event_parameters.keys()))
        posterior_samples['snr'] = snr
 
        if tides:
            posterior_samples = self._tidal_fisher_to_component(posterior_samples)
 
        if enforce_physicality:
            posterior_samples = self._apply_physicality_cuts(posterior_samples, tides)
 
        convert = (bilby.gw.conversion.generate_all_bns_parameters if tides
                   else bilby.gw.conversion.generate_all_bbh_parameters)
        return convert(posterior_samples, npool=npool)

    def _validate(self, parameters):
        # Scream loudly if the user asks for something that doesn't exist
        unknown = [p for p in parameters if p not in H5_KEY_MAP and p not in DERIVED]
        if unknown:
            logging.error(f"Unknown parameters: {unknown}.\nValid: {ALL_PARAMETERS}")

    def _read_arrays(self, h5_to_load, cov_to_load):
        with h5py.File(self.file_path, 'r') as f:
            snr = f['snr'][:]
            sky_area_90 = f['sky_area_90'][:]
            is_detected = f['is_detected'][:]

            raw_arrays = {}
            for param in h5_to_load:
                h5_key = H5_KEY_MAP[param]
                if h5_key in f['event_parameters']:
                    raw_arrays[param] = f['event_parameters'][h5_key][:]
                else:
                    logging.warning(f"'{h5_key}' not found in event_parameters, skipping '{param}'")

            # Pull only the diagonal elements we need: cov[i, i, :]
            cov_diag = {param: f['covariance'][idx, idx, :] for param, idx in cov_to_load.items()}

        return raw_arrays, cov_diag, snr, sky_area_90, is_detected

    def _build_events(self, snr, sky_area_90, is_detected,
                      raw_arrays, cov_diag,
                      requested_raw, requested_derived):
        events = []
        for k in numpy.where(is_detected)[0]:
            # Scalar snapshots for this event
            raw  = {p: float(raw_arrays[p][k]) for p in raw_arrays}
            cov  = {p: float(cov_diag[p][k]) for p in cov_diag}

            ev = {'snr': float(snr[k]),
                'sky_area_90': float(sky_area_90[k])}
            for p in requested_raw:
                if p in raw:
                    ev[p] = raw[p]

            for d in requested_derived:
                _, _, compute_fn = DERIVED[d]
                try:
                    ev[d] = compute_fn(raw, cov)
                except Exception as e:
                    logging.warning(f"Could not compute '{d}' for event {k}: {e}")
                    ev[d] = float('nan')

            events.append(ev)
        return events

    @staticmethod
    def _to_tidal_fisher_basis(lambda_1, lambda_2, chirp_mass, eta):
        """Convert (lambda_1, lambda_2) -> (lambda_tilde, delta_lambda_tilde)."""
        mass_ratio = bilby.gw.conversion.symmetric_mass_ratio_to_mass_ratio(eta)
        mass_1, mass_2 = bilby.gw.conversion.chirp_mass_and_mass_ratio_to_component_masses(chirp_mass, mass_ratio)
        lambda_tilde = bilby.gw.conversion.lambda_1_lambda_2_to_lambda_tilde(lambda_1, lambda_2, mass_1, mass_2)
        delta_lambda_tilde = bilby.gw.conversion.lambda_1_lambda_2_to_delta_lambda_tilde(lambda_1, lambda_2, mass_1, mass_2)
        return lambda_tilde, delta_lambda_tilde

    @staticmethod
    def _tidal_fisher_to_component(posterior_samples):
        """Convert sampled (lambda_tilde, delta_lambda_tilde) -> (lambda_1, lambda_2)."""
        mass_ratio = bilby.gw.conversion.symmetric_mass_ratio_to_mass_ratio(posterior_samples['symmetric_mass_ratio'])
        mass_1, mass_2 = bilby.gw.conversion.chirp_mass_and_mass_ratio_to_component_masses(posterior_samples['chirp_mass'], mass_ratio)
        lambda_1, lambda_2 = bilby.gw.conversion.lambda_tilde_delta_lambda_tilde_to_lambda_1_lambda_2(posterior_samples['lambda_tilde'],
                                                                                                      posterior_samples['delta_lambda_tilde'],
                                                                                                      mass_1, mass_2,)
        posterior_samples['lambda_1'] = lambda_1
        posterior_samples['lambda_2'] = lambda_2
        return posterior_samples.drop(columns=['lambda_tilde', 'delta_lambda_tilde'])

    @staticmethod
    def _fix_covariance(covariance, event_index, check_covariance):
        if not check_covariance:
            return covariance
        cov = (covariance + covariance.T) / 2
        eigenvalues, eigenvectors = numpy.linalg.eigh(cov)
        if numpy.any(eigenvalues <= 0):
            logging.warning(
                f"Covariance for event {event_index} is not positive definite. "
                f"Clipping {numpy.sum(eigenvalues <= 0)} non-positive eigenvalues."
            )
            eigenvalues = numpy.clip(eigenvalues, a_min=1e-30, a_max=None)
            covariance = eigenvectors @ numpy.diag(eigenvalues) @ eigenvectors.T
        return covariance

    @staticmethod
    def _apply_physicality_cuts(posterior_samples, tides):
        mask = (
            (posterior_samples['chirp_mass'] > 0) &
            (posterior_samples['symmetric_mass_ratio'] > 0) &
            (posterior_samples['symmetric_mass_ratio'] <= 0.25) &
            (posterior_samples['luminosity_distance'] > 0) &
            (posterior_samples['chi_1'].between(-0.99, 0.99)) &
            (posterior_samples['chi_2'].between(-0.99, 0.99))
        )
        if tides:
            mask &= (posterior_samples['lambda_1'] >= 0) & (posterior_samples['lambda_2'] >= 0)
        return posterior_samples[mask]