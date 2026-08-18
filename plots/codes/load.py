import h5py, json, logging, numpy, pandas
from dataclasses import dataclass, field
from math import sqrt

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# Which index corresponds to which parameter in the covariance matrix.
#
# Indices 3, 4 and 5 are gwfast's (theta, phi, iota) -- see FISHER_BASIS_INDEX
# below.  They are labelled (ra, dec, theta_jn) here because ra = phi exactly,
# dec = pi/2 - theta (so the marginal sigma is identical), and iota = theta_jn
# for the aligned-spin waveforms in these catalogues.  The delta_* quantities
# derived from the covariance diagonal are therefore correct under either name;
# anything that needs the *gwfast* basis (e.g. extracting a Fisher sub-block)
# must use FISHER_BASIS_INDEX instead.
PARAM_IDX = {
    "chirp_mass": 0,
    "symmetric_mass_ratio": 1,
    "luminosity_distance": 2,
    "ra": 3,
    "dec": 4,
    "theta_jn": 5,
    "psi": 6,
    "geocent_time": 7,
    "phase": 8,
    "chi_1": 9,
    "chi_2": 10,
    "lambda_tilde": 11,
    "delta_lambda_tilde": 12,
}

# I hate GWFAST names! I will name them in a more bilby way which is easy enough for stupid Koustav to remember
H5_KEY_MAP = {
    "chirp_mass": "Mc",
    "symmetric_mass_ratio": "eta",
    "luminosity_distance": "dL",
    "ra": "ra",
    "dec": "dec",
    "theta_jn": "thetaJN",
    "psi": "psi",
    "geocent_time": "tcoal",
    "phase": "Phicoal",
    "chi_1": "chi1z",
    "chi_2": "chi2z",
    "lambda_1": "Lambda1",
    "lambda_2": "Lambda2",
    "redshift": "z",
    "mass_1_source": "m1_src",
    "mass_2_source": "m2_src",
}

# gwfast's ordering of the 11x11 per-event Fisher/covariance matrices.  Used as
# a fallback when an H5 file predates the 'parameters_indices' attribute; the
# attribute is authoritative and is read by ``FisherResults._read_metadata``.
DEFAULT_FISHER_BASIS_INDEX = {
    "Mc": 0,
    "eta": 1,
    "dL": 2,
    "theta": 3,
    "phi": 4,
    "iota": 5,
    "psi": 6,
    "tcoal": 7,
    "Phicoal": 8,
    "chi1z": 9,
    "chi2z": 10,
}

# Source-frame parameters the population pipeline works in, in canonical order.
# These are the keys of the ``events`` dict consumed by pop_models.PopModel.
POPULATION_KEYS = ["mass_1_source", "mass_2_source", "redshift"]

# The (Mc, eta, dL) sub-block of the per-event Fisher is the one that carries
# information about (mass_1_source, mass_2_source, redshift).
_DETECTOR_FRAME_KEYS = ["Mc", "eta", "dL"]

# Parameters whose marginal sigma comes directly from the covariance diagonal.
# These are exactly the parameters in PARAM_IDX (the Fisher basis).
_SIGMA_PARAMS = list(PARAM_IDX)

# Parameters for which a fractional uncertainty makes physical sense
# (i.e. the parameter is strictly positive and lives in H5_KEY_MAP).
_FRACTIONAL_PARAMS = [
    "chirp_mass",
    "luminosity_distance",
    "symmetric_mass_ratio",
]


def _make_sigma(param):
    """compute_fn: sqrt(cov[param, param])"""
    return lambda raw, cov: sqrt(cov[param])


def _make_relative_difference(param):
    """compute_fn: sqrt(cov[param, param]) / raw[param]"""
    return lambda raw, cov: sqrt(cov[param]) / raw[param]


# delta_X -> absolute uncertainty from Fisher diagonal
DERIVED = {f"delta_{p}": ([], [p], _make_sigma(p)) for p in _SIGMA_PARAMS}

# delta_X_fractional -> relative uncertainty; needs X loaded from h5 as well
DERIVED.update(
    {
        f"delta_{p}_fractional": ([p], [p], _make_relative_difference(p))
        for p in _FRACTIONAL_PARAMS
    }
)

ALL_PARAMETERS = list(H5_KEY_MAP) + list(DERIVED)


# ---------------------------------------------------------------------------
# Source-frame -> detector-frame (gwfast Fisher basis) conversions
# ---------------------------------------------------------------------------


def _default_cosmology():
    """Planck18, imported lazily so that importing this module stays cheap."""
    from astropy.cosmology import Planck18

    return Planck18


def detector_frame_from_source_frame(
    mass_1_source, mass_2_source, redshift, cosmology=None
):
    """
    Map source-frame parameters onto gwfast's Fisher basis.  Vectorised.
    """
    from astropy import units

    if cosmology is None:
        cosmology = _default_cosmology()

    total_mass = mass_1_source + mass_2_source
    chirp_mass = (mass_1_source * mass_2_source) ** 0.6 / total_mass**0.2 * (
        1 + redshift
    )
    symmetric_mass_ratio = mass_1_source * mass_2_source / total_mass**2
    luminosity_distance = (
        cosmology.luminosity_distance(redshift).to(units.Gpc).value
    )
    return chirp_mass, symmetric_mass_ratio, luminosity_distance


def jacobian_source_frame_to_detector_frame(
    mass_1_source, mass_2_source, redshift, cosmology=None, h_rel=1e-4, h_abs=1e-6
):
    """
    Forward Jacobian J_{ab} = d f_a / d x_b where

        f = (chirp_mass, symmetric_mass_ratio, luminosity_distance)
        x = (mass_1_source, mass_2_source, redshift)

    Computed via centred finite differences.

    Returns
    -------
    jacobian : (N, 3, 3) ndarray
    """
    if cosmology is None:
        cosmology = _default_cosmology()

    number_of_events = len(mass_1_source)
    jacobian = numpy.zeros((number_of_events, 3, 3))

    for column, base_array in enumerate([mass_1_source, mass_2_source, redshift]):
        step = numpy.maximum(h_rel * numpy.abs(base_array), h_abs)

        arguments_plus = [mass_1_source.copy(), mass_2_source.copy(), redshift.copy()]
        arguments_minus = [mass_1_source.copy(), mass_2_source.copy(), redshift.copy()]
        arguments_plus[column] = base_array + step
        arguments_minus[column] = base_array - step

        f_plus = detector_frame_from_source_frame(*arguments_plus, cosmology=cosmology)
        f_minus = detector_frame_from_source_frame(
            *arguments_minus, cosmology=cosmology
        )
        for row in range(3):
            jacobian[:, row, column] = (f_plus[row] - f_minus[row]) / (2.0 * step)

    return jacobian


def _detector_frame_chirp_eta(mass_1_det, mass_2_det):
    """Detector-frame chirp mass and symmetric mass ratio (cosmology-free)."""
    total_mass = mass_1_det + mass_2_det
    chirp_mass = (mass_1_det * mass_2_det) ** 0.6 / total_mass ** 0.2
    symmetric_mass_ratio = mass_1_det * mass_2_det / total_mass ** 2
    return chirp_mass, symmetric_mass_ratio


def jacobian_detector_masses(
    mass_1_det, mass_2_det, distance, h_rel=1e-4, h_abs=1e-6
):
    """
    Forward Jacobian J_{ab} = d f_a / d x_b where

        f = (chirp_mass_det, symmetric_mass_ratio, luminosity_distance)
        x = (mass_1_det, mass_2_det, luminosity_distance)
    Computed via centred finite differences.
    Returns
    -------
    jacobian : (N, 3, 3) ndarray
    """
    number_of_events = len(mass_1_det)
    jacobian = numpy.zeros((number_of_events, 3, 3))
    base = [numpy.asarray(mass_1_det, float), numpy.asarray(mass_2_det, float),
            numpy.asarray(distance, float)]

    def evaluate(m1, m2, dl):
        chirp, eta = _detector_frame_chirp_eta(m1, m2)
        return chirp, eta, dl

    for column, array in enumerate(base):
        step = numpy.maximum(h_rel * numpy.abs(array), h_abs)
        plus = [a.copy() for a in base]
        minus = [a.copy() for a in base]
        plus[column] = array + step
        minus[column] = array - step
        f_plus = evaluate(*plus)
        f_minus = evaluate(*minus)
        for row in range(3):
            jacobian[:, row, column] = (f_plus[row] - f_minus[row]) / (2.0 * step)

    return jacobian


# ---------------------------------------------------------------------------
# Population catalogue container
# ---------------------------------------------------------------------------


@dataclass
class PopulationCatalogue:
    """
    A detected catalogue in the form the population Fisher pipeline consumes.

    Attributes
    ----------
    events : dict
        Per-event source-frame column arrays keyed by bilby-style names
        ('mass_1_source', 'mass_2_source', 'redshift', and the derived
        'mass_ratio' = mass_2_source / mass_1_source).
    snr : ndarray
        Per-event optimal SNR, aligned element-for-element with ``events``.
    fisher : ndarray or None
        (N, 3, 3) per-event Fisher matrix in the
        (mass_1_source, mass_2_source, redshift) basis.  ``None`` when the
        catalogue was loaded with ``with_fisher=False``.
    number_detected : int
        Size of the analysis sample: ``is_detected & snr >= snr_threshold``.
        This is the number of events in ``events`` / ``snr`` / ``fisher``.
    number_above_threshold : int
        Injections with ``snr >= snr_threshold``, with **no** ``is_detected``
        filter.  This is the numerator of P_det, kept separate on purpose:
        inferring it from the already-filtered analysis sample collapses
        P_det to 1 and silently removes the 1/P_det amplification in Terms
        III and IV.
    number_total : int
        Total injections in the catalogue (P_det denominator).
    snr_threshold : float
    metadata : dict
        The file-level metadata from ``FisherResults.metadata``.
    """

    events: dict
    snr: numpy.ndarray
    fisher: numpy.ndarray
    number_detected: int
    number_above_threshold: int
    number_total: int
    snr_threshold: float
    metadata: dict = field(default_factory=dict)

    @property
    def p_det(self):
        """P_det(lambda) = number_above_threshold / number_total."""
        return float(self.number_above_threshold) / float(self.number_total)

    def __len__(self):
        return self.number_detected

    def select(self, mask):
        """
        A new catalogue restricted to ``mask`` (boolean or index array).

        ``number_above_threshold`` and ``number_total`` are deliberately left
        untouched: they describe the injection campaign, not the analysis
        sample, and trimming them would force P_det = 1.
        """
        return PopulationCatalogue(
            events={key: numpy.asarray(value)[mask] for key, value in self.events.items()},
            snr=numpy.asarray(self.snr)[mask],
            fisher=None if self.fisher is None else numpy.asarray(self.fisher)[mask],
            number_detected=int(len(numpy.asarray(self.snr)[mask])),
            number_above_threshold=self.number_above_threshold,
            number_total=self.number_total,
            snr_threshold=self.snr_threshold,
            metadata=dict(self.metadata),
        )


class FisherResults:
    """
    I probably did an overkill here. But meh!
    """

    def __init__(self, file_path):
        self.file_path = file_path
        self._read_metadata()

    def _read_metadata(self):
        from plot_utils import parse_combo

        with h5py.File(self.file_path, "r") as f:
            self.snr_threshold = float(f.attrs["snr_threshold"])
            self.total_number_of_events = int(f.attrs["total_events"])
            self.detected_events = int(f.attrs["detected_events"])
            self.detectors = parse_combo(f.attrs["detectors"])
            # gwfast writes its Fisher-basis ordering into the file; trust it
            # over the hardcoded fallback.
            if "parameters_indices" in f.attrs:
                self.fisher_basis_index = {
                    name: int(index)
                    for name, index in json.loads(f.attrs["parameters_indices"]).items()
                }
            else:
                logging.warning(
                    "No 'parameters_indices' attribute in the file; assuming "
                    "gwfast's default Fisher basis ordering."
                )
                self.fisher_basis_index = dict(DEFAULT_FISHER_BASIS_INDEX)
        logging.info(
            "Loaded Fisher file:\n"
            f"{self.file_path}\n"
            f" which has detectors : {self.detectors}\n"
            f" If I put an SNR threshold of: {self.snr_threshold}\n"
            f" then I detect : {self.detected_events} / {self.total_number_of_events} events"
        )

    @property
    def metadata(self):
        return {
            "snr_threshold": self.snr_threshold,
            "total_number_of_events": self.total_number_of_events,
            "detected_events": self.detected_events,
            "detectors": self.detectors,
        }

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
        h5_side_dependencies = set()
        cov_side_dependencies = set()
        for d in requested_derived:
            h5_dependencies, cov_dependencies, _ = DERIVED[d]
            h5_side_dependencies.update(h5_dependencies)
            cov_side_dependencies.update(cov_dependencies)

        h5_to_load = set(requested_raw) | h5_side_dependencies
        cov_to_load = {p: PARAM_IDX[p] for p in cov_side_dependencies if p in PARAM_IDX}

        raw_arrays, cov_diag, snr, sky_area_90, is_detected = self._read_arrays(
            h5_to_load, cov_to_load
        )

        events = self._build_events(
            snr,
            sky_area_90,
            is_detected,
            raw_arrays,
            cov_diag,
            requested_raw,
            requested_derived,
        )

        return {"metadata": self.metadata, "events": events}

    def load_population_catalogue(
        self,
        snr_threshold,
        parameters=None,
        cosmology=None,
        with_fisher=True,
    ):
        """
        Load a detected catalogue for the population (hyperparameter) pipeline.

        Opens the file once and returns everything the population Fisher needs:
        source-frame column arrays, the per-event SNR, the per-event Fisher
        matrix rotated into the source-frame basis, and the three event counts.

        Parameters
        ----------
        snr_threshold : float
            Analysis threshold.  The returned arrays are restricted to
            ``is_detected & (snr >= snr_threshold)``.
        parameters : list[str], optional
            Source-frame parameters to load, in bilby-style naming.  Defaults
            to ``POPULATION_KEYS``.  ``mass_ratio`` is derived automatically
            when both component masses are present.
        cosmology : astropy cosmology, optional
            Used for the source-frame basis rotation.  Defaults to Planck18.
        with_fisher : bool, optional
            When False, skip reading and rotating the 11x11 per-event Fisher
            matrices (the expensive part) and return ``fisher=None``.

        Returns
        -------
        PopulationCatalogue
        """
        if parameters is None:
            parameters = list(POPULATION_KEYS)

        unknown = [p for p in parameters if p not in H5_KEY_MAP]
        if unknown:
            raise ValueError(
                f"Unknown population parameters: {unknown}.  "
                f"Valid: {sorted(H5_KEY_MAP)}"
            )

        events = {}
        with h5py.File(self.file_path, "r") as f:
            is_detected = f["is_detected"][:]
            snr_all = f["snr"][:]
            number_total = int(snr_all.shape[0])

            above_threshold = snr_all >= snr_threshold
            # Analysis sample: gwfast produced a valid FIM (is_detected) AND
            # the event passes the user's threshold.
            analysis_mask = is_detected & above_threshold
            number_detected = int(analysis_mask.sum())
            # P_det numerator: threshold only, so that P_det stays a pure
            # function of snr_threshold.  See PopulationCatalogue's docstring.
            number_above_threshold = int(above_threshold.sum())

            snr = snr_all[analysis_mask]

            for parameter in parameters:
                h5_key = H5_KEY_MAP[parameter]
                if h5_key not in f["event_parameters"]:
                    logging.warning(
                        f"'{h5_key}' not found in event_parameters; "
                        f"skipping '{parameter}'."
                    )
                    continue
                events[parameter] = f["event_parameters"][h5_key][analysis_mask]

            fisher_source_frame = None
            if with_fisher:
                fisher_detector_frame = f["fisher"][:, :, analysis_mask]

        if "mass_1_source" in events and "mass_2_source" in events:
            events["mass_ratio"] = events["mass_2_source"] / events["mass_1_source"]

        if with_fisher:
            fisher_source_frame = self._rotate_fisher_to_source_frame(
                fisher_detector_frame, events, number_detected, cosmology
            )

        logging.info(
            f"Loaded {number_detected} analysis events "
            f"(is_detected & snr>={snr_threshold}) / "
            f"{number_above_threshold} above threshold / "
            f"{number_total} total injections from {self.file_path}  "
            f"[P_det = {number_above_threshold/number_total:.4f}]"
        )

        return PopulationCatalogue(
            events=events,
            snr=snr,
            fisher=fisher_source_frame,
            number_detected=number_detected,
            number_above_threshold=number_above_threshold,
            number_total=number_total,
            snr_threshold=float(snr_threshold),
            metadata=self.metadata,
        )

    def _rotate_fisher_to_source_frame(
        self, fisher_detector_frame, events, number_detected, cosmology
    ):
        """
        Push the (Mc, eta, dL) sub-block of the per-event Fisher into the
        (mass_1_source, mass_2_source, redshift) basis via Gamma = J^T Gamma J.

        Parameters
        ----------
        fisher_detector_frame : (11, 11, N) ndarray as stored in the H5 file
        events : dict – must contain the three source-frame parameters
        number_detected : int
        cosmology : astropy cosmology or None

        Returns
        -------
        (N, 3, 3) ndarray
        """
        missing = [key for key in POPULATION_KEYS if key not in events]
        if missing:
            raise ValueError(
                f"Cannot rotate the per-event Fisher into the source frame "
                f"without {missing}.  Either request them via `parameters` "
                f"or pass with_fisher=False."
            )

        basis_indices = [self.fisher_basis_index[key] for key in _DETECTOR_FRAME_KEYS]
        block = numpy.ix_(basis_indices, basis_indices, numpy.arange(number_detected))
        fisher_sub_block = fisher_detector_frame[block].transpose(2, 0, 1)

        jacobian = jacobian_source_frame_to_detector_frame(
            events["mass_1_source"],
            events["mass_2_source"],
            events["redshift"],
            cosmology=cosmology,
        )

        # einsum: sum over a, b of J[n,a,i] * fisher_sub_block[n,a,b] * J[n,b,j]
        fisher_source_frame = numpy.einsum(
            "nai,nab,nbj->nij", jacobian, fisher_sub_block, jacobian
        )
        logging.info(
            f"Rotated per-event Fisher into the source frame: "
            f"{fisher_source_frame.shape}"
        )
        return fisher_source_frame

    def load_event_parameters(
        self,
        event_index,
        number_of_samples,
        rng=250114,
        check_covariance=False,
        tides=True,
        enforce_physicality=True,
        npool=10,
    ):
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
        import bilby

        numpy.random.seed(rng)

        parameter_map = {
            "chirp_mass": "Mc",
            "symmetric_mass_ratio": "eta",
            "luminosity_distance": "dL",
            "ra": "ra",
            "dec": "dec",
            "theta_jn": "thetaJN",
            "psi": "psi",
            "geocent_time": "tcoal",
            "phase": "Phicoal",
            "chi_1": "chi1z",
            "chi_2": "chi2z",
        }

        with h5py.File(self.file_path, "r") as f:
            snr = f["snr"][event_index]
            event_parameters = {
                param: f["event_parameters"][h5_key][event_index]
                for param, h5_key in parameter_map.items()
            }
            if tides:
                lambda_1_true = f["event_parameters"]["Lambda1"][event_index]
                lambda_2_true = f["event_parameters"]["Lambda2"][event_index]
            covariance = f["covariance"][:, :, event_index]

        if numpy.any(numpy.isnan(covariance)):
            raise ValueError(f"Covariance for event {event_index} contains NaN values.")

        if tides:
            event_parameters["lambda_tilde"], event_parameters["delta_lambda_tilde"] = (
                self._to_tidal_fisher_basis(
                    lambda_1_true,
                    lambda_2_true,
                    event_parameters["chirp_mass"],
                    event_parameters["symmetric_mass_ratio"],
                )
            )

        covariance = self._fix_covariance(covariance, event_index, check_covariance)

        samples = numpy.random.multivariate_normal(
            mean=list(event_parameters.values()),
            cov=covariance,
            size=int(number_of_samples),
        )
        posterior_samples = pandas.DataFrame(
            samples, columns=list(event_parameters.keys())
        )
        posterior_samples["snr"] = snr

        if tides:
            posterior_samples = self._tidal_fisher_to_component(posterior_samples)

        if enforce_physicality:
            posterior_samples = self._apply_physicality_cuts(posterior_samples, tides)

        convert = (
            bilby.gw.conversion.generate_all_bns_parameters
            if tides
            else bilby.gw.conversion.generate_all_bbh_parameters
        )
        return convert(posterior_samples, npool=npool)

    def _validate(self, parameters):
        # Scream loudly if the user asks for something that doesn't exist
        unknown = [p for p in parameters if p not in H5_KEY_MAP and p not in DERIVED]
        if unknown:
            logging.error(f"Unknown parameters: {unknown}.\nValid: {ALL_PARAMETERS}")

    def _read_arrays(self, h5_to_load, cov_to_load):
        with h5py.File(self.file_path, "r") as f:
            snr = f["snr"][:]
            sky_area_90 = f["sky_area_90"][:]
            is_detected = f["is_detected"][:]

            raw_arrays = {}
            for param in h5_to_load:
                h5_key = H5_KEY_MAP[param]
                if h5_key in f["event_parameters"]:
                    raw_arrays[param] = f["event_parameters"][h5_key][:]
                else:
                    logging.warning(
                        f"'{h5_key}' not found in event_parameters, skipping '{param}'"
                    )

            # BBH catalogues carry no tidal parameters, so their covariance is
            # 11x11 while PARAM_IDX also names lambda_tilde (11) and
            # delta_lambda_tilde (12).  Skip whatever the file does not have
            # rather than raising an IndexError on the default parameter list.
            covariance_size = f["covariance"].shape[0]
            out_of_range = {
                param: idx for param, idx in cov_to_load.items() if idx >= covariance_size
            }
            if out_of_range:
                logging.warning(
                    f"Covariance is {covariance_size}x{covariance_size}; "
                    f"skipping {sorted(out_of_range)} (index "
                    f"{sorted(out_of_range.values())} out of range)."
                )

            # Pull only the diagonal elements we need: cov[i, i, :]
            cov_diag = {
                param: f["covariance"][idx, idx, :]
                for param, idx in cov_to_load.items()
                if idx < covariance_size
            }

        return raw_arrays, cov_diag, snr, sky_area_90, is_detected

    def _build_events(
        self,
        snr,
        sky_area_90,
        is_detected,
        raw_arrays,
        cov_diag,
        requested_raw,
        requested_derived,
    ):
        events = []
        for k in numpy.where(is_detected)[0]:
            # Scalar snapshots for this event
            raw = {p: float(raw_arrays[p][k]) for p in raw_arrays}
            cov = {p: float(cov_diag[p][k]) for p in cov_diag}

            ev = {"snr": float(snr[k]), "sky_area_90": float(sky_area_90[k])}
            for p in requested_raw:
                if p in raw:
                    ev[p] = raw[p]

            for d in requested_derived:
                _, _, compute_fn = DERIVED[d]
                try:
                    ev[d] = compute_fn(raw, cov)
                except Exception as e:
                    logging.warning(f"Could not compute '{d}' for event {k}: {e}")
                    ev[d] = float("nan")

            events.append(ev)
        return events

    @staticmethod
    def _to_tidal_fisher_basis(lambda_1, lambda_2, chirp_mass, eta):
        """Convert (lambda_1, lambda_2) -> (lambda_tilde, delta_lambda_tilde)."""
        import bilby

        mass_ratio = bilby.gw.conversion.symmetric_mass_ratio_to_mass_ratio(eta)
        mass_1, mass_2 = (
            bilby.gw.conversion.chirp_mass_and_mass_ratio_to_component_masses(
                chirp_mass, mass_ratio
            )
        )
        lambda_tilde = bilby.gw.conversion.lambda_1_lambda_2_to_lambda_tilde(
            lambda_1, lambda_2, mass_1, mass_2
        )
        delta_lambda_tilde = (
            bilby.gw.conversion.lambda_1_lambda_2_to_delta_lambda_tilde(
                lambda_1, lambda_2, mass_1, mass_2
            )
        )
        return lambda_tilde, delta_lambda_tilde

    @staticmethod
    def _tidal_fisher_to_component(posterior_samples):
        """Convert sampled (lambda_tilde, delta_lambda_tilde) -> (lambda_1, lambda_2)."""
        import bilby

        mass_ratio = bilby.gw.conversion.symmetric_mass_ratio_to_mass_ratio(
            posterior_samples["symmetric_mass_ratio"]
        )
        mass_1, mass_2 = (
            bilby.gw.conversion.chirp_mass_and_mass_ratio_to_component_masses(
                posterior_samples["chirp_mass"], mass_ratio
            )
        )
        lambda_1, lambda_2 = (
            bilby.gw.conversion.lambda_tilde_delta_lambda_tilde_to_lambda_1_lambda_2(
                posterior_samples["lambda_tilde"],
                posterior_samples["delta_lambda_tilde"],
                mass_1,
                mass_2,
            )
        )
        posterior_samples["lambda_1"] = lambda_1
        posterior_samples["lambda_2"] = lambda_2
        return posterior_samples.drop(columns=["lambda_tilde", "delta_lambda_tilde"])

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
            (posterior_samples["chirp_mass"] > 0)
            & (posterior_samples["symmetric_mass_ratio"] > 0)
            & (posterior_samples["symmetric_mass_ratio"] <= 0.25)
            & (posterior_samples["luminosity_distance"] > 0)
            & (posterior_samples["chi_1"].between(-0.99, 0.99))
            & (posterior_samples["chi_2"].between(-0.99, 0.99))
        )
        if tides:
            mask &= (posterior_samples["lambda_1"] >= 0) & (
                posterior_samples["lambda_2"] >= 0
            )
        return posterior_samples[mask]


# ---------------------------------------------------------------------------
# Module-level convenience
# ---------------------------------------------------------------------------


def load_population_catalogue(
    file_path, snr_threshold, parameters=None, cosmology=None, with_fisher=True
):
    """
    ``FisherResults(file_path).load_population_catalogue(...)`` in one call.

    For scripts that do not otherwise need the ``FisherResults`` instance.
    See ``FisherResults.load_population_catalogue`` for the arguments.
    """
    return FisherResults(file_path).load_population_catalogue(
        snr_threshold,
        parameters=parameters,
        cosmology=cosmology,
        with_fisher=with_fisher,
    )


def load_injected_masses(file_path):
    """
    All injected source-frame component masses from a gwfast catalogue.

    Returns the *true* (mass_1_source, mass_2_source) of every injection with no
    detection or SNR mask -- the astrophysical population before selection, used
    to fit the fiducial mass-model shape (analogous to
    ``infer_time_delay.load_injected_redshifts`` for the redshift sector).  This
    is distinct from ``load_population_catalogue``, which returns only the
    *detected* events together with their per-event measurement Fisher.

    Parameters
    ----------
    file_path : str
        Catalogue H5 path.

    Returns
    -------
    mass_1_source, mass_2_source : (N,) ndarray
        Injected source-frame primary and secondary masses [M_sun].
    """
    with h5py.File(file_path, "r") as handle:
        group = handle["event_parameters"]
        return (
            group[H5_KEY_MAP["mass_1_source"]][:],
            group[H5_KEY_MAP["mass_2_source"]][:],
        )


def load_detector_frame_fisher(file_path, snr_threshold):
    """
    Per-event measurement Fisher in the **detector-frame** observable basis
    ``(mass_1_det, mass_2_det, luminosity_distance)`` for the analysis sample.

    Unlike ``load_population_catalogue`` (which rotates to the *source* frame
    using a fixed cosmology), this keeps the measurement in the detector frame,
    where it is cosmology-independent -- required by the spectral-siren Fisher,
    which treats the cosmology ``(H0, Omega_m)`` as a hyperparameter and so must
    not fold cosmology into the per-event Fisher.

    The ``(Mc, eta, dL)`` sub-block of the stored 11x11 Fisher is sliced (the same
    convention as ``_rotate_fisher_to_source_frame``, i.e. conditioning on the
    other parameters) and rotated to ``(mass_1_det, mass_2_det, dL)`` via the
    cosmology-free mass Jacobian ``jacobian_detector_masses``.

    Parameters
    ----------
    file_path : str
    snr_threshold : float

    Returns
    -------
    fisher_detector : (N, 3, 3) ndarray
        Per-event Fisher in ``(mass_1_det, mass_2_det, luminosity_distance)``.
    events : dict
        ``mass_1_det``, ``mass_2_det``, ``luminosity_distance`` [Gpc], plus the
        source-frame ``mass_1_source``, ``mass_2_source``, ``redshift`` for
        reference.  Detector masses use the injected ``d_L`` and redshift.
    snr : (N,) ndarray
    number_detected, number_above_threshold, number_total : int
    """
    basis = FisherResults(file_path).fisher_basis_index
    indices = [basis[key] for key in _DETECTOR_FRAME_KEYS]  # Mc, eta, dL
    with h5py.File(file_path, "r") as handle:
        is_detected = handle["is_detected"][:]
        snr_all = handle["snr"][:]
        above_threshold = snr_all >= snr_threshold
        analysis_mask = is_detected & above_threshold
        number_total = int(is_detected.shape[0])
        number_above_threshold = int(above_threshold.sum())

        group = handle["event_parameters"]
        mass_1_source = group["m1_src"][:][analysis_mask]
        mass_2_source = group["m2_src"][:][analysis_mask]
        redshift = group["z"][:][analysis_mask]
        distance = group["dL"][:][analysis_mask]
        snr = snr_all[analysis_mask]
        fisher = handle["fisher"][:]  # (11, 11, N_total)

    number_detected = int(analysis_mask.sum())
    sub_block = fisher[numpy.ix_(indices, indices)][:, :, analysis_mask].transpose(
        2, 0, 1
    )  # (N, 3, 3) in (Mc, eta, dL)

    one_plus_z = 1.0 + redshift
    mass_1_det = mass_1_source * one_plus_z
    mass_2_det = mass_2_source * one_plus_z

    jacobian = jacobian_detector_masses(mass_1_det, mass_2_det, distance)
    fisher_detector = numpy.einsum("nai,nab,nbj->nij", jacobian, sub_block, jacobian)

    events = {
        "mass_1_det": mass_1_det,
        "mass_2_det": mass_2_det,
        "luminosity_distance": distance,
        "mass_1_source": mass_1_source,
        "mass_2_source": mass_2_source,
        "redshift": redshift,
    }
    return (
        fisher_detector, events, snr,
        number_detected, number_above_threshold, number_total,
    )
