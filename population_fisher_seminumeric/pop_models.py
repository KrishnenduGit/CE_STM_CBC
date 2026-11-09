import numpy, logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------


class PopModel:
    """
    Base class for GW population models.

    Subclasses must define
    ----------------------
    parameter_names  : list[str]
        Ordered list of free hyperparameter names.
    fiducial : dict
        Fiducial (reference) hyperparameter values.

    and implement
    -------------
    log_prob(events, params) -> ndarray(N_events,)
        Log (unnormalised) probability per detected event.
        Return -numpy.inf for events outside the model support.

    analytic_score(events, params) -> ndarray(N_events, N_params)
        Closed-form score d ln p / d lambda per event; required by the
        population Fisher.
    """

    parameter_names = []
    fiducial = {}
    phys_event_keys = []  # physical event params this model depends on

    def log_prob(self, events, params):
        raise NotImplementedError

    def parameters_to_vector(self, params):
        """Ordered parameter vector from a dict."""
        return numpy.array([params[n] for n in self.parameter_names], dtype=float)

    def vector_to_parameters(self, vec):
        """Parameter dict from an ordered vector."""
        return {n: float(v) for n, v in zip(self.parameter_names, vec)}

    def __repr__(self):
        return (
            f"{type(self).__name__}("
            + ", ".join(f"{k}={v}" for k, v in self.fiducial.items())
            + ")"
        )


# ---------------------------------------------------------------------------
# Joint model
# ---------------------------------------------------------------------------


class JointModel(PopModel):
    """
    Product of independent models: p(theta | lambda) = prod_k p_k(theta_k | lambda_k).

    Parameters
    ----------
    *models : PopModel instances
        Independent sub-models to combine.

    Examples
    --------
    >>> joint = JointModel(BrokenPowerLawPlusTwoPeaksMass(), MadauDickinsonRedshift())
    >>> joint.parameter_names
    ['alpha', 'beta', 'z_peak', 'alpha_m', 'beta_q', 'm_min', 'm_max']
    """

    def __init__(self, *models):
        self._models = list(models)
        seen = {}
        for model in self._models:
            for name in model.parameter_names:
                if name in seen:
                    raise ValueError(
                        f"Duplicate hyperparameter '{name}' in "
                        f"{seen[name].__class__.__name__} and "
                        f"{model.__class__.__name__}.  Rename one."
                    )
                seen[name] = model
        self.parameter_names = [
            name for model in self._models for name in model.parameter_names
        ]
        self.fiducial = {
            name: value
            for model in self._models
            for name, value in model.fiducial.items()
        }
        # Union of physical event keys
        # (canonical order: mass_1_source, mass_2_source, redshift)
        _canonical_order = ["mass_1_source", "mass_2_source", "redshift"]
        seen_keys = set()
        for model in self._models:
            for key in getattr(model, "phys_event_keys", []):
                seen_keys.add(key)
        self.phys_event_keys = [key for key in _canonical_order if key in seen_keys]

    def log_prob(self, events, params):
        total = numpy.zeros(len(next(iter(events.values()))))
        for model in self._models:
            sub_params = {name: params[name] for name in model.parameter_names}
            total = total + model.log_prob(events, sub_params)
        return total

    def analytic_score(self, events, params):
        """Concatenate the analytic scores of the sub-models (all must have one)."""
        cols = []
        for model in self._models:
            sub_params = {name: params[name] for name in model.parameter_names}
            cols.append(model.analytic_score(events, sub_params))
        return numpy.hstack(cols)

    def __repr__(self):
        return "JointModel(" + ", ".join(repr(m) for m in self._models) + ")"
