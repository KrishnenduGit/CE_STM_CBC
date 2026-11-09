"""
Redshift models.  ``MadauDickinsonRedshift``: merger rate following the
Madau-Dickinson star-formation-rate shape, with the Planck18 comoving volume.
"""

import numpy

from pop_models import PopModel


def _log_dVc_dz_over_1plusz(redshift, cosmo=None):
    if cosmo is None:
        from astropy.cosmology import Planck18 as cosmo

    dVc_dz = cosmo.differential_comoving_volume(redshift).value  # Mpc^3/sr (up to constant factors, irrelevant)
    return numpy.log(dVc_dz) - numpy.log1p(redshift)

class MadauDickinsonRedshift(PopModel):
    """
    BBH merger rate following the Madau-Dickinson star-formation rate (SFR).

    Physical event parameters this model depends on (needed for higher-order
    population Fisher correction terms):
        z = source redshift

    The observed merger rate per unit redshift is

        dN/dz ∝ psi_MD(z) * dV_c/dz(z) / (1+z)

    with the Madau-Dickinson SFR shape

        psi_MD(z) = (1+z)^alpha / [1 + ((1+z)/(1+z_peak))^(alpha+beta)]

    Folding in the 1/(1+z) time-dilation, the full log probability per event is

        ln p(z | alpha, beta, z_peak)
            = alpha ln(1+z) - ln[1 + ((1+z)/(1+z_peak))^(alpha+beta)]
              + ln[dV_c/dz(z)] - ln(1+z)

    The dV_c/dz(z)/(1+z) piece is cosmology-fixed (lambda-independent), so it
    drops out of d ln p / d lambda exactly -- `analytic_score` below omits it
    and only encodes the shape term. It is NOT omitted from `log_prob`,
    however, because `log_prob` is also finite-differenced in z (not just in
    lambda) to build the population-curvature Hessian H_theta used in Fisher
    terms II-V (see `_H_theta`/`_P_theta` in pop_fisher_higher_order.py). 

    Analytic scores (d ln p / d lambda) are provided (no finite differences
    needed) since the added cosmological term is lambda-independent.

    Parameters
    ----------
    alpha  : low-z power-law slope (fiducial 2.7)
    beta   : high-z fall-off index (fiducial 2.9)
    z_peak : redshift at the SFR peak (fiducial 1.9)

    Required event key
    ------------------
    'redshift' : source redshift array
    """

    def __init__(
        self, 
        parameter_names = ['alpha', 'beta', 'z_peak'],
        fiducial={"alpha": 2.7, "beta": 2.9, "z_peak": 1.9}, phys_event_keys=["redshift"]
    ):

        self.parameter_names = parameter_names
        self.fiducial = fiducial
        self.phys_event_keys = phys_event_keys

    def log_prob(self, events, params):
        redshift = events["redshift"]
        alpha = params["alpha"]
        beta = params["beta"]
        z_peak = params["z_peak"]

        # log psi_MD(z); the 1/(1+z) time dilation lives in the helper below and
        # must NOT be folded in here as well.
        log_shape = alpha * numpy.log1p(redshift) - numpy.log1p(
            ((1.0 + redshift) / (1.0 + z_peak)) ** (alpha + beta)
        )
        return log_shape + _log_dVc_dz_over_1plusz(redshift)

    def analytic_score(self, events, params):
        """
        Analytic score d ln p / d lambda for each event.
        (Look at Koustav's copy for the derivation.)

        Let u = (1+z)/(1+z_peak),  r = u^(alpha+beta).

        d ln p / d alpha  = ln(1+z) - r/(1+r) * ln(u)
        d ln p / d beta   =         - r/(1+r) * ln(u)
        d ln p / d z_peak = (alpha+beta) * r / ((1+r) * (1+z_peak))
        """
        redshift = events["redshift"]
        alpha = params["alpha"]
        beta = params["beta"]
        z_peak = params["z_peak"]

        ratio = (1.0 + redshift) / (1.0 + z_peak)
        r = ratio ** (alpha + beta)
        ln_ratio = numpy.log(ratio)
        frac = r / (1.0 + r)

        d_alpha = numpy.log1p(redshift) - frac * ln_ratio
        d_beta = -frac * ln_ratio
        d_zpeak = (alpha + beta) * frac / (1.0 + z_peak)

        return numpy.column_stack([d_alpha, d_beta, d_zpeak])

