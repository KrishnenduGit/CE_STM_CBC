"""
This module contains whatever is needed to move from detector frame mass to source frame 
mass and luminosity distance to redshift, for a flat wCDM cosmology parameterized by H0, 
Omega_m and the dark-energy equation of state w0:

    E(z)^2 = Om (1+z)^3 + (1 - Om) (1+z)^{3(1+w0)}

w0 = -1 is a cosmological constant and reproduces flat LCDM exactly, so every function
here defaults to it and the LCDM results are recovered bit for bit.

Note that the numbers differ a bit from the astropy's Planck18 cosmology. 
I have set the Tcmb0 to 0 and folded the massive neutrinos into cold matter, which is a 
good approximation for the redshifts we care about.

The disagreement is small and grows with redshift:

    z          0.1      1.0      5.0     10.0
    d_L      0.0006%  0.007%   0.025%   0.038%
    dV_c/dz  0.0025%  0.030%   0.12%    0.21%

so meh! if you ask me. 

It is inspired by Jose Ezquiaga's ``gwcosmo`` trick and is built for speed!

Note that because GWFast loves distance in Gpc, we use Gpc here too.  H0 is in km/s/Mpc.
I cache the per-cosmology objects (the astropy model and the d_L -> z interpolant) on 
(H0, Omega_m, w0) so that the finite-difference scores in the Fisher stay cheap.

"""

import numpy
from astropy import units
from astropy.cosmology import FlatwCDM, Planck18

FIDUCIAL_COSMOLOGY = {
    "H0": float(Planck18.H0.value),
    "Om": float(Planck18.Om0 + Planck18.Onu0), # include massive neutrinos in Om, since we treat them as cold matter
    "w0": -1.0,                                 # cosmological constant; free in the spectral-siren Fisher
}

# Redshift grid on which d_L(z) is tabulated to build the inverse z(d_L).
# z = 20 is far past the catalogue's largest redshift (~10) and covers every
# trial cosmology in the Fisher finite differences.
_Z_GRID = numpy.concatenate(
    [numpy.linspace(1e-4, 10.0, 6000), numpy.linspace(10.0, 20.0, 1000)[1:]]
)

# Cache keyed on rounded (H0, Om): astropy model + (d_L grid, z grid) interpolant.
_CACHE = {}


def _entry(H0, Om, w0=-1.0):
    # Rounded to 8 decimals, which is finer than every finite-difference step the
    # Fisher uses (Om ~ 3.1e-5, w0 ~ 1e-4), so trial cosmologies never collide.
    key = (round(float(H0), 8), round(float(Om), 8), round(float(w0), 8))
    entry = _CACHE.get(key)
    if entry is None:
        from scipy.interpolate import CubicSpline

        model = FlatwCDM(H0=key[0], Om0=key[1], w0=key[2])
        distance_grid = model.luminosity_distance(_Z_GRID).to(units.Gpc).value
        # Cubic (C2) inverse z(d_L): the higher-order Fisher terms take *second*
        # theta-derivatives through z(d_L), so a piecewise-linear ``numpy.interp``
        # (piecewise-constant second derivative) is not smooth enough. Thanks Claudio!
        redshift_spline = CubicSpline(distance_grid, _Z_GRID)
        entry = (model, distance_grid, redshift_spline)
        _CACHE[key] = entry
    return entry


def luminosity_distance(redshift, H0, Om, w0=-1.0):
    """Luminosity distance [Gpc] at ``redshift`` for flat LCDM ``(H0, Om)``."""
    model, _, _ = _entry(H0, Om, w0)
    return model.luminosity_distance(redshift).to(units.Gpc).value


def redshift_of_distance(distance, H0, Om, w0=-1.0):
    """
    Redshift at a given luminosity ``distance`` [Gpc] (inverse of
    ``luminosity_distance``), by C2 cubic-spline interpolation of ``d_L(z)``.
    """
    _, _, redshift_spline = _entry(H0, Om, w0)
    return redshift_spline(distance)


def differential_comoving_volume(redshift, H0, Om, w0=-1.0):
    """
    ``dV_c/dz`` [Gpc^3] (full sky) at ``redshift`` for flat LCDM ``(H0, Om)``.

    astropy returns ``dV_c/dz/dOmega`` per steradian; multiply by ``4 pi`` for
    the sky-integrated element.  The overall constant is cosmology-independent
    and cancels from any score, but the cosmology *dependence* -- which carries
    real ``H0``/``Om`` information -- is kept exact.
    """
    model, _, _ = _entry(H0, Om, w0)
    per_steradian = model.differential_comoving_volume(redshift).to(
        units.Gpc ** 3 / units.sr
    ).value
    return 4.0 * numpy.pi * per_steradian


def ddL_dz(redshift, H0, Om, w0=-1.0):
    """
    ``d d_L / dz`` [Gpc], analytic for flat LCDM:

        d_L = (1+z) D_C(z),   dD_C/dz = D_H / E(z)
        => d d_L/dz = D_C(z) + (1+z) D_H / E(z)
    """
    model, _, _ = _entry(H0, Om, w0)
    comoving = model.comoving_distance(redshift).to(units.Gpc).value
    hubble_distance = model.hubble_distance.to(units.Gpc).value
    return comoving + (1.0 + redshift) * hubble_distance / model.efunc(redshift)


def background_terms(redshift, H0, Om, w0=-1.0):
    """
    Flat-wCDM background quantities needed for analytic cosmology derivatives.

    Everything comes from the same cached astropy model as the rest of this
    module, so the analytic scores stay consistent with `luminosity_distance`
    and `ddL_dz`.

    Returns
    -------
    comoving : ndarray   D_C(z) [Gpc]
    hubble_distance : float   D_H = c / H0 [Gpc]
    efunc : ndarray      E(z) = sqrt(Om (1+z)^3 + (1 - Om) (1+z)^{3(1+w0)})
    defunc_dz : ndarray  dE/dz, the wCDM form below

    Notes
    -----
    ``dE/dz`` must carry the dark-energy term.  The LCDM expression
    ``1.5 Om (1+z)^2 / E`` is only the w0 = -1 special case, and using it under
    wCDM is wrong by 26% at w0 = -0.8 and 46% at w0 = -1.2.  It feeds ``d2dL_dz2``
    and ``dmeasure_dz`` in ``SpectralSirenModel.analytic_score``, so getting it
    wrong would silently corrupt the *H0* column too, not just the w0 one.
    """
    model, _, _ = _entry(H0, Om, w0)
    efunc = model.efunc(redshift)
    one_plus_z = 1.0 + numpy.asarray(redshift, dtype=float)
    matter = 3.0 * model.Om0 * one_plus_z ** 2
    dark_energy = (
        3.0 * (1.0 + w0) * (1.0 - model.Om0) * one_plus_z ** (3.0 * (1.0 + w0) - 1.0)
    )
    return (
        model.comoving_distance(redshift).to(units.Gpc).value,
        model.hubble_distance.to(units.Gpc).value,
        efunc,
        (matter + dark_energy) / (2.0 * efunc),
    )


def detector_to_source_frame(mass_1_det, mass_2_det, distance, H0, Om, w0=-1.0):
    """
    Detector-frame -> source-frame, as in ``gwcosmo.detector_to_source_frame``.

    Parameters
    ----------
    mass_1_det, mass_2_det : ndarray
        Redshifted (detector-frame) component masses.
    distance : ndarray
        Luminosity distance [Gpc].
    H0, Om : float

    Returns
    -------
    mass_1_source, mass_2_source, redshift : ndarray
    """
    redshift = redshift_of_distance(distance, H0, Om, w0)
    one_plus_z = 1.0 + redshift
    return mass_1_det / one_plus_z, mass_2_det / one_plus_z, redshift
