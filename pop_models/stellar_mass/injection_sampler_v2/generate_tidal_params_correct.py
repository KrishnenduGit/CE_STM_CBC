import os
import numpy as np
import bilby
from scipy.integrate import solve_ivp
from scipy.interpolate import PchipInterpolator
from bilby.gw.eos.tov_solver import IntegrateTOV

# bilby's default rel_err=1e-4 is under-converged and makes Lambda non-monotonic
TOV_RTOL = 1e-6
TOV_ATOL = 0.0


def _integrate_TOV_tight(self):
    """Replacement for IntegrateTOV.integrate_TOV with a converged tolerance."""
    result = solve_ivp(
        self._IntegrateTOV__tov_eqns,
        (self.pseudo_enthalpy, 1e-16),
        self.y,
        rtol=TOV_RTOL,
        atol=TOV_ATOL,
    )
    m_fin = result.y[0, -1]
    r_fin = result.y[1, -1]
    H_fin = result.y[2, -1]
    B_fin = result.y[3, -1]
    k_2 = self._IntegrateTOV__calc_k2(r_fin, B_fin, H_fin, m_fin / r_fin)
    return m_fin, r_fin, k_2


IntegrateTOV.integrate_TOV = _integrate_TOV_tight

# Central pressures for the TOV family; do not exceed ~4000 or M_max collapses to ~0.2 M_sun.
EOS_FAMILY_NPTS = 1000

# Solar mass in geometric units [m]; EOSFamily.mass is stored in these units.
MSUN_GEOM = (bilby.utils.gravitational_constant * bilby.utils.solar_mass
             / bilby.utils.speed_of_light**2)


def load_sfho_eos(eos_dir):
    """
    Loads SFHo EOS directly from CompOSE files (no Pycompose).
    Caches converted pressure and energy arrays to {eos_dir}/sfho_p_e_geom.txt.

    Args:
        eos_dir (str): path to CompOSE EOS directory.

    Returns:
        bilby.gw.eos.EOSFamily: EOS family wrapper.
    """

    cache_path = os.path.join(eos_dir, "sfho_p_e_geom.txt")

    if os.path.exists(cache_path):
        print(f"Loading Pressure - Energy from cache: {cache_path}")
        data = np.loadtxt(cache_path)
        pressure, energy = data[:, 0], data[:, 1]
        return bilby.gw.eos.EOSFamily(
            bilby.gw.eos.TabularEOS(np.column_stack((pressure, energy))),
            npts=EOS_FAMILY_NPTS
        )

    thermo_ns = os.path.join(eos_dir, "eos.thermo.ns")
    nb_ns = os.path.join(eos_dir, "eos.nb.ns")

    # Column indices for Q1 (pressure per baryon) and Q7 (internal energy per baryon).
    IDX_Q1, IDX_Q7 = 3, 9

    with open(thermo_ns) as fh:
        m_n = float(fh.readline().split()[0])

    thermo = np.genfromtxt(thermo_ns, skip_header=1)
    nb_raw = np.genfromtxt(nb_ns)

    # Strip the two leading index markers from eos.nb if present.
    nb = nb_raw[2:] if (nb_raw.ndim == 1 and len(nb_raw) > thermo.shape[0]) else nb_raw
    nb = np.atleast_1d(nb).ravel()

    n = min(len(nb), thermo.shape[0])
    nb = nb[:n]
    thermo = thermo[:n]

    G_SI = bilby.utils.gravitational_constant
    c_SI = bilby.utils.speed_of_light
    m_n_MeV = 939.5654

    MeVfm3_to_Pa = 1.602176634e32
    MeVfm3_to_gcm3 = MeVfm3_to_Pa / c_SI**2 / 1000.0
    gcm3_to_geo_m2 = (G_SI / c_SI**2) * 1e3

    p_MeV = thermo[:, IDX_Q1] * nb
    e_MeV = (thermo[:, IDX_Q7] + 1.0) * nb * m_n_MeV

    mask = (p_MeV > 0) & (e_MeV > 0)
    p_MeV, e_MeV = p_MeV[mask], e_MeV[mask]

    o = np.argsort(p_MeV)
    p_MeV, e_MeV = p_MeV[o], e_MeV[o]

    # Convert to geometric units (m^-2).
    pressure = p_MeV * MeVfm3_to_gcm3 * gcm3_to_geo_m2
    energy = e_MeV * MeVfm3_to_gcm3 * gcm3_to_geo_m2

    print(f"Saving EOS cache to: {cache_path}")
    np.savetxt(cache_path, np.column_stack((pressure, energy)),
               header="Pressure [m^-2]    Energy_Density [m^-2]")

    return bilby.gw.eos.EOSFamily(
        bilby.gw.eos.TabularEOS(np.column_stack((pressure, energy))),
        npts=EOS_FAMILY_NPTS
    )


def family_nodes(fam, mmin=0.0):
    """
    Extract (mass, Lambda) from the TOV family in physical units.

    Args:
        fam (bilby.gw.eos.EOSFamily): TOV family.
        mmin (float): discard nodes below this mass [M_sun].

    Returns:
        tuple[np.ndarray, np.ndarray]: (mass [M_sun], Lambda), sorted by mass.
    """
    mass = np.asarray(fam.mass) / MSUN_GEOM
    lam = np.asarray(fam.tidal_deformability)

    order = np.argsort(mass)
    mass, lam = mass[order], lam[order]

    good = np.isfinite(mass) & np.isfinite(lam) & (lam > 0) & (mass >= mmin)
    mass, lam = mass[good], lam[good]

    # PchipInterpolator requires strictly increasing mass.
    if len(mass) > 1:
        keep = np.concatenate(([True], np.diff(mass) > 0))
        mass, lam = mass[keep], lam[keep]

    return mass, lam


class LambdaCalculator:
    """Compute dimensionless tidal deformability Lambda(m) for an EOS."""

    def __init__(self, eos_dir, method="interp", mmin=1.0, grid_size=500):
        self.fam = load_sfho_eos(eos_dir)
        self.method = method
        self.mmin = mmin
        self.interp = None
        self.eos_max_mass = self.fam.maximum_mass
        self.eos_max_mass = np.floor(self.fam.maximum_mass * 100.0) / 100.0

        if method == "interp":
            # PCHIP is shape preserving, so it cannot overshoot between TOV nodes.
            m_nodes, lam_nodes = family_nodes(self.fam, mmin=mmin)

            if len(m_nodes) < 4:
                raise ValueError(
                    f"Only {len(m_nodes)} usable TOV nodes above {mmin} M_sun; "
                    "cannot build the Lambda interpolant."
                )

            # Should be 0; nonzero means the rtol patch did not take effect.
            n_bad = int(np.sum(np.diff(lam_nodes) > 0))
            if n_bad:
                print(f"WARNING: {n_bad} of {len(m_nodes) - 1} node intervals "
                      f"above {mmin} M_sun have Lambda increasing with mass. "
                      f"Expected 0 at rtol={TOV_RTOL:g}.")

            self.interp = PchipInterpolator(m_nodes, lam_nodes, extrapolate=False)

    def _strict_lambda(self, m):
        """Compute Lambda for a single mass."""
        try:
            lam = self.fam.lambda_from_mass(m)
            if np.isfinite(lam) and lam > 0:
                return lam
        except Exception:
            pass

        if m < (self.eos_max_mass - 1e-3):
            raise ValueError(f"EOS Error: Mass {m:.4f} failed (< Limit {self.eos_max_mass:.4f})")

        return np.nan

    def get_lambda(self, mass):
        """Compute Lambda for scalar or array of masses."""
        m = np.atleast_1d(np.asarray(mass, dtype=float))
        lam = np.zeros_like(m)

        mask = (m >= self.mmin) & (m <= self.eos_max_mass)

        if np.any(mask):
            if self.interp is not None:
                lam[mask] = self.interp(m[mask])
            else:
                lam[mask] = [self._strict_lambda(x) for x in m[mask]]

        lam = np.nan_to_num(lam, nan=0.0)
        lam = np.maximum(lam, 0.0)
        return lam.item() if np.ndim(mass) == 0 else lam


def add_lambdas(samples, calc, mapping):
    """Add Lambda values to a samples dictionary."""
    for out_key, mass_key in mapping.items():
        if mass_key in samples:
            samples[out_key] = calc.get_lambda(samples[mass_key])
    return samples


if __name__ == "__main__":
    eos_dir = "/ligo/home/ligo.org/sanika.khadkikar/Projects/stm/post_processing_ns/CE_STM_CBC/pop_models/stellar_mass/injection_sampler_v2/sfho_compose_files/"
    calc = LambdaCalculator(eos_dir)
    print(f"SFHo Max Mass: {calc.eos_max_mass:.4f}")

    test_m = [1.4, 2.05, 2.06001, 2.2]
    print(f"Test {test_m} -> {calc.get_lambda(test_m)}")