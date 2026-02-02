import numpy as np
import bilby
from compose.eos import Metadata, Table
from scipy.interpolate import interp1d


def load_sfho_eos(eos_dir):
    """
    Loads SFHo EOS from CompOSE files and converts it into a bilby tabularEOS.

    Args:
        eos_dir (str): path to CompOSE EOS directory.

    Returns:
        bilby.gw.eos.EOSFamily: EOS family wrapper.
    """
    # Load CompOSE metadata
    md = Metadata(
        pairs={
            0: ("e", "electron"), 10: ("n", "neutron"), 11: ("p", "proton"),
            4002: ("He4", "helium-4"), 3002: ("He3", "helium-3"),
            3001: ("H3", "tritium"), 2001: ("H2", "deuteron"),
        },
        quads={999: ("N", "average nucleous")},
    )
    
    # Load the EOS table
    eos = Table(md)
    eos.read(eos_dir)
    eos.compute_cs2(floor=1e-6)
    eos.compute_abar()
    eos.validate()

    # Slice T=0 for beta-equilibrium
    eos_be = eos.slice_at_t_idx(0).make_beta_eq_table()
    nb = eos_be.nb.flatten()
    
    # Constants
    c_cgs = 2.99792458e10 # cm/s
    mevfm3_gcm3 = 1.602176634e33 / (c_cgs**2) # MeV/fm^3 to g/cm^3
    m_n = 939.5654 # Neutron mass in MeV

    c_SI = bilby.utils.speed_of_light
    G_SI = bilby.utils.gravitational_constant
    MSUN_SI = bilby.utils.solar_mass
    
    # Geometric conversion factor
    cgs_to_geom = 1000 * (G_SI / c_SI**2)

    # Build pressure and density arrays
    pressure = eos_be.thermo['Q1'].flatten() * nb * mevfm3_gcm3 * cgs_to_geom
    energy = (eos_be.thermo['Q7'].flatten() + 1) * nb * m_n * mevfm3_gcm3 * cgs_to_geom

    return bilby.gw.eos.EOSFamily(bilby.gw.eos.TabularEOS(
        np.column_stack((pressure, energy))
    ))


class LambdaCalculator:
    """
    Compute dimensionless tidal deformability Lambda(m) for an EOS.

    If method="interp", precompute Lambda on a mass grid and interpolate.
    Otherwise compute Lambda directly via the EOS solver (slower).
    """
    def __init__(self, eos_dir, method="interp", mmin=1.0, grid_size=500):
        self.fam = load_sfho_eos(eos_dir)
        self.method = method
        self.mmin = mmin
        self.interp = None
        
        # Maximum stable NS mass for SFHO EOS (~2.06 M_sun)
        self.eos_max_mass = self.fam.maximum_mass
        
        if method == "interp":
            m_grid = np.linspace(mmin, self.eos_max_mass, grid_size)
            lam_grid = np.array([self._strict_lambda(m) for m in m_grid])
            
            # Cubic interpolation routine
            valid = np.isfinite(lam_grid) & (lam_grid > 0)
            self.interp = interp1d(
                m_grid[valid], 
                lam_grid[valid], 
                kind='cubic', 
                bounds_error=False, 
                fill_value=0.0
            )

    def _strict_lambda(self, m):
        """
        Compute Lambda for a single mass.
        
        Args:
            m (float): mass in m_sun.

        Returns:
            float: Lambda if valid and 0.0 if BH.

        Raises:
            ValueError: If the computation of Lambda faills well below the maximum stable mass.
        """
        try:
            lam = self.fam.lambda_from_mass(m)
            if np.isfinite(lam) and lam > 0:
                return lam
        except Exception:
            pass
        
        if m < (self.eos_max_mass - 1e-3): 
            raise ValueError(f"EOS Error: Mass {m:.4f} failed calculation (< Limit {self.eos_max_mass:.4f})")
        
        return 0.0

    def get_lambda(self, mass):
        """
        Compute Lambda for a scalr or an array of masses.

        Args:
            mass (float or array-like): mass(es) in m_sun.
        
        Returns:
            float or np.ndarray: Lambda(s)
        """

        m = np.atleast_1d(np.asarray(mass, dtype=float))
        lam = np.zeros_like(m)
        
        mask = (m >= self.mmin) & (m <= self.eos_max_mass)
        
        if np.any(mask):
            if self.interp is not None:
                lam[mask] = self.interp(m[mask])
            else:
                lam[mask] = [self._strict_lambda(x) for x in m[mask]]
        
        lam = np.maximum(lam, 0.0)
        return lam.item() if np.ndim(mass) == 0 else lam


def add_lambdas(samples, calc, mapping):
    """
    Add Lambda values to a dictionary.
    
    Args:
        samples (dict) : dictionary containing masses
        calc (LambdaCalculator): tidal deformability calculator
        mapping (dict): {'lambda_1: 'mass_1, 'lambda_2': 'mass_2'}

    Returns:
        dict: updated samples dict with Lambdas.
    """
    for out_key, mass_key in mapping.items():
        if mass_key in samples:
            samples[out_key] = calc.get_lambda(samples[mass_key])
    return samples

if __name__ == "__main__":
    eos_dir = "/ligo/home/ligo.org/shiksha.pandey/tidal_priors/stm/sfho_compose_files/"
    calc = LambdaCalculator(eos_dir)
    print(f"SFHo Max Mass: {calc.eos_max_mass:.4f}")
    
    test_m = [1.4, 2.05, 2.1, 2.5]
    print(f"Test {test_m} -> {calc.get_lambda(test_m)}")
