import numpy as np
from scipy.stats import truncnorm
from gwpopulation.models.spin import independent_spin_orientation_gaussian_isotropic

def sample_gwpop_gaussian_component(n_binaries, spin_parameters):
    """
    Samples the Gaussian Component Spins model by directly evaluating the 
    native gwpopulation 2D joint probability functions.
    """

    mu_chi = spin_parameters['mu_chi']
    sigma_chi = spin_parameters['sigma_chi']
    mu_spin = spin_parameters['mu_spin']
    sigma_spin = spin_parameters['sigma_spin']
    xi_spin = spin_parameters['xi_spin']
    
    # ---------------------------------------------------------
    # 1. Spin Magnitudes (Independent)
    # Since these are fully independent Truncated Normals, 
    # scipy is mathematically exact and much faster than rejection sampling.
    # ---------------------------------------------------------
    a_lower = (0.0 - mu_chi) / sigma_chi
    a_upper = (1.0 - mu_chi) / sigma_chi
    
    a1 = truncnorm.rvs(a_lower, a_upper, loc=mu_chi, scale=sigma_chi, size=n_binaries)
    a2 = truncnorm.rvs(a_lower, a_upper, loc=mu_chi, scale=sigma_chi, size=n_binaries)

    # ---------------------------------------------------------
    # 2. Spin Tilts (Joint 2D Rejection Sampling)
    # We use gwpopulation to calculate the true 2D probability surface
    # ---------------------------------------------------------
    accepted_ct1 = []
    accepted_ct2 = []
    
    # Find the maximum possible probability height (the peak of the 2D Gaussian)
    # This happens when both spins sit exactly at mu_spin
    max_dataset = {'cos_tilt_1': np.array([mu_spin]), 'cos_tilt_2': np.array([mu_spin])}
    max_p = independent_spin_orientation_gaussian_isotropic(
        max_dataset, 
        xi_spin=xi_spin, 
        sigma_1=sigma_spin, sigma_2=sigma_spin, 
        mu_1=mu_spin, mu_2=mu_spin
    )[0] # Extract the single float value

    # We propose large batches of random pairs to speed up the loop
    batch_size = n_binaries * 5 
    
    while len(accepted_ct1) < n_binaries:
        # Step A: Propose random uniform pairs in 2D space
        prop_ct1 = np.random.uniform(-1, 1, batch_size)
        prop_ct2 = np.random.uniform(-1, 1, batch_size)
        
        # Step B: Evaluate the true joint probability of these pairs using gwpopulation
        dataset_prop = {'cos_tilt_1': prop_ct1, 'cos_tilt_2': prop_ct2}
        p_joint = independent_spin_orientation_gaussian_isotropic(
            dataset_prop, 
            xi_spin=xi_spin, 
            sigma_1=sigma_spin, sigma_2=sigma_spin, 
            mu_1=mu_spin, mu_2=mu_spin
        )
        
        # Step C: Roll a uniform random die for each pair (the height)
        u = np.random.uniform(0, max_p, batch_size)
        
        # Step D: Keep the pairs where the die roll falls UNDER the 2D surface
        accept_mask = u < p_joint
        
        accepted_ct1.extend(prop_ct1[accept_mask])
        accepted_ct2.extend(prop_ct2[accept_mask])
        
    return a1, a2, np.array(accepted_ct1[:n_binaries]), np.array(accepted_ct2[:n_binaries])
