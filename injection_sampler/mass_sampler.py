#!/ligo/home/ligo.org/shiksha.pandey/.conda/envs/gwfast311/bin/python
import numpy as np
import sys
import os
import inspect
import argparse
import popsummary
from scipy.interpolate import RegularGridInterpolator


DATADIR = "/ligo/home/ligo.org/shiksha.pandey/tidal_priors/data/"
SCRIPTS_DIR = "/ligo/home/ligo.org/shiksha.pandey/tidal_priors/scripts"

sys.path.insert(0, SCRIPTS_DIR)
import pdb_external

def load_hyperparameters(pdb_file, model_class):
    """
    Loads required hyperparameters from a popsummary H5 file. 

    Args:
        pdb_file (str): path to the H5 file.
        model_class: population model class.

    Returns:
        dict: dictionary containing the median values of hyperparameters.

    Raises:
        ValueError: if any required hyperparameters are missing. 

    """
    print(f"Loading hyperparameters from {pdb_file}...")
    pdb_result = popsummary.popresult.PopulationResult(fname=pdb_file)
    
    # Inspect the model class to find strictly required arguments
    model_args = [k for k in inspect.getfullargspec(model_class.__call__).args 
                  if k not in ['self', 'dataset']]
    
    inj_params = {}
    missing_params = []
    
    for key in model_args:
        try:
            val = pdb_result.get_hyperparameter_samples(hyperparameters=[key])
            if isinstance(val, dict):
                inj_params[key] = np.median(val[key])
            elif hasattr(val, 'dtype') and val.dtype.names:
                inj_params[key] = np.median(val[key])
            else:
                inj_params[key] = np.median(val)
        except Exception:
            missing_params.append(key)
    
    if missing_params:
        raise ValueError(
            f"Missing required parameters in H5 file: {missing_params}\n"
            "Cannot generate physical samples with 0.0 defaults."
        )
    return inj_params

def generate_mass_ratio_samples(n_samples, pdb_file, source_type):
    """
    Generates mass samples using a grid method.

    Args:
        n_samples (int): number of samples to generate.
        pdb_file (str): path to the H5 file.
        source_type (str): 'bns', 'bbh', or 'nsbh'

    Returns:
        tuple: (m1_out, p_m1_out, q_out, p_q_given_m1) as numpy arrays.
    """
    # Setup model and hyperparameters
    model = pdb_external.NotchFilterBinnedPairingMassDistribution()
    params = load_hyperparameters(pdb_file, pdb_external.NotchFilterBinnedPairingMassDistribution)
    
    # Scale grid size with sqrt(N). Floor at 500, cap at 2000.
    n_grid = int(np.sqrt(n_samples * 1.5))
    n_grid = np.clip(n_grid, 500, 2000)
    
    print(f"Building {n_grid}x{n_grid} grid for {source_type.upper()}...")

    # Define grid ranges for different source types
    if source_type == 'bns':
        m_grid = np.linspace(1.0, 2.3, n_grid)
        q_grid = np.linspace(0.001, 1.0, n_grid)
    elif source_type == 'bbh':
        m_grid = np.geomspace(5.0, 500.0, n_grid)
        q_grid = np.linspace(0.001, 1.0, n_grid)
    elif source_type == 'nsbh':
        m_grid = np.geomspace(5.0, 500.0, n_grid)
        q_grid = np.geomspace(0.001, 0.5, n_grid)
    else:
        raise ValueError("Invalid source-type")

    # Evaluate joint density p(m1, m2)
    mc = (m_grid[:-1] + m_grid[1:]) / 2
    qc = (q_grid[:-1] + q_grid[1:]) / 2
    M, Q = np.meshgrid(mc, qc, indexing='ij')
    
    dataset = {'mass_1': M.flatten(), 'mass_2': (M * Q).flatten()}
    rate_density = model(dataset, **params).reshape(M.shape)
    
    # Jacobian transform to p(m1, q): dm2/dq = m1
    pdf_m1_q = rate_density * M 
    
    # Apply physical cuts. May need to change ns_max to 2.06 as that is the SFHO cut-off.
    ns_max, bh_min = 2.25, 5.0
    m2 = M * Q
    
    if source_type == 'bns': 
        mask = (M <= ns_max) & (m2 <= ns_max)
    elif source_type == 'bbh': 
        mask = (M >= bh_min) & (m2 >= bh_min)
    elif source_type == 'nsbh': 
        mask = (M >= bh_min) & (m2 <= ns_max)
    
    pdf_m1_q[~mask] = 0.0
    
    dm = np.diff(m_grid)
    dq = np.diff(q_grid)
    dM, dQ = np.meshgrid(dm, dq, indexing='ij')
    # Probability mass = Density * Volume (for Sampling)
    prob_mass = pdf_m1_q * dM * dQ
    total_prob = np.sum(prob_mass)
    if total_prob == 0: 
        raise ValueError("Grid contained no valid samples! Check grid ranges vs cuts.")
    prob_mass /= total_prob
    
    # Marginal p(m1): integrate p(m1,q) over q (Sum density * dq)
    marginal_pm1_grid = np.sum(pdf_m1_q * dQ, axis=1)
    marginal_pm1_grid /= np.trapz(marginal_pm1_grid, mc) # Normalization

    # Sample from the grid with jitter inside cells
    print(f"Sampling {n_samples} events...")
    n_draw = int(n_samples * 1.1)
    
    idx = np.random.choice(prob_mass.size, size=n_draw, p=prob_mass.flatten())
    idx_m, idx_q = np.unravel_index(idx, M.shape)
    
    m1_s = m_grid[idx_m] + np.random.rand(n_draw) * dm[np.minimum(idx_m, len(dm)-1)]
    q_s = q_grid[idx_q] + np.random.rand(n_draw) * dq[np.minimum(idx_q, len(dq)-1)]
    q_s = np.minimum(q_s, 1.0)
    
    # Final filter
    m2_s = m1_s * q_s
    if source_type == 'bns': 
        keep = (m1_s <= ns_max) & (m2_s <= ns_max) & (q_s <= 1.0)
    elif source_type == 'bbh': 
        keep = (m1_s >= bh_min) & (m2_s >= bh_min) & (q_s <= 1.0)
    elif source_type == 'nsbh': 
        keep = (m1_s >= bh_min) & (m2_s <= ns_max) & (q_s <= 1.0)
    
    m1_out = m1_s[keep][:n_samples]
    q_out = q_s[keep][:n_samples]
    
    if len(m1_out) < n_samples:
        print(f"Warning: Only generated {len(m1_out)} valid samples. Grid/Jitter edge effects.")

    # Interpolate probabilities at sample locations
    interp_joint = RegularGridInterpolator(
        (mc, qc), 
        pdf_m1_q, 
        bounds_error=False, 
        fill_value=0.0
    )
    
    # Clamping forces edge samples to nearest grid center
    m1_clip = np.clip(m1_out, mc[0], mc[-1])
    q_clip  = np.clip(q_out,  qc[0], qc[-1])
    p_m1_out = np.interp(m1_clip, mc, marginal_pm1_grid) # 1D marginal internpolation
    p_joint_out = interp_joint(np.column_stack((m1_clip, q_clip))) # 2D joint interpolation
    
    p_joint_out = np.nan_to_num(p_joint_out, nan=0.0, posinf=0.0, neginf=0.0)
    p_joint_out = np.maximum(p_joint_out, 0.0)
    
    # Calculate conditional p(q|m1) = p(m1,q) / p(m1)
    p_q_given_m1 = np.divide(
        p_joint_out, 
        p_m1_out, 
        out=np.zeros_like(p_joint_out), 
        where=p_m1_out > 1e-50
    )
    
    return m1_out, p_m1_out, q_out, p_q_given_m1

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-type", type=str, required=True, choices=['bns', 'bbh', 'nsbh'])
    parser.add_argument("--n-samples", type=int, default=10000)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    # Run Generation
    m1, p_m1, q, p_q = generate_mass_ratio_samples(
        args.n_samples, 
        os.path.join(DATADIR, 'AllCBC_FullPop.h5'), 
        args.source_type
    )

    # Save to File
    fname = args.output or f"m1_pm1_q_pq_{args.source_type}.dat"
    print(f"\nSaving {len(m1)} samples to {fname}...")
    
    header = (f"m1\tp(m1)\tq\tp(q|m1)\n"
              f"Source: {args.source_type} | Samples: {len(m1)}")
              
    np.savetxt(fname, np.c_[m1, p_m1, q, p_q], header=header, delimiter='\t')
    print("Done.")
