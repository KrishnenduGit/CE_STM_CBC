#!/ligo/home/ligo.org/sanika.khadkikar/.conda/envs/ns_pe/bin/python

import numpy as np
import sys
import os
import argparse
import popsummary
from scipy.stats import norm
from scipy.interpolate import RegularGridInterpolator

DATADIR = "/ligo/home/ligo.org/sanika.khadkikar/Projects/et-centric/fisher_scripts/data/"

# ------------------------------------------------------------
# FullPop mass model GWTC4.0
# ------------------------------------------------------------

def truncated_normal(m, mu, sigma, mmin, mmax):
    norm_const = norm.cdf((mmax - mu) / sigma) - norm.cdf((mmin - mu) / sigma)
    pdf = norm.pdf((m - mu) / sigma) / (sigma * norm_const)
    pdf[(m < mmin) | (m > mmax)] = 0.0
    return pdf

def l(m, mcut, eta):
    return 1.0 / (1.0 + (m / mcut)**eta)

def h(m, mcut, eta):
    return 1.0 - l(m, mcut, eta)

def n1(m, m_NSmax, m_BHmin, eta_NSmax, eta_BHmin, A):
    return 1.0 - A * l(m, m_NSmax, eta_NSmax) * h(m, m_BHmin, eta_BHmin)

def n2(m, m_UMGmin, m_UMGmax, eta_UMGmin, eta_UMGmax, A2):
    return 1.0 - A2 * l(m, m_UMGmin, eta_UMGmin) * h(m, m_UMGmax, eta_UMGmax)

def pi_m(
    m,
    alpha_1, alpha_dip, alpha_2,
    m_NSmin, m_NSmax, m_BHmin, m_BHmax,
    m_UMGmin, m_UMGmax,
    eta_NSmin, eta_NSmax, eta_BHmin, eta_BHmax,
    eta_UMGmin, eta_UMGmax,
    c1, mu1, sig1,
    c2, mu2, sig2,
    A, A2,
    mmin, mmax,
):
    g1 = truncated_normal(m, mu1, sig1, mmin, mmax)
    g2 = truncated_normal(m, mu2, sig2, mmin, mmax)
    peak = 1.0 + c1 * g1 + c2 * g2

    power = np.zeros_like(m)
    mask1 = m < m_NSmax
    mask2 = (m >= m_NSmax) & (m < m_BHmin)
    mask3 = m >= m_BHmin

    power[mask1] = m[mask1]**alpha_1
    power[mask2] = m[mask2]**alpha_dip * m_NSmax**(alpha_1 - alpha_dip)
    power[mask3] = (
        m[mask3]**alpha_2
        * m_NSmax**(alpha_1 - alpha_dip)
        * m_BHmin**(alpha_dip - alpha_2)
    )

    return (
        peak
        * n1(m, m_NSmax, m_BHmin, eta_NSmax, eta_BHmin, A)
        * n2(m, m_UMGmin, m_UMGmax, eta_UMGmin, eta_UMGmax, A2)
        * h(m, m_NSmin, eta_NSmin)
        * l(m, m_BHmax, eta_BHmax)
        * power
    )

# ------------------------------------------------------------
# Mass-ratio sampler
# ------------------------------------------------------------

def generate_mass_ratio_samples(n_samples, pdb_file, source_type):

    print(f"Loading max-likelihood hyperparameters from {pdb_file}")
    res = popsummary.popresult.PopulationResult(fname=pdb_file)

    logl = res.get_hyperparameter_samples(hyperparameters=["log_likelihood"])
    maxl = np.argmax(logl)

    def get(name):
        return res.get_hyperparameter_samples(
            hyperparameters=[name]
        )[maxl]

    params = dict(
        alpha_1=get("alpha_1"), alpha_dip=get("alpha_dip"), alpha_2=get("alpha_2"),
        m_NSmin=get("NSmin"), m_NSmax=get("NSmax"),
        m_BHmin=get("BHmin"), m_BHmax=get("BHmax"),
        m_UMGmin=get("UPPERmin"), m_UMGmax=get("UPPERmax"),
        eta_NSmin=get("n0"), eta_NSmax=get("n1"),
        eta_BHmin=get("n2"), eta_UMGmin=get("n3"),
        eta_UMGmax=get("n4"), eta_BHmax=get("n5"),
        c1=get("mix1"), mu1=get("mu1"), sig1=get("sig1"),
        c2=get("mix2"), mu2=get("mu2"), sig2=get("sig2"),
        A=get("A"), A2=get("A2"),
    )

    # Grid size logic (UNCHANGED)
    n_grid = int(np.sqrt(n_samples * 1.5))
    n_grid = np.clip(n_grid, 500, 2000)
    print(f"Building {n_grid}x{n_grid} grid for {source_type.upper()}...")

    sfho_max_mass = 2.064501812848886

    if source_type == "bns":
        m_grid = np.linspace(1.0, sfho_max_mass, n_grid)
        q_grid = np.linspace(0.1, 1.0, n_grid)

    elif source_type == "bbh":
        m_grid = np.geomspace(sfho_max_mass, 500.0, n_grid)
        q_grid = np.linspace(0.1, 1.0, n_grid)

    elif source_type == "nsbh":
        m_grid = np.geomspace(1.0, 500.0, n_grid)
        q_grid = np.geomspace(0.1, 0.5, n_grid)
        
    else:
        raise ValueError("Invalid source-type")

    mc = (m_grid[:-1] + m_grid[1:]) / 2
    qc = (q_grid[:-1] + q_grid[1:]) / 2
    M, Q = np.meshgrid(mc, qc, indexing="ij")

    p_m1 = pi_m(mc, **params, mmin=1.0, mmax=500.0)
    p_m2 = pi_m(mc, **params, mmin=1.0, mmax=500.0)
    rate_density = np.outer(p_m1, p_m2)

    pdf_m1_q = rate_density * M

    ns_max, bh_min = sfho_max_mass, sfho_max_mass
    m2 = M * Q

    if source_type == "bns":
        mask = (M <= ns_max) & (m2 <= ns_max)
    elif source_type == "bbh":
        mask = (M >= bh_min) & (m2 >= bh_min)
    elif source_type == "nsbh":
        mask = (M >= bh_min) & (m2 <= ns_max)

    pdf_m1_q[~mask] = 0.0

    dm = np.diff(m_grid)
    dq = np.diff(q_grid)
    dM, dQ = np.meshgrid(dm, dq, indexing="ij")

    prob_mass = pdf_m1_q * dM * dQ
    prob_mass /= np.sum(prob_mass)

    marginal_pm1 = np.sum(pdf_m1_q * dQ, axis=1)
    marginal_pm1 /= np.trapz(marginal_pm1, mc)

    print(f"Sampling {n_samples} events...")
    idx = np.random.choice(prob_mass.size, size=int(1.1 * n_samples), p=prob_mass.flatten())
    idx_m, idx_q = np.unravel_index(idx, M.shape)

    m1_s = m_grid[idx_m] + np.random.rand(len(idx)) * dm[np.minimum(idx_m, len(dm)-1)]
    q_s = q_grid[idx_q] + np.random.rand(len(idx)) * dq[np.minimum(idx_q, len(dq)-1)]
    q_s = np.minimum(q_s, 1.0)

    m2_s = m1_s * q_s
    if source_type == "bns":
        keep = (m1_s <= ns_max) & (m2_s <= ns_max)
    elif source_type == "bbh":
        keep = (m1_s >= bh_min) & (m2_s >= bh_min)
    else:
        keep = (m1_s >= bh_min) & (m2_s <= ns_max)

    m1_out = m1_s[keep][:n_samples]
    q_out = q_s[keep][:n_samples]

    interp_joint = RegularGridInterpolator((mc, qc), pdf_m1_q, bounds_error=False, fill_value=0.0)
    p_joint = interp_joint(np.column_stack((m1_out, q_out)))
    p_m1_out = np.interp(m1_out, mc, marginal_pm1)

    p_q_given_m1 = np.divide(
        p_joint, p_m1_out,
        out=np.zeros_like(p_joint),
        where=p_m1_out > 1e-50
    )

    return m1_out, p_m1_out, q_out, p_q_given_m1

# ------------------------------------------------------------
# CLI
# ------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-type", type=str, required=True, choices=["bns", "bbh", "nsbh"])
    parser.add_argument("--n-samples", type=int, default=10000)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    m1, p_m1, q, p_q = generate_mass_ratio_samples(
        args.n_samples,
        os.path.join(DATADIR, "AllCBC_FullPop.h5"),
        args.source_type,
    )

    fname = args.output or f"m1_pm1_q_pq_{args.source_type}.dat"
    print(f"\nSaving {len(m1)} samples to {fname}...")

    header = (
        "m1\tp(m1)\tq\tp(q|m1)\n"
        f"Source: {args.source_type} | Samples: {len(m1)}"
    )

    np.savetxt(fname, np.c_[m1, p_m1, q, p_q], header=header, delimiter="\t")
    print("Done.")
