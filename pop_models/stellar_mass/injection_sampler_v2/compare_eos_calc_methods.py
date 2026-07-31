"""
Combined M-R and Lambda Comparison: Direct CompOSE vs PyCompose vs Direct Parsing

Plots the EOSFamily's OWN nodes (fam.mass, fam.radius, fam.tidal_deformability).
No mass grid, no truncation, no interpolation: whatever the TOV solver produced
is what gets drawn.

Note fam.mass and fam.radius are in GEOMETRIC units (metres).
  mass [M_sun] = fam.mass / (G*M_sun/c^2) = fam.mass / 1476.625
  radius [km]  = fam.radius / 1000

Runs each loader independently and deletes the cache between them so neither
inherits the other's converted table.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import bilby

# ============================================================================
# Configuration
# ============================================================================

eos_dir = "/ligo/home/ligo.org/sanika.khadkikar/Projects/stm/post_processing_ns/CE_STM_CBC/pop_models/stellar_mass/injection_sampler_v2/sfho_compose_files"
mr_file = os.path.join(eos_dir, "eos.mr")
output_dir = "./mr_lambda_comparison"

CACHE_FILES = [
    os.path.join(eos_dir, "sfho_p_e_geom.txt"),
    os.path.join(eos_dir, "sfho_p_e_geom_compose.txt"),
]

G_SI = bilby.utils.gravitational_constant
c_SI = bilby.utils.speed_of_light
MSUN_SI = bilby.utils.solar_mass
MSUN_GEOM = G_SI * MSUN_SI / c_SI**2      # metres per solar mass


def clear_cache(label):
    for path in CACHE_FILES:
        if os.path.exists(path):
            try:
                os.remove(path)
                print(f"  [{label}] deleted cache: {path}")
            except Exception as e:
                print(f"  [{label}] could not delete {path}: {e}")


def family_nodes(fam):
    """
    Raw solver output converted to physical units, sorted by mass.
    Returns (mass_msol, radius_km, lambda).
    """
    m = np.asarray(fam.mass) / MSUN_GEOM
    r = np.asarray(fam.radius) / 1.0e3
    lam = np.asarray(fam.tidal_deformability)
    order = np.argsort(m)
    m, r, lam = m[order], r[order], lam[order]
    keep = np.isfinite(m) & np.isfinite(r) & np.isfinite(lam)
    return m[keep], r[keep], lam[keep]


def report(label, m, lam):
    """Count Lambda monotonicity violations above 1.0 M_sun."""
    w = m >= 1.0
    if np.sum(w) < 3:
        return
    dl = np.diff(lam[w])
    n_v = int(np.sum(dl > 0))
    print(f"      nodes: {len(m)}, on stable branch above 1.0: {int(np.sum(w))}")
    print(f"      Lambda monotonicity violations above 1.0 M_sun: {n_v}")
    if n_v:
        mm = m[w]
        for i in np.where(dl > 0)[0][:5]:
            print(f"        m {mm[i]:.4f} -> {mm[i+1]:.4f} : "
                  f"Lambda {lam[w][i]:.2f} -> {lam[w][i+1]:.2f}")


if not os.path.exists(output_dir):
    os.makedirs(output_dir)

print("="*80)
print("COMBINED M-R AND LAMBDA COMPARISON (raw family nodes)")
print("="*80)
print(f"\nEOS directory: {eos_dir}")

print("\nClearing any pre-existing cache...")
clear_cache("pre")

# ============================================================================
# Approach 1: PyCompose (generate_tidal_params.py)
# ============================================================================

print("\n[1/3] PYCOMPOSE (generate_tidal_params.py)")
print("-"*80)

pyc = None
try:
    from generate_tidal_params import load_sfho_eos as load_pyc

    fam_pyc = load_pyc(eos_dir)
    m_p, r_p, l_p = family_nodes(fam_pyc)
    pyc = dict(m=m_p, r=r_p, lam=l_p, mmax=fam_pyc.maximum_mass)
    print(f"      M_max = {fam_pyc.maximum_mass:.4f} M_sun")
    print(f"      mass range   {m_p.min():.4f} to {m_p.max():.4f} M_sun")
    print(f"      radius range {r_p.min():.4f} to {r_p.max():.4f} km")
    report("pycompose", m_p, l_p)
except ImportError:
    print("      generate_tidal_params.py not found, skipping")
except Exception as e:
    print(f"      Error: {type(e).__name__}: {e}")

print("\n  Clearing cache after PyCompose run...")
clear_cache("post-pyc")

# ============================================================================
# Approach 2: Direct parsing (generate_tidal_params_correct.py)
# ============================================================================

print("\n[2/3] DIRECT PARSING (generate_tidal_params_correct.py)")
print("-"*80)

dpr = None
try:
    from generate_tidal_params_correct import load_sfho_eos as load_correct

    fam_dir = load_correct(eos_dir)
    m_d, r_d, l_d = family_nodes(fam_dir)
    dpr = dict(m=m_d, r=r_d, lam=l_d, mmax=fam_dir.maximum_mass)
    print(f"      M_max = {fam_dir.maximum_mass:.4f} M_sun")
    print(f"      mass range   {m_d.min():.4f} to {m_d.max():.4f} M_sun")
    print(f"      radius range {r_d.min():.4f} to {r_d.max():.4f} km")
    report("direct", m_d, l_d)
except ImportError:
    print("      generate_tidal_params_correct.py not found, skipping")
except Exception as e:
    print(f"      Error: {type(e).__name__}: {e}")

print("\n  Clearing cache after direct-parsing run...")
clear_cache("post-dir")

# ============================================================================
# Approach 3: Direct M-R file
# ============================================================================

print("\n[3/3] DIRECT M-R FILE (eos.mr)")
print("-"*80)

mrf = None
try:
    d = np.genfromtxt(mr_file)
    mrf = dict(r=d[:, 0], m=d[:, 1])
    print(f"      {len(mrf['m'])} points, M_max = {np.max(mrf['m']):.4f} M_sun")
except Exception as e:
    print(f"      Failed: {e}")

# ============================================================================
# Plots
# ============================================================================

print("\n" + "-"*80)
print("Plotting raw family nodes")
print("-"*80)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# ---- M-R ----
if mrf is not None:
    ax1.plot(mrf['r'], mrf['m'], lw=2.5, color='teal', ls='--',
             label='Direct M-R (eos.mr)', alpha=0.9)
if pyc is not None:
    ax1.plot(pyc['r'], pyc['m'], lw=2.0, color='orange', ls='-',
             label='PyCompose EOS', alpha=0.85)
if dpr is not None:
    ax1.plot(dpr['r'], dpr['m'], lw=2.0, color='purple', ls='-.',
             label='Direct Parsing EOS', alpha=0.85)

ax1.grid(alpha=0.4, ls='--')
ax1.set_xlabel(r'Radius [km]', fontsize=12, fontweight='bold')
ax1.set_ylabel(r'Mass [$M_\odot$]', fontsize=12, fontweight='bold')
ax1.set_title('M-R Curves (full solver output)', fontsize=13, fontweight='bold')
ax1.tick_params(axis='both', which='major', labelsize=11)
ax1.legend(frameon=False, fontsize=10, loc='best')

# ---- Lambda ----
if pyc is not None:
    ax2.semilogy(pyc['m'], pyc['lam'], lw=2.0, color='orange', ls='-',
                 label='PyCompose EOS', alpha=0.85)
if dpr is not None:
    ax2.semilogy(dpr['m'], dpr['lam'], lw=2.0, color='purple', ls='-.',
                 label='Direct Parsing EOS', alpha=0.85)

ax2.grid(alpha=0.4, ls='--', which='both')
ax2.set_xlabel(r'Mass [$M_\odot$]', fontsize=12, fontweight='bold')
ax2.set_ylabel(r'Dimensionless Tidal Deformability $\Lambda$',
               fontsize=12, fontweight='bold')
ax2.set_title('Lambda Curves (full solver output)', fontsize=13, fontweight='bold')
ax2.tick_params(axis='both', which='major', labelsize=11)
ax2.legend(frameon=False, fontsize=10, loc='best')

fig.suptitle('SFHo EOS: M-R and Lambda Comparison (raw family nodes)',
             fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()

plot_path = os.path.join(output_dir, 'mr_lambda_comparison.png')
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
print(f"  Saved: {plot_path}")
plt.show()

# ---- second figure: zoomed to the BNS-relevant range, linear axes ----
fig2, (bx1, bx2) = plt.subplots(1, 2, figsize=(14, 5))

if mrf is not None:
    bx1.plot(mrf['r'], mrf['m'], lw=2.5, color='teal', ls='--',
             label='Direct M-R (eos.mr)', alpha=0.9)
if pyc is not None:
    bx1.plot(pyc['r'], pyc['m'], lw=2.0, color='orange', ls='-',
             label='PyCompose EOS', alpha=0.85)
if dpr is not None:
    bx1.plot(dpr['r'], dpr['m'], lw=2.0, color='purple', ls='-.',
             label='Direct Parsing EOS', alpha=0.85)
bx1.set_xlim(9, 14)
bx1.set_ylim(0.8, 2.5)
bx1.grid(alpha=0.4, ls='--')
bx1.set_xlabel(r'Radius [km]', fontsize=12, fontweight='bold')
bx1.set_ylabel(r'Mass [$M_\odot$]', fontsize=12, fontweight='bold')
bx1.set_title('M-R, BNS range', fontsize=13, fontweight='bold')
bx1.legend(frameon=False, fontsize=10, loc='best')

if pyc is not None:
    w = pyc['m'] >= 1.0
    bx2.plot(pyc['m'][w], pyc['lam'][w], lw=2.0, color='orange', ls='-',
             label='PyCompose EOS', alpha=0.85)
if dpr is not None:
    w = dpr['m'] >= 1.0
    bx2.plot(dpr['m'][w], dpr['lam'][w], lw=2.0, color='purple', ls='-.',
             label='Direct Parsing EOS', alpha=0.85)
bx2.grid(alpha=0.4, ls='--')
bx2.set_xlabel(r'Mass [$M_\odot$]', fontsize=12, fontweight='bold')
bx2.set_ylabel(r'$\Lambda$', fontsize=12, fontweight='bold')
bx2.set_title('Lambda, above 1.0 M_sun', fontsize=13, fontweight='bold')
bx2.legend(frameon=False, fontsize=10, loc='best')

fig2.suptitle('SFHo EOS: zoomed to BNS-relevant range',
              fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()

plot_path2 = os.path.join(output_dir, 'mr_lambda_zoom.png')
plt.savefig(plot_path2, dpi=300, bbox_inches='tight')
print(f"  Saved: {plot_path2}")
plt.show()

# ============================================================================
# Summary
# ============================================================================

print("\n" + "="*80)
print("SUMMARY")
print("="*80)

print(f"\n{'Source':<24}{'M_max':>10}{'R @ M_max':>12}{'nodes':>8}")
print("-"*54)
if mrf is not None:
    i = int(np.argmax(mrf['m']))
    print(f"{'Direct M-R file':<24}{np.max(mrf['m']):>10.4f}"
          f"{mrf['r'][i]:>12.4f}{len(mrf['m']):>8}")
if pyc is not None:
    i = int(np.argmax(pyc['m']))
    print(f"{'PyCompose (TOV)':<24}{pyc['mmax']:>10.4f}"
          f"{pyc['r'][i]:>12.4f}{len(pyc['m']):>8}")
if dpr is not None:
    i = int(np.argmax(dpr['m']))
    print(f"{'Direct parsing (TOV)':<24}{dpr['mmax']:>10.4f}"
          f"{dpr['r'][i]:>12.4f}{len(dpr['m']):>8}")

if pyc is not None and dpr is not None:
    print("\nLambda at probe masses (interpolated from nodes only for this table):")
    print(f"\n{'mass':>6}{'PyCompose':>13}{'DirectParse':>13}{'diff %':>10}")
    print("-"*42)
    for m in [1.1, 1.2, 1.4, 1.6, 1.8, 2.0]:
        a = float(np.interp(m, pyc['m'], pyc['lam']))
        b = float(np.interp(m, dpr['m'], dpr['lam']))
        pct = 100 * (a - b) / b if b else np.nan
        print(f"{m:>6.2f}{a:>13.2f}{b:>13.2f}{pct:>9.2f}%")

print("\nNote: the two loaders may use different EOSFamily npts. If the Lambda")
print("curves differ, check EOS_FAMILY_NPTS in each file before blaming the EOS.")
print("="*80)