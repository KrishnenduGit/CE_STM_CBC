"""
Does rtol=1e-6 remove the Lambda monotonicity violations at REALISTIC node
density? Verification before touching the pipeline.

Why this test is needed: the previous sweep produced only 88 nodes above
1.0 M_sun (spacing ~0.012 M_sun), where the genuine Lambda decrease per node is
4-5% and 0.4% solver noise can never flip the ordering. It reported 0.0%
violations for every config, which proved nothing. The real bilby family has
~405 nodes above 1.0 M_sun (spacing ~0.0026 M_sun) where the genuine change is
~1% and 0.4% noise DOES flip it, giving the observed 23 violations.

So: reproduce the real node density, then compare rtol=1e-4 against 1e-6/1e-8.

Established by the corrected sweep:
  - rtol=1e-4 is off by 0.18-0.46% from converged (NOT converged)
  - rtol=1e-6 agrees with 1e-8 to 0.017%
  - DOP853 identical to RK45; h_stop irrelevant
  - densifying the EOS table shifts M_max 2.0603 -> 2.0578 and Lambda by
    0.5-1.5%: a systematic change, not a fix

Correct eos_dir. Cache cleared before and after.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import bilby
from bilby.gw.eos.tov_solver import IntegrateTOV

eos_dir = "/ligo/home/ligo.org/sanika.khadkikar/Projects/stm/post_processing_ns/CE_STM_CBC/pop_models/stellar_mass/injection_sampler_v2/sfho_compose_files"
output_dir = "./tov_rtol_verification"

CACHE_FILES = [
    os.path.join(eos_dir, "sfho_p_e_geom.txt"),
    os.path.join(eos_dir, "sfho_p_e_geom_compose.txt"),
]

G_SI = bilby.utils.gravitational_constant
c_SI = bilby.utils.speed_of_light
MSUN_SI = bilby.utils.solar_mass
MSUN_GEOM = G_SI * MSUN_SI / c_SI**2

PROBE = [1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.8]


def clear_cache(label):
    for p in CACHE_FILES:
        if os.path.exists(p):
            try:
                os.remove(p)
                print(f"  [{label}] deleted cache: {p}")
            except Exception as e:
                print(f"  [{label}] could not delete {p}: {e}")


if not os.path.exists(output_dir):
    os.makedirs(output_dir)

print("="*80)
print("VERIFICATION: does rtol=1e-6 remove violations at realistic node density?")
print("="*80)
clear_cache("pre")

# ------------------------------------------------------------ build EOS table
thermo_path = os.path.join(eos_dir, "eos.thermo.ns")
thermo = np.genfromtxt(thermo_path, skip_header=1)
with open(thermo_path) as fh:
    m_n = float(fh.readline().split()[0])
nb_raw = np.genfromtxt(os.path.join(eos_dir, "eos.nb.ns"))
nb = nb_raw[2:] if (nb_raw.ndim == 1 and len(nb_raw) > thermo.shape[0]) else nb_raw
nb = np.atleast_1d(nb).ravel()
n = min(len(nb), thermo.shape[0])
nb, thermo = nb[:n], thermo[:n]

MeVfm3_to_Pa = 1.602176634e32
to_geom = (MeVfm3_to_Pa / c_SI**2 / 1000.0) * ((G_SI / c_SI**2) * 1e3)

p_raw = thermo[:, 3] * nb
e_raw = (thermo[:, 9] + 1.0) * nb * m_n
mk = (p_raw > 0) & (e_raw > 0)
p_raw, e_raw = p_raw[mk], e_raw[mk]
o = np.argsort(p_raw)
pressure, energy = p_raw[o] * to_geom, e_raw[o] * to_geom

eos = bilby.gw.eos.TabularEOS(np.column_stack((pressure, energy)))
print(f"\nEOS table: {len(pressure)} points (native, unmodified)")


class TunableTOV(IntegrateTOV):
    def __init__(self, eos, eps_0, rtol=1e-4, atol=0.0, method="RK45"):
        super().__init__(eos, eps_0)
        self.rtol, self.atol, self.method = rtol, atol, method

    def integrate_TOV(self):
        res = solve_ivp(self._IntegrateTOV__tov_eqns,
                        (self.pseudo_enthalpy, 1e-16), self.y,
                        rtol=self.rtol, atol=self.atol, method=self.method)
        m, r = res.y[0, -1], res.y[1, -1]
        H, B = res.y[2, -1], res.y[3, -1]
        k2 = self._IntegrateTOV__calc_k2(r, B, H, m / r)
        return m, r, k2, res.nfev


def run_branch(eps_grid, **kw):
    mass, lam, nfev, nfail = [], [], 0, 0
    for eps0 in eps_grid:
        try:
            m, r, k2, nf = TunableTOV(eos, eps0, **kw).integrate_TOV()
            nfev += nf
            if not (np.isfinite(m) and np.isfinite(r) and np.isfinite(k2)):
                nfail += 1
                continue
            C = m / r
            L = (2.0 / 3.0) * k2 * C**-5
            if not (np.isfinite(L) and L > 0):
                nfail += 1
                continue
            mass.append(m / MSUN_GEOM)
            lam.append(L)
        except Exception:
            nfail += 1
    mass, lam = np.asarray(mass), np.asarray(lam)
    if len(mass) < 10:
        return None
    ipk = int(np.argmax(mass))              # peak in central-density order
    mass, lam = mass[:ipk + 1], lam[:ipk + 1]
    idx = np.argsort(mass)
    mass, lam = mass[idx], lam[idx]
    keep = np.concatenate(([True], np.diff(mass) > 0))
    return dict(mass=mass[keep], lam=lam[keep], nfev=nfev, nfail=nfail)


# ---------------------- calibrate grid size to hit ~405 nodes above 1.0 M_sun
print("\n" + "-"*80)
print("Step 1: calibrate central-density grid to the real family's node density")
print("-"*80)
print("\nTarget: ~405 nodes above 1.0 M_sun, spacing ~0.0026 M_sun")
print("(this is what bilby produces at npts=1000)\n")

lo, hi = np.log10(np.max(energy) * 2e-3), np.log10(np.max(energy) * 0.98)
n_grid = None
for trial in (400, 900, 1500, 2200):
    g = np.logspace(lo, hi, trial)
    r = run_branch(g, rtol=1e-4)
    if r is None:
        continue
    n_above = int(np.sum(r["mass"] >= 1.0))
    sp = np.median(np.diff(r["mass"][r["mass"] >= 1.0])) if n_above > 2 else np.nan
    print(f"  grid {trial:>5}: {n_above:>4} nodes above 1.0, spacing {sp:.5f} M_sun")
    n_grid = trial
    if n_above >= 380:
        break

eps_grid = np.logspace(lo, hi, n_grid)
print(f"\n  using grid = {n_grid}")

# ------------------------------------------------- Step 2: the actual test
print("\n" + "-"*80)
print("Step 2: violations vs rtol at realistic density")
print("-"*80)

configs = [
    ("rtol=1e-4 (bilby default)", dict(rtol=1e-4)),
    ("rtol=1e-5",                 dict(rtol=1e-5)),
    ("rtol=1e-6",                 dict(rtol=1e-6)),
    ("rtol=1e-7",                 dict(rtol=1e-7)),
    ("rtol=1e-8",                 dict(rtol=1e-8)),
]

runs = []
print(f"\n{'config':<28}{'nodes':>7}{'viol':>7}{'viol %':>9}"
      f"{'max rise %':>12}{'nfev':>12}{'rel cost':>10}")
print("-"*85)

base_nfev = None
for label, kw in configs:
    r = run_branch(eps_grid, **kw)
    if r is None:
        print(f"{label:<28}  failed")
        continue
    w = r["mass"] >= 1.0
    mm, ll = r["mass"][w], r["lam"][w]
    d = np.diff(ll)
    up = np.where(d > 0)[0]
    max_rise = 100 * np.max(d[up] / ll[:-1][up]) if len(up) else 0.0
    if base_nfev is None:
        base_nfev = r["nfev"]
    r.update(label=label, n=len(mm), nviol=len(up), max_rise=max_rise,
             mm=mm, ll=ll)
    runs.append(r)
    print(f"{label:<28}{len(mm):>7}{len(up):>7}{100*len(up)/max(1,len(d)):>8.2f}%"
          f"{max_rise:>12.3f}{r['nfev']:>12}{r['nfev']/base_nfev:>9.1f}x")

# ------------------------------------------------- Step 3: Lambda stability
print("\n" + "-"*80)
print("Step 3: does tightening rtol change the science?")
print("-"*80)

if runs:
    ref = runs[-1]
    hdr = f"{'config':<28}" + "".join(f"{f'L({m})':>10}" for m in PROBE)
    print("\n" + hdr)
    print("-"*len(hdr))
    for r in runs:
        print(f"{r['label']:<28}" +
              "".join(f"{float(np.interp(m, r['mm'], r['ll'])):>10.3f}" for m in PROBE))

    print(f"\nPercent difference from '{ref['label']}':\n")
    print(hdr)
    print("-"*len(hdr))
    for r in runs:
        cells = ""
        for m in PROBE:
            a = float(np.interp(m, r["mm"], r["ll"]))
            b = float(np.interp(m, ref["mm"], ref["ll"]))
            cells += f"{100*(a-b)/b:>9.3f}%"
        print(f"{r['label']:<28}{cells}")

# ------------------------------------------------------------------ plots
print("\n" + "-"*80)
print("Step 4: plotting")
print("-"*80)

if runs:
    fig, axes = plt.subplots(1, 3, figsize=(19, 5))

    ax = axes[0]
    ax.bar(range(len(runs)), [r["nviol"] for r in runs], color='crimson', alpha=0.85)
    ax.set_xticks(range(len(runs)))
    ax.set_xticklabels([r["label"] for r in runs], rotation=45, ha='right', fontsize=7)
    ax.set_ylabel('monotonicity violations', fontsize=11, fontweight='bold')
    ax.set_title('Violations vs rtol (realistic density)', fontsize=12, fontweight='bold')
    ax.grid(alpha=0.3, axis='y')

    ax = axes[1]
    ax.loglog([1e-4, 1e-5, 1e-6, 1e-7, 1e-8][:len(runs)],
              [max(r["nfev"], 1) for r in runs], 'o-', color='navy', lw=2)
    ax.set_xlabel('rtol', fontsize=11, fontweight='bold')
    ax.set_ylabel('total nfev', fontsize=11, fontweight='bold')
    ax.set_title('Cost vs rtol', fontsize=12, fontweight='bold')
    ax.grid(alpha=0.3, which='both')

    ax = axes[2]
    for r, c in zip(runs, plt.cm.viridis(np.linspace(0, 0.9, len(runs)))):
        w = (r["mm"] >= 1.0) & (r["mm"] <= 1.45)
        ax.plot(r["mm"][w], r["ll"][w], '-', lw=1.4, color=c,
                label=r["label"], alpha=0.85)
    ax.set_xlabel(r'Mass [$M_\odot$]', fontsize=11, fontweight='bold')
    ax.set_ylabel(r'$\Lambda$', fontsize=11, fontweight='bold')
    ax.set_title('Lambda, 1.0 to 1.45', fontsize=12, fontweight='bold')
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)

    fig.suptitle('Does tightening rtol remove the Lambda violations?',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    pp = os.path.join(output_dir, 'rtol_verification.png')
    plt.savefig(pp, dpi=300, bbox_inches='tight')
    print(f"  Saved: {pp}")
    plt.show()

clear_cache("post")

print("\n" + "="*80)
print("READING THE RESULT")
print("="*80)
print("Step 2 is the test. If violations fall sharply from rtol=1e-4 to 1e-6 at")
print("  ~400 nodes, tightening the tolerance is the fix and the kinks go away.")
print("If violations stay high at every rtol, the noise is NOT integration error")
print("  and the dedp spline derivative is the remaining suspect.")
print("Step 3 shows what tightening costs you in Lambda: if <0.5%, it is safe.")
print("'rel cost' in Step 2 is the runtime penalty, paid once per EOS load.")
print("="*80)