#!/usr/bin/env python3

import os
import glob
import argparse
from itertools import groupby
import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from network_adapter import load_mapped_networks

snr_th = 10.0
ce_tech = ['hAp', 'hAL', 'lAp', 'lAL']  # Technology choices for CE detectors
fmins = [7, 10, 15]

# Define broad themes and metrics to be presented
themes = [
    ('Population\nreach', [
        ('det_rate', 'Detection rate (%)'),
        ('hi_z', r'$z > 5$'),
        ('loud', r'SNR $> 100$'),
    ]),
    ('Black hole\nspin', [
        ('chi1', r'$\sigma_{\chi_{1z}} < 0.05$'),
    ]),
    ('Multi-\nmessenger', [
        ('sa10', r'Sky area < 10 $\mathrm{deg}^2$'),
        ('sa1', r'Sky area < 1 $\mathrm{deg}^2$'),
        ('dl10', r'$\sigma_{d_L}/d_L < 0.1$'),
    ]),
]


def columns():
    """
    The full list of columns to be plotted with entries: net_name, column_label, family, fmin_label

    net_name must match the prefix of the network HDF5 filename.
    For example, net1_ce10__CE40hAp_10_CE20hAp_10_LIAp_10.hdf5.
    """

    # Make the 'l' for low circulating power of 1 MW distinguishable from capital I.
    ell = lambda c: c.replace('l', r'$\ell$')

    out = []
    # For the first network family: CE40 + CE20 + Ip, create columns for each CE fmin.
    for f in fmins:
        for i, c in enumerate(ce_tech):
            out.append((f'net{i + 1}_ce{f}', f'CE40{ell(c)}+CE20{ell(c)}+Ip',
                        'CE40 + CE20 + Ip', f'{f} Hz'))

    # CE40 + ET + Ip
    for f in fmins:
        for nm, c in zip(['net5a', 'net5b', 'net6a', 'net6b'], ce_tech):
            out.append((f'{nm}_ce{f}', f'CE40{ell(c)}+ET+Ip',
                        'CE40 + ET + Ip', f'{f} Hz'))

    # One column per fmin for the following.
    single = [
        ('net7', 'CE40hAp+CE20hAp+ET', 'CE40 + CE20 + ET'),
        ('net8a', 'CE40hAp+ET', 'CE40 + ET'),
        ('net8b', 'CE40hAp+ET-2L', 'CE40 + ET-2L'),
    ]

    for prefix, label, family in single:
        for f in fmins:
            out.append((f'{prefix}_ce{f}', label, family, f'{f} Hz'))

    # Current-gen detectors - no CE so no fmin label necessary.
    out.append(('net9a', 'HLI A+', 'HLI', ''))
    out.append(('net9b', 'HLI A#', 'HLI', ''))

    return out


def x_positions(cols, gap_fmin=0.4, gap_fam=1.2):
    """ Creating spaces between columns when family or sub-group changes for better distinction."""

    xs = []
    x = 0.0

    for j, (_, _, family, fmin) in enumerate(cols):
        if j:
            prev_family, prev_fmin = cols[j - 1][2], cols[j - 1][3]

            if family != prev_family:
                x += 1 + gap_fam     # new family -> big gap
            elif fmin != prev_fmin:
                x += 1 + gap_fmin    # same family, new fmin -> small gap
            else:
                x += 1               # same subgroup

        xs.append(x)
    return xs

def spans(cols, field):
    """
    Group consecutive columns that share the same family or fmin label.
    Eg. ('CE40 + CE20 + Ip', 0, 12)  # This family spans columns 0 to 11.
    """

    out, i = [], 0
    for value, grp in groupby(range(len(cols)), key=lambda j: cols[j][field]):
        n = len(list(grp))
        out.append((value, i, i + n))
        i += n

    return out


def text_color(rgba):
    # Pick black or white text depending on the brightness of cells using luminance weights.
    r, g, b, _ = rgba
    return 'black' if (0.299 * r + 0.587 * g + 0.114 * b) > 0.55 else 'white'


def load_networks(networks_dir):
    # One HDF5 file per network. The key is the filename prefix before '__'.
    nets = {}

    for fp in sorted(glob.glob(os.path.join(networks_dir, '*.hdf5'))):
        short = os.path.splitext(os.path.basename(fp))[0].split('__')[0]
        # Loading errors
        with h5py.File(fp, 'r') as f:
            nets[short] = dict(
                snr=f['snr'][:],
                is_det=f['is_detected'][:],
                sky=f['sky_area_90'][:],
                sig_dL=f['errors/sigma_dL'][:],
                sig_chi1=f['errors/sigma_chi1z'][:],
                dL=f['event_parameters/dL'][:],
                z=f['event_parameters/z'][:],
                n_total=int(f.attrs['total_events']),
                n_det=int(f.attrs['detected_events']),
            )

    return nets


def metrics_for(d):
    """Turn one network's raw arrays into the metrics we display. (Apply appropriate guards)"""
    snr, det, sky, z = d['snr'], d['is_det'], d['sky'], d['z']

    # Population reach: detection rate, reach and loudness
    m = {
        'det_rate': 100.0 * d['n_det'] / d['n_total'],
        'hi_z': int((det & (z > 5)).sum()),
        'loud': int((snr > 100).sum()),
    }

    # BH spin:
    chi1 = d['sig_chi1']
    ok = det & np.isfinite(chi1) & (chi1 > 0)
    m['chi1'] = int((ok & (chi1 < 0.05)).sum())

    # Multi-messenger: sky area and fractional distance uncertainty
    with np.errstate(divide='ignore', invalid='ignore'):
        frac_dL = d['sig_dL'] / d['dL']

    sky_ok = det & np.isfinite(sky) & (sky > 0)
    dL_ok = det & np.isfinite(frac_dL) & (frac_dL > 0)

    m['sa10'] = int((sky_ok & (sky < 10)).sum())
    m['sa1'] = int((sky_ok & (sky < 1)).sum())
    m['dl10'] = int((dL_ok & (frac_dL < 0.1)).sum())

    return m


def cell_text(key, val):
    # Detection rate is in percentage, other rows are counts.
    if key == 'det_rate':
        if val >= 1: return f'{val:.0f}%'
        if val >= 0.01: return f'{val:.2f}%' # To avoid getting 0 for HLI
        return f'{val:.0e}%' if val > 0 else '0%'

    if val >= 1e4: return f'{val / 1000:.0f}k'
    if val >= 1e3: return f'{val / 1000:.1f}k'

    return f'{int(val)}'


def plot(nets, out_path):
    # Setting columns and display geometry
    cols = columns()
    col_x = x_positions(cols)
    supers = spans(cols, 2)       # family brackets
    fmin_spans = spans(cols, 3)   # fmin sub-labels

    # Flatten the theme rows and record their span
    rows, row_spans, i0 = [], [], 0

    for theme, items in themes:
        for key, label in items:
            rows.append((theme, key, label))

        row_spans.append((theme, i0, i0 + len(items)))
        i0 += len(items)

    nrow, ncol = len(rows), len(cols)

    # Compute every network's metrics once, then fill the value grid.
    M = {}
    for col in cols:
        net = col[0]
        if net in nets: M[net] = metrics_for(nets[net])
        else: print(f'WARNING: {net} not found in networks')

    values = np.zeros((nrow, ncol))
    for i, row in enumerate(rows):
        key = row[1]
        for j, col in enumerate(cols):
            net = col[0]
            if net in M:
                values[i, j] = M[net].get(key, 0)

    # Drawing rows bottom-to-top with some gaps.
    row_y, y = [0.0] * nrow, 0.0

    for i in range(nrow - 1, -1, -1):
        row_y[i] = y
        y += 1

        if i and rows[i][0] != rows[i - 1][0]: y += 0.4

    W = col_x[-1] + 1 if col_x else 1   # grid width in cell units
    H = max(row_y) + 1 if row_y else 1  # grid height in cell units

    fig, ax = plt.subplots(figsize=(26, 11))
    #cmap = plt.cm.RdYlGn
    cmap = mcolors.LinearSegmentedColormap.from_list(
    "custom_RdYlGn",
    [
        "#830f07",
        "#d73027",
        "#f46d43",
        "#fdae61",
        "#fee08b",
        "#ffffbf",
        "#d9ef8b",
        "#a6d96a",
        "#66bd63",
        "#12723c",
        "#022716",
    ]
)

    # Draw cells one row at a time
    for i in range(nrow):
        key = rows[i][1]
        rv = values[i]

        # Color scale is chosen per row.
        if key == 'det_rate':
            norm = mcolors.Normalize(0, 100)
        elif rv.max() > 0:
            vmax = rv.max()
            pos = rv[rv > 0]
            vmin = max(pos.min() if pos.size else 1, 1)
            norm = (mcolors.LogNorm(vmin, vmax) if vmax > 10 * vmin
                    else mcolors.Normalize(0, vmax))
        else:
            norm = mcolors.Normalize(0, 1)  # for whole row zero

        for j in range(ncol):
            v = values[i, j]
            color = cmap(norm(v)) if v > 0 else cmap(0.0)

            ax.add_patch(plt.Rectangle((col_x[j], row_y[i]), 1, 1,
                                       facecolor=color, edgecolor='white', lw=0.6))
            ax.text(col_x[j] + 0.5, row_y[i] + 0.5, cell_text(key, v),
                    ha='center', va='center', color=text_color(color), fontsize=8)

    # Family brackets and names
    for label, s, e in supers:
        xl, xr = col_x[s], col_x[e - 1] + 1
        ax.plot([xl, xr], [H + 0.9, H + 0.9], color='black', lw=1.3)
        ax.text(0.5 * (xl + xr), H + 1.4, label,
                ha='center', va='bottom', fontsize=11, fontweight='bold')

    # Sub-labels for fmins
    for label, s, e in fmin_spans:
        if not label:
            continue

        xl, xr = col_x[s], col_x[e - 1] + 1
        ax.text(0.5 * (xl + xr), H + 0.3, label,
                ha='center', va='bottom', fontsize=9, style='italic', color='dimgray')

    # Network labels to be displayed at the bottom
    for j, col in enumerate(cols):
       label = col[1]
       ax.text(col_x[j] + 0.5, -0.4, label,
                ha='right', va='top', fontsize=8, rotation=70)

    # Theme names to be displayed on the left of the summary plot
    theme_x = -7.1

    for theme, s, e in row_spans:
        y_mid = 0.5 * (row_y[s] + 1 + row_y[e - 1])
        ax.text(theme_x, y_mid, theme,
                ha='left', va='center', fontsize=11, fontweight='bold', color='dimgray')

    for i, row in enumerate(rows):
       label = row[2]
       ax.text(-0.3, row_y[i] + 0.5, label, ha='right', va='center', fontsize=10)

    ax.set_xlim(theme_x - 0.3, W + 0.5)
    ax.set_ylim(-4.5, H + 2.1 + 0.5)
    ax.set_xticks([])
    ax.set_yticks([])

    for sp in ax.spines.values():
        sp.set_visible(False)

    # Display the catalog size and the detection threshold in the title.
    totals = sorted({d['n_total'] for d in nets.values()})
    cat = f'N = {totals[0]:,} events'

    ax.set_title('Network Science Metrics for NSBH', fontsize = 15, fontweight = 'bold', pad = 32)
    ax.text(0.5, 0.99, f'1-year catalog: {cat}\n'
            f'Detection threshold: network SNR > {snr_th:g}',
            transform = ax.transAxes, ha = 'center', va='bottom', fontsize = 12)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, bbox_inches='tight', dpi=200)
    print(f'Saved: {out_path}')
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--networks_dir',
                    default=None)
    ap.add_argument('--out_path',
                    default=None)
    ap.add_argument('--name_map_csv', default=None,
                    help='Load externally-formatted networks using this CSV map.')
    args = ap.parse_args()

    print(f'Loading networks from {args.networks_dir}')
    if args.name_map_csv:
        nets = load_mapped_networks(args.networks_dir, args.name_map_csv)
    else:
        nets = load_networks(args.networks_dir)

    print(f'Loaded {len(nets)} networks')
    plot(nets, args.out_path)


if __name__ == '__main__':
    main()
