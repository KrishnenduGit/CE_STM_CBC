#!/usr/bin/env python3
"""
Adapter for differently formatted network files into the dictionary
shape the heatmap plotter expects.
The external files are not modified. Instead, a CSV maps each external filename stem to
the short net key used by the plot script, such as net1_ce7.
"""
import os
import csv
import json
import glob
import h5py
import numpy as np


def _stem(path_or_name):
    base = os.path.basename(path_or_name.strip())
    for ext in ('.hdf5', '.h5'):
        if base.endswith(ext):
            return base[:-len(ext)]
    return base


def load_name_map(csv_path):
    """
    Read mapping CSV.
    Required columns: ref_net_name,net_key
    """
    out = {}
    with open(csv_path, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            ref_name = _stem(row['ref_net_name'])
            net_key = row['net_key'].strip()
            out[ref_name] = net_key
    return out


def _read_parameters_indices(f):
    raw = f.attrs['parameters_indices']
    if isinstance(raw, bytes):
        raw = raw.decode()
    return json.loads(raw)


def _sigma_from_cov(cov, idx):
    diag = cov[idx, idx, :]
    sigma = np.full(diag.shape, np.nan)
    ok = np.isfinite(diag) & (diag > 0)
    sigma[ok] = np.sqrt(diag[ok])
    return sigma


def load_mapped_networks(networks_dir, name_map_csv):
    """
    Load differently formatted network files and return: {net_key: data_dict}
    The returned data_dict has the same fields as the plotter's load_networks().
    """
    name_map = load_name_map(name_map_csv)
    nets = {}
    files = sorted(
        glob.glob(os.path.join(networks_dir, '*.h5')) +
        glob.glob(os.path.join(networks_dir, '*.hdf5'))
    )
    for fp in files:
        base = _stem(fp)
        if base not in name_map:
            print(f'WARNING: no mapping for file {base}')
            continue
        short = name_map[base]
        with h5py.File(fp, 'r') as f:
            par = _read_parameters_indices(f)
            for needed in ('dL', 'chi1z'):
                if needed not in par:
                    raise KeyError(
                        f"{base}: parameters_indices has no '{needed}' "
                        f"(found {sorted(par)})."
                    )
            cov = f['covariance'][:]
            nets[short] = dict(
                snr=f['snr'][:],
                is_det=f['is_detected'][:],
                sky=f['sky_area_90'][:],
                sig_dL=_sigma_from_cov(cov, par['dL']),
                sig_chi1=_sigma_from_cov(cov, par['chi1z']),
                dL=f['event_parameters/dL'][:],
                z=f['event_parameters/z'][:],
                n_total=int(f.attrs['total_events']),
                n_det=int(f.attrs['detected_events']),
            )
    return nets
