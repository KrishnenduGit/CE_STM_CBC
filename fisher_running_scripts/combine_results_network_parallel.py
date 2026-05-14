import numpy as np
import glob
import os
import h5py
import json
import argparse
from multiprocessing import Pool, cpu_count
from gwfast.fisherTools import compute_localization_region, CheckFisher, CovMatr, compute_inversion_error

def _process_chunk(args):
    """
    Worker function to process a chunk of Fisher matrices in parallel.
    Defined at the top level so it can be pickled by multiprocessing.Pool.
    """
    fishers_chunk, theta_chunk, par_dict = args
    
    # Initialize default NaN arrays in case of chunk failure or empty chunk
    Cov_dL = np.full(fishers_chunk.shape, np.nan)
    eps_dL = np.full(fishers_chunk.shape[-1], np.nan)
    my_sky_area_90 = np.full(fishers_chunk.shape[-1], np.nan)
    cond_numbers = np.full(fishers_chunk.shape[-1], np.nan)

    if fishers_chunk.shape[-1] == 0:
        return Cov_dL, eps_dL, my_sky_area_90, cond_numbers

    # CheckFisher returns condition numbers as the third output
    try:
        _, _, cond_numbers = CheckFisher(fishers_chunk, use_mpmath=True)
    except Exception as e:
        print(f"CheckFisher error on chunk: {e}")

    try:
        Cov_dL, eps_dL = CovMatr(
            fishers_chunk,
            invMethodIn='cho', 
            condNumbMax=1e50, 
            svals_thresh=1e-15, 
            truncate=False, 
            verbose=False
        )
        my_sky_area_90 = compute_localization_region(
            Cov_dL, par_dict, theta_chunk, perc_level=90, units='SqDeg'
        )
    except Exception as e:
        print(f"Inversion/Localization error on chunk: {e}")

    return Cov_dL, eps_dL, my_sky_area_90, cond_numbers

def combine_fishers_network(det_dict, popname, catalog_path, base_path, out_path, snr_threshold=10.0, n_cores=48):
    print(f"Combining network: {' + '.join(det_dict.keys())}")
    
    # Loading events
    event_file = os.path.join(catalog_path, f"{popname}".upper() + f"_cat_1yr.h5")
    with h5py.File(event_file, 'r') as f:
        events = {key: f[key][:] for key in f.keys()}

    nevents = len(events['m1_src'])
    print(f"Total events in full catalog: {nevents}")

    # Calculate detector-frame total mass
    Mz = (events['m1_src'] + events['m2_src']) * (1 + events['z'])
    
    # Calculate Schwarzschild ISCO GW frequency
    GMsun_over_c3 = 4.925491025543575903411922162094833998e-6 # seconds
    f_isco=1./(np.sqrt(6.)*6.*np.pi*GMsun_over_c3)/ Mz

    snr_net_sq = np.zeros(nevents)
    fisher_net = None  # Will initialize after we know the shape from the first detector

    detector_names = list(det_dict.keys())
    inband_matrix = np.zeros((nevents, len(detector_names)), dtype=bool)

    # Parameter indices in the Fisher. 
    par_dict = {'Mc': 0, 'eta': 1, 'dL': 2, 'theta': 3, 'phi': 4, 
                'iota': 5, 'psi': 6, 'tcoal': 7, 'Phicoal': 8, 
                'chi1z': 9, 'chi2z': 10}
    
    # Loop through each detector
    for det_idx, (det, fmin) in enumerate(det_dict.items()):
        print(f"\n--- Processing {det} (fmin = {fmin} Hz) ---")

        mask = (4.0 * f_isco) > fmin
        expected_events = np.sum(mask)
        print(f"Events in-band for {det}: {expected_events}")

        inband_matrix[mask, det_idx] = True

        det_dir = os.path.join(base_path, f"results_{popname}s_{det}", f"results_{popname}s_{det}_{fmin}hz")
        detname = 'net' if 'ET' in det_dir else det.split(':')[0]

        fisher_pattern = os.path.join(det_dir, "all_fishers_0_to_*.hdf5")
        fisher_files = glob.glob(fisher_pattern)
        snr_pattern = os.path.join(det_dir, "all_snrs_0_to_*.hdf5")
        snr_files = glob.glob(snr_pattern)
        
        if not fisher_files:
            print(f"\n[ERROR] Could not find any files matching: {fisher_pattern}")
            try:
                print(os.listdir(det_dir)) 
            except FileNotFoundError:
                print(f" -> Wait, the directory {det_dir} doesn't even exist! Check your path variables.")
            raise FileNotFoundError("Stopping script due to missing Fisher file.")

        fisher_file = fisher_files[0]
        snr_file = snr_files[0]
        
        try:
            with h5py.File(fisher_file, 'r') as f:
                F_det = f['fisher'][detname][:]
            with h5py.File(snr_file, 'r') as f:
                snr_det = f['snr'][detname][:]
        except FileNotFoundError:
            raise FileNotFoundError(f"Could not find Fisher/SNR files in {det_dir}.")
        
        if len(snr_det) != expected_events:
            raise ValueError(f"Shape mismatch! Expected {expected_events} events based on fmin={fmin}, but found {len(snr_det)} in output.")

        if fisher_net is None:
            npar = F_det.shape[0]  
            fisher_net = np.zeros((npar, npar, nevents))
            
        snr_net_sq[mask] += snr_det**2
        fisher_net[:, :, mask] += F_det

    # Calculate final network SNR
    snr_net = np.sqrt(snr_net_sq)
    
    # Apply Network SNR Threshold
    detected_mask = snr_net > snr_threshold
    n_detected = np.sum(detected_mask)
    n_total = len(snr_net)
    
    events_detected = {key: val[detected_mask] for key, val in events.items()}
    fishers_detected = fisher_net[:, :, detected_mask]
    print(f"Events detected (Network SNR > {snr_threshold}): {n_detected} / {n_total}")
    
    # --- PARALLEL MATRIX COMPUTATION START ---
    if n_detected > 0:
        # Cap cores at available system CPUs to prevent oversubscription issues just in case
        usable_cores = min(n_cores, cpu_count())
        print(f"Computing localization region and covariances in parallel across {usable_cores} cores...")

        # Split along the event axis (last axis)
        fisher_chunks = np.array_split(fishers_detected, usable_cores, axis=-1)
        theta_chunks = np.array_split(events_detected["theta"], usable_cores)
        
        # Package arguments for the worker pool
        pool_args = [(fc, tc, par_dict) for fc, tc in zip(fisher_chunks, theta_chunks)]

        with Pool(processes=usable_cores) as pool:
            results = pool.map(_process_chunk, pool_args)

# Recombine the returned chunks (These correspond ONLY to detected events)
        Cov_dL_det = np.concatenate([res[0] for res in results], axis=-1)
        eps_dL_det = np.concatenate([res[1] for res in results])
        my_sky_area_90_det = np.concatenate([res[2] for res in results])
        cond_numbers_det = np.concatenate([res[3] for res in results])
    else:
        # Handle the edge case where no events pass the SNR threshold
        print("No events detected. Skipping matrix inversions.")
        npar = fisher_net.shape[0]
        Cov_dL_det = np.empty((npar, npar, 0))
        eps_dL_det = np.empty(0)
        my_sky_area_90_det = np.empty(0)
        cond_numbers_det = np.empty(0)
    # --- PARALLEL MATRIX COMPUTATION END ---

    # --- PADDING TO FULL CATALOG SIZE ---
    print("Padding uncalculated events with NaNs to match catalog size...")
    
    # Initialize arrays of the full catalog size (nevents) filled with NaNs
    Cov_dL = np.full(fisher_net.shape, np.nan)
    eps_dL = np.full(nevents, np.nan)
    my_sky_area_90 = np.full(nevents, np.nan)
    cond_numbers = np.full(nevents, np.nan)

    # Insert the calculated arrays into the indices where detected_mask is True.
    # We use the ellipsis (...) for Cov_dL to apply the mask only to the last axis.
    Cov_dL[..., detected_mask] = Cov_dL_det
    eps_dL[detected_mask] = eps_dL_det
    my_sky_area_90[detected_mask] = my_sky_area_90_det
    cond_numbers[detected_mask] = cond_numbers_det
    # -------------------------------------

    # Save the combined network results
    os.makedirs(out_path, exist_ok=True)
    name_parts = [f"{det}_{fmin}hz" for det, fmin in det_dict.items()]
    outfile_name = f"network_{popname}_{'_'.join(name_parts)}.h5"
    outfile_path = os.path.join(out_path, outfile_name)

    with h5py.File(outfile_path, 'w') as f:
        f.create_dataset('snr', data=snr_net)
        f.create_dataset('fisher', data=fisher_net)
        f.create_dataset('covariance', data=Cov_dL)
        f.create_dataset('inversion_errors', data=eps_dL)
        f.create_dataset('sky_area_90', data=my_sky_area_90)
        f.create_dataset('condition_numbers', data=cond_numbers)
        f.create_dataset('is_in_band', data=inband_matrix)
        f.create_dataset('is_detected', data=detected_mask)

        param_group = f.create_group('event_parameters')
        for key, val in events.items():
            param_group.create_dataset(key, data=val)
        
        f.attrs['detectors'] = ",".join(detector_names) 
        f.attrs['snr_threshold'] = snr_threshold
        f.attrs['total_events'] = n_total
        f.attrs['detected_events'] = n_detected
        f.attrs['parameters_indices'] = json.dumps(par_dict)
    
    print(f"Network calculation complete! Results saved to {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Combine individual gwfast detector runs into a network.")
    parser.add_argument("--detectors", nargs='+', required=True, help="Format -> DET:FMIN (e.g., CE40:10.0 CE20:15.0 ET:5.0)")
    parser.add_argument("--popname", type=str, required=True, help="Population name")
    parser.add_argument("--catalog_path", type=str, required=True, help="Path containing the full event catalog")
    parser.add_argument("--base_path", type=str, required=True, help="Path containing the results folders")
    parser.add_argument("--out_path", type=str, required=True, help="Where to save the combined network results")
    parser.add_argument("--snr_th", type=float, default=10.0, help="Network SNR threshold for detection")
    parser.add_argument("--cores", type=int, default=48, help="Number of CPU cores to use for parallelizing Fisher matrix operations")
    
    args = parser.parse_args()

    det_dict = {}
    for item in args.detectors:
        det, fmin_str = item.split(':')
        det_dict[det] = float(fmin_str)

    combine_fishers_network(det_dict, args.popname, args.catalog_path, args.base_path, args.out_path, args.snr_th, args.cores)