import numpy as np
import glob
import os
import h5py
import json
import argparse
from gwfast.fisherTools import compute_localization_region, CheckFisher, CovMatr, compute_inversion_error

def combine_fishers_network(det_dict, popname, catalog_path, base_path, out_path, snr_threshold=10.0):
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
    # Hard coded for IMRPhenomXPHM with aligned spins. Adjust if using a different waveform or parameter set!
    par_dict = {'Mc': 0,
                'eta': 1,
                'dL': 2,
                'theta': 3,
                'phi': 4,
                'iota': 5,
                'psi': 6,
                'tcoal': 7,
                'Phicoal': 8,
                'chi1z': 9,
                'chi2z': 10}
    
    # Loop through each detector
    for det_idx, (det, fmin) in enumerate(det_dict.items()):
        print(f"\n--- Processing {det} (fmin = {fmin} Hz) ---")

        # Determine which events from the full catalog survived this detector's fmin cut
        mask = (4.0 * f_isco) > fmin
        expected_events = np.sum(mask)
        print(f"Events in-band for {det}: {expected_events}")

        # Matrix which flags which events are in-band for each detector
        inband_matrix[mask, det_idx] = True

        det_dir = os.path.join(base_path, f"results_{popname}s_{det}", f"results_{popname}s_{det}_{fmin}hz")
        if 'ET' in det_dir:
            detname = 'net'
        else:
            detname = det.split('_')[0]

        # Load the Fisher matrices and SNRs
        fisher_pattern = os.path.join(det_dir, "all_fishers_0_to_*.hdf5")
        fisher_files = glob.glob(fisher_pattern)
        snr_pattern = os.path.join(det_dir, "all_snrs_0_to_*.hdf5")
        snr_files = glob.glob(snr_pattern)
        if not fisher_files:
            print(f"\n[ERROR] Could not find any files matching: {fisher_pattern}")
            print(f"Let's see what is actually inside the directory '{det_dir}':")
            try:
                # Print all files in that folder so you can check the spelling/extension
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
        
        # Safety Check: Ensure the loaded arrays match our mask length
        if len(snr_det) != expected_events:
            raise ValueError(f"Shape mismatch! Expected {expected_events} events based on fmin={fmin}, but found {len(snr_det)} in output.")

        if fisher_net is None:
            npar = F_det.shape[0]  # Assuming shape is (npar, npar, nevents)
            fisher_net = np.zeros((npar, npar, nevents))
            
        # Add SNRs in quadrature
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
    
    _, _, cond_numbers = CheckFisher(fishers_detected, use_mpmath=True)
    
    Cov_dL = np.full( fishers_detected.shape, np.nan)
    eps_dL = np.full( fishers_detected.shape[-1], np.nan)
    my_sky_area_90 = np.full( fishers_detected.shape[-1], np.nan)
    
    try:
        Cov_dL, eps_dL = CovMatr( fishers_detected,
                                                            invMethodIn='cho', 
                                                            condNumbMax=1e50, 
                                                            svals_thresh=1e-15, 
                                                            truncate=False, 
                                                            verbose=False
                                                            )
        
        print('Computing localization region...')
        my_sky_area_90 = compute_localization_region(Cov_dL, par_dict, events_detected["theta"], perc_level=90, units='SqDeg')
    except Exception as e:
        print(e)
        print()
        Cov_dL = np.full( fishers_detected.shape, np.nan)
        eps_dL = np.full(fishers_detected.shape[-1], np.nan)
        my_sky_area_90 = np.full(fishers_detected.shape[-1], np.nan)
        cond_numbers = np.full(fishers_detected.shape[-1], np.nan)

    # Save the combined network results
    os.makedirs(out_path, exist_ok=True)
    name_parts = [f"{det}_{fmin}hz" for det, fmin in det_dict.items()]
    outfile_name = f"network_{popname}_{'_'.join(name_parts)}.h5"
    outfile_path = os.path.join(out_path, outfile_name)

    with h5py.File(outfile_path, 'w') as f:
            # Create datasets for the mathematical arrays
            f.create_dataset('snr', data=snr_net)
            f.create_dataset('fisher', data=fisher_net)
            f.create_dataset('covariance', data=Cov_dL)
            f.create_dataset('inversion_errors', data=eps_dL)
            f.create_dataset('sky_area_90', data=my_sky_area_90)
            f.create_dataset('condition_numbers', data=cond_numbers)
            f.create_dataset('is_in_band', data=inband_matrix)
            f.create_dataset('is_detected', data=detected_mask)

            # Save true parameters for all the events
            param_group = f.create_group('event_parameters')
            for key, val in events.items():
                param_group.create_dataset(key, data=val)
            
            # Save metadata as attributes
            f.attrs['detectors'] = ",".join(detector_names) 
            f.attrs['snr_threshold'] = snr_threshold
            f.attrs['total_events'] = n_total
            f.attrs['detected_events'] = n_detected
            f.attrs['event_parameters'] = json.dumps(par_dict)
    
    print(f"Network calculation complete! Results saved to {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Combine individual gwfast detector runs into a network.")
    # Use nargs='+' to accept a list of detectors (e.g., --detectors CE40 CE20 ET)
    parser.add_argument("--detectors", nargs='+', required=True, help="Format -> DET:FMIN (e.g., CE40:10.0 CE20:15.0 ET:5.0)")
    parser.add_argument("--popname", type=str, required=True, help="Population name")
    parser.add_argument("--catalog_path", type=str, required=True, help="Path containing the full event catalog")
    parser.add_argument("--base_path", type=str, required=True, help="Path containing the results folders")
    parser.add_argument("--out_path", type=str, required=True, help="Where to save the combined network results")
    parser.add_argument("--snr_th", type=float, default=10.0, help="Network SNR threshold for detection")
    
    args = parser.parse_args()

    det_dict = {}
    for item in args.detectors:
        det, fmin_str = item.split(':')
        det_dict[det] = float(fmin_str)

    combine_fishers_network(det_dict, args.popname, args.catalog_path, args.base_path, args.out_path, args.snr_th)