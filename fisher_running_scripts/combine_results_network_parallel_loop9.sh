#!/bin/bash
#SBATCH --job-name=combine_loop9
#SBATCH -p parallel
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48          
#SBATCH -A berti                  
#SBATCH --time=0-03:00:00          
#SBATCH --output=combine_loop9.out 
#SBATCH --error=combine_loop9.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=lreali1@jhu.edu

# Define the arrays
POPS=("pbh" "imbh" "popiii")
LIGO_CONFIGS=(
    "A+" 
    "Asharp" 
)
FMIN="10.0"

echo "Starting sequential processing of network configurations..."
echo "============================================================"

# Nested loops to iterate over every combination
for POP in "${POPS[@]}"; do
    for CONF in "${LIGO_CONFIGS[@]}"; do
        # Construct the detector string
        DETECTORS="LH${CONF}:${FMIN} LL${CONF}:${FMIN} LI${CONF}:${FMIN}"
        
        echo "Processing -> Pop: ${POP} | LIGO Config: ${CONF}"
        
        # Run the Python script (it will block and wait to finish before the loop continues)
        python combine_results_network_parallel.py \
            --detectors $DETECTORS \
            --popname "${POP}" \
            --catalog_path "/home/lreali1/scr16_berti/luca/stm_cbc_runs/${POP}s" \
            --base_path "/home/lreali1/scr16_berti/luca/stm_cbc_runs/${POP}s/single_detectors" \
            --out_path "/home/lreali1/scr16_berti/luca/stm_cbc_runs/${POP}s/networks/networks9" \
            --snr_th 10.0 \
            --cores 48
            
        echo "Done with ${POP} - ${CONF}"
        echo "------------------------------------------------------------"
            
    done
done

echo "All network combinations have been successfully processed!"