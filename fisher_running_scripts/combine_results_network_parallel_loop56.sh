#!/bin/bash
#SBATCH --job-name=combine_loop56
#SBATCH -p parallel
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48          
#SBATCH -A berti                  
#SBATCH --time=0-03:00:00          
#SBATCH --output=combine_loop56.out 
#SBATCH --error=combine_loop56.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=lreali1@jhu.edu

# Define the arrays
POPS=("pbh" "imbh" "popiii")
CE_CONFIGS=(
    "1p5MW_Aplus_coat" 
    "1p0MW_Aplus_coat" 
    "1p5MW_aLIGO_coat" 
    "1p0MW_aLIGO_coat"
)
CE_FMINS=("5.0" "7.0" "10.0" "15.0")
LI_Aplus="LIA+:10.0"
ET_triangle="ETD:5.0"

echo "Starting sequential processing of 48 network configurations..."
echo "============================================================"

# Nested loops to iterate over every combination
for POP in "${POPS[@]}"; do
    for CONF in "${CE_CONFIGS[@]}"; do
        for FMIN in "${CE_FMINS[@]}"; do
            
            # Construct the detector string
            DETECTORS="CE40km_${CONF}:${FMIN} ${ET_triangle} ${LI_Aplus}"
            
            echo "Processing -> Pop: ${POP} | CE Config: ${CONF} | CE fmin: ${FMIN} Hz"
            
            # Run the Python script (it will block and wait to finish before the loop continues)
            python combine_results_network_parallel.py \
                --detectors $DETECTORS \
                --popname "${POP}" \
                --catalog_path "/home/lreali1/scr16_berti/luca/stm_cbc_runs/${POP}s" \
                --base_path "/home/lreali1/scr16_berti/luca/stm_cbc_runs/${POP}s/single_detectors" \
                --out_path "/home/lreali1/scr16_berti/luca/stm_cbc_runs/${POP}s/networks/networks56" \
                --snr_th 10.0 \
                --cores 48
                
            echo "Done with ${POP} - ${CONF} - ${FMIN} Hz"
            echo "------------------------------------------------------------"
            
        done
    done
done

echo "All network combinations have been successfully processed!"