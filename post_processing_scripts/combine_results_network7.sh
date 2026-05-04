#!/bin/bash
#SBATCH --job-name=combine_networks7
#SBATCH -p parallel
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH -A berti                  
#SBATCH --time=0-01:00:00            
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

# Define arrays for the parameters
POPS=("pbh" "imbh" "popiii")

# The Cosmic Explorer PSD configurations
CE_CONFIGS=(
    "1p5MW_Aplus_coat" 
)

# The low-frequency limits for CE
CE_FMINS=("5.0" "7.0" "10.0" "15.0")

# Fixed ET configuration
ET_DET="ET_triangle:5.0"

echo "Starting network combination loops..."
echo "=================================================="

# Loop over populations
for pop in "${POPS[@]}"; do
    
    # Loop over CE PSD designs
    for ce_config in "${CE_CONFIGS[@]}"; do
        
        # Loop over CE low-frequency cutoffs
        for fmin in "${CE_FMINS[@]}"; do
            
            # Construct the detector string
            DETECTORS="CE40_${ce_config}:${fmin} CE20_${ce_config}:${fmin} ${ET_DET}"
            
            echo "Processing -> Pop: ${pop} | CE Config: ${ce_config} | CE fmin: ${fmin} Hz"

            mkdir -p "/home/lreali1/scr16_berti/luca/stm_cbc_runs/${pop}s/networks/network7"
            
            python combine_results_network.py \
                --detectors $DETECTORS \
                --popname "${pop}" \
                --catalog_path "/home/lreali1/scr16_berti/luca/stm_cbc_runs/${pop}s" \
                --base_path "/home/lreali1/scr16_berti/luca/stm_cbc_runs/${pop}s/single_detectors" \
                --out_path "/home/lreali1/scr16_berti/luca/stm_cbc_runs/${pop}s/networks/network7" \
                --snr_th 10.0 
                
            echo "--------------------------------------------------"
        done
    done
done

echo "=================================================="
echo "All network combinations successfully combined!"