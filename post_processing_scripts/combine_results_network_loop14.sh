#!/bin/bash
#SBATCH --job-name=combine_array
#SBATCH -p parallel
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --array=0-47                  # Spawns 48 independent jobs simultaneously: 4 CE configs x 4 fmin values x 3 populations = 48 total combinations
#SBATCH -A berti                  
#SBATCH --time=0-00:30:00             
#SBATCH --output=logs/combine_%A_%a.out 
#SBATCH --error=logs/combine_%A_%a.err

# IMPORTANT: Make sure you run 'mkdir -p logs' in your directory before submitting!

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

# Map the SLURM_ARRAY_TASK_ID to the specific grid combination
# TASK_ID ranges from 0 to 47
TASK_ID=$SLURM_ARRAY_TASK_ID

# Use modulo and integer division to find the correct index for each array
FMIN_IDX=$(( TASK_ID % 4 ))
CONF_IDX=$(( (TASK_ID / 4) % 4 ))
POP_IDX=$(( (TASK_ID / 16) % 3 ))

# Extract the exact parameters for this specific job clone
POP=${POPS[$POP_IDX]}
CONF=${CE_CONFIGS[$CONF_IDX]}
FMIN=${CE_FMINS[$FMIN_IDX]}

# Construct the detector string
DETECTORS="CE40_${CONF}:${FMIN} CE20_${CONF}:${FMIN} ${LI_Aplus}"

echo "=================================================="
echo "Array Task ID: $TASK_ID"
echo "Processing -> Pop: ${POP} | CE Config: ${CONF} | CE fmin: ${FMIN} Hz"
echo "=================================================="

# Run the Python script for just this single combination
python combine_results_network.py \
    --detectors $DETECTORS \
    --popname "${POP}" \
    --catalog_path "/home/lreali1/scr16_berti/luca/stm_cbc_runs/${POP}s" \
    --base_path "/home/lreali1/scr16_berti/luca/stm_cbc_runs/${POP}s/single_detectors" \
    --out_path "/home/lreali1/scr16_berti/luca/stm_cbc_runs/${POP}s/networks/networks14" \
    --snr_th 10.0 

echo "Task $TASK_ID complete!"