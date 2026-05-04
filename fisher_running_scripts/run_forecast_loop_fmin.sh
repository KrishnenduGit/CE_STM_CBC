#!/bin/bash
#SBATCH --job-name=fisher_run
#SBATCH -p parallel
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH -A berti                  
#SBATCH --time=0-04:00:00            
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=lreali1@jhu.edu

# Assign arguments to readable variables
#POP=pbh
#POP=imbh
POP=popiii

#DET=CE40_1p5MW_Aplus_coat  
#DET=CE40_1p0MW_Aplus_coat
#DET=CE40_1p5MW_aLIGO_coat
#DET=CE40_1p0MW_aLIGO_coat
#DET=CE20_1p5MW_Aplus_coat
#DET=CE20_1p0MW_Aplus_coat
#DET=CE20_1p5MW_aLIGO_coat
#DET=CE20_1p0MW_aLIGO_coat

# Convert population to uppercase for the catalog filename (pbh -> PBH)
POP_UPPER=${POP^^}

echo "=================================================="
echo "Initializing Fisher Pipeline"
echo "Population: ${POP_UPPER}"
echo "Detector:   ${DET}"
echo "=================================================="

# Prevent CPU Thread Contention
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# Dynamically construct the base paths
pop_folder="/home/lreali1/scr16_berti/luca/stm_cbc_runs/${POP}s"
net_folder="/home/lreali1/scr16_berti/luca/stm_cbc_runs/detectors"
NETJSON="${net_folder}/${DET}.json"

cd "$pop_folder"

mkdir -p "single_detectors/results_${POP}s_${DET}"

# Define the array of low-frequency cutoffs
FMIN_VALUES=(5.0 7.0 10.0 15.0)

# Loop through each frequency
for fmin in "${FMIN_VALUES[@]}"; do
    echo "--------------------------------------------------"
    echo "Starting run for fmin = ${fmin} Hz"
    
    # Dynamically construct catalog and output names
    CATALOG="${pop_folder}/${POP_UPPER}_cat_1yr_filtered_${fmin}hz.h5"
    OUTDIR="${pop_folder}/single_detectors/results_${POP}s_${DET}/results_${POP}s_${DET}_${fmin}hz"
    
    mkdir -p "$OUTDIR"
    
    srun -n 1 /usr/bin/time -v \
    python -m run.calculate_forecasts_from_catalog \
      --fname_obs "$CATALOG" \
      --fout "$OUTDIR" \
      --wf_model 'LAL-IMRPhenomXPHM' \
      --batch_size 5 \
      --npools 47 \
      --snr_th 1e-5 \
      --idx_in 0 \
      --fmin "$fmin" \
      --compute_fisher 1 \
      --netfile "$NETJSON" \
      --mpi 0 \
      --duty_factor 1. \
      --concatenate 1 \
      --rot 1 \
      --lalargs 'HM' \
      --return_all 1 \
      --jit_Fisher 0 
      
    echo "Finished run for fmin = ${fmin} Hz. Results in: $OUTDIR"
done

echo "=================================================="
echo "Pipeline complete for ${POP_UPPER} on ${DET}!"
