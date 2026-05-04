#!/bin/bash
#SBATCH --job-name=pbhs_CE40aligo
#SBATCH -p parallel
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH -A berti                  
#SBATCH --time=0-01:00:00            
#SBATCH --output=pbhs_CE40aligo.out
#SBATCH --error=pbhs_CE40aligo.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=lreali1@jhu.edu

# Prevent CPU Thread Contention
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

pop_folder='/home/lreali1/scr16_berti/luca/stm_cbc_runs/pbhs'
net_folder='/home/lreali1/scr16_berti/luca/stm_cbc_runs/detectors/gwfast_jsons'

CATALOG="$pop_folder/PBH_cat_1yr_filtered_5.0hz.h5"
OUTDIR="$pop_folder/single_detectors/results_pbhs_CE40_1p0MW_aLIGO_coat/results_pbhs_CE40_1p0MW_aLIGO_coat_5.0hz"
NETJSON="$net_folder/CE40_1p0MW_aLIGO_coat.json"

mkdir -p "$OUTDIR"
cd "$pop_folder"

srun -n 1 /usr/bin/time -v \
python -m run.calculate_forecasts_from_catalog \
  --fname_obs "$CATALOG" \
  --fout "$OUTDIR" \
  --wf_model 'LAL-IMRPhenomXPHM' \
  --batch_size 5 \
  --npools 47 \
  --snr_th 1e-5 \
  --idx_in 0 \
  --fmin 5. \
  --compute_fisher 1 \
  --netfile "$NETJSON" \
  --mpi 0 \
  --duty_factor 1. \
  --concatenate 1 \
  --rot 1 \
  --lalargs 'HM' \
  --return_all 1 \
  --jit_Fisher 0 
