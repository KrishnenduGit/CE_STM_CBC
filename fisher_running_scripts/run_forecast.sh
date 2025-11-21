#!/bin/bash
#SBATCH --job-name=gwfast_full_GR_nojit
#SBATCH -p parallel
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH -A berti                  
#SBATCH --time=1-00:00:00            
#SBATCH --output=gwfast_full_GR_bs10nojit%j.out
#SBATCH --error=gwfast_full_GR_bs10nojit%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=tayyaban414@gmail.com


module load anaconda3/2023.09-0
eval "$(conda shell.bash hook)"
conda activate gwfast-env

CATALOG="/home/ext-tayyaba/gwfast/events_20250910_064359.h5"
OUTDIR="/home/ext-tayyaba/gwfast/results_full_GR_bs10_jit0"
NETJSON="/home/ext-tayyaba/gwfast/CEET_network.json"

mkdir -p "$OUTDIR"
cd /home/ext-tayyaba/gwfast

srun -n 1 /usr/bin/time -v \
python -m run.calculate_forecasts_from_catalog \
  --fname_obs "$CATALOG" \
  --fout "$OUTDIR" \
  --wf_model IMRPhenomD \
  --batch_size 10 \
  --npools 47 \
  --snr_th 12. \
  --idx_in 0 \
  --fmin 2. \
  --compute_fisher 1 \
  --netfile "$NETJSON" \
  --mpi 0 \
  --duty_factor 1. \
  --concatenate 1 \
  --rot 1 \
  --jit_Fisher 0 
