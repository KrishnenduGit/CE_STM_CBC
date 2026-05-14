#!/bin/bash
#SBATCH --job-name=combine_network_par
#SBATCH -p parallel
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48          
#SBATCH -A berti                  
#SBATCH --time=0-01:00:00            
#SBATCH --output=combine_network_parallel.out
#SBATCH --error=combine_network_parallel.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=lreali1@jhu.edu

POP=bbh
#POP=pbh
#POP=imbh
#POP=popiii

# Available detectors and fmin:
#DET=CE40km_1p5MW_Aplus_coat  
#DET=CE40km_1p0MW_Aplus_coat
#DET=CE40km_1p5MW_aLIGO_coat
#DET=CE40km_1p0MW_aLIGO_coat
#DET=CE20km_1p5MW_Aplus_coat
#DET=CE20km_1p0MW_Aplus_coat
#DET=CE20km_1p5MW_aLIGO_coat
#DET=CE20km_1p0MW_aLIGO_coat
#DET=ETD
#DET=LIA+

# 5, 7, 10, 15 Hz for CE20 and CE40, 5 Hz for ET

python combine_results_network_parallel.py \
    --detectors CE40km_1p5MW_Aplus_coat:5.0 CE20km_1p5MW_Aplus_coat:5.0 LIA+:10.0 \
    --popname ${POP} \
    --catalog_path /home/lreali1/scr16_berti/luca/stm_cbc_runs/${POP}s \
    --base_path /home/lreali1/scr16_berti/luca/stm_cbc_runs/${POP}s/single_detectors \
    --out_path /home/lreali1/scr16_berti/luca/stm_cbc_runs/${POP}s/networks \
    --snr_th 10.0 \
    --cores 48                     # <-- CHANGED: Tell the Python script to use 48 cores