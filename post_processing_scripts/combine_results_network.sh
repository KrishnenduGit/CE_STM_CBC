#!/bin/bash
#SBATCH --job-name=test_combine_network
#SBATCH -p parallel
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH -A berti                  
#SBATCH --time=0-00:30:00            
#SBATCH --output=test_combine_network.out
#SBATCH --error=test_combine_network.err

POP=pbh
#POP=imbh
#POP=popiii

# Available detectors and fmin:
#DET=CE40_1p5MW_Aplus_coat  
#DET=CE40_1p0MW_Aplus_coat
#DET=CE40_1p5MW_aLIGO_coat
#DET=CE40_1p0MW_aLIGO_coat
#DET=CE20_1p5MW_Aplus_coat
#DET=CE20_1p0MW_Aplus_coat
#DET=CE20_1p5MW_aLIGO_coat
#DET=CE20_1p0MW_aLIGO_coat
#DET=ET_triangle

# 5, 7, 10, 15 Hz for CE20 and CE40, 5 Hz for ET

python combine_results_network.py \
    --detectors CE40_1p5MW_Aplus_coat:5.0 CE20_1p5MW_Aplus_coat:5.0 ET_triangle:5.0 \
    --popname ${POP} \
    --catalog_path /home/lreali1/scr16_berti/luca/stm_cbc_runs/${POP}s \
    --base_path /home/lreali1/scr16_berti/luca/stm_cbc_runs/${POP}s/single_detectors \
    --out_path /home/lreali1/scr16_berti/luca/stm_cbc_runs/${POP}s/networks \
    --snr_th 10.0