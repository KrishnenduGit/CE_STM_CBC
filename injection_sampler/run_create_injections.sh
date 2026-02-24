source /home/divyajyoti/miniconda3/bin/activate
conda activate spe_ngloc_mygwf_mypycbc

OUT_DIR="/home/divyajyoti/ACADEMIC/Projects/CE_STM/CBC/data/population_samples/pop1"
OUT_FILE_PREFIX="pop1"
CONFIG_FILE="/home/divyajyoti/ACADEMIC/Projects/CE_STM/CBC/CE_STM_CBC/injection_sampler/config_pop1.ini"

python create_injections.py \
    --config-file "$CONFIG_FILE" \
    --out-dir "$OUT_DIR" \
    --out-file-prefix pop1 \
    --reference-frequency 5.0 \
    --max-NS-mass 2.8 \
    --save-config