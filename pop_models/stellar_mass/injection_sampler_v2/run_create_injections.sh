source /home/divyajyoti/miniconda3/bin/activate
conda activate spe_ngloc_mygwf_mypycbc

OUT_DIR="./"
OUT_FILE_PREFIX="pop"
EOS_DIR="/home/divyajyoti/ACADEMIC/Projects/CE_STM/CBC/data/eos_dir"
CONFIG_FILE="/home/divyajyoti/ACADEMIC/Projects/CE_STM/CBC/CE_STM_CBC/injection_sampler/config_pop_v2.ini"

python create_injections.py \
    --config-file "$CONFIG_FILE" \
    --out-dir "$OUT_DIR" \
    --out-file-prefix "$OUT_FILE_PREFIX" \
    --eos-dir "$EOS_DIR" \
    --reference-frequency 5.0 \
    --max-NS-mass 2.06 \
    --save-config
