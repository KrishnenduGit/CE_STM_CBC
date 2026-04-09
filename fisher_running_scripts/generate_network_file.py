import json

fname = 'CE40CE20ET.json'

my_network = {
    "CE40": {
        "lat": 46.,               # CE40 Latitude
        "long": -125.,            # CE40 Longitude
        "xax": 145.,              # CE40 Orientation
        "shape": "L",             # L-shaped
        "psd_path": "/home/lreali1/scr16_berti/luca/stm_cbc_runs/psds/CE40km_1p5MW_Aplus_coat_strain.txt"
    },
    "CE20": {
        "lat": 29.,               # CE20 Latitude
        "long": -94.,             # CE20 Longitude
        "xax": 205.,              # CE20 Orientation
        "shape": "L",             # L-shaped
        "psd_path": "/home/lreali1/scr16_berti/luca/stm_cbc_runs/psds/CE20km_1p5MW_Aplus_coat_strain.txt"
    },
    "ET": {
        "lat": 43.722376,       # ET Latitude
        "long": 9.475483,       # ET Longitude
        "xax": 0.,              # ET Orientation
        "shape": "T",           # Triangular shape (3 nested detectors)
        "psd_path": "/home/lreali1/scr16_berti/luca/stm_cbc_runs/psds/ET_cryo_10km_asd.txt"
    }
}

with open(fname, 'w') as fp:
    json.dump(my_network, fp)