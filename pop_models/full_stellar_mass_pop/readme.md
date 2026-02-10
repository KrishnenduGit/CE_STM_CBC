## Sampling mass for FullPop-4.0 using GWForge
- [sample_masses_with_GWForge.ipynb](https://github.com/KrishnenduGit/CE_STM_CBC/blob/main/pop_models/full_stellar_mass_pop/sample_masses_with_GWForge.ipynb) can be used with this fork of GWForge https://github.com/divyajyoti09/gwforge/tree/main
    - A pull request has been created to merge this into the main GWForge branch
    - At the moment the sampler uses rejection sampling, but two 2D interpolation methods have also been added to GWForge. These can be implemented in the `population.mass` module of the package if needed.
- The default hyperparameters for FullPop-4.0 used in GWTC-4 populations paper are stored in []
    - These can be obtained using the script [write_fullpop_hyperpars_to_ini.ipynb](https://github.com/KrishnenduGit/CE_STM_CBC/blob/main/pop_models/full_stellar_mass_pop/write_fullpop_hyperpars_to_ini.ipynb). The data release h5 file which reads these parameters is available on this [Zenodo link](https://zenodo.org/records/16911563)
