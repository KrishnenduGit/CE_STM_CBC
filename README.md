# Cosmic Explorer Sceince Trace-ability matrix studies 
## Network Details

| Detector Configurations | Lower cut-off frequencies | Other |
|-------------------------|---------------------------|---------------------------|
| ET triangle 10 km + CE 20 km [1.5 MW PSDs Aplus coating] | Flow = 15 Hz, 10 Hz, 5 Hz |  
| ET triangle 10 km + CE 40 km [1.5 MW PSDs Aplus coating] |  Flow = 15 Hz, 10 Hz, 5 Hz |  
| LIO Asharp + two 3G [CE40 + ET triangle] | Flow = 15 Hz, 10 Hz, 5 Hz |  
| LIO Asharp + two 3G [CE40 + CE20] | Flow = 15 Hz, 10 Hz, 5 Hz | 
| HLI at Asharp sensitivity | Flow = 15 Hz, 10 Hz, 5 Hz | 

## Location for CE and ET
- [MPSAC Document](https://dcc.cosmicexplorer.org/public/0163/T2300003/002/CE_Detectors_for_MPSAC.pdf)
- [COBA document](https://cds.cern.ch/record/2855729/files/2303.15923.pdf)
- ET triangle Sardinia, CE at Texas and Washington state
- Last column is the x azimuth then add 90 to get the y-arm azimuth
- Updating BlueBEAR `/rds/projects/n/naderivk-siqm-test/venv/lib/python3.10/site-packages/bilby/gw/detector/`

![Screenshot_2025-05-02_at_09.06.56](/uploads/e9e0f715d256dc290a98bdb5729b451b/Screenshot_2025-05-02_at_09.06.56.png)
# PSD Details

- From [this page](https://apps.et-gw.eu/tds/?content=3&r=18213), [this page](https://github.com/cosmic-explorer/mpsac_detector_networks) and [Bilby](https://git.ligo.org/lscsoft/bilby/-/tree/master/bilby/gw/detector/noise_curves?ref_type=heads)

![ET_MPSAC_COBA](/uploads/7bef53bf08a9a3176aed75aa9a919ca4/ET_MPSAC_COBA.png)![ET_Bilby_Comp](/uploads/7fa481ebf16c5be9b5303d216704757e/ET_Bilby_Comp.png)![CE_PSDs](/uploads/bdc8410865442d3ee43d9c73e1627215/CE_PSDs.png)


- In the ET file(s), the two columns correspond to high-frequency (HF) and low frequency (LF) respectively, but we need the combined one. 

## Population Analysis Workflow & Data Products

| **Stage 1** | Type of the source | **Stage 2** | **Stage 3** | **Stage 4** |
|------------|------------|------------|------------|------------|
| **Initial population data assuming mass, spin and redshift models** | Stellar mass BHs | **Fisher errors:**<br>different network configurations,<br>lower cut-off frequencies | Population analysis: Stellar mass BHs | **Population hyperparameter estimates and implications:**<br>Redshift evolution of current and early Universe BH populations<br>Star formation rate density estimate<br>Merger rate density estimates<br>Etc. |
| **Initial population data assuming mass, spin and redshift models** | BNS | **Fisher errors:**<br>different network configurations,<br>lower cut-off frequencies | Population analysis: BNS | **Population hyperparameter estimates and implications** <br>NS maximum mass estimates and distinguishing NS population from BHs<br>Etc.|
| **Initial population data assuming mass, spin and redshift models** | NSBH | **Fisher errors:**<br>different network configurations,<br>lower cut-off frequencies | Population analysis: NSBH | **Population hyperparameter estimates and implications** <br>NS maximum mass estimates and distinguishing NS population from BHsStar formation rate density estimate<br>Merger rate density estimates<br>Etc.|
| **Initial population data assuming mass, spin and redshift models** | IMBBHS | **Fisher errors:**<br>different network configurations,<br>lower cut-off frequencies | Population analysis: IMBBHS | **Population hyperparameter estimates and implications** <br>Redshift evolution of current and early Universe BH populations<br>Star formation rate density estimate<br>Merger rate density estimates<br>Etc. |
| **Initial population data assuming mass, spin and redshift models** | popIII | **Fisher errors:**<br>different network configurations,<br>lower cut-off frequencies | Population analysis: popIII | **Population hyperparameter estimates and implications** <br>Redshift evolution of current and early Universe BH populations<br>Star formation rate density estimate<br>Merger rate density estimates<br>Etc. |
| **Initial population data assuming mass, spin and redshift models** | pBHs | **Fisher errors:**<br>different network configurations,<br>lower cut-off frequencies | Population analysis: pBHs | **Population hyperparameter estimates and implications** <br>NS maximum mass<br>Redshift evolution of current and early Universe BH populations<br>Star formation rate density estimate<br>Merger rate density estimates<br>Etc.|

> **Caption:**  
> Schematic overview of population modeling, parameter estimation, and astrophysical implications.  
> The CBC group performs the analyses in five stages and produces data products that can be shared with other relevant groups, including MMA, Cosmology, EoS, and TGR.


## Source Classes and Population Models

| **Source class** | **Stellar mass BHs** | **BNS** | **NSBH** | **IMBBHs** | **popIII** | **Primordial BHs** |
|------------------|---------------------|---------|----------|------------|------------|--------------------|
| **Spin model** | Isotropically oriented and uniform magnitude [0,1] | Non-spin | BH spin is Gaussian with mean and dispersion ~ [0.1, 0.1] | Isotropically oriented and uniform magnitude [0,1] | Isotropically oriented and uniform magnitude [0,1] | Isotropically oriented and uniform magnitude [0,1] |
| **Mass model** | Piecewise power law with Gaussian peaks, smoothed mass cutoffs, and notch filters modelling mass gaps | Piecewise power law with Gaussian peaks, smoothed mass cutoffs modelling mass gaps | Piecewise power law with Gaussian peaks, smoothed mass cutoffs, and notch filters modelling mass gaps | Truncated power law for primary mass (max ~25,000 M☉, min 100 M☉); power law for mass ratio with ±1 | Santoliquido mass and redshift | Kink–Y model for mass and redshift |
| **Main data products from Stage 2** | Fisher errors on all parameters including mass, spins, redshift, and luminosity distance | Fisher error on effective tidal deformability parameters, mass ratio and total mass <br> Fisher error on the sky localisation | Fisher error on effective tidal deformability parameters, mass ratio and total mass <br> Fisher error on the sky localisation and redshift | Fisher errors on all parameters including mass, spins, redshift, and luminosity distance | Fisher errors on all parameters including mass, spins, redshift, and luminosity distance | Fisher errors on all parameters including mass, spins, redshift, and luminosity distance |
| **Main data products from Stage 3** | Population hyper PE|Population hyper PE|Population hyper PE|Population hyper PE|Population hyper PE|Population hyper PE|
| **Main data products from Stage 4** | |||||

