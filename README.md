# Cosmic Explorer Sceince Trace-ability matrix studies 

## Initial Population Data Details: Where do you find the correct data?

| Population Type | Location | Comments |
|-------------------------|---------------------------|---------------------------|
| Stellar mass BHs, NSBH, BNS | steall mass BBHs, BNSs, NSBHs [here](https://github.com/KrishnenduGit/CE_STM_CBC/tree/main/pop_models/full_stellar_mass_pop) |  |
| popIII | [pop3](https://github.com/KrishnenduGit/CE_STM_CBC/tree/main/pop_models/pop3) | |
| pBH | [pbh](https://github.com/KrishnenduGit/CE_STM_CBC/tree/main/pop_models/pbh) | |
| IMBBHs | [imbhs](https://github.com/KrishnenduGit/CE_STM_CBC/tree/main/pop_models/imbhs) | |


## Network Details

| All | Two of the following | Extreme Cases |
|-------------------------|---------------------------|---------------------------|
| 1 (CE40hA+, CE20hA+, LI) | 5a (CE40hA+, ET-D, LI) | 7   (CE40hA+, CE20hA+, ET-D)
| 2 (CE40hAL, CE20hAL, LI) | 5b (CE40hAL, ET-D, LI) | 8a  (CE40hA+, ET-D) 
| 3 (CE40𝓵A+, CE20𝓵A+, LI) | 6a (CE40𝓵A+, ET-D, LI) | 8b  (CE40hA+, ET-2Ls) 
| 4 (CE40𝓵AL, CE20𝓵AL, LI) | 6b (CE40𝓵A+, ET-D, LI) | 9ab (LH, LL, LI) at A+ and A# sensitivity

- CE40 and CE20 denote Cosmic Explorer detectors with 40Km and 20Km arm length, respectively. h/𝓵 refer to the circulating laser power (high 1.5 MW, low 1.0 MW). A+/AL denote different coating choices. The detectors are located at the same fiducial sites as the MPSAC studies. We consider low-frequency cutoffs of 7, 10 and 15Hz.
- ET-D refers to Einstein Telescope in its triangular configuration, located in the Mediterranean. ET-2Ls refers to Einstein Telescope in its 2L configuration, with detectors located in Sardinia and Meuse-Rhine. We consider a low-frequency cutoff of 5Hz.
- LH, LL, and LI denote LIGO Hanford, LIGO Livingston and LIGO-India detectors. LI is always considered at A+ sensitivity, except for the 9ab networks where we compare A+ and A# sensitivities for all three detectors. The low-frequency cutoff is 10Hz. 

## Location for CE and ET
- [MPSAC Document](https://dcc.cosmicexplorer.org/public/0163/T2300003/002/CE_Detectors_for_MPSAC.pdf)
- [COBA document](https://cds.cern.ch/record/2855729/files/2303.15923.pdf)
- ET triangle Sardinia, CE at Texas and Washington state
- Last column is the x azimuth then add 90 to get the y-arm azimuth

```
name = 'CEA'
power_spectral_density = PowerSpectralDensity(psd_file='T2000017-v6_cosmic_explorer_strain.txt')
minimum_frequency = 5
maximum_frequency = 2048
length = 40
latitude = 46
longitude = -125
elevation = 142.554
xarm_azimuth = 260.0
yarm_azimuth = 350.0
# xarm_tilt = -6.195e-4
# yarm_tilt = 1.25e-5

```

and 

```
# MPSAC
name = 'CEB'
power_spectral_density = PowerSpectralDensity(psd_file='T2000017-v6_cosmic_explorer_20km_strain.txt')
minimum_frequency = 5
maximum_frequency = 2048
length = 20
latitude = 29
longitude = -94
elevation = 142.554
xarm_azimuth = 200.0
yarm_azimuth = 290.0
# xarm_tilt = -6.195e-4
# yarm_tilt = 1.25e-5
```

finally, ET

```
# Proposed ET at Serdinia, Italy, coordinates from GWBench and converted to degrees
name = 'ETS1'
power_spectral_density = PowerSpectralDensity(psd_file='18213_ET10kmcolumns_first2cols.txt')
length = 10
minimum_frequency = 5 # changing to 5 from 10
maximum_frequency = 2048
latitude = 40 + 31./60 # conversions for second and milli sec
longitude = 9 +  25./60
elevation = 51.884
xarm_azimuth = 90.0
yarm_azimuth = 180.0
shape = 'Triangle'
``` 

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
| **Redshift model** | Madau-Dickinson with inverse time delay | Madau-Dickinson with inverse time delay | Madau-Dickinson with inverse time delay | Madau-Dickinson, with alpha_z=2.7, beta_z=2.9, zp=1.9 in the notation of the MPSAC paper. Local rate R0=1 Gpc^-3 yr^-1| Santoliquido redshift | Kink–Y model for redshift |
| **Main data products from Stage 2** | Fisher errors on all parameters including mass, spins, redshift, and luminosity distance | Fisher error on effective tidal deformability parameters, mass ratio and total mass <br> Fisher error on the sky localisation | Fisher error on effective tidal deformability parameters, mass ratio and total mass <br> Fisher error on the sky localisation and redshift | Fisher errors on all parameters including mass, spins, redshift, and luminosity distance | Fisher errors on all parameters including mass, spins, redshift, and luminosity distance | Fisher errors on all parameters including mass, spins, redshift, and luminosity distance |
| **Main data products from Stage 3** | Population hyper PE|Population hyper PE|Population hyper PE|Population hyper PE|Population hyper PE|Population hyper PE|
| **Main data products from Stage 4** | |||||

