# Cosmic Explorer Sceince Trace-ability matrix studies 
Git repository for Cosmic Explorer Science Trace-ability matrix studies

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
| **Mass model** | Piecewise power law with Gaussian peaks, smoothed mass cutoffs, and notch filters modelling mass gaps | Piecewise power law with Gaussian peaks, smoothed mass cutoffs modelling mass gaps | Piecewise power law with Gaussian peaks, smoothed mass cutoffs, and notch filters modelling mass gaps | Truncated power law for primary mass (max ~25,000 M☉, min 100 M☉); power law for mass ratio with ±1 | Monolithic mass and redshift | Kink–Y model for mass and redshift |
| **Main data products from Stage 2** | Fisher errors on all parameters including mass, spins, redshift, and luminosity distance | Fisher errors on effective tidal parameters, mass, spin, and luminosity distance | Fisher errors on effective tidal parameters, mass, spin, and luminosity distance | Fisher errors on all parameters including mass, spins, redshift, and luminosity distance | Fisher errors on all parameters including mass, spins, redshift, and luminosity distance | Fisher errors on all parameters including mass, spins, redshift, and luminosity distance |

