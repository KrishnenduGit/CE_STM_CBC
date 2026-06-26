# Notes for forgetful Koustav

## 1. Environment setup

Assuming Conda is installed, create the environment with:

```bash
conda env create -f codes/environment.yaml
```

This creates a Conda environment named `cbc-stm` with the required plotting and analysis dependencies.

## 2. What is inside `codes/`

Main files:

- `codes/environment.yaml`
  - Conda environment definition for this plotting workflow.

- `codes/load.py`
  - Provides the `FisherResults` class to read Fisher `.h5` files.
  - Reads metadata (SNR threshold, total events, detected events, detector combo!!!).
  - `load(parameters=[...])` returns detected-event summaries with:
    - always-available: `snr`, `sky_area_90`
    - requested raw parameters (for example `chirp_mass`, `luminosity_distance`, `theta_jn`, etc.)
    - derived uncertainties (for example `delta_chirp_mass`, `delta_luminosity_distance_fractional`, etc.)
  - `load_event_parameters(...)` draws Fisher samples for one event from the Fisher covariance, supports:
    - tidal and non-tidal conversions
    - covariance repair check (optional)
    - physicality cuts (optional; but Koustav has set it to True!!!)
    - conversion to bilby-style derived parameters.

- `codes/plot_utils.py`
  - Shared plotting + formatting helpers.
  - Detector-label parsing utilities (`parse_detector`, `parse_combo`) for readable legend labels.
  - Label map (`GWlatex_labels`) for consistent GW axis labels.
  - Plot-style helper (`new_rcParams`) and color helper (`create_custom_colormap`).
  - `plot_corner(...)` wrapper around `corner.corner` for posterior corner plots with consistent labels and defaults.

## 3. Usage examples (from `check.ipynb`)

### 3.1 Import the modules

```python
import sys, os
sys.path.insert(0, os.path.abspath("codes"))

from load import FisherResults
from plot_utils import plot_corner
```

### 3.2 Load event-level summaries for CDF-style plots

```python
results = FisherResults(file_path=file_path).load(
    parameters=[
        "luminosity_distance",
        "chirp_mass",
        "theta_jn",
        "delta_luminosity_distance_fractional",
        "delta_chirp_mass_fractional",
        "delta_theta_jn",
    ]
)
```

Then in the notebook, these loaded arrays are used with `plot_1cdf(...)` to compare detector networks for:
- `snr`
- `sky_area_90`
- distance/mass fractional errors
- inclination error (`delta_theta_jn`)

### 3.3 Draw posterior samples for a single event + make a corner plot

```python
fisher_results = FisherResults(file_path=file_path)
posterior_samples = fisher_results.load_event_parameters(
    event_index,
    number_of_samples,
    rng=250114,
    npool=10,
    tides=True,
    enforce_physicality=True,
)

figure = plot_corner(
    posterior_samples,
    save=False,
    parameters=[
        "chirp_mass",
        "mass_ratio",
        "luminosity_distance",
        "chi_eff",
        "lambda_tilde",
        "delta_lambda_tilde",
    ],
)
```

## 4. Quick reminders

- Use `sys.path.insert(... "codes")` in notebooks/scripts run from `plots/`.
- `FisherResults.load(...)` is for population/event-summary comparisons.
- `FisherResults.load_event_parameters(...)` is for single-event posterior sampling.
- `plot_corner(...)` is the fastest way to get a publication-style corner plot from those samples.


## Main note:  👷🏻

- 🚧 This is heavily under development. So mistakes are expected!