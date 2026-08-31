# Population Fisher forecasts

Hyperparameter (population) Fisher matrices for CE/ET catalogues, following
Gair et al. 2022 ([arXiv:2205.07893](https://arxiv.org/abs/2205.07893)) Eq. 21.

The Term-I score is always the model's closed-form `analytic_score` (no
finite-difference fallback).  Finite differences remain only where no closed
form exists: the `Om`/`w0` columns of the spectral-siren score, and the
Terms II-V machinery in `pop_fisher_higher_order.py`.

Every script takes `--fisher-file`, defaulting to the CE40+CE20+ET catalogue in
this directory, and `--help` works everywhere.

## Forecasts

```bash
# Madau-Dickinson redshift only.  All five Gair terms, and it reports how much
# each one contributes.  Writes its figures to the current directory.
python redshift_forecast.py

# Mass + redshift.  Term I only, so it is directly comparable to a Bayesian run
# with an injected value.  -> forecast_mass_redshift_output/
python forecast_mass_redshift.py

# Spectral sirens: population + (H0, Om, w0) in the detector frame, plus a
# measurement-error bracket on sigma(H0).  -> spectral_sirens_output/
python spectral_sirens_forecast.py
```

`--output-directory`, `--snr-thresholds`, `--n-samples` and `--seed` are
available on the latter two.

## Checks

```bash
python validate_gwtc5_model.py      # the mass model against GWTC-5 B10-B14
python validate_analytic_scores.py  # analytic scores against the validators' own FD reference
```

## The mass model, and one convention that matters

`BrokenPowerLawPlusTwoPeaksMass` implements GWTC-5
([arXiv:2605.27226](https://arxiv.org/abs/2605.27226)) Eqs. B10-B14: a broken
power law plus two left-truncated Gaussians, with the Planck taper applied to
both component masses.
