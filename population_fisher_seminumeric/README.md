# Population Fisher forecasts

Running the scripts is pretty straightforward.

For example, if you just want to get Fisher errors for the Madau--Dickinson population, you can start with:

```python
python redshift_forecast.py --help
```

The script expects a Fisher results file, which you can provide with:

```python
python validate_pop_fisher.py --fisher-file network_bbh_CE40km_1p5MW_Aplus_coat_5.0hz_CE20km_1p5MW_Aplus_coat_5.0hz_ETD_5.0hz.h5
```

This spits out the relevant numbers and generates some nice plots. This calculation includes all terms in the Population Fisher matrix and also calculates the relative contribution of each term.

Similarly, if you are interested in a mass + redshift forecast, simply run:

```python
python forecast_mass_redshift.py --fisher-file network_bbh_CE40km_1p5MW_Aplus_coat_5.0hz_CE20km_1p5MW_Aplus_coat_5.0hz_ETD_5.0hz.h5
```

This one only calculates the first Fisher term and ignores all higher-order corrections. So, by construction, it is equivalent to Sylvia's Bayesian runs with an injected value 🤷

The generated results are saved in an output directory, which by default is:

```text
forecast_mass_redshift_output/
```

Again, beautiful plots come for free!

Finally, for the spectral siren forecast, you can do the same thing:

```python
python spectral_sirens_forecast.py --fisher-file network_bbh_CE40km_1p5MW_Aplus_coat_5.0hz_CE20km_1p5MW_Aplus_coat_5.0hz_ETD_5.0hz.h5
```
There is also another directory created for you with beautiful plots and results.


You will need a few packages here and there, but overall it is fast and largely good!!!
