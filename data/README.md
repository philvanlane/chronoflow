```prot_catalogue.csv``` is a csv copy of the rotator catalogue that was presented in Van-Lane et al. ([2025](https://ui.adsabs.harvard.edu/abs/2024arXiv241212244V/abstract)).

Note that we include stars here which were not included in the analysis described in Van-Lane et al. ([2025](https://ui.adsabs.harvard.edu/abs/2024arXiv241212244V/abstract)). To capture only the subset that was used in that analysis (and to train ChronoFlow), the following filters should be applied:

* ```cmd_exclude = 0```; this removes post-MS stars that we visually identified using CMDs.
* ```dered_exclude = 0```; this removes stars that we were not able to get extinction measurements for.
* ```phot_exclude = 0```; this removes stars that have incomplete Gaia DR3 photometry.
* ```dr3_ruwe < 1.4 ```; this removes stars that we excluded due to possible binarity.>