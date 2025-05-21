# ChronoFlow

📄 **[Van-Lane et al. (2025)](https://ui.adsabs.harvard.edu/abs/2024arXiv241212244V/abstract)**

[![DOI](https://zenodo.org/badge/895726142.svg)](https://doi.org/10.5281/zenodo.15482094)

## Overview

ChronoFlow is a an empirical gyrochonology model that was developed using machine learning to characterize the rotational evolution of stars over the age and colour parameter spaces; specifically the de-reddened Gaia DR3 colour $(BP-RP)_0$. It also incorporate the effect of photometric uncertainty. We encourage you to read the full paper cited above to understand how the model was developed and tested, and its applicability to any use cases you may have.

Here we show ChronoFlow's ability to model the probability density of a population of stars in rotation period-colour space as a function of time:

<p align="center">
  <img width = "600" src="./figures/cf_animation.gif"/>
</p>


## Applications

ChronoFlow can be used to forward model the rotational evolution of stars given age, de-reddened Gaia DR3 colour $(BP-RP)_0$, and photometric uncertainty. It can also be used to estimate stellar ages given observed $(BP-RP)_0$ and rotation period. 

Refer to the ***tutorials*** folder for notebook examples that can be used to infer stellar ages and forward model rotational evolution with ChronoFlow. You can also refer to the `README.md` file in that folder for directions to set up the code in your local environment.

### Software dependencies

To use ChronoFlow, the following dependencies are required:

* [numpy](https://pypi.org/project/numpy/) (version `1.24.0` was used to develop ChronoFlow)
* [scipy](https://pypi.org/project/scipy/) (this is used to calculate posterior distributions; version `1.13.1` is used in these tutorials; only `scipy.special` is required)
* [PyTorch](https://pypi.org/project/torch/) (version `1.13.1 ` was used to develop ChronoFlow; only the `torch` component is required)
* [zuko](https://pypi.org/project/zuko/) (version `1.1.0` was used to develop ChronoFlow)

Please refer to each package for installation instructions (all are available using `pip`).

## Citation

If your work uses ChronoFlow, please cite [Van-Lane et al. (2025)](https://ui.adsabs.harvard.edu/abs/2024arXiv241212244V/abstract).

## Bug Reports and Questions

If you are having any issues with this code, feel free to open a Git Issue!