# Optical Turbulence

Library with optical turbulence expressions implemented for satellite-ground/slanted atmopsheric propagation.


## Required Libraries

```
colorama
matplotlib
mpmath
numpy
scipy
seaborn
```

## List of Expressions Available

> For a more thorough list of expressions with the mathematical expressions implemented go to [docs.](docs/parameters_expressions.md)

- Scintillation index downlink and uplink for flat and round earth
- Rytov variance downlink and uplink for flat and round earth
- OpticalBeam class using Andrews diffractive and refractive parameters
- Refractive-index structure altitude models
- RMS windspeed for the RIS models
- Statistical models for downlink and uplink
    - Exponentiated Weibull
    - Lognormal
    - Gamma
    - GammaGamma
    - ModulatedGamma (WIP)

## Installation

Currently, the only way to install the library is by downloading it directly from the repository or using an installer such as uv (https://docs.astral.sh/uv/) or pip.

### UV Installation example
Optical turbulence can be installed using uv:
```cli
uv add git+https://github.com/ak-aurora/optical_turbulence --branch main
```

### PIP Installation example
Optical turbulence can be installed using pip:
```cli
pip install git+https://github.com/ak-aurora/optical_turbulence
```

## Information Sources

Every function should have its own reference; nonetheless, most of the expressions can be found in one of the following references

- [1] L. C. Andrews and M. Beason, **Laser beam propagation in random media: new and advanced topics**. Bellingham, Washington, USA: SPIE Press, 2023.
- [2] L. C. Andrews, **Field Guide to Atmospheric Optics**, Second Edition. in Field Guide. Bellingham: Society of Photo-Optical Instrumentation Engineers, 2019.
- [3] L. C. Andrews, **Laser Beam Propagation Through Random Media**, 2nd ed. in Press Monograph. Bellingham: SPIE, 2005.