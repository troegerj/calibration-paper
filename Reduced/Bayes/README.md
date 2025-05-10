# Bayesian inference for model calibration

This repository provides the research code for model calibration using Bayesian inference and Markov chain Monte Carlo (MCMC) computations.

Two examples are provided:
- Calibration of linear elasticity from full-field displacement data using synthetic experimental data 
- Calibration of elasto-plasticity using experimental stress and lateral strain data obtained from uniaxial tensile tests

> [!NOTE] 
> The experimental datasets are stored in the top-level directory [*Experimental_Data*](https://github.com/troegerj/calibration-paper/tree/main/Experimental_Data).

The MCMC-calibration is carried out using a MATLAB-implementation of the affine-invariant ensemble sampler *emcee* by [Goodman and Weare (2010)](https://msp.org/camcos/2010/5-1/p04.xhtml). The sampler has only one free parameter (the stretch scale). All calibration settings can be found at the top of the main-routines *calib_cube.m* and *calib_plate.m*.

## Citing the code

If you use this research code, please cite the [related scientific publication](#related-scientific-publication) as specified below.

## Related scientific publication

[*"Reduced and All-at-Once Approaches for Model Calibration and Discovery in Computational Solid Mechanics"*](https://doi.org/10.1115/1.4066118)

```
@article{roemer_modelCalibration_2024,
    title={Reduced and all-at-once approaches for model calibration and discovery in computational solid mechanics},
    author={Römer, Ulrich and Hartmann, Stefan and Tröger, Jendrik-Alexander and Anton, David and Wessels, Henning and Flaschel, Moritz and De Lorenzis, Laura},
    year={2024},
    journal={Applied Mechanics Reviews},
    volume={77},
    number={04},
    pages={040801},
    doi={https://doi.org/10.1115/1.4066118}
}
```