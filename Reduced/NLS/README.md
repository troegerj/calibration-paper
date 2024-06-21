# Nonlinear Least-Squares for model calibration

This repository provides the research code for model calibration using nonlinear least-squares and uncertainty quantification in the frequentist setting.

Routines for **uncertainty quantification** comprise uncertainty quantification in single-step calibrations using **Gaussian error propagation (first-order second moment method)** and **Monte Carlo method**. The uncertainty quantification in two-step calibrations using nonlinear least-squares is provided in the subdirectory *twoStepUQ*, see Appendix G of the [related scientific publication](#related-scientific-publication) for the theoretical details.

Three examples are provided:
- Calibration of linear elasticity from full-field displacement data using synthetic experimental data 
- Calibration of different hyperelasticity models from full-field displacement data using artificially noised data
- Calibration of elasto-plasticity using experimental stress and lateral strain data obtained from uniaxial tensile tests

> [!NOTE] 
> The experimental datasets are stored in the top-level directory [*Experimental_Data*](https://github.com/troegerj/calibration-paper/tree/main/Experimental_Data). The experimental data of the hyperelasticity example can be found in this [GitHub repository](https://github.com/EUCLID-code/EUCLID-hyperelasticity) and the links therein.

The nonlinear least-squares calibration is done using the MATLAB-implementation of trust-region reflective algorithm, which is a gradient-based solver for nonlinear optimization problems. The calibrations are carried out by using the main routines *calib_linElas_plate_nls.m*, *calib_hyperelas_plate_nls.m*, and *calib_plasticity_cube_nls.m*. All calibration settings can be set at the beginning of this main files.

## Citing the code

If you use this research code, please cite the [related scientific publication](#related-scientific-publication) as specified below.

## Related scientific publication

[*"Reduced and All-at-Once Approaches for Model Calibration and Discovery in Computational Solid Mechanics"*](https://arxiv.org/abs/2404.16980)

```
@article{roemer_modelCalibration_2024,
    title={Reduced and all-at-once approaches for model calibration and discovery in computational solid mechanics},
    author={Römer, Ulrich and Hartmann, Stefan and Tröger, Jendrik-Alexander and Anton, David and Wessels, Henning and Flaschel, Moritz and De Lorenzis, Laura},
    year={2024},
    journal={arXiv preprint},
    doi={https://doi.org/10.48550/arXiv.2404.16980}
}
```