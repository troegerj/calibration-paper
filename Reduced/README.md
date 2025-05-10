# Reduced approaches for model calibration

This repository provides the research code for the so-called reduced approaches:
- NLS-FEM (nonlinear least-squares and finite elements)
- Bayes-FEM (Bayesian inference and finite elements)
- NLS-PINN (nonlinear least-squares and parametric physics-informed neural networks)
- Bayes-PINN (Bayesian inference and parametric physics-informed neural networks)

All theoretical details regarding the different approaches are provided in the [related scientific publication](#related-scientific-publication). 

To **reproduce the reported results**, follow the installation details in the corresponding subdirectories. The provided codes comprise routines for deterministic and stochastic calibration using **nonlinear least-squares** and **Bayesian inference**. To evaluate the model, a **finite element code** is available for solving small and finite strain problems. Moreover, code for **parametric physics-informed neural networks** is provided as well, which could be used for model evaluation instead of the FEM code.

> [!NOTE] 
> The intention of the finite element code is to supply executable examples without many modifications. If you intent to use the code for calibration problems involving significantly non-linear geometrical or physical behavior, we highly recommend to use any of the available open source or commercial finite element codes for the model evaluations to perform the calibration within reasonable time.

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