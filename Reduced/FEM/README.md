# Finite element code for model calibration

This repository provides the finite element code, which is used for the deterministic and stochastic model calibration examples.

The code is applicable to two- and three-dimensional small and finite strain problems of computational solid mechanics. Currently, the implemented constitutive models cover **linear elasticity**, **weakly compressible hyperelasticity (Neo-Hooke, Isihara, Haines-Wilson)** and **elasto-viscoplasticity**. The required *inputfiles* for the examples in the [related scientific publication](#related-scientific-publication) are provided in the respective subdirectory *inputfiles*. Since the code is used in a model calibration context, a string denoting the inputfile and the material parameters are required as inputs.

> [!NOTE] 
> The intention of the finite element code is to supply executable examples without many modifications. If you intent to use the code for calibration problems involving significantly non-linear geometrical or physical behavior, we highly recommend to use any of the available open source or commercial finite element codes for the model evaluations to perform the calibration within reasonable time.

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


