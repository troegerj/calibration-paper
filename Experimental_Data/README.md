# Experimental data for model calibration

This directory comprises the experimental data for reproducing the results reported in the [related scientific publication](#related-scientific-publication).

This comprises an **artificially noised full-field displacement dataset of a linear elastic two-dimensional plate with a hole**. The data is generated using forward FEM simulations. Then, the displacements are superimposed with Gaussian noise and linearly interpolated onto a spatial grid. Since the exact material parameters are known, the data are employed for **re-identification** purposes to test the noise sensitivity of different methods.

The second dataset comprises five repetitions of **uniaxial tensile tests on the steel** alloy *TS275* using common dog-bone like tensile specimens. We provide the **axial and lateral strain**, which were measured with optical measurement techniques. Moreover, the **axial stress** is given to calibrate elasto-plastic constitutive models based on this dataset.

## Citing the data

If you use this research data, please cite the [related scientific publication](#related-scientific-publication) as specified below.

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