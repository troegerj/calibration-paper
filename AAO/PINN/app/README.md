# Inverse physics-informed neural networks for model calibration

This repository provides the research code for the calibration of a linear elastic constitutive model from full-field displacement data using inverse physics-informed neural networks.

This code is supposed to be executed in a [*Singularity container*](https://sylabs.io). You can find the [installation instructions](#installation) below.


## Related scientific publication

[*"Reduced and All-at-Once Approaches for Model Calibration and Discovery in Computational Solid Mechanics"*](https://arxiv.org/abs/2404.16980)

**Citing the publication**:

    @article{roemer_modelCalibration_2024,
        title={Reduced and all-at-once approaches for model calibration and discovery in computational solid mechanics},
        author={Römer, Ulrich and Hartmann, Stefan and Tröger, Jendrik-Alexander and Anton, David and Wessels, Henning and Flaschel, Moritz and De Lorenzis, Laura},
        year={2024},
        journal={arXiv preprint},
        doi={https://doi.org/10.48550/arXiv.2404.16980}
    }

The **results in this publication can be reproduced** with the following script, which can be found at the top level of the *app* directory (see file structure below):
- *inverse_pinn_2D_linearelasticity_withnoise_2e-04.py* 
- *inverse_pinn_2D_linearelasticity_withnoise_4e-04.py* 
- *inverse_pinn_2D_linearelasticity_withoutnoise.py* 

> [!NOTE]
> The three simulations differ only in the level of noise in the full-field displacement data.



## Installation


1. For strict separation of input/output data and the source code, the project requires the following file structure:

Repository \
├── AAO \
&nbsp;&nbsp;&nbsp;&nbsp;├── PINN \
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├── app \
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├── input \
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;└── output \
&nbsp;&nbsp;&nbsp;&nbsp;└── ... \
└── ...

> [!IMPORTANT]
> If the output directory (see file structure above) does not already exist, it must be created manually before the simulation is started.

2. Clone the repository into the *Repository* directory via:

        git clone https://github.com/troegerj/calibration-paper.git .

3. Install the software dependencies. This code is supposed to be executed in a [*Singularity container*](#singularity). In addition, due to the high computational costs, we recommend running the simulations on a GPU. 

4. Run the code.


### Singularity

You can find the singularity definition file in the *.devcontainer* directory. To build the image, navigate to your `<repository directory>/AAO/PINN` (see directory tree above) and run:

    singularity build calibrationpaper.sif app/.devcontainer/container.def

Once the image is built, you can run the scripts via:

    singularity run --nv calibrationpaper.sif python3 <full-path-to-script>/<script-name>

Please replace `<full-path-to-script>` and `<script-name>` in the above command according to your file structure and the script you want to execute.

> [!IMPORTANT]
> You may have to use the *fakreroot* option of singularity if you do not have root rights on your system. In this case, you can try building the image by running the command `singularity build --fakeroot calibrationpaper.sif app/.devcontainer/container.def`. However, the fakeroot option must be enabled by your system administrator. For further information, please refer to the [Singularity documentation](https://sylabs.io/docs/).



## Citing the code


If you use this research code, please cite the [related scientific publication](#related-scientic-publication) as specified above.