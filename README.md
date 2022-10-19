# Task 1 - Adapt diffusion model for map generation.

<!-- #region -->
<p align="center">
<img  src="contents/_ddpm_sample_19.png">
</p>

The task was performed within the contest for the position of ML-engenner at Skoltech Applied AI Center.
Goal of the project was to adapt existing diffusion DL model (https://github.com/cloneofsimo/minDiffusion) for map generation purposes. 

Scoring is shown below:
- Model should generate pictures similar to te real Earth maps with dimmention of (64x64) (10 point);
- Dataset preparation quality will be reviewed (5 points);
- Modifications of the existing code should be shown.

# Introduction

Diffusion models became very promising nowadays, as being recently introduced it is already shown perfect results. 
The core concept behind these models is deceptively simple:

- Take a starting image;
- Add some noise, iteratively degrading the image until almost nothing but noise remains;
- Train a model to 'undo' these noise steps;
- To generate, start from pure noise and repeatedly apply the model to 'denoise' our way back to a plausible image.

Being alternative to GANs, today's Diffusion models show: 

- [x] **High-quality sampling**

- [x] **Mode coverage and sample diversity**

- [ ] **Slow and computationally expensive sampling**

# Getting started
Current DL code was tested localy with NVIDIA RTX 2060 mobile (cuda 11.6) and python 3.9.12. 

It presents refactored initial repo with adaptation for map generation

Requiremnet can be found at:

```
$ requirements.txt
```

Map dataset was taken from Awesome Satellite Imagery Datasets (https://github.com/chrieke/awesome-satellite-imagery-datasets).
This dataset is a composition of scenes taken by SPOT sensor in 2005 over four counties in Brazil. dataset includes scenes with different plant ages and/or with spectral distortions caused by shadows. There is a total of four orthoimages, one for each county, and also four mask images with pixel level annotation of coffee/non-coffee class.

# ADDED:







<!-- #region -->
<p align="center">
<img  src="contents/_ddpm_sample_cifar43.png">
</p>

Above result took about 2 hours of training on single 3090 GPU. Top 8 images are generated, bottom 8 are ground truth.

Here is another example, trained on 100 epochs (about 1.5 hours)

<p align="center">
<img  src="contents/_ddpm_sample_cifar100.png">
</p>



# Updates of the existing repo

- requirements.txt was added, as well as short intro for position of diffusion models in today's DL
- some typos as cuda:1 exchenged to cuda:0, 
- some additional helpfull annotations were added to the code
- compositon of the classes and functions refactored, the redundant ones have been removed
- loss functions were testd, NAdam
- map pictures for dataset loader framework was added

WHAT CAN BE CHANGED ELSE

- lear variance, not only mean of gausian noise
- class traking (conditional sampling)
- β sheduler can be more complex (e.g. cosine sheduler)
