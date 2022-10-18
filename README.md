# Task 1 of contest (modify minDiffusion) by Nikolaev O.V.

<!-- #region -->
<p align="center">
<img  src="contents/_ddpm_sample_19.png">
</p>

Goal of this educational repository is to provide a self-contained, minimalistic implementation of diffusion models using Pytorch.

Many implementations of diffusion models can be a bit overwhelming. Here, `superminddpm` : under 200 lines of code, fully self contained implementation of DDPM with Pytorch is a good starting point for anyone who wants to get started with Denoising Diffusion Models, without having to spend time on the details.

Simply:

```
$ python superminddpm.py
```
# ADDED:
# Getting started
1) Code is tested with python 3.9.12 and pythorch 1.12.1, with cuda 11.6 installed
2) NVIDIA RTX 2060 
Please install the necessary dependencies, pip is a recommended way to do this.
3) device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
4) cuda:0 вместо cuda:1


Above script is self-contained. (Of course, you need to have pytorch and torchvision installed. Latest version should suffice. We do not use any cutting edge features.)

If you want to use the bit more refactored code, that runs CIFAR10 dataset:

```
$ python train_cifar10.py
```

<!-- #region -->
<p align="center">
<img  src="contents/_ddpm_sample_cifar43.png">
</p>

Above result took about 2 hours of training on single 3090 GPU. Top 8 images are generated, bottom 8 are ground truth.

Here is another example, trained on 100 epochs (about 1.5 hours)

<p align="center">
<img  src="contents/_ddpm_sample_cifar100.png">
</p>

Currently has:

- [x] Tiny implementation of DDPM
- [x] MNIST, CIFAR dataset.
- [x] Simple unet structure. + Simple Time embeddings.
- [x] CelebA dataset.

TODOS

- [ ] DDIM
- [ ] Classifier Guidance
- [ ] Multimodality

# Updates!

- Using more parameter yields better result for MNIST.
- More comments in superminddpm.py
