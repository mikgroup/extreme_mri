# Extreme MRI

This repo contains scripts to reproduce some experiments in [Extreme MRI](https://arxiv.org/abs/1909.13482). It uses the Python package [sigpy](https://github.com/mikgroup/sigpy).

# Colab Demo

To start, we recommend trying the demo notebook in Google Colab: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mikgroup/extreme_mri/blob/master/colab-demo.ipynb).

# Data

An example DCE dataset can be found on Zenodo (This is the second DCE dataset in paper):
https://zenodo.org/record/3647820

For more description about the variable names, please see the `Variables` section below.

# Running locally on command line

## Installation

Install `sigpy` using pip:

	pip install sigpy
	
GPU is recommended for running extreme MRI reconstruction. To do so, you will need to install [cupy](https://cupy.chainer.org), a numpy-like package for CUDA, either through conda or pip:

	pip install cupy
	

## Variables

- `ksp`: kspace data array of shape [# of channels, # of TRs, readout lengths].
- `coord`: kspace coordinate array of shape [# of TRs, readout lengths, # of dimensions].
- `dcf`: density compensation factor of shape [# of TRs, readout lengths].
- `mps`: sensitivity maps of shape [# of channels, nx, ny, nz]
- `img`: reconstructed image of shape [# of frames, nx, ny, nz]


## Processing pipeline

To run the reconstruction, the general pipeline is to:

- run setup script to the folder containing numpy arrays (`ksp.npy`, `coord.npy`, `dcf.npy`).
- automatically select FOV to account for leakage from slab selection.
- perform gridding reconstruction to look at image.
- estimate sensitivity maps using JSENSE.
- running the low rank reconstruction.


## Example Usages

Auto FOV and estimate sensitivity maps

	source setup.sh path/to/folder/
	python autofov.py $ksp $coord $dcf --device 0
	python jsense_recon.py $ksp $coord $dcf $mps --device 0
	
Estimate respiratory signal and soft-gating weights with TR of 7.7 ms

	python estimate_resp.py $ksp 0.0077 $resp
	python soft_gating_weights.py $resp $sgw
	
Low rank reconstruction with 500 frames

	python multi_scale_low_rank_recon.py $ksp $coord $dcf $mps 500 $img --device 0
	
Low rank reconstruction with 20 frames and soft-gating weights

	python multi_scale_low_rank_recon.py $ksp $coord $dcf $mps 20 $img --device 0 --sgw_file $sgw

Motion resolved reconstruction with 5 bin

	python motion_resolved_recon.py $ksp $coord $dcf $mps $resp 5 $mrimg --device 0
