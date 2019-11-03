# Extreme MRI

This repo contains scripts to perform extreme MRI reconstruction. It uses the python package [sigpy](https://github.com/mikgroup/sigpy)


## Installation

Install `sigpy` using pip:

	pip install sigpy
	
To use GPUs, you will need to install `cupy`, a numpy-like package for CUDA, either through conda or pip:

	pip install cupy
	

## Variables

- `ksp`: kspace data array of shape [# of channels, # of TRs, readout lengths].
- `coord`: kspace coordinate array of shape [# of TRs, readout lengths, # of dimensions].
- `dcf`: density compensation factor of shape [# of TRs, readout lengths].
- `mps`: sensitivity maps of shape [# of channels, nx, ny, ...]
- `img`: reconstructed image of shape [# of frames, nx, ny, ...]


## Processing pipeline

To run the reconstruction, the general pipeline is to:

- run setup script to folder with appropriate numpy arrays (`ksp.npy`, `coord.npy`, `dcf.npy`). (`setup.sh`).
- automatically select FOV to account for leakage from slab selection (`autofov.py`).
- perform gridding reconstruction to look at image (`gridding.py`).
- estimate sensitivity maps using JSENSE (`jsense_recon.py`).
- running the low rank reconstruction (`low_rank_recon.py`).


## Example Usages

	source setup.sh path/to/numpy/arrays
	python autofov.py $ksp $coord $dcf --device 0
	python gridding_recon.py $ksp $coord $dcf $grd --device 0
	python jsense_recon.py $ksp $coord $dcf $mps --device 0
	python low_rank_recon.py $ksp $coord $dcf $mps 30 $img --device 0
