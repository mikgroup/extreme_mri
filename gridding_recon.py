import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--num_frames', type=int, default=1)
parser.add_argument('--device', type=int, default=-1)
parser.add_argument('--multi_gpu', action='store_true')
parser.add_argument('--img_shape', nargs='+', type=int)
parser.add_argument('ksp_file', type=str)
parser.add_argument('coord_file', type=str)
parser.add_argument('dcf_file', type=str)
parser.add_argument('img_file', type=str)
args = parser.parse_args()

import os
import numpy as np
import sigpy as sp
import sigpy.mri as mr
import logging

logging.basicConfig(level=logging.DEBUG)

# Choose device
comm = sp.Communicator()
if args.multi_gpu:
    device = sp.Device(comm.rank)
else:
    device = sp.Device(args.device)

xp = device.xp

logging.info('Reading data.')
ksp = np.load(args.ksp_file, 'r')
coord = np.load(args.coord_file, 'r')
dcf = np.load(args.dcf_file, 'r')

ndim = coord.shape[-1]
num_coils, num_tr, num_ro = ksp.shape
tr_per_frame = num_tr // args.num_frames
if args.img_shape:
    img_shape = args.img_shape
else:
    img_shape = sp.estimate_shape(coord)

logging.info(f'Image shape: {img_shape}.')

with device:
    if comm.rank == 0:
        img = []

    for t in range(args.num_frames):
        coord_t = sp.to_device(coord[t * tr_per_frame: (t + 1) * tr_per_frame], device)
        dcf_t = sp.to_device(dcf[t * tr_per_frame: (t + 1) * tr_per_frame], device)
        
        img_t = 0
        for c in range(comm.rank, num_coils, comm.size):
            logging.info(f'Reconstructing time {t}, coil {c}')
            ksp_tc = sp.to_device(ksp[c, t * tr_per_frame: (t + 1) * tr_per_frame, :], device)
            
            img_t += xp.abs(sp.nufft_adjoint(ksp_tc * dcf_t, coord_t, img_shape))**2

        comm.reduce(img_t, root=0)
        if comm.rank == 0:
            img_t = img_t**0.5
            img.append(sp.to_device(img_t))

    if comm.rank == 0:
        logging.info('Saving data.')
        np.save(args.img_file, img)
