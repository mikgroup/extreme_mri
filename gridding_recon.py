import numpy as np
import sigpy as sp
import logging
import argparse


def gridding_recon(ksp, coord, dcf, T=1, device=sp.cpu_device):
    """ Gridding reconstruction.

    Args:
        ksp (array): k-space measurements of shape (C, num_tr, num_ro, D).
            where C is the number of channels,
            num_tr is the number of TRs, num_ro is the readout points,
            and D is the number of spatial dimensions.
        coord (array): k-space coordinates of shape (num_tr, num_ro, D).
        dcf (array): density compensation factor of shape (num_tr, num_ro).
        mps (array): sensitivity maps of shape (C, N_D, ..., N_1).
            where (N_D, ..., N_1) represents the image shape.
        T (int): number of frames.

    Returns:
        img (array): image of shape (T, N_D, ..., N_1).
    """
    device = sp.Device(device)
    xp = device.xp
    num_coils, num_tr, num_ro = ksp.shape
    tr_per_frame = num_tr // T
    img_shape = sp.estimate_shape(coord)

    with device:
        img = []
        for t in range(T):
            tr_start = t * tr_per_frame
            tr_end = (t + 1) * tr_per_frame
            coord_t = sp.to_device(
                coord[tr_start:tr_end], device)
            dcf_t = sp.to_device(dcf[tr_start:tr_end], device)

            img_t = 0
            for c in range(num_coils):
                logging.info(f'Reconstructing time {t}, coil {c}')
                ksp_tc = sp.to_device(ksp[c, tr_start:tr_end, :], device)

                img_t += xp.abs(sp.nufft_adjoint(
                    ksp_tc * dcf_t, coord_t, img_shape))**2

            img_t = img_t**0.5
            img.append(sp.to_device(img_t))

    img = np.stack(img)
    return img


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--T', type=int, default=1)
    parser.add_argument('--device', type=int, default=-1)
    parser.add_argument('ksp_file', type=str)
    parser.add_argument('coord_file', type=str)
    parser.add_argument('dcf_file', type=str)
    parser.add_argument('img_file', type=str)
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG)

    logging.info('Reading data.')
    ksp = np.load(args.ksp_file)
    coord = np.load(args.coord_file)
    dcf = np.load(args.dcf_file)

    img = gridding_recon(ksp, coord, dcf, T=args.T, device=args.device)
    np.save(args.img_file, img)
