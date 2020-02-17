import argparse
import numpy as np
import sigpy as sp
import logging


def autofov(ksp, coord, dcf, num_ro=100, device=sp.cpu_device,
            thresh=0.1):
    """Automatic estimation of FOV.

    FOV is estimated by thresholding a low resolution gridded image.
    coord will be modified in-place.

    Args:
        ksp (array): k-space measurements of shape (C, num_tr, num_ro, D).
            where C is the number of channels,
            num_tr is the number of TRs, num_ro is the readout points,
            and D is the number of spatial dimensions.
        coord (array): k-space coordinates of shape (num_tr, num_ro, D).
        dcf (array): density compensation factor of shape (num_tr, num_ro).
        num_ro (int): number of read-out points.
        device (Device): computing device.
        thresh (float): threshold between 0 and 1.

    """
    device = sp.Device(device)
    xp = device.xp
    with device:
        kspc = ksp[:, :, :num_ro]
        coordc = coord[:, :num_ro, :]
        dcfc = dcf[:, :num_ro]
        coordc2 = sp.to_device(coordc * 2, device)

        num_coils = len(kspc)
        imgc_shape = np.array(sp.estimate_shape(coordc))
        imgc2_shape = sp.estimate_shape(coordc2)
        imgc2_center = [i // 2 for i in imgc2_shape]
        imgc2 = sp.nufft_adjoint(sp.to_device(dcfc * kspc, device),
                                 coordc2, [num_coils] + imgc2_shape)
        imgc2 = xp.sum(xp.abs(imgc2)**2, axis=0)**0.5
        if imgc2.ndim == 3:
            imgc2_cor = imgc2[:, imgc2.shape[1] // 2, :]
            thresh *= imgc2_cor.max()
        else:
            thresh *= imgc2.max()

        boxc = imgc2 > thresh

        boxc = sp.to_device(boxc)
        boxc_idx = np.nonzero(boxc)
        boxc_shape = np.array([int(np.abs(boxc_idx[i] - imgc2_center[i]).max()) * 2
                               for i in range(imgc2.ndim)])

        img_scale = boxc_shape / imgc_shape
        coord *= img_scale


if __name__ == '__main__':

    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument('--num_ro', type=int, default=100)
    parser.add_argument('--device', type=int, default=-1)
    parser.add_argument('--thresh', type=float, default=0.1)

    parser.add_argument('ksp_file', type=str)
    parser.add_argument('coord_file', type=str)
    parser.add_argument('dcf_file', type=str)

    args = parser.parse_args()

    ksp = np.load(args.ksp_file)
    coord = np.load(args.coord_file)
    dcf = np.load(args.dcf_file)

    autofov(ksp, coord, dcf, num_ro=args.num_ro, device=args.device,
            thresh=args.thresh)

    logging.info('Image shape: {}'.format(sp.estimate_shape(coord)))

    logging.info('Saving data.')
    np.save(args.coord_file, coord)
