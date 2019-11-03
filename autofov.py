import argparse
import numpy as np
import sigpy as sp
import logging


def autofov(ksp, coord, dcf, nro=100, device=sp.cpu_device,
            thresh=0.2, pad=1.0):
    device = sp.Device(device)
    xp = device.xp
    with device:
        kspc = ksp[:, :, :nro]
        coordc = coord[:, :nro, :]
        dcfc = dcf[:, :nro]
        coordc2 = coordc * 2

        num_coils = kspc.shape[0]
        imgc_shape = np.array(sp.estimate_shape(coordc))
        imgc2_shape = sp.estimate_shape(coordc2)
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
        boxc_bound = np.array([[min(boxc_idx[i]), max(boxc_idx[i])]
                               for i in range(3)])
        boxc_shape = boxc_bound[:, 1] - boxc_bound[:, 0]
        boxc_center = (boxc_bound[:, 1] + boxc_bound[:, 0]) / 2
        boxc_shift = boxc_center - np.array(imgc2_shape) // 2

        img_scale = boxc_shape / imgc_shape
        coord *= img_scale * pad
        new_img_shape = sp.estimate_shape(coord)

        phase_shift = boxc_shift / imgc_shape
        lin_phase = np.exp(1j * 2 * np.pi * np.sum(
            coord * phase_shift, axis=-1))
        ksp *= lin_phase
        shift = phase_shift * new_img_shape

        return shift


if __name__ == '__main__':

    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument('--nro', type=int, default=100)
    parser.add_argument('--device', type=int, default=-1)
    parser.add_argument('--thresh', type=float, default=0.2)
    parser.add_argument('--pad', type=float, default=1.0)

    parser.add_argument('ksp_file', type=str)
    parser.add_argument('coord_file', type=str)
    parser.add_argument('dcf_file', type=str)

    args = parser.parse_args()

    ksp = np.load(args.ksp_file)
    coord = np.load(args.coord_file)
    dcf = np.load(args.dcf_file)

    autofov(ksp, coord, dcf, nro=args.nro, device=args.device,
            thresh=args.thresh, pad=args.pad)

    logging.info('Image shape: {}'.format(sp.estimate_shape(coord)))

    logging.info('Saving data.')
    np.save(args.ksp_file, ksp)
    np.save(args.coord_file, coord)
