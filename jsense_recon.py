import argparse
import numpy as np
import sigpy as sp
import sigpy.mri as mr
import logging


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--mps_ker_width', type=int, default=16)
    parser.add_argument('--ksp_calib_width', type=int, default=24)
    parser.add_argument('--lamda', type=float, default=0)

    parser.add_argument('--max_iter', type=int, default=10)
    parser.add_argument('--max_inner_iter', type=int, default=10)
    parser.add_argument('--device', type=int, default=-1)
    parser.add_argument('--multi_gpu', action='store_true')

    parser.add_argument('ksp_file', type=str)
    parser.add_argument('coord_file', type=str)
    parser.add_argument('dcf_file', type=str)
    parser.add_argument('mps_file', type=str)
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG)

    logging.info('Reading data.')
    ksp = np.load(args.ksp_file, mmap_mode='r')
    coord = np.load(args.coord_file)
    dcf = np.load(args.dcf_file)

    # Choose device
    comm = sp.Communicator()
    if args.multi_gpu:
        device = comm.rank
    else:
        device = args.device

    logging.info('Jsense Recon.')
    ksp = np.array_split(ksp, comm.size)[comm.rank]
    mps = mr.app.JsenseRecon(ksp,
                             coord=coord, weights=dcf,
                             mps_ker_width=args.mps_ker_width,
                             ksp_calib_width=args.ksp_calib_width,
                             lamda=args.lamda,
                             device=device,
                             comm=comm,
                             max_iter=args.max_iter,
                             max_inner_iter=args.max_inner_iter).run()

    img_shape = mps.shape[1:]
    mps = comm.gatherv(mps, root=0)
    if comm.rank == 0:
        logging.info('Saving data.')
        mps = mps.reshape((-1, ) + img_shape)
        np.save(args.mps_file, mps)
