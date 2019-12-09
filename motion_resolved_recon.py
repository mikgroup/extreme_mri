import argparse
import logging
import numpy as np
import sigpy as sp
from tqdm.auto import tqdm


class MotionResolvedRecon(object):
    def __init__(self, ksp, coord, dcf, mps, resp, B,
                 lamda=1e-6, alpha=1, beta=0.5,
                 max_power_iter=10, max_iter=300,
                 device=sp.cpu_device, margin=10,
                 coil_batch_size=None, comm=None, show_pbar=True, **kwargs):
        self.B = B
        self.C = len(mps)
        self.mps = mps
        self.device = sp.Device(device)
        self.xp = device.xp
        self.alpha = alpha
        self.beta = beta
        self.lamda = lamda
        self.max_iter = max_iter
        self.max_power_iter = max_power_iter
        self.comm = comm
        if comm is not None:
            self.show_pbar = show_pbar and comm.rank == 0

        self.img_shape = list(mps.shape[1:])

        bins = np.percentile(resp, np.linspace(0 + margin, 100 - margin, B + 1))
        self.bksp = []
        self.bcoord = []
        self.bdcf = []
        for b in range(B):
            idx = (resp >= bins[b]) & (resp < bins[b + 1])
            self.bksp.append(
                sp.to_device(ksp[:, idx], self.device))
            self.bcoord.append(
                sp.to_device(coord[idx], self.device))
            self.bdcf.append(
                sp.to_device(dcf[idx], self.device))

        self._normalize()

    def _normalize(self):
        # Normalize using first phase.
        with device:
            mrimg_adj = 0
            for c in range(self.C):
                mrimg_c = sp.nufft_adjoint(
                    self.bksp[0][c] * self.bdcf[0], self.bcoord[0],
                    self.img_shape)
                mrimg_c *= self.xp.conj(sp.to_device(mps[c], device))
                mrimg_adj += mrimg_c

            if comm is not None:
                comm.allreduce(mrimg_adj)

            # Get maximum eigenvalue.
            F = sp.linop.NUFFT(self.img_shape, self.bcoord[0])
            W = sp.linop.Multiply(F.oshape, self.bdcf[0])
            max_eig = sp.app.MaxEig(F.H * W * F,
                                    max_iter=self.max_power_iter,
                                    dtype=ksp.dtype, device=device,
                                    show_pbar=self.show_pbar).run()

            # Normalize
            self.alpha /= max_eig
            self.lamda *= max_eig * self.xp.abs(mrimg_adj).max().item()

    def gradf(self, mrimg):
        out = self.xp.zeros_like(mrimg)
        for b in range(self.B):
            for c in range(self.C):
                mps_c = sp.to_device(self.mps[c], self.device)
                out[b] += sp.nufft_adjoint(
                    self.bdcf[b] * (sp.nufft(mrimg[b] * mps_c, self.bcoord[b])
                                    - self.bksp[b][c]),
                    self.bcoord[b],
                    oshape=mrimg.shape[1:]) * self.xp.conj(mps_c)

        if self.comm is not None:
            self.comm.allreduce(out)

        eps = 1e-31
        for b in range(self.B):
            if b > 0:
                diff = mrimg[b] - mrimg[b - 1]
                sp.axpy(out[b], self.lamda, diff / (self.xp.abs(diff) + eps))

            if b < self.B - 1:
                diff = mrimg[b] - mrimg[b + 1]
                sp.axpy(out[b], self.lamda, diff / (self.xp.abs(diff) + eps))

        return out

    def run(self):
        done = False
        while not done:
            try:
                with tqdm(total=self.max_iter, desc='MotionResolvedRecon',
                          disable=not self.show_pbar) as pbar:
                    with self.device:
                        mrimg = self.xp.zeros([self.B] + self.img_shape,
                                             dtype=self.mps.dtype)
                        for it in range(self.max_iter):
                            g = self.gradf(mrimg)
                            sp.axpy(mrimg, -self.alpha, g)

                            gnorm = self.xp.linalg.norm(g.ravel()).item()
                            if np.isnan(gnorm) or np.isinf(gnorm):
                                raise OverflowError('LowRankRecon diverges.')

                            pbar.set_postfix(gnorm=gnorm)
                            pbar.update()

                        done = True
            except OverflowError:
                self.alpha *= self.beta

        return mrimg


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='Motion Resolved Reconstruction.')
    parser.add_argument('--lamda', type=float, default=1e-6,
                        help='Regularization.')
    parser.add_argument('--max_iter', type=int, default=300,
                        help='Maximum epochs.')
    parser.add_argument('--device', type=int, default=-1,
                        help='Computing device.')
    parser.add_argument('--multi_gpu', action='store_true',
                        help='Toggle multi-gpu. Require MPI. '
                        'Ignore device when toggled.')

    parser.add_argument('ksp_file', type=str,
                        help='k-space file.')
    parser.add_argument('coord_file', type=str,
                        help='Coordinate file.')
    parser.add_argument('dcf_file', type=str,
                        help='Density compensation file.')
    parser.add_argument('mps_file', type=str,
                        help='Sensitivity maps file.')
    parser.add_argument('resp_file', type=str,
                        help='Respiratory signal file.')
    parser.add_argument('B', type=int,
                        help='Number of frames.')
    parser.add_argument('mrimg_file', type=str,
                        help='Output image file.')

    args = parser.parse_args()
    logging.basicConfig(level=logging.DEBUG)

    ksp = np.load(args.ksp_file, 'r')
    coord = np.load(args.coord_file)
    dcf = np.load(args.dcf_file)
    mps = np.load(args.mps_file, 'r')
    resp = np.load(args.resp_file)

    comm = sp.Communicator()
    if args.multi_gpu:
        device = sp.Device(comm.rank)
    else:
        device = sp.Device(args.device)

    # Split between nodes.
    ksp = ksp[comm.rank::comm.size]
    mps = mps[comm.rank::comm.size]
    mrimg = MotionResolvedRecon(ksp, coord, dcf, mps, resp, args.B,
                               max_iter=args.max_iter, lamda=args.lamda,
                               device=device, comm=comm).run()

    if comm.rank == 0:
        xp = sp.get_array_module(mrimg)
        xp.save(args.mrimg_file, mrimg)
