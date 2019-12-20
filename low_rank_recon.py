import argparse
import logging
import numpy as np
import sigpy as sp
from math import ceil
from tqdm.auto import tqdm
from low_rank_image import LowRankImage

try:
    import mkl
    mkl.set_num_threads(1)
except:
    pass


class LowRankRecon(object):
    r"""Low rank reconstruction.

    Considers the objective function,

    .. math::
        f(l, r) = sum_t \| ksp_t - \mathcal{A}(L, R_t) \|_2^2 +
        \lambda ( \| L \|_F^2 + \| R_t \|_F^2)

    where :math:`\mathcal{A}_t` is the forward operator for time :math:`t`.

    Args:
        ksp (array): k-space measurements of shape (num_tr, num_ro, img_ndim).
        coord (array): coordinates.
        dcf (array): dcf.
        mps (array): sensitivity maps of shape (C, ...).
        T (int): number of frames.
        J (int): number of multi-scale levels.

    """
    def __init__(self, ksp, coord, dcf, mps, T, lamda,
                 blk_widths=[32, 64, 128], alpha=1, beta=0.5, sgw=None,
                 device=sp.cpu_device, comm=None, seed=0,
                 max_epoch=90, decay_epoch=30, max_power_iter=10,
                 show_pbar=True):
        self.ksp = ksp
        self.coord = coord
        self.dcf = dcf
        self.mps = mps
        self.sgw = sgw
        self.blk_widths = blk_widths
        self.T = T
        self.lamda = lamda
        self.alpha = alpha
        self.beta = beta
        self.device = sp.Device(device)
        self.comm = comm
        self.seed = seed
        self.max_epoch = max_epoch
        self.max_power_iter = max_power_iter
        self.show_pbar = show_pbar and (comm is None or comm.rank == 0)

        np.random.seed(self.seed)
        self.xp = self.device.xp
        with self.device:
            self.xp.random.seed(self.seed)

        self.dtype = self.ksp.dtype
        self.C, self.num_tr, self.num_ro = self.ksp.shape
        self.tr_per_frame = self.num_tr // self.T
        self.img_shape = self.mps.shape[1:]
        self.D = len(self.img_shape)
        self.J = len(self.blk_widths)
        if self.sgw is not None:
            self.dcf *= np.expand_dims(self.sgw, -1)

        with self.device:
            self.B = self._get_B()
            self.G = self._get_G()

        self._normalize()

    def _get_B(self):
        B = []
        for j in range(self.J):
            b_j = [min(i, self.blk_widths[j]) for i in self.img_shape]
            s_j = [(b + 1) // 2 for b in b_j]

            i_j = [ceil((i - b + s) / s) * s + b - s
                   for i, b, s in zip(self.img_shape, b_j, s_j)]

            C_j = sp.linop.Resize(self.img_shape, i_j)
            B_j = sp.linop.BlocksToArray(i_j, b_j, s_j)
            w_j = sp.hanning(b_j, dtype=self.dtype, device=self.device)**0.5
            W_j = sp.linop.Multiply(B_j.ishape, w_j)
            B.append(C_j * B_j * W_j)

        return B

    def _get_G(self):
        G = []
        for j in range(self.J):
            b_j = [min(i, self.blk_widths[j]) for i in self.img_shape]
            s_j = [(b + 1) // 2 for b in b_j]

            i_j = [ceil((i - b + s) / s) * s + b - s
                   for i, b, s in zip(self.img_shape, b_j, s_j)]
            n_j = [(i - b + s) // s for i, b, s in zip(i_j, b_j, s_j)]

            M_j = sp.prod(b_j)
            P_j = sp.prod(n_j)
            G.append(M_j**0.5 + self.T**0.5 + (2 * np.log(P_j))**0.5)

        return G

    def _normalize(self):
        with self.device:
            # Estimate maximum eigenvalue.
            coord_t = sp.to_device(self.coord[:self.tr_per_frame], self.device)
            dcf_t = sp.to_device(self.dcf[:self.tr_per_frame], self.device)
            F = sp.linop.NUFFT(self.img_shape, coord_t)
            W = sp.linop.Multiply(F.oshape, dcf_t)

            max_eig = sp.app.MaxEig(F.H * W * F, max_iter=self.max_power_iter,
                                    dtype=self.dtype, device=self.device,
                                    show_pbar=self.show_pbar).run()
            self.dcf /= max_eig

            # Estimate scaling.
            img_adj = 0
            dcf = sp.to_device(self.dcf, self.device)
            for c in range(self.C):
                mps_c = sp.to_device(self.mps[c], self.device)
                ksp_c = sp.to_device(self.ksp[c], self.device)
                img_adj_c = sp.nufft_adjoint(ksp_c * dcf, self.coord, self.img_shape)
                img_adj_c *= self.xp.conj(mps_c)
                img_adj += img_adj_c

            if self.comm is not None:
                self.comm.allreduce(img_adj)

            img_adj_norm = self.xp.linalg.norm(img_adj).item()
            self.ksp /= img_adj_norm

    def _init_vars(self):
        # Initialize L.
        self.L = []
        self.R = []
        for j in range(self.J):
            L_j_shape = self.B[j].ishape
            L_j = self.xp.random.standard_normal(L_j_shape).astype(self.dtype)
            sp.axpy(L_j, 1j,
                    self.xp.random.standard_normal(L_j_shape).astype(self.dtype))
            L_j_norm = self.xp.sum(self.xp.abs(L_j)**2,
                                   axis=range(-self.D, 0), keepdims=True)**0.5
            L_j /= L_j_norm

            R_j_shape = (self.T, ) + L_j_norm.shape
            R_j = self.xp.zeros(R_j_shape, dtype=self.dtype)
            self.L.append(L_j)
            self.R.append(R_j)

    def _power_method(self):
        for it in range(self.max_power_iter):
            # R = A^H(y)^H L
            with tqdm(desc='PowerIter R {}/{}'.format(it + 1, self.max_power_iter),
                      total=self.T, disable=not self.show_pbar, leave=True) as pbar:
                for t in range(self.T):
                    self._AHyH_L(t)
                    pbar.update()

            # Normalize R
            for j in range(self.J):
                R_j_norm = self.xp.sum(self.xp.abs(self.R[j])**2,
                                       axis=0, keepdims=True)**0.5
                self.R[j] /= R_j_norm

            # L = A^H(y) R
            with tqdm(desc='PowerIter L {}/{}'.format(it + 1, self.max_power_iter),
                      total=self.T, disable=not self.show_pbar, leave=True) as pbar:
                for j in range(self.J):
                    self.L[j].fill(0)

                for t in range(self.T):
                    self._AHy_R(t)
                    pbar.update()

            # Normalize L.
            sigma = []
            for j in range(self.J):
                L_j_norm = self.xp.sum(self.xp.abs(self.L[j])**2,
                                       axis=range(-self.D, 0), keepdims=True)**0.5
                self.L[j] /= L_j_norm
                sigma.append(L_j_norm)

        sigma_max = 0
        for j in range(self.J):
            self.L[j] *= sigma[j]**0.5
            self.R[j] *= sigma[j]**0.5
            sigma_max = max(sigma_max, sigma[j].max().item())

        self.alpha /= sigma_max

    def _AHyH_L(self, t):
        # Download k-space arrays.
        tr_start = t * self.tr_per_frame
        tr_end = (t + 1) * self.tr_per_frame
        coord_t = sp.to_device(self.coord[tr_start:tr_end], self.device)
        dcf_t = sp.to_device(self.dcf[tr_start:tr_end], self.device)
        ksp_t = sp.to_device(self.ksp[:, tr_start:tr_end], self.device)

        # A^H(y_t)
        AHy_t = 0
        for c in range(self.C):
            mps_c = sp.to_device(self.mps[c], self.device)
            AHy_tc = sp.nufft_adjoint(dcf_t * ksp_t[c], coord_t,
                                      oshape=self.img_shape)
            AHy_tc *= self.xp.conj(mps_c)
            AHy_t += AHy_tc

        if self.comm is not None:
            self.comm.allreduce(AHy_t)

        for j in range(self.J):
            AHy_tj= self.B[j].H(AHy_t)
            self.R[j][t] = self.xp.sum(AHy_tj * self.xp.conj(self.L[j]),
                                       axis=range(-self.D, 0), keepdims=True)

    def _AHy_R(self, t):
        # Download k-space arrays.
        tr_start = t * self.tr_per_frame
        tr_end = (t + 1) * self.tr_per_frame
        coord_t = sp.to_device(self.coord[tr_start:tr_end], self.device)
        dcf_t = sp.to_device(self.dcf[tr_start:tr_end], self.device)
        ksp_t = sp.to_device(self.ksp[:, tr_start:tr_end], self.device)

        # A^H(y_t)
        AHy_t = 0
        for c in range(self.C):
            mps_c = sp.to_device(self.mps[c], self.device)
            AHy_tc = sp.nufft_adjoint(dcf_t * ksp_t[c], coord_t,
                                      oshape=self.img_shape)
            AHy_tc *= self.xp.conj(mps_c)
            AHy_t += AHy_tc

        if self.comm is not None:
            self.comm.allreduce(AHy_t)

        for j in range(self.J):
            AHy_tj = self.B[j].H(AHy_t)
            self.L[j] += AHy_tj * self.xp.conj(self.R[j][t])

    def run(self):
        with self.device:
            self._init_vars()
            self._power_method()
            self.L_init = []
            self.R_init = []
            for j in range(self.J):
                self.L_init.append(sp.to_device(self.L[j]))
                self.R_init.append(sp.to_device(self.R[j]))

            done = False
            while not done:
                try:
                    self.L = []
                    self.R = []
                    for j in range(self.J):
                        self.L.append(sp.to_device(self.L_init[j], self.device))
                        self.R.append(sp.to_device(self.R_init[j], self.device))

                    self._sgd()
                    done = True
                except OverflowError:
                    self.alpha *= self.beta
                    if self.show_pbar:
                        tqdm.write('\nReconstruction diverged. '
                                   'Restart with alpha={:.3g}.'.format(self.alpha))

            if self.comm is None or self.comm.rank == 0:
                return LowRankImage(
                    [sp.to_device(L_j, sp.cpu_device) for L_j in self.L],
                    [sp.to_device(R_j, sp.cpu_device) for R_j in self.R],
                    self.img_shape)

    def _sgd(self):
        for self.epoch in range(self.max_epoch):
            desc = 'Epoch {}/{}'.format(self.epoch + 1, self.max_epoch)
            disable = not self.show_pbar
            total = self.T
            with tqdm(desc=desc, total=total,
                      disable=disable, leave=True) as pbar:
                obj = 0
                for i, t in enumerate(np.random.permutation(self.T)):
                    obj += self._update(t)
                    pbar.set_postfix(obj=obj * self.T / (i + 1))
                    pbar.update()

    def _update(self, t):
        # Form image.
        img_t = 0
        for j in range(self.J):
            img_t += self.B[j](self.L[j] * self.R[j][t])

        # Download k-space arrays.
        tr_start = t * self.tr_per_frame
        tr_end = (t + 1) * self.tr_per_frame
        coord_t = sp.to_device(self.coord[tr_start:tr_end], self.device)
        dcf_t = sp.to_device(self.dcf[tr_start:tr_end], self.device)
        ksp_t = sp.to_device(self.ksp[:, tr_start:tr_end], self.device)

        # Data consistency.
        e_t = 0
        obj_t = 0
        for c in range(self.C):
            mps_c = sp.to_device(self.mps[c], self.device)
            e_tc = sp.nufft(img_t * mps_c, coord_t)
            e_tc -= ksp_t[c]
            e_tc *= dcf_t**0.5
            obj_t += self.xp.linalg.norm(e_tc)**2
            e_tc *= dcf_t**0.5
            e_tc = sp.nufft_adjoint(e_tc, coord_t, oshape=self.img_shape)
            e_tc *= self.xp.conj(mps_c)
            e_t += e_tc

        if self.comm is not None:
            self.comm.allreduce(e_t)
            self.comm.allreduce(obj_t)

        obj_t = obj_t.item()

        # Compute gradient.
        for j in range(self.J):
            lamda_j = self.lamda * self.G[j]

            # L gradient.
            g_L_j = self.B[j].H(e_t)
            g_L_j *= self.xp.conj(self.R[j][t])
            sp.axpy(g_L_j, lamda_j / self.T, self.L[j])
            g_L_j *= self.T

            # R gradient.
            g_R_jt = self.B[j].H(e_t)
            g_R_jt *= self.xp.conj(self.L[j])
            g_R_jt = self.xp.sum(g_R_jt, axis=range(-self.D, 0), keepdims=True)
            sp.axpy(g_R_jt, lamda_j, self.R[j][t])

            # Objective value.
            obj_t += lamda_j / self.T * self.xp.linalg.norm(self.L[j]).item()**2
            obj_t += lamda_j * self.xp.linalg.norm(self.R[j][t]).item()**2
            if np.isinf(obj_t) or np.isnan(obj_t):
                raise OverflowError

            # Add.
            sp.axpy(self.L[j],
                    -self.alpha * self.beta**(self.epoch // self.decay_epoch), g_L_j)
            sp.axpy(self.R[j][t], -self.alpha, g_R_jt)

        obj_t /= 2
        return obj_t


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Low rank reconstruction.')
    parser.add_argument('--blk_widths', type=int, nargs='+', default=[32, 64, 128],
                        help='Block widths for low rank.')
    parser.add_argument('--alpha', type=float, default=1,
                        help='Step-size')
    parser.add_argument('--beta', type=float, default=0.5,
                        help='Step-size decay.')
    parser.add_argument('--max_epoch', type=int, default=90,
                        help='Maximum epochs.')
    parser.add_argument('--decay_epoch', type=int, default=30,
                        help='Decay epochs.')
    parser.add_argument('--max_power_iter', type=int, default=10,
                        help='Maximum power iterations.')
    parser.add_argument('--device', type=int, default=-1,
                        help='Computing device.')
    parser.add_argument('--multi_gpu', action='store_true',
                        help='Toggle multi-gpu. Require MPI. '
                        'Ignore device when toggled.')
    parser.add_argument('--sgw_file', type=str,
                        help='Soft gating weights.')

    parser.add_argument('ksp_file', type=str,
                        help='k-space file.')
    parser.add_argument('coord_file', type=str,
                        help='Coordinate file.')
    parser.add_argument('dcf_file', type=str,
                        help='Density compensation file.')
    parser.add_argument('mps_file', type=str,
                        help='Sensitivity maps file.')
    parser.add_argument('T', type=int,
                        help='Number of frames.')
    parser.add_argument('lamda', type=float,
                        help='Regularization. Recommend 1e-8 to start.')
    parser.add_argument('img_file', type=str,
                        help='Output image file.')

    args = parser.parse_args()
    logging.basicConfig(level=logging.DEBUG)

    ksp = np.load(args.ksp_file, 'r')
    coord = np.load(args.coord_file)
    dcf = np.load(args.dcf_file)
    mps = np.load(args.mps_file, 'r')

    comm = sp.Communicator()
    if args.multi_gpu:
        device = sp.Device(comm.rank)
    else:
        device = sp.Device(args.device)

    if args.sgw_file is not None:
        sgw = np.load(args.sgw_file)
    else:
        sgw = None

    # Split between nodes.
    ksp = ksp[comm.rank::comm.size].copy()
    mps = mps[comm.rank::comm.size].copy()

    app = LowRankRecon(ksp, coord, dcf, mps, args.T, args.lamda,
                       sgw=sgw,
                       blk_widths=args.blk_widths,
                       alpha=args.alpha,
                       beta=args.beta,
                       max_epoch=args.max_epoch,
                       decay_epoch=args.decay_epoch,
                       max_power_iter=args.max_power_iter,
                       device=device, comm=comm)
    img = app.run()

    if comm.rank == 0:
        img.save(args.img_file)
