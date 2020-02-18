import argparse
import logging
import numpy as np
import sigpy as sp
from math import ceil
from tqdm.auto import tqdm
from multi_scale_low_rank_image import MultiScaleLowRankImage

try:
    import mkl
    mkl.set_num_threads(1)
except:
    pass


class MultiScaleLowRankRecon:
    r"""Multi-scale low rank reconstruction.

    Considers the objective function,

    .. math::
        f(l, r) = sum_t \| ksp_t - \mathcal{A}(L, R_t) \|_2^2 +
        \lambda ( \| L \|_F^2 + \| R_t \|_F^2)

    where :math:`\mathcal{A}_t` is the forward operator for time :math:`t`.

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
        lamda (float): regularization parameter.
        blk_widths (tuple of ints): block widths for multi-scale low rank.
        beta (float): step-size decay factor.
        sgw (None or array): soft-gating weights.
            Shape should be compatible with dcf.
        device (sp.Device): computing device.
        comm (None or sp.Communicator): distributed communicator.
        seed (int): random seed.
        max_epoch (int): maximum number of epochs.
        decay_epoch (int): number of epochs to decay step-size.
        max_power_iter (int): maximum number of power iteration.
        show_pbar (bool): show progress bar.

    """
    def __init__(self, ksp, coord, dcf, mps, T, lamda,
                 blk_widths=[32, 64, 128], beta=0.5, sgw=None,
                 device=sp.cpu_device, comm=None, seed=0,
                 max_epoch=60, decay_epoch=20, max_power_iter=5,
                 show_pbar=True):
        self.ksp = ksp
        self.coord = coord
        self.dcf = dcf
        self.mps = mps
        self.sgw = sgw
        self.blk_widths = blk_widths
        self.T = T
        self.lamda = lamda
        self.beta = beta
        self.device = sp.Device(device)
        self.comm = comm
        self.seed = seed
        self.max_epoch = max_epoch
        self.decay_epoch = decay_epoch
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

        self.B = [self._get_B(j) for j in range(self.J)]
        self.G = [self._get_G(j) for j in range(self.J)]

        self._normalize()

    def _get_B(self, j):
        b_j = [min(i, self.blk_widths[j]) for i in self.img_shape]
        s_j = [(b + 1) // 2 for b in b_j]

        i_j = [ceil((i - b + s) / s) * s + b - s
               for i, b, s in zip(self.img_shape, b_j, s_j)]

        C_j = sp.linop.Resize(self.img_shape, i_j,
                              ishift=[0] * self.D, oshift=[0] * self.D)
        B_j = sp.linop.BlocksToArray(i_j, b_j, s_j)
        with self.device:
            w_j = sp.hanning(b_j, dtype=self.dtype, device=self.device)**0.5
        W_j = sp.linop.Multiply(B_j.ishape, w_j)
        return C_j * B_j * W_j

    def _get_G(self, j):
        b_j = [min(i, self.blk_widths[j]) for i in self.img_shape]
        s_j = [(b + 1) // 2 for b in b_j]

        i_j = [ceil((i - b + s) / s) * s + b - s
               for i, b, s in zip(self.img_shape, b_j, s_j)]
        n_j = [(i - b + s) // s for i, b, s in zip(i_j, b_j, s_j)]

        M_j = sp.prod(b_j)
        P_j = sp.prod(n_j)
        return M_j**0.5 + self.T**0.5 + (2 * np.log(P_j))**0.5

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
            coord = sp.to_device(self.coord, self.device)
            for c in range(self.C):
                mps_c = sp.to_device(self.mps[c], self.device)
                ksp_c = sp.to_device(self.ksp[c], self.device)
                img_adj_c = sp.nufft_adjoint(ksp_c * dcf, coord, self.img_shape)
                img_adj_c *= self.xp.conj(mps_c)
                img_adj += img_adj_c

            if self.comm is not None:
                self.comm.allreduce(img_adj)

            img_adj_norm = self.xp.linalg.norm(img_adj).item()
            self.ksp /= img_adj_norm

    def _init_vars(self):
        self.L = []
        self.R = []
        for j in range(self.J):
            L_j_shape = self.B[j].ishape
            L_j = sp.randn(L_j_shape, dtype=self.dtype, device=self.device)
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
            with tqdm(desc='PowerIter R {}/{}'.format(
                    it + 1, self.max_power_iter),
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
            with tqdm(desc='PowerIter L {}/{}'.format(
                    it + 1, self.max_power_iter),
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

        for j in range(self.J):
            self.L[j] *= sigma[j]**0.5
            self.R[j] *= sigma[j]**0.5

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
            self._sgd()

            if self.comm is None or self.comm.rank == 0:
                return MultiScaleLowRankImage(
                    (self.T, ) + self.img_shape,
                    [sp.to_device(L_j, sp.cpu_device) for L_j in self.L],
                    [sp.to_device(R_j, sp.cpu_device) for R_j in self.R])

    def _sgd(self):
        for self.epoch in range(self.max_epoch):
            desc = 'Epoch {}/{}'.format(self.epoch + 1, self.max_epoch)
            disable = not self.show_pbar
            total = self.T
            with tqdm(desc=desc, total=total,
                      disable=disable, leave=True) as pbar:
                loss = 0
                for i, t in enumerate(np.random.permutation(self.T)):
                    loss += self._update_R(t)
                    loss += self._update_L(t)
                    pbar.set_postfix(loss=loss * self.T / (i + 1) / 2)
                    pbar.update()

    def _update_R(self, t):
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
        loss_t = 0
        for c in range(self.C):
            mps_c = sp.to_device(self.mps[c], self.device)
            e_tc = sp.nufft(img_t * mps_c, coord_t)
            e_tc -= ksp_t[c]
            e_tc *= dcf_t**0.5
            loss_t += self.xp.linalg.norm(e_tc)**2
            e_tc *= dcf_t**0.5
            e_tc = sp.nufft_adjoint(e_tc, coord_t, oshape=self.img_shape)
            e_tc *= self.xp.conj(mps_c)
            e_t += e_tc

        if self.comm is not None:
            self.comm.allreduce(e_t)
            self.comm.allreduce(loss_t)

        loss_t = loss_t.item()

        # Compute gradient.
        for j in range(self.J):
            lamda_j = self.lamda * self.G[j]

            # R gradient.
            g_R_jt = self.B[j].H(e_t)
            g_R_jt *= self.xp.conj(self.L[j])
            g_R_jt = self.xp.sum(g_R_jt, axis=range(-self.D, 0), keepdims=True)
            g_R_jt += lamda_j * self.R[j][t]

            # Loss.
            loss_t += lamda_j / self.T * self.xp.linalg.norm(self.L[j]).item()**2
            loss_t += lamda_j * self.xp.linalg.norm(self.R[j][t]).item()**2
            if np.isinf(loss_t) or np.isnan(loss_t):
                raise OverflowError

            # Precondition.
            L_j_norm2 = self.xp.sum(
                self.xp.abs(self.L[j])**2, axis=range(-self.D, 0), keepdims=True)
            g_R_jt /= self.J * L_j_norm2 + lamda_j

            # Update.
            self.R[j][t] -= g_R_jt

        loss_t /= 2
        return loss_t

    def _update_L(self, t):
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
        loss_t = 0
        for c in range(self.C):
            mps_c = sp.to_device(self.mps[c], self.device)
            e_tc = sp.nufft(img_t * mps_c, coord_t)
            e_tc -= ksp_t[c]
            e_tc *= dcf_t**0.5
            loss_t += self.xp.linalg.norm(e_tc)**2
            e_tc *= dcf_t**0.5
            e_tc = sp.nufft_adjoint(e_tc, coord_t, oshape=self.img_shape)
            e_tc *= self.xp.conj(mps_c)
            e_t += e_tc

        if self.comm is not None:
            self.comm.allreduce(e_t)
            self.comm.allreduce(loss_t)

        loss_t = loss_t.item()

        # Compute gradient.
        for j in range(self.J):
            lamda_j = self.lamda * self.G[j]

            # L gradient.
            g_L_j = self.B[j].H(e_t)
            g_L_j *= self.xp.conj(self.R[j][t])
            g_L_j += lamda_j / self.T * self.L[j]
            g_L_j *= self.T

            # Loss.
            loss_t += lamda_j / self.T * self.xp.linalg.norm(self.L[j]).item()**2
            loss_t += lamda_j * self.xp.linalg.norm(self.R[j][t]).item()**2
            if np.isinf(loss_t) or np.isnan(loss_t):
                raise OverflowError

            # Precondition.
            R_j_norm2 = self.xp.sum(self.xp.abs(self.R[j])**2, axis=0)
            g_L_j /= self.J * R_j_norm2 + lamda_j
            g_L_j *= self.beta**(self.epoch // self.decay_epoch)

            # Update.
            self.L[j] -= g_L_j

        loss_t /= 2
        return loss_t


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='Multi-Scale Low rank reconstruction.')
    parser.add_argument('--blk_widths', type=int, nargs='+', default=[32, 64, 128],
                        help='Block widths for low rank.')
    parser.add_argument('--beta', type=float, default=0.5,
                        help='Step-size decay.')
    parser.add_argument('--max_epoch', type=int, default=60,
                        help='Maximum epochs.')
    parser.add_argument('--decay_epoch', type=int, default=20,
                        help='Decay epochs.')
    parser.add_argument('--max_power_iter', type=int, default=5,
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

    app = MultiScaleLowRankRecon(ksp, coord, dcf, mps, args.T, args.lamda,
                                 sgw=sgw,
                                 blk_widths=args.blk_widths,
                                 beta=args.beta,
                                 max_epoch=args.max_epoch,
                                 decay_epoch=args.decay_epoch,
                                 max_power_iter=args.max_power_iter,
                                 device=device, comm=comm)
    img = app.run()

    if comm.rank == 0:
        img.save(args.img_file)
