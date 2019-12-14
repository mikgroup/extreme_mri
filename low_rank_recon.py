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
                 blk_widths=[32, 64, 128], alpha=1, beta=0.1, sgw=None,
                 device=sp.cpu_device, comm=None, seed=0, eps=1e-3,
                 max_epoch=100, max_power_iter=10,
                 show_pbar=True, save_objective_values=False):
        self.ksp = ksp
        self.coord = coord
        self.dcf = dcf
        self.mps = mps
        self.sgw = sgw
        self.eps = eps
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
        self.save_objective_values = save_objective_values

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
        self.B = _get_B(self.img_shape, self.T, self.blk_widths)
        if self.sgw is not None:
            self.dcf *= np.expand_dims(self.sgw, -1)

        self._normalize()

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
            self.ksp *= (sp.prod(self.img_shape) * self.T)**0.5 / img_adj_norm

    def _init_LR(self):
        with self.device:
            self.L = []
            self.R = []
            self.L_ref = []
            self.R_ref = []
            self.gradf_L_ref = []
            self.gradf_R_ref = []
            for j in range(self.J):
                L_j_shape = self.B[j].ishape
                L_j = self.xp.random.standard_normal(L_j_shape).astype(self.dtype)
                sp.axpy(L_j, 1j,
                        self.xp.random.standard_normal(L_j_shape).astype(self.dtype))
                L_j_norm = self.xp.sum(self.xp.abs(L_j)**2,
                                       axis=range(-self.D, 0), keepdims=True)**0.5
                L_j /= L_j_norm
                L_j *= self.eps

                R_j_shape = (self.T, ) + L_j_norm.shape
                R_j = self.xp.random.standard_normal(R_j_shape).astype(self.dtype)
                sp.axpy(R_j, 1j,
                        self.xp.random.standard_normal(R_j_shape).astype(self.dtype))
                R_j_norm = self.xp.sum(self.xp.abs(R_j)**2, axis=0, keepdims=True)**0.5
                R_j /= R_j_norm
                R_j *= self.eps

                self.L.append(L_j)
                self.R.append(R_j)

                self.L_ref.append(self.xp.empty_like(L_j))
                self.R_ref.append(self.xp.empty_like(R_j))
                self.gradf_L_ref.append(self.xp.empty_like(L_j))
                self.gradf_R_ref.append(self.xp.empty_like(R_j))

    def run(self):
        with self.device:
            done = False
            while not done:
                try:
                    self._init_LR()
                    self._svrg()
                    done = True
                except OverflowError:
                    self.alpha *= self.beta
                    if self.show_pbar:
                        self.pbar.close()
                        tqdm.write(
                            '\nReconstruction diverged. '
                            'Restart with alpha={}.'.format(self.alpha))

            if self.comm is None or self.comm.rank == 0:
                return LowRankImage(
                    [sp.to_device(L_j, sp.cpu_device) for L_j in self.L],
                    [sp.to_device(R_j, sp.cpu_device) for R_j in self.R],
                    self.img_shape)

    def _svrg(self):
        for epoch in range(self.max_epoch):
            with tqdm(desc='Epoch {}/{} GD'.format(epoch + 1, self.max_epoch),
                      total=self.T * self.C, disable=not self.show_pbar,
                      leave=True) as self.pbar:
                for j in range(self.J):
                    self.xp.copyto(self.L_ref[j], self.L[j])
                    self.xp.copyto(self.R_ref[j], self.R[j])
                    self.gradf_L_ref[j].fill(0)
                    self.gradf_R_ref[j].fill(0)

                loss = 0
                for i in range(self.T * self.C):
                    t = i // self.C
                    c = i % self.C
                    loss += self._gd_update(t, c)
                    self.pbar.update()

                self.pbar.set_postfix(loss=loss)

            with tqdm(desc='Epoch {}/{} SGD'.format(epoch + 1, self.max_epoch),
                      total=self.T * self.C, disable=not self.show_pbar,
                      leave=True) as self.pbar:
                for i in np.random.permutation(self.T * self.C):
                    t = i // self.C
                    c = i % self.C
                    self._sgd_update(t, c)
                    self._vr_update(t, c)
                    self.pbar.update()

    def _gd_update(self, t, c):
        # Form image.
        img_t = 0
        for j in range(self.J):
            img_t += self.B[j](self.L_ref[j] * self.R_ref[j][t])

        # Download k-space arrays.
        tr_start = t * self.tr_per_frame
        tr_end = (t + 1) * self.tr_per_frame
        coord_t = sp.to_device(self.coord[tr_start:tr_end], self.device)
        dcf_t = sp.to_device(self.dcf[tr_start:tr_end], self.device)
        ksp_tc = sp.to_device(self.ksp[c, tr_start:tr_end], self.device)
        mps_c = sp.to_device(self.mps[c], self.device)

        # Data consistency.
        loss_tc = 0
        img_t *= mps_c
        e_tc = sp.nufft(img_t, coord_t)
        e_tc -= ksp_tc
        e_tc *= dcf_t**0.5
        loss_tc += self.xp.linalg.norm(e_tc)**2
        e_tc *= dcf_t**0.5
        e_tc = sp.nufft_adjoint(e_tc, coord_t, oshape=self.img_shape)
        e_tc *= self.xp.conj(mps_c)
        del img_t, coord_t, dcf_t, ksp_tc, mps_c

        if self.comm is not None:
            self.comm.allreduce(e_tc)
            self.comm.allreduce(loss_tc)

        loss_tc = loss_tc.item()

        # Gradient update.
        for j in range(self.J):
            i_j, b_j, s_j, n_j, G_j = _get_bparams(
                self.img_shape, self.T, self.blk_widths[j])
            lamda_j = self.lamda * G_j

            # L gradient.
            g_L_ref_j = self.B[j].H(e_tc)
            g_L_ref_j *= self.xp.conj(self.R_ref[j][t])
            sp.axpy(g_L_ref_j, lamda_j / (self.T * self.C), self.L_ref[j])
            self.gradf_L_ref[j] += g_L_ref_j
            loss_tc += lamda_j / (self.T * self.C) * self.xp.linalg.norm(
                self.L_ref[j]).item()**2

            # R gradient.
            g_R_ref_jt = self.B[j].H(e_tc)
            g_R_ref_jt *= self.xp.conj(self.L_ref[j])
            g_R_ref_jt = self.xp.sum(g_R_ref_jt, axis=range(-self.D, 0), keepdims=True)
            sp.axpy(g_R_ref_jt, lamda_j / self.C, self.R_ref[j][t])
            self.gradf_R_ref[j][t] += g_R_ref_jt
            loss_tc += lamda_j / self.C * self.xp.linalg.norm(self.R_ref[j]).item()**2

            if np.isinf(loss_tc) or np.isnan(loss_tc):
                raise OverflowError

        loss_tc /= 2
        return loss_tc

    def _sgd_update(self, t, c):
        # Form image.
        img_t = 0
        for j in range(self.J):
            img_t += self.B[j](self.L[j] * self.R[j][t])

        # Download k-space arrays.
        tr_start = t * self.tr_per_frame
        tr_end = (t + 1) * self.tr_per_frame
        coord_t = sp.to_device(self.coord[tr_start:tr_end], self.device)
        dcf_t = sp.to_device(self.dcf[tr_start:tr_end], self.device)
        ksp_tc = sp.to_device(self.ksp[c, tr_start:tr_end], self.device)
        mps_c = sp.to_device(self.mps[c], self.device)

        # Data consistency.
        img_t *= mps_c
        e_tc = sp.nufft(img_t, coord_t)
        e_tc -= ksp_tc
        e_tc *= dcf_t
        e_tc = sp.nufft_adjoint(e_tc, coord_t, oshape=self.img_shape)
        e_tc *= self.xp.conj(mps_c)
        del img_t, coord_t, dcf_t, ksp_tc, mps_c

        if self.comm is not None:
            self.comm.allreduce(e_tc)

        # Gradient update.
        for j in range(self.J):
            i_j, b_j, s_j, n_j, G_j = _get_bparams(
                self.img_shape, self.T, self.blk_widths[j])
            lamda_j = self.lamda * G_j

            # L gradient.
            g_L_j = self.B[j].H(e_tc)
            g_L_j *= self.xp.conj(self.R[j][t])
            sp.axpy(g_L_j, lamda_j / (self.T * self.C), self.L[j])
            g_L_j *= self.T * self.C

            # R gradient.
            g_R_jt = self.B[j].H(e_tc)
            g_R_jt *= self.xp.conj(self.L[j])
            g_R_jt = self.xp.sum(g_R_jt, axis=range(-self.D, 0), keepdims=True)
            sp.axpy(g_R_jt, lamda_j / self.C, self.R[j][t])
            g_R_jt *= self.C

            # Update.
            alpha_j = self.alpha / G_j
            sp.axpy(self.L[j], -alpha_j, g_L_j)
            sp.axpy(self.R[j][t], -alpha_j, g_R_jt)

    def _vr_update(self, t, c):
        # Form image.
        img_t = 0
        for j in range(self.J):
            img_t += self.B[j](self.L_ref[j] * self.R_ref[j][t])

        # Download k-space arrays.
        tr_start = t * self.tr_per_frame
        tr_end = (t + 1) * self.tr_per_frame
        coord_t = sp.to_device(self.coord[tr_start:tr_end], self.device)
        dcf_t = sp.to_device(self.dcf[tr_start:tr_end], self.device)
        ksp_tc = sp.to_device(self.ksp[c, tr_start:tr_end], self.device)
        mps_c = sp.to_device(self.mps[c], self.device)

        # Data consistency.
        img_t *= mps_c
        e_tc = sp.nufft(img_t, coord_t)
        e_tc -= ksp_tc
        e_tc *= dcf_t
        e_tc = sp.nufft_adjoint(e_tc, coord_t, oshape=self.img_shape)
        e_tc *= self.xp.conj(mps_c)
        del img_t, coord_t, dcf_t, ksp_tc, mps_c

        if self.comm is not None:
            self.comm.allreduce(e_tc)

        # Gradient update.
        for j in range(self.J):
            i_j, b_j, s_j, n_j, G_j = _get_bparams(
                self.img_shape, self.T, self.blk_widths[j])
            lamda_j = self.lamda * G_j

            # L gradient.
            g_L_ref_j = self.B[j].H(e_tc)
            g_L_ref_j *= self.xp.conj(self.R_ref[j][t])
            sp.axpy(g_L_ref_j, lamda_j / (self.T * self.C), self.L_ref[j])
            g_L_ref_j *= self.T * self.C

            # R gradient.
            g_R_ref_jt = self.B[j].H(e_tc)
            g_R_ref_jt *= self.xp.conj(self.L_ref[j])
            g_R_ref_jt = self.xp.sum(g_R_ref_jt, axis=range(-self.D, 0), keepdims=True)
            sp.axpy(g_R_ref_jt, lamda_j / self.C, self.R_ref[j][t])
            g_R_ref_jt *= self.C

            # Update.
            alpha_j = self.alpha / G_j
            sp.axpy(self.L[j], -alpha_j, self.gradf_L_ref[j] - g_L_ref_j)
            sp.axpy(self.R[j][t], -alpha_j, self.gradf_R_ref[j][t] - g_R_ref_jt)


def _get_B(img_shape, T, blk_widths, dtype=np.complex64):
    """Get block to array linear operator.

    """
    B = []
    J = len(blk_widths)
    for j in range(J):
        i_j, b_j, s_j, n_j, G_j = _get_bparams(
            img_shape, T, blk_widths[j])

        C_j = sp.linop.Resize(img_shape, i_j)
        B_j = sp.linop.BlocksToArray(i_j, b_j, s_j)
        W_j = sp.linop.Multiply(B_j.ishape, sp.triang(b_j, dtype=dtype))
        B.append(C_j * B_j * W_j)

    return B


def _get_bparams(img_shape, T, blk_width):
    """Calculate block parameters for scale j.

    Args:
        img_shape (tuple of ints): image shape.
        T (int): number of frames.
        j (int): scale index.

    Returns:
       tuple of ints: block shape. img_shape // 2**j
       tuple of ints: block strides. b_j // 2
       tuple of ints: number of blocks.
           (img_shape - b_j + s_j) // s_j.
       int: number of frames for each block.
           T // 2**j

    """
    b_j = [min(i, blk_width) for i in img_shape]
    s_j = [(b + 1) // 2 for b in b_j]

    i_j = [ceil((i - b + s) / s) * s + b - s
           for i, b, s in zip(img_shape, b_j, s_j)]
    n_j = [(i - b + s) // s for i, b, s in zip(i_j, b_j, s_j)]

    M_j = 1
    for d in range(len(img_shape)):
        M_j *= np.sum(sp.triang(b_j[d])**2)

    P_j = sp.prod(n_j)
    G_j = M_j**0.5 + T**0.5 + (2 * np.log(P_j))**0.5

    return i_j, b_j, s_j, n_j, G_j


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Low rank reconstruction.')
    parser.add_argument('--blk_widths', type=int, nargs='+',
                        default=[32, 64, 128],
                        help='Block widths for low rank.')
    parser.add_argument('--alpha', type=float, default=1,
                        help='Step-size')
    parser.add_argument('--beta', type=float, default=0.1,
                        help='Step-size decay')
    parser.add_argument('--max_epoch', type=int, default=100,
                        help='Maximum epochs.')
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
                        help='Regularization. Recommend 1e-4 to start.')
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
    C = (len(ksp) + comm.size - 1) // comm.size
    ksp = ksp[comm.rank::comm.size].copy()
    mps = mps[comm.rank::comm.size].copy()
    ksp = sp.resize(ksp, (C, ) + ksp.shape[1:])
    mps = sp.resize(mps, (C, ) + mps.shape[1:])

    img = LowRankRecon(ksp, coord, dcf, mps, args.T, args.lamda,
                       sgw=sgw,
                       blk_widths=args.blk_widths,
                       alpha=args.alpha, beta=args.beta,
                       max_epoch=args.max_epoch,
                       device=device, comm=comm).run()

    if comm.rank == 0:
        img.save(args.img_file)
