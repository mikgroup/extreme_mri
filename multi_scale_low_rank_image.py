import numpy as np
import h5py
import sigpy as sp
from math import ceil


class MultiScaleLowRankImage(object):
    """Multi-scale low rank image.

    Args:
        shape (tuple of ints): image shape.
        L (list of arrays): Left singular vectors of length J.
            Each scale is of shape n_j + b_j.
            where n_j represents the number of blocks,
            and b_j represents the block shape.
        R (list of arrays): Right singular vectors of length J.
            Each scale is of shape [T] + n_j + [1] * D
            where T is the number of frames,
            n_j represents the number of blocks,
            and D represents the number of spatial dimensions.
        res (None of tuple of floats): resolution.

    """
    def __init__(self, shape, L, R, res=None):
        self.shape = tuple(shape)
        self.img_shape = self.shape[1:]
        self.T = self.shape[0]
        self.size = sp.prod(self.shape)
        self.ndim = len(self.shape)
        self.dtype = L[0].dtype
        self.J = len(L)
        self.D = self.ndim - 1
        self.blk_widths = [max(L[j].shape[-self.D:]) for j in range(self.J)]
        self.L = L
        self.R = R
        self.device = sp.cpu_device
        if res is None:
            self.res = (1, ) * self.D

    def use_device(self, device):
        self.device = sp.Device(device)
        self.L = [sp.to_device(L_j, self.device) for L_j in self.L]
        self.R = [sp.to_device(R_j, self.device) for R_j in self.R]

    def _get_B(self, j):
        b_j = [min(i, self.blk_widths[j]) for i in self.img_shape]
        s_j = [(b + 1) // 2 for b in b_j]

        i_j = [ceil((i - b + s) / s) * s + b - s
               for i, b, s in zip(self.img_shape, b_j, s_j)]

        C_j = sp.linop.Resize(self.img_shape, i_j,
                              ishift=[0] * self.D, oshift=[0] * self.D)
        B_j = sp.linop.BlocksToArray(i_j, b_j, s_j)
        w_j = sp.hanning(b_j, dtype=self.dtype, device=self.device)**0.5
        W_j = sp.linop.Multiply(B_j.ishape, w_j)
        return C_j * B_j * W_j

    def __len__(self):
        return self.T

    def _get_img(self, t, idx=None):
        with self.device:
            img_t = 0
            for j in range(self.J):
                B_j = self._get_B(j)
                img_t += B_j(self.L[j] * self.R[j][t])[idx]

        img_t = sp.to_device(img_t, sp.cpu_device)
        return img_t

    def __getitem__(self, index):
        if isinstance(index, slice):
            return np.stack([self._get_img(t) for t in range(self.T)[index]])
        elif isinstance(index, tuple):
            tslc = index[0]
            if isinstance(tslc, slice):
                return np.stack([self._get_img(t, index[1:])
                                 for t in range(self.T)[tslc]])
            else:
                return self._get_img(tslc, index[1:])
        else:
            return self._get_img(index)

    @classmethod
    def load(cls, filename):
        with h5py.File(filename, 'r') as hf:
            shape = hf['shape'][:]
            res = hf['res'][:]
            J = hf.attrs['J']
            L = []
            R = []
            for j in range(J):
                L.append(hf['L_{}'.format(j)][:])
                R.append(hf['R_{}'.format(j)][:])

            return cls(shape, L, R, res=res)

    def save(self, filename):
        with h5py.File(filename, 'w') as hf:
            hf.attrs['data_format'] = 'multi_scale_low_rank'
            hf.create_dataset('shape', data=self.shape)
            hf.create_dataset('res', data=self.res)
            hf.attrs['J'] = self.J
            for j in range(self.J):
                hf.create_dataset('L_{}'.format(j), data=self.L[j])
                hf.create_dataset('R_{}'.format(j), data=self.R[j])
