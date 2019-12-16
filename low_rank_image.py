import numpy as np
import h5py
import sigpy as sp
from math import ceil


class LowRankImage(object):
    """Low rank image representation.

    Args:
        L (np.array): Left singular vectors of length J.
            Each scale is of shape [T] + n_j + b_j.
        R (np.array): Right singular vectors of length J.
            Each scale is of shape [T] + n_j + [1] * num_img_dim.
        img_shape (tuple of ints): Image shape.

    """
    def __init__(self, L, R, img_shape):
        self.img_shape = img_shape
        self.T = len(R[0])
        self.shape = (self.T, ) + tuple(img_shape)
        self.size = sp.prod(self.shape)
        self.ndim = len(self.shape)
        self.dtype = L[0].dtype
        self.J = len(L)
        self.D = len(img_shape)
        self.blk_widths = [max(L[j].shape[-self.D:]) for j in range(self.J)]
        self.B = _get_B(img_shape, self.T, self.blk_widths)
        self.L = L
        self.R = R
        self.device = sp.cpu_device
        
    def use_device(self, device):
        self.device = sp.Device(device)
        self.L = [sp.to_device(L_j, self.device) for L_j in self.L]
        self.R = [sp.to_device(R_j, self.device) for R_j in self.R]
        
    def __len__(self):
        return self.T

    def _get_img(self, t, idx=None):
        with self.device:
            img_t = 0
            for j in range(self.J):
                img_t += self.B[j](self.L[j] * self.R[j][t])[idx]

        img_t = sp.to_device(img_t, sp.cpu_device)
        return img_t

    def __getitem__(self, index):
        if isinstance(index, int):
            return self._get_img(index)
        elif isinstance(index, slice):
            return np.stack([self._get_img(t) for t in range(self.T)[index]])
        elif isinstance(index, tuple):
            tslc = index[0]
            if isinstance(tslc, int):
                return self._get_img(tslc, index[1:])
            elif isinstance(tslc, slice):
                return np.stack([self._get_img(t, index[1:])
                                 for t in range(self.T)[tslc]])

    def save(self, filename):
        with h5py.File(filename, 'w') as hf:
            hf.attrs['J'] = self.J
            for j in range(self.J):
                hf.create_dataset('L_{}'.format(j), data=self.L[j])
                hf.create_dataset('R_{}'.format(j), data=self.R[j])

            hf.create_dataset('img_shape', data=self.img_shape)

    @classmethod
    def load(cls, filename):
        with h5py.File(filename, 'r') as hf:
            J = hf.attrs['J']
            L = []
            R = []
            for j in range(J):
                L.append(hf['L_{}'.format(j)][:])
                R.append(hf['R_{}'.format(j)][:])

            img_shape = hf['img_shape'][:]
            return cls(L, R, img_shape)


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
        W_j = sp.linop.Multiply(B_j.ishape, sp.hanning(b_j, dtype=dtype)**0.5)
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

    M_j = sp.prod(b_j)
    P_j = sp.prod(n_j)
    G_j = M_j**0.5 + T**0.5 + (2 * np.log(P_j))**0.5

    return i_j, b_j, s_j, n_j, G_j
