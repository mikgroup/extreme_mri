import argparse
import numpy as np
from scipy.signal import firls, convolve


def estimate_resp(dc, tr, n=9999, fl=0.1, fh=1, fw=0.01):
    """Estimate respiratory signal from DC.

    The function performs:
    1) Filter DC with a band-pass filter with symmetric extension.
    2) Normalize each channel by a robust estimation of mean and variance.
    3) Return the channel with the maximum variance.

    Args:
        dc (array): multi-channel DC array of shape [num_coils, num_tr].
        tr (float): TR in seconds.
        n (int): length of band-pass filter.
        fl (float): lower cut-off of band-pass filter in Hz.
        fh (float): higher cut-off of band-pass filter in Hz.
        fw (float): transition width of band-pass filter.

    Returns:
        array: respiratory signal of length num_tr.
    """
    dc = np.abs(dc)
    fs = 1 / tr
    bands = [0, fl - fw, fl, fh, fh + fw, fs / 2]
    desired = [0, 0, 1, 1, 0, 0]

    filt = firls(n, bands, desired, fs=fs)
    sigma_max = 0
    for c in range(len(dc)):
        dc_pad = np.pad(dc[c], [n // 2, n // 2], mode='reflect')
        resp_c = convolve(dc_pad, filt, mode='valid')
        sigma_c = 1.4826 * np.median(np.abs(resp_c - np.median(resp_c)))

        if sigma_c > sigma_max:
            resp = (resp_c - np.median(resp_c)) / sigma_c
            sigma_max = sigma_c

    return resp


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='Estimate respiratory signal.')

    parser.add_argument('ksp_file', type=str,
                        help='k-space file.')
    parser.add_argument('tr', type=float,
                        help='TR in seconds.')
    parser.add_argument('resp_file', type=str,
                        help='Output respiratory signal file.')

    args = parser.parse_args()

    ksp = np.load(args.ksp_file)
    dc = ksp[:, :, 0]
    resp = estimate_resp(dc, args.tr)
    np.save(args.resp_file, resp)
