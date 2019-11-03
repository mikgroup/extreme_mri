import argparse
import numpy as np


def soft_gating_weight(resp, percentile=30, alpha=1, flip=False):
    sigma = 1.4628 * np.median(np.abs(resp - np.median(resp)))
    resp = (resp - np.median(resp)) / sigma
    thresh = np.percentile(resp, percentile)
    if flip:
        resp *= -1
    return np.exp(-alpha * np.maximum((resp - thresh), 0))


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='Estimate respiratory signal.')

    parser.add_argument('--flip', action='store_true',
                        help='Flip signal.')
    parser.add_argument('resp_file', type=str,
                        help='Output respiratory signal file.')
    parser.add_argument('sgw_file', type=str,
                        help='Soft gating weights')

    args = parser.parse_args()

    resp = np.load(args.resp_file)
    sgw = soft_gating_weight(resp, flip=args.flip)
    np.save(args.sgw_file, sgw)
