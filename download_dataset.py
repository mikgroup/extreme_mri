from tqdm.auto import tqdm
import requests
import pathlib


def download(url, path):
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        r = requests.get(url, stream=True)
        total_size = int(r.headers.get('content-length', 0))
        block_size = 1024 #1 Kibibyte
        with path.open('wb') as f:
            with tqdm(desc='Downloading ' + path.name, total=total_size,
                      unit='iB', unit_scale=True) as pbar:
                for data in r.iter_content(block_size):
                    f.write(data)
                    pbar.update(len(data))


def download_dce_dataset():
    dcf_path = pathlib.Path('data/dce/dcf.npy')
    coord_path = pathlib.Path('data/dce/coord.npy')
    ksp_path = pathlib.Path('data/dce/ksp.npy')
    download('https://zenodo.org/record/3647820/files/dcf.npy', dcf_path)
    download('https://zenodo.org/record/3647820/files/coord.npy', coord_path)
    download('https://zenodo.org/record/3647820/files/ksp.npy', ksp_path)


def download_lung_dataset():
    dcf_path = pathlib.Path('data/lung/dcf.npy')
    coord_path = pathlib.Path('data/lung/coord.npy')
    ksp_path = pathlib.Path('data/lung/ksp.npy')
    download('https://zenodo.org/record/3672170/files/dcf.npy', dcf_path)
    download('https://zenodo.org/record/3672170/files/coord.npy', coord_path)
    download('https://zenodo.org/record/3672170/files/ksp.npy', ksp_path)
