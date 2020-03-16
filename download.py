#!/usr/bin/env python3

import argparse
import glob
import os
import subprocess
import sys
from os.path import join, realpath, dirname
import urllib.request
import shutil
import ssl
import io
import multiprocessing

try:
    from PIL import Image
except ImportError:
    print('Please install Pillow to run this script! (pip install Pillow)')
    sys.exit()
try:
    import numpy as np
except ImportError:
    print('Please install NumPy to run this script! (pip install numpy)')
    sys.exit()
try:
    from tqdm import tqdm
except ImportError:
    tqdm = None
    print('Install tqdm (pip install tqdm) for a fancy progressbar with estimated remaining time - falling back to basic progressbar')

# ssl._create_default_https_context = ssl._create_unverified_context  # dirty fix
script_dir = dirname(realpath(__file__))


def load_urls(urls_file):
    urls = list(open(urls_file).readlines())
    urls = [_.strip('\r\n') for _ in urls]  # strip linebreaks
    # Removing empty lines from list of URLS (without shifting line indices, which determine filenames)
    iurls = [(index, url) for index, url in enumerate(urls) if url]
    return iurls


def download_and_check_image(iurl):
    index, url = iurl
    save_file = join(images_dir, '%08d.jpg' % index)
    try:
        if args.force or not os.path.isfile(save_file):
            # Load into bytes
            img_data = urllib.request.urlopen(url).read()

            # Convert to RGB if necessary
            img = Image.open(io.BytesIO(img_data))
            if img.mode != 'RGB':
                print('Grayscale -> RGB : %s' % save_file)
                arr = np.asarray(img)
                assert len(arr.shape) == 2
                arr = np.stack([arr] * 3, axis=-1)
                rgb_img = Image.fromarray(arr, mode='RGB')
                rgb_img.save(save_file)
            else:
                open(save_file, 'wb').write(img_data)
            img.close()
        else:
            global redownloading_warning
            if not redownloading_warning:
                print('[WARN] Some images already exist, and are not being redownloaded. Use --force to redownload these')
                redownloading_warning = True
    except KeyboardInterrupt:
        print('KeyboardInterrupt: Exiting early!')
        sys.exit(130)  # Avoid humungous backtraces when ctrl+c is pressed
    except Exception as e:
        print('Download failed with {} for {}-th image from {}'.format(index, e, url))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Download the KaoKore Dataset.')
    parser.add_argument(
        '--dir',
        type=str,
        help='Directory in which the downloaded dataset is stored',
        default='kaokore'
    )
    parser.add_argument(
        '--force',
        help='Force redownloading of already downloaded images',
        action='store_true'
    )
    parser.add_argument(
        '--threads',
        type=int,
        help='Number of simultaneous threads to use for downloading',
        default=16
    )
    parser.add_argument(
        '--ssl_unverified_context',
        help='Force to use unverified context for SSL',
        action='store_true'
    )
    args = parser.parse_args()

    if args.ssl_unverified_context:
        print('[WARN] Use unverified context for SSL as requested. Use at your own risk')
        ssl._create_default_https_context = ssl._create_unverified_context

    script_dir = dirname(realpath(__file__))

    urls_file = join(script_dir, 'dataset', 'urls.txt')
    iurls = load_urls(urls_file)

    redownloading_warning = False

    print('Downloading {} images using {} threads'.format(len(iurls), args.threads))
    images_dir = join(args.dir, 'images_256')
    os.makedirs(images_dir, exist_ok=True)

    pool = multiprocessing.Pool(args.threads)

    if tqdm:  # Use tqdm progressbar
        bar = tqdm(total=len(iurls))
        for i, _ in enumerate(pool.imap_unordered(download_and_check_image, iurls)):
            bar.update()
    else:  # Use a basic status print
        for i, _ in enumerate(pool.imap_unordered(download_and_check_image, iurls)):
            print('Download images: %7d / %d Done' % (i + 1, len(iurls)), end='\r', flush=True)
    print()

    # TODO: Download these files if not already present
    for file in [
            'labels.csv',
            'original_tags.txt',
            'labels.metadata.en.txt',
            'labels.metadata.ja.txt',
    ]:
        shutil.copy(join(script_dir, 'dataset', file), args.dir)
