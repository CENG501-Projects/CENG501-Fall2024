#!/usr/bin/env python3

import os
import lmdb
import gdown
import zipfile
import tempfile
import argparse
import subprocess
import torchvision

from torchvision import transforms
from PIL import Image

from dataset_util import download_file, extract_archive

# ----------------------------------------------------------------------------
# Helper functions to download each dataset.
# ----------------------------------------------------------------------------

def download_cifar10(root_dir, test_only):
    """
    Downloads CIFAR-10 dataset into root_dir.
    """
    print("Downloading CIFAR-10...")
    if not test_only:
        _ = torchvision.datasets.CIFAR10(
            root=root_dir,
            train=True,
            download=True,
            transform=transforms.ToTensor()
        )
    _ = torchvision.datasets.CIFAR10(
        root=root_dir,
        train=False,
        download=True,
        transform=transforms.ToTensor()
    )
    print("CIFAR-10 download complete.\n")

def download_cifar100(root_dir, test_only):
    """
    Downloads CIFAR-100 dataset into root_dir.
    """
    print("Downloading CIFAR-100...")
    if not test_only:
        _ = torchvision.datasets.CIFAR100(
            root=root_dir,
            train=True,
            download=True,
            transform=transforms.ToTensor()
        )
    _ = torchvision.datasets.CIFAR100(
        root=root_dir,
        train=False,
        download=True,
        transform=transforms.ToTensor()
    )
    print("CIFAR-100 download complete.\n")

def download_svhn(root_dir, test_only):
    """
    Downloads SVHN dataset into root_dir.
    """
    print("Downloading SVHN...")
    if not test_only:
        _ = torchvision.datasets.SVHN(
            root=root_dir,
            split='train',
            download=True,
            transform=transforms.ToTensor()
        )
    _ = torchvision.datasets.SVHN(
        root=root_dir,
        split='test',
        download=True,
        transform=transforms.ToTensor()
    )
    print("SVHN download complete.\n")

def download_places365(root_dir, test_only):
    """
    Downloads Places365 dataset into root_dir using torchvision.datasets.Places365.
    """
    print("Downloading Places365 (this is large, please be patient)...")
    if not test_only:
        _ = torchvision.datasets.Places365(
            root=root_dir,
            split='train-standard',
            small=True,
            download=True
        )
    _ = torchvision.datasets.Places365(
        root=root_dir,
        split='val',
        small=True,
        download=True
    )
    print("Places365 download complete.\n")

def list_categories():
    url = 'http://dl.yf.io/lsun/categories.txt'
    with urlopen(Request(url)) as response:
        return response.read().decode().strip().split('\n')
    
def export_mdb_images(db_path, out_dir=None, flat=True, limit=-1, size=256):
    out_dir = out_dir
    env = lmdb.open(
        db_path, map_size=1099511627776,
        max_readers=1000, readonly=True
    )
    count = 0
    with env.begin(write=False) as txn:
        cursor = txn.cursor()
        for key, val in cursor:
            key = str(key, 'utf-8')
            # decide image out directory
            if not flat:
                image_out_dir = os.path.join(out_dir, '/'.join(key[:6]))
            else:
                image_out_dir = out_dir

            # create the directory if an image out directory doesn't exist
            if not os.path.exists(image_out_dir):
                os.makedirs(image_out_dir)

            with tempfile.NamedTemporaryFile('wb') as temp:
                temp.write(val)
                temp.flush()
                temp.seek(0)
                image_out_path = os.path.join(image_out_dir, key + '.jpg')
                Image.open(temp.name).resize((size, size)).save(image_out_path)
            count += 1
            if count == limit:
                break
            if count % 1000 == 0:
                print('Finished', count, 'images')

def download_lsun(root_dir, test_only, category='bedroom', set_name='train'):
    """
    Downloads LSUN dataset into root_dir using Dropbox.
    """
    print("Downloading LSUN dataset...")
    url = 'http://dl.yf.io/lsun/scenes/' \
          '{set_name}_lmdb.zip'.format(**locals())
    if test_only:
        out_name = 'test_lmdb.zip'
        url = 'http://dl.yf.io/lsun/scenes/test_lmdb.zip'
    else:
        out_name = '{category}_{set_name}_lmdb.zip'.format(**locals())
    out_path = os.path.join(root_dir, out_name)
    cmd = ['curl', '-C', '-', url, '-o', out_path]
    print('Downloading', category, set_name, 'set')
    subprocess.call(cmd)
    print("LSUN dataset download complete.")

    # Extract the zip file
    print(f"Extracting {out_name}...")
    with zipfile.ZipFile(out_path, 'r') as zip_ref:
        zip_ref.extractall(root_dir)
    
    print(f"Extraction complete. Dataset saved to {root_dir}.")

    export_mdb_images(os.path.join(root_dir,out_name.split(".zip")[0]), os.path.join(root_dir,out_name.split(".zip")[0])) 

def download_iSUN(root_dir,test_only):
    """
    Placeholder for iSUN dataset download.
    iSUN is not in torchvision.datasets, so you need a custom routine.
    e.g., you could use 'wget' from an official mirror or your own hosting server.
    """
    print("Downloading iSUN dataset...")
    url = f"https://drive.google.com/uc?id=16wArJ5Uhj04sJQB9bXZ5jaywNrtza4ct"
    out_name = 'iSUN.tar.gz'.format(**locals())
    out_path = os.path.join(root_dir, out_name)
    gdown.download(url, out_path, quiet=False)
    extract_archive(out_path, root_dir)
    
    print(f"Extraction complete. Dataset saved to {root_dir}.")

def download_texture(root_dir, test_only):
    """
    Downloads Texture dataset (DTD) from the Oxford VGG site.
    """
    print("Downloading Texture dataset...")
    os.makedirs(root_dir, exist_ok=True)

    url = "https://www.robots.ox.ac.uk/~vgg/data/dtd/download/dtd-r1.0.1.tar.gz"
    save_path = os.path.join(root_dir, "dtd.tar.gz")

    try:
        download_file(url, save_path)
        extract_archive(save_path, root_dir)
    except Exception as e:
        print(f"Error downloading Texture dataset: {e}")
    print("Texture dataset download complete.")

# ----------------------------------------------------------------------------
# Main function to parse arguments and handle dataset downloads.
# ----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Script to download standard datasets for OOD detection."
    )
    parser.add_argument("--root_dir", type=str, default="datasets",
                        help="Root directory where datasets will be saved.")
    parser.add_argument("--cifar10", action="store_true",
                        help="Download CIFAR-10 dataset.")
    parser.add_argument("--cifar100", action="store_true",
                        help="Download CIFAR-100 dataset.")
    parser.add_argument("--svhn", action="store_true",
                        help="Download SVHN dataset.")
    parser.add_argument("--lsun", action="store_true",
                        help="Download LSUN dataset. (Requires older TorchVision or custom script.)")
    parser.add_argument("--places365", action="store_true",
                        help="Download Places365 dataset.")
    parser.add_argument("--isun", action="store_true",
                        help="Download iSUN dataset (placeholder).")
    parser.add_argument("--texture", action="store_true",
                        help="Download Texture dataset (placeholder).")
    parser.add_argument("--all", action="store_true",
                        help="Download all datasets mentioned above.")
    parser.add_argument("--test", action="store_true",
                        help="Only download test or validation sets.")

    args = parser.parse_args()
    os.makedirs(args.root_dir, exist_ok=True)

    if args.cifar10 or args.all:
        download_cifar10(args.root_dir,args.test)
    if args.cifar100 or args.all:
        download_cifar100(args.root_di,args.test)
    if args.svhn or args.all:
        download_svhn(args.root_dir,args.test)
    if args.lsun or args.all:
        download_lsun(args.root_dir,args.test)
    if args.places365 or args.all:
        download_places365(args.root_dir,args.test)
    if args.isun or args.all:
        download_iSUN(args.root_dir,args.test)
    if args.texture or args.all:
        download_texture(args.root_dir,args.test)

    print("All requested downloads are complete.")

if __name__ == "__main__":
    main()
