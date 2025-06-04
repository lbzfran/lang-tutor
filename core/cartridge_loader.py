#!/usr/bin/env python
"""
Simple cartridge loader.
"""
import hashlib
import tarfile
import os


def _compute_directory_hash(path):
    sha1_hash = hashlib.sha1()

    for root, _, files in os.walk(path):
        for fname in sorted(files):
            fpath = os.path.join(root, fname)

            with open(fpath, 'rb') as f:
                while chunk := f.read(8192):
                    sha1_hash.update(chunk)

    return sha1_hash.hexdigest()


def cartridge_load(src, dst):
    with tarfile.open(src, 'r:gz') as tar:
        tar.extractall(path=dst)

    print(src, dst)
    cmp_file = f"{os.path.basename(src).split('.')[0]}_hash.txt"

    with open(os.path.join(dst, cmp_file)) as f:
        stored_hash = f.read().split(":")[1].strip()

    recomputed_hash = _compute_directory_hash(dst)

    if stored_hash != recomputed_hash:
        print("Integrity check failed! Directory has changed:\n{}\n{}",
              stored_hash, recomputed_hash)
