#!/usr/bin/env python
"""
Simple reused utility code.
"""
import hashlib
import os
from typing import List
from sentence_transformers import SentenceTransformer

model_id = "all-MiniLM-L6-v2"


def set_model(new_id):
    global model_id

    model_id = new_id or model_id or "all-MiniLM-L6-v2"


def compute_directory_hash(path):
    sha1_hash = hashlib.sha1()

    for root, _, files in os.walk(path):
        for fname in sorted(files):
            fpath = os.path.join(root, fname)

            with open(fpath, 'rb') as f:
                while chunk := f.read(8192):
                    sha1_hash.update(chunk)

    return sha1_hash.hexdigest()


def generate_embeddings(text_chunks: List[str]) -> List[List[float]]:
    global model_id

    model = SentenceTransformer(model_id)

    embeddings = model.encode(text_chunks)
    return embeddings
