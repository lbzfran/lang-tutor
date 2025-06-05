#!/usr/bin/env python
"""
Focuses on breaking down text into chunks.
"""
from typing import List, Dict
from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_community.vectorstores import FAISS
import faiss
import numpy as np
import json
import os
import tarfile
import core.util as util


def load_data_json(path: str) -> Dict[str, any]:
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    return data


def split_text_into_chunks(data: Dict[str, any]) -> List[str]:
    chunks = []
    for entry in data["content"]:
        text = entry.get("korean", "") + "\n" + entry.get("english", "")
        chunks.append(text)

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunked_text = splitter.split_text("\n".join(chunks))

    return chunked_text



def store_vectors(embeddings: List[List[float]], chunked_text: List[str], storage_path: str) -> None:
    embeddings_np = np.array(embeddings).astype("float32")

    index = faiss.IndexFlatL2(embeddings_np.shape[1])
    index.add(embeddings_np)

    faiss.write_index(index, os.path.join(storage_path, "faiss.index"))

    with open(os.path.join(storage_path, "chunks.txt"), "w", encoding="utf-8") as f:
        for chunk in chunked_text:
            f.write(f"{chunk}\n")

    cartridge_compile(storage_path, storage_path + '.cart')


def cartridge_compile(src, dst):
    dir_hash = util.compute_directory_hash(src)

    hash_file_path = f'{os.path.basename(src)}_hash.txt'
    with open(hash_file_path, 'w') as f:
        f.write(f"SHA-1:{dir_hash}\n")

    with tarfile.open(dst, 'w:gz') as tar:
        tar.add(src, arcname=os.path.basename(src))
        tar.add(hash_file_path, arcname=hash_file_path)

    os.remove(hash_file_path)
