#!/usr/bin/env python
"""
Simple cartridge loader.
"""
import tarfile
import os
import core.util as util
import faiss


def cartridge_load(src, dst) -> str:
    with tarfile.open(src, 'r:gz') as tar:
        tar.extractall(path=dst)

    # print(src, dst)
    dir_name = f"{os.path.basename(src).split('.')[0]}"
    cmp_file = f"{dir_name}_hash.txt"

    with open(os.path.join(dst, cmp_file)) as f:
        stored_hash = f.read().split(":")[1].strip()

    recomputed_hash = util.compute_directory_hash(os.path.join(dst, dir_name))

    if stored_hash != recomputed_hash:
        print("Integrity check failed! Directory has changed:\n{}\n{}".format(
              stored_hash, recomputed_hash))

    return dir_name


def load_faiss_index(index_path):
    index = faiss.read_index(index_path)
    return index


def load_document_chunks(chunks_path):
    with open(chunks_path, "r", encoding="utf-8") as file:
        chunks = file.readlines()
    return [chunk.strip() for chunk in chunks]


def retrieve_relevant_documents(query_vector, faiss_index, k=5):

    assert query_vector.shape[1] == faiss_index.d, f"FAISS index dim mismatch: {faiss_index.d} over {query_vector.shape[1]}"

    distances, indices = faiss_index.search(query_vector, k)
    return indices[0]
