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
import pymupdf


def load_data_json(path: str) -> Dict[str, any]:
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    return data


def load_data_pdf(path: str) -> str:
    doc = pymupdf.open(path)
    text = ""
    for pnum in range(doc.page_count):
        page = doc.load_page(pnum)
        text += page.get_text("text")

    return text


def split_text_into_chunks(data: Dict[str, any] | str, ext: str) -> List[str]:
    text_to_split: str = ""
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

    if ext == 'json':
        chunks = []
        for entry in data:
            text = entry.get("korean", "") + "\n" + entry.get("english", "")
            chunks.append(text)
        text_to_split = "\n".join(chunks)
    elif ext == 'pdf':
        text_to_split = data
    else:
        print("extension not supported!")
        return [""]

    chunked_text = splitter.split_text(text_to_split)

    return chunked_text


def index_append(f_index, embeddings: List[List[float]], chunked_text: List[str], storage_path: str):
    embeddings_np = np.array(embeddings).astype("float32")

    chunked_text_path = os.path.join(storage_path, "chunks.txt")

    if f_index is None:
        f_index = faiss.IndexFlatL2(embeddings_np.shape[1])
        with open(chunked_text_path, "w", encoding="utf-8") as f:
            pass

    f_index.add(embeddings_np)

    with open(chunked_text_path, "a", encoding="utf-8") as f:
        for chunk in chunked_text:
            f.write(f"{chunk}\n")

    return f_index


def store_vectors(f_index, storage_path: str) -> None:
    faiss.write_index(f_index, os.path.join(storage_path, "faiss.index"))
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
