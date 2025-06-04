#!/usr/bin/env python
"""
Focuses on breaking down text into chunks.
"""
from typing import List, Dict
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
import faiss
import numpy as np
import json
import os
from sentence_transformers import SentenceTransformer


def split_text_into_chunks(data: Dict[str, any]) -> List[str]:
    chunks = []
    for entry in data["content"]:
        text = entry.get("korean", "") + "\n" + entry.get("english", "")
        chunks.append(text)

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunked_text = splitter.split_text("\n".join(chunks))

    return chunked_text


def generate_embeddings(text_chunks: List[str]) -> List[List[float]]:
    model = SentenceTransformer("all-MiniLM-L6-v2")

    embeddings = model.encode(text_chunks)
    return embeddings


def store_vectors(embeddings: List[List[float]], chunked_text: List[str], storage_path: str) -> None:
    embeddings_np = np.array(embeddings).astype("float32")

    index = faiss.IndexFlatL2(embeddings_np.shape[1])
    index.add(embeddings_np)

    faiss.write_index(index, os.path.join(storage_path, "faiss.index"))

    with open(os.path.join(storage_path, "chunks.txt"), "w", encoding="utf-8") as f:
        for chunk in chunked_text:
            f.write(f"{chunk}\n")

def load_cartridge(path: str) -> Dict[str, any]:
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    return data

if __name__ == "__main__":
    path = os.path.join(os.getcwd(), "cartridges", "kr_en",
                        "korean_travel_basics.json")

    data = load_cartridge(path)

    text_chunks = split_text_into_chunks(data)

    embeddings = generate_embeddings(text_chunks)

    store_vectors(embeddings, text_chunks, os.path.join(
        os.getcwd(), "vectors", "kr_en"))
