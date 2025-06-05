#!/usr/bin/env python
"""
Makes the prompting.
"""

# import ollama
import core.cartridge_loader as loader
import os
import numpy as np
import core.util as util

# ollama.init()


def generate_response(query, context):
    combined_input = f"Context: {context}\n\nQuestion: {query}"
    # response = ollama.chat(combined_input)
    # return response['text']
    return combined_input


def perform_rag(query, faiss_index_path, chunks_path):
    faiss_index = loader.load_faiss_index(faiss_index_path)
    document_chunks = loader.load_document_chunks(chunks_path)

    query_vector = util.generate_embeddings([query]).astype(np.float32)

    relevant_indices = loader.retrieve_relevant_documents(
        query_vector, faiss_index, k=3)

    context = " ".join([document_chunks[i] for i in relevant_indices])

    answer = generate_response(query, context)
    return answer


if __name__ == "__main__":
    cartridge_path = os.path.join(os.getcwd(), "cache", "kr_en")
    faiss_index_path = "faiss.index"
    chunks_path = "chunks.txt"

    result = perform_rag(input("> "), os.path.join(
        cartridge_path, faiss_index_path), os.path.join(cartridge_path, chunks_path))

    print(result)
