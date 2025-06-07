#!/usr/bin/env python
"""
Makes the prompting.
"""

import ollama
import core.loader as loader
import os
import numpy as np
import core.util as util

# ollama.init()
# model_name = "huihui_ai/deepseek-r1-abliterated:1.5b-qwen-distill-q8_0"
model_name = "smollm2:1.7b-instruct-q8_0"


def generate_response(context):
    combined_input = f"""
        You are an AI language tutor. Use the context below to create a lesson for a beginner learner.

        CONTEXT:
        <<<< {context} >>>>

        LESSON STRUCTURE:
        1. Vocabulary List
        2. Grammar Explanation
        3. Practice Sentences
        4. Translation Exercise
        """
    print("Context: '{}'".format(context))
    response = ollama.chat(
        model=model_name,
        messages=[
            {
                'role': 'user',
                'content': combined_input,
            },
        ])
    return response['message']['content']


def perform_rag(query, faiss_index_path, chunks_path):
    print("db: {}\nchunks: {}".format(faiss_index_path, chunks_path))

    faiss_index = loader.load_faiss_index(faiss_index_path)
    document_chunks = loader.load_document_chunks(chunks_path)

    query_vector = util.generate_embeddings([query]).astype(np.float32)

    relevant_indices = loader.retrieve_relevant_documents(
        query_vector, faiss_index, k=3)

    context = " ".join([document_chunks[i] for i in relevant_indices])

    return context

