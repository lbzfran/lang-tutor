#!/usr/bin/env python
"""
Makes the prompting.
"""

import requests
import core.loader as loader
import numpy as np
import core.util as util

# ollama.init()
# chat_model_name = "huihui_ai/deepseek-r1-abliterated:1.5b-qwen-distill-q8_0"
chat_model_api = 'ollama'
chat_model_name = "smollm2:1.7b-instruct-q8_0"
chat_model_server = 'http://localhost:11434'

chat = None
if chat_model_api == 'ollama':
    import ollama

    def chat_internal_(model_name: str, messages: list):
        response = ollama.chat(
            model=model_name,
            messages=messages)
        return response['message']['content']
    chat = chat_internal_

elif chat_model_api == 'openai':
    from openai import OpenAI

    open_client_ = OpenAI(base_url=chat_model_server, api_key='<INSERT_KEY_HERE>')

    def chat_internal_(model_name: str, messages: list):
        response = open_client_.chat.completions.create(
            model=model_name,
            messages=messages
        )
        return response.choices[0].message.content
    chat = chat_internal_



def check_model_server():
    global chat_model_server
    try:
        response = requests.get(f"{chat_model_server}/api/tags", timeout=2)
        return response.status_code == 200
    except requests.RequestException:
        return False


def generate_response(prompt, context):
    global chat_model_name

    combined_input = f"""
        [[[ {prompt} ]]]

        CONTEXT:
        <<<< {context} >>>>
    """
    # print("Context: '{}'".format(context))

    response = chat(
        model=chat_model_name,
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
        query_vector, faiss_index, k=5)

    context = " ".join([document_chunks[i] for i in relevant_indices])

    return context


def perform_cag(query, chunks_path):
    print("db: <nil>\nchunks: {}".format(chunks_path))

    # faiss_index = loader.load_faiss_index(faiss_index_path)
    document_chunks = loader.load_document_chunks(chunks_path)

    context = " ".join(document_chunks)

    return context
