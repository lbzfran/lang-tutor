import os
import shutil
from typing import List
import core.util as util
import core.vectorizer as vectorizer
import core.loader as loader
import core.prompter as prompter

last_cartridge_id = ""
current_f_index = None
cache_dir_path = os.path.join(os.getcwd(), "cache")
cartridge_dir_path = os.path.join(os.getcwd(), "cartridges")
data_dir_path = os.path.join(os.getcwd(), "data")
chat_use_CAG = False
chat_prompt = """
You are an AI Language Tutor.
Generate a simple test based on the following information.
For each question generated, include both question AND the answer, and
clearly label them 'QUESTION' and 'ANSWER'.
If an answer contains an explanation, make the explanation general, and assume
the test-taker cannot see the context you are provided with.
"""


def add_command(cartridge_id: str = None, *file_names: List[str]):
    """
    Adds one or more valid (pdf/json) files to a cartridge.
    The cartridge id must be specified, and will create a new
    directory in the cartridge directory if not found.
    """
    global last_cartridge_id, current_f_index

    cartridge_id = cartridge_id or last_cartridge_id
    if cartridge_id is None:
        print("'add' expects id i.e., 'compile <cartridge_id>'.")
        return
    last_cartridge_id = cartridge_id

    output_dir = os.path.join(cartridge_dir_path, cartridge_id)
    os.makedirs(output_dir, exist_ok=True)

    for file_name in file_names:
        data_file_path = os.path.join(data_dir_path, file_name)
        if not os.path.exists(data_file_path):
            print("'{}' not found in '{}'.".format(file_name, cartridge_id))
            continue

        file_ext = file_name.split('.')[1]
        print("using path: '{}'.".format(data_file_path))

        text_chunks: List[str] = [""]
        if file_ext == 'json':
            data = vectorizer.load_data_json(data_file_path)
        elif file_ext == 'pdf':
            data = vectorizer.load_data_pdf(data_file_path)
        else:
            print("file of type '{}' not supported!".format(file_ext))
            return

        text_chunks = vectorizer.split_text_into_chunks(data, file_ext)
        embeddings = util.generate_embeddings(text_chunks)

        current_f_index = vectorizer.index_append(
            current_f_index, embeddings, text_chunks, output_dir)

        print("'{}' added to '{}'.".format(file_name, cartridge_id))


def compile_command(cartridge_id: str = None):
    """
    Prepares a cartridge to be used by the program.
    The specified cartridge id is expected to already exist
    within the cartridge directory.
    Outputs a 'cart' file.
    """
    global current_f_index

    cartridge_id = cartridge_id or last_cartridge_id
    if cartridge_id is None:
        print("'compile' expects id i.e., 'compile <cartridge_id>'.")
        return

    output_dir = os.path.join(cartridge_dir_path, cartridge_id)
    if not os.path.isdir(output_dir):
        print("ERROR: called 'compile' when id not established in memory. try 'add' first.")
        return

    if current_f_index:
        vectorizer.store_vectors(current_f_index, output_dir)
        current_f_index = None

    print("compiled vector file at '{}'.".format(
        os.path.join(cartridge_dir_path, cartridge_id + '.cart')))


def load_command(cartridge_id: str = None, cartridge_path: str = None, output_dir: str = cache_dir_path):
    """
    Loads a valid cartridge file into the cache directory.
    Also automatically sets the cartridge id to be used.
    """
    global last_cartridge_id

    cartridge_id = cartridge_id or last_cartridge_id
    if cartridge_id is None:
        print("ERROR: cartridge id was not previously set.")
        return

    cartridge_path = cartridge_path or os.path.join(
        cartridge_dir_path, cartridge_id + '.cart')
    os.makedirs(output_dir, exist_ok=True)

    print("using path: '{}'.".format(cartridge_path))

    last_cartridge_id = loader.cartridge_load(cartridge_path, output_dir)

    print("Loaded cartridge! Data dumped in '{}'.".format(data_dir_path))


def set_command(cartridge_id: str = None):
    """
    Manually set the cartridge id to be used.
    If no arguments are passed, an interactive process
    will guide user to choose from the available cartridges
    already loaded in the cache directory.
    """
    global last_cartridge_id

    possible_ids = []
    if cartridge_id is None:
        for _, dirs, _ in os.walk(cache_dir_path):
            for d in dirs:
                possible_ids.append(d)

        print("Possible Ids in cache:", possible_ids)
        while True:
            print("Enter an id:")
            user_input = input(">> ")

            if user_input in possible_ids:
                cartridge_id = user_input
                break

            print("Input is not valid! Try again.")

    print("Cartridge set to '{}'.".format(cartridge_id))
    last_cartridge_id = cartridge_id


def unload_command(cartridge_id: str = None):
    """
    Clears out the cache directory.
    """
    print("This will clear out all files within the cache. Continue?")

    answer = input("(y/N): ")
    if answer.lower() == 'y':
        shutil.rmtree(os.path.join(cache_dir_path, cartridge_id))
        print("Unloaded '{}' from cache!".format(cartridge_id))

def prompt_command(*new_prompt):
    """
    Simple setter for the prompt used by the chat model.
    """
    global chat_prompt

    if new_prompt is None:
        print("WARN: no prompt provided. Nothing changed.")
        return

    old_prompt = chat_prompt
    chat_prompt = " ".join(new_prompt)
    print("prompt: '{}' -> '{}'.".format(old_prompt, chat_prompt))


def context_command(*query):
    """
    Generates a context from a loaded cartridge based on a given query.
    """
    global last_cartridge_id, chat_use_CAG

    cartridge_id = last_cartridge_id
    if cartridge_id is None:
        print("ERROR: cartridge id was not previously set via 'load'.")
        return

    cartridge_dir_path = os.path.join(cache_dir_path, cartridge_id)

    query_joined = " ".join(query)

    context_lang = "Language: {}\n".format(cartridge_id)

    context_info: str = ""
    document_chunks_path = os.path.join(cartridge_dir_path, "chunks.txt")
    if chat_use_CAG:
        context_info = prompter.perform_cag(query_joined, document_chunks_path)
    else:
        faiss_path = os.path.join(cartridge_dir_path, "faiss.index")
        context_info = prompter.perform_rag(query_joined, faiss_path, document_chunks_path)

    context = context_lang + context_info

    print("Query: {}\nContext:\n".format(query_joined) + context)

    return context


def cag_command(*query):
    """
    Experimental switch that toggles the use of 'Cache-augmented Generation'
    over 'Retrieval-Augmented Generation'.
    """
    global chat_use_CAG

    chat_use_CAG = not chat_use_CAG
    print("TOGGLE: 'use_CAG' -> {}.".format(chat_use_CAG))


def generate_command(*query):
    """
    Calls the chat model to generate based on context generated from
    the query, and with a prompt specified in advance.
    """
    global last_cartridge_id, chat_prompt

    if not prompter.check_model_server():
        print("ERROR: Can't connect to model's server.")
        return

    print("===\n\n")
    context = context_command(*query)
    result = prompter.generate_response(chat_prompt, context)

    print("Generated:\n" + result)
    print("\n\n===")


def exit_command():
    """
    Called before the program exits.
    Could be used for deinitializing/cleanup.
    """
    pass


def help_command():
    """
    lists the available commands.
    """
    global available_commands

    for c in available_commands.keys():
        print(f"\t- {c}")


available_commands = {
    "add": add_command,
    "compile": compile_command,
    "help": help_command,
    "list": help_command,
    "load": load_command,
    "set": set_command,
    "unload": unload_command,
    "context": context_command,
    "generate": generate_command,
    "cag": cag_command,
    "exit": exit_command
}


def main():
    print("""
    Welcome!
    Type 'help' to list available commands.
    """)

    while True:
        cmd_inputs = input("> ").strip()
        parts = cmd_inputs.split()

        command = parts[0].lower()
        arguments = parts[1:]

        if command in available_commands.keys():
            available_commands[command](*arguments)
            if command == "exit":
                break
        else:
            print("Command '{}' not found.".format(command))


if __name__ == "__main__":
    main()
