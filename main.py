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

# NOTE(liam): path is expected


def add_command(cartridge_id: str = None, *file_names: List[str]):
    global last_cartridge_id, current_f_index

    cartridge_id = cartridge_id or last_cartridge_id
    if cartridge_id is None:
        print("'add' expects id i.e., 'compile <cartridge_id>'.")
        return
    last_cartridge_id = cartridge_id

    output_dir = os.path.join(cartridge_dir_path, cartridge_id)
    os.makedirs(output_dir, exist_ok=True)

    for file_name in file_names:
        data_file_path = os.path.join(data_dir_path, cartridge_id, file_name)
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

    shutil.rmtree(os.path.join(cache_dir_path, cartridge_id))

    print("Unloaded '{}' from cache!".format(cartridge_id))


def context_command(*query):
    global last_cartridge_id

    cartridge_id = last_cartridge_id
    if cartridge_id is None:
        print("ERROR: cartridge id was not previously set via 'load'.")
        return

    cartridge_dir_path = os.path.join(cache_dir_path, cartridge_id)

    query = " ".join(query)

    result = prompter.perform_rag(query, os.path.join(cartridge_dir_path, "faiss.index"),
                                  os.path.join(cartridge_dir_path, "chunks.txt"))

    print("Context generated:\n" + result)


def generate_command():
    pass


def exit_command():
    pass


def help_command():
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
