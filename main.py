import os
from typing import List
import core.util as util
import core.vectorizer as vectorizer
import core.loader as loader
import core.prompter as prompter

current_cartridge_id = ""
cache_dir_path = os.path.join(os.getcwd(), "cache")
cartridge_dir_path = os.path.join(os.getcwd(), "cartridges")
data_dir_path = os.path.join(os.getcwd(), "data")

# NOTE(liam): path is expected


def compile_command(cartridge_id: str = None, file_name: str = None):
    if cartridge_id is None or file_name is None:
        print("'compile' expects id and file name. i.e., 'compile <cartridge_id> <file_name>'.")
        return

    data_file_path = os.path.join(data_dir_path, cartridge_id, file_name)
    output_dir = os.path.join(cartridge_dir_path, cartridge_id)

    os.makedirs(output_dir, exist_ok=True)

    print("using path: '{}'.".format(data_file_path))

    data = vectorizer.load_data_json(data_file_path)
    text_chunks = vectorizer.split_text_into_chunks(data)
    embeddings = util.generate_embeddings(text_chunks)

    vectorizer.store_vectors(embeddings, text_chunks, output_dir)

    print("compiled vector file at '{}'.".format(
        os.path.join(cartridge_dir_path, cartridge_id + '.cart')))


def load_command(cartridge_id: str = None, cartridge_path: str = None, output_dir: str = cache_dir_path):
    global current_cartridge_id

    cartridge_path = cartridge_path or os.path.join(
        cartridge_dir_path, cartridge_id + '.cart')
    os.makedirs(output_dir, exist_ok=True)

    print("using path: '{}'.".format(cartridge_path))

    current_cartridge_id = loader.cartridge_load(cartridge_path, output_dir)

    print("Loaded cartridge! Data dumped in '{}'.".format(data_dir_path))


def set_command(cartridge_id: str = None):
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
    current_cartridge_id = cartridge_id

def context_command(query: str = "", cartridge_id: str = None):
    global current_cartridge_id

    cartridge_id = cartridge_id or current_cartridge_id
    if cartridge_id is None:
        print("cartridge not specified.")

    cartridge_dir_path = os.path.join(cache_dir_path, cartridge_id)

    result = prompter.perform_rag(query, os.path.join(cartridge_dir_path, "faiss.index"),
                                  os.path.join(cartridge_dir_path, "chunks.txt"))

    print("Context generated:\n" + result)


def help_command():
    global available_commands

    for c in available_commands.keys():
        print(f"\t- {c}")

    print("\t- exit")


available_commands = {
    "compile": compile_command,
    "help": help_command,
    "list": help_command,
    "load": load_command,
    "set": set_command,
    "context": context_command
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

        if command == "exit":
            break
        elif command in available_commands.keys():
            available_commands[command](*arguments)
        else:
            print("Command '{}' not found.".format(command))


if __name__ == "__main__":
    main()
