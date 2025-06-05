import os
from typing import List
import core.vectorizer as vectorizer
import core.cartridge_loader as loader
import core.prompter as prompter

current_cartridge_id = None
cache_path = os.path.join(os.getcwd(), "cache")
cartridge_path = os.path.join(os.getcwd(), "cartridges")

# NOTE(liam): path is expected
def vectorize_command(data_file_path: str, cartridge_id: str = None):
    cartridge_id = cartridge_id or "default"
    output_dir = os.path.join(cartridge_path, cartridge_id)

    os.makedirs(output_dir, exist_ok=True)

    print("using path: {}".format(data_file_path))

    data = vectorizer.load_data_json(data_file_path)
    text_chunks = vectorizer.split_text_into_chunks(data)
    embeddings = vectorizer.generate_embeddings(text_chunks)

    vectorizer.store_vectors(embeddings, text_chunks, output_dir)

    print("Vectorized file!")


def load_command(path: str = None, output_dir: str = None):
    global current_cartridge_id

    path = path or os.path.join(
        os.getcwd(), "cartridges", "kr_en.cart")
    output_dir = output_dir or cache_path
    os.makedirs(output_dir, exist_ok=True)

    print("using path: {}".format(path))

    current_cartridge_id = loader.cartridge_load(path, output_dir)

    print("Loaded cartridge!")


def context_command(query: str = "", cartridge_id: str = ""):
    global current_cartridge_id

    cartridge_id = cartridge_id or current_cartridge_id
    cartridge_path = os.path.join(cache_path, cartridge_id)

    result = prompter.perform_rag(query, os.path.join(cartridge_path, "faiss.index"),
                                  os.path.join(cartridge_path, "chunks.txt"))

    print("Context generated:\n" + result)


def help_command():
    print("placeholder text.")
    pass


def get_commands() -> List[str]:
    result = {
        "vectorize": vectorize_command,
        "help": help_command,
        "load": load_command,
        "context": context_command
    }

    return result


def main():
    print("""
    Welcome!
    Type 'help' to list available commands.
    """)

    commands = get_commands()
    while True:
        cmd_inputs = input("> ").strip()
        parts = cmd_inputs.split()

        command = parts[0].lower()
        arguments = parts[1:]

        if command == "exit":
            break
        elif command == "list":
            i = 0
            for c in commands.keys():
                print(f"\t{i + 1}. {c}")
                i += 1

            print(f"\t{i + 1}. list\n\t{i + 2}. exit")

        elif command in commands.keys():
            commands[command](*arguments)
        else:
            print("Command '{}' not found.".format(command))


if __name__ == "__main__":
    main()
