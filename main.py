import os
from typing import List
import core.vectorizer as vectorizer
import core.cartridge_loader as loader

def vectorize_command(path: str = None, output_dir: str = None):
    path = path or os.path.join(
        os.getcwd(), "data", "kr_en", "korean_travel_basics.json")
    output_dir = output_dir or os.path.join(os.getcwd(), "cartridges", "kr_en")

    os.makedirs(output_dir, exist_ok=True)

    print("using path: {}".format(path))

    data = vectorizer.load_data_json(path)
    text_chunks = vectorizer.split_text_into_chunks(data)
    embeddings = vectorizer.generate_embeddings(text_chunks)

    vectorizer.store_vectors(embeddings, text_chunks, output_dir)

    print("Vectorized file!")

def load_command(path: str = None, output_dir: str = None):
    path = path or os.path.join(
        os.getcwd(), "cartridges", "kr_en.cart")
    output_dir = output_dir or os.path.join(os.getcwd(), "temp")
    os.makedirs(output_dir, exist_ok=True)

    print("using path: {}".format(path))

    loader.cartridge_load(path, output_dir)

    print("Loaded cartridge!")

def help_command():
    print("placeholder text.")
    pass

def get_commands() -> List[str]:
    result = {
        "vectorize": vectorize_command,
        "help": help_command,
        "load": load_command
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
