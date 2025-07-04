
# import core.vectorizer as vectorizer
import pandas as pd
import os

if __name__ == "__main__":
    path = os.path.join(os.getcwd(), "data", "kr_en", "2024_01.csv")

    print(path)

    df = pd.read_csv(path, encoding="utf-8")
    # vectorizer.load_data_csv(path)
