import pickle
import pandas as pd


with open("ja_chars.csv", "r") as f:
    data = pd.read_csv(f, names=["char"])
    data = data.values.flatten().tolist()


with open("../pickle/ja_chars.pkl", "wb") as wf:
    pickle.dump(data, wf)
