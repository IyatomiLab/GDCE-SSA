import pickle


def load_data(path):
    with open(path, "rb") as bf:
        data = pickle.load(bf)

    return data
