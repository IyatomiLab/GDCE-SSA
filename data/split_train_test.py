import pickle
import numpy as np
from sklearn.model_selection import train_test_split


def load_pickle(path):
    with open(path, "rb") as bf:
        data = pickle.load(bf)
    return data


def save_pickle(path, data):
    with open(path, "wb") as f:
        pickle.dump(data, f)


if __name__ == "__main__":
    # data_name_list = ["newspaper", "livedoor"]
    data_name_list = ["livedoor"]
    for data_name in data_name_list:
        path = f"./pickle/{data_name}.pkl"
        data = load_pickle(path)

        X = np.array(
            [text for text in data["title" if data_name == "livedoor" else "body"]]
        )
        label = [
            category
            for category in data["label" if data_name == "livedoor" else "newspaper"]
        ]
        label_names = list(set(label))
        label_names.sort()
        y = np.array([label_names.index(label[i]) for i in range(len(label))])

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, stratify=y, test_size=0.2, random_state=42
        )
        num_class = len(label_names)

        train_data = {"data": list(zip(X_train, y_train)), "num_class": num_class}
        test_data = {"data": list(zip(X_test, y_test)), "num_class": num_class}

        save_pickle(f"./pickle/train_{data_name}.pkl", train_data)
        save_pickle(f"./pickle/test_{data_name}.pkl", test_data)
