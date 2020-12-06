import pickle
from pathlib import Path

data_parent_dir = Path("text")

category_path_list = [
    category_dir for category_dir in data_parent_dir.iterdir() if category_dir.is_dir()
]

title_list = []
text_list = []
label_list = []
for category_path in category_path_list:
    label = category_path.name
    for category_data_path in list(category_path.glob("*.txt")):
        if category_data_path.name == "LICENSE.txt":
            continue
        with open(category_data_path, "r") as f:
            sentences = f.readlines()

            title = sentences[2].strip().replace("\n", "").replace("\u3000", "")
            text = (
                "".join(sentences[3:]).strip().replace("\n", "").replace("\u3000", "")
            )

            title_list.append(title)
            text_list.append(text)
            label_list.append(label)

dataset = {
    "title": title_list,
    "text": text_list,
    "label": label_list,
}

with open("../pickle/livedoor.pkl", "wb") as bf:
    pickle.dump(dataset, bf)
