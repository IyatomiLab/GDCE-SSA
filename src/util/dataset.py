import random
import torch
import numpy as np
from base.char2img import Char2Img
from util.load_data import load_data


class JapaneseCharacters(Char2Img):
    def __init__(self, path):
        super().__init__(path["ja_chars"], path["font"])

    def __len__(self):
        return len(self.chars)

    def __getitem__(self, i):
        char = self.chars[i]
        char_img = self.font_img_dict[char]

        char_img = np.asarray(char_img, dtype=np.float32)
        char_img = np.expand_dims(char_img, 0)
        char_img = torch.from_numpy(char_img).div(255)

        return char_img, char


class Newspaper(Char2Img):
    def __init__(
        self,
        args,
        path,
        test=False,
        slide=False,
    ):
        super().__init__(path["ja_chars"], path["font"])

        data_path = path["newspaper"]["test"] if test else path["newspaper"]["train"]
        data = load_data(data_path)
        self.data = data["data"]
        self.num_class = data["num_class"]

        self.char_len = args.char_len
        self.test = test
        self.slide = slide

        if args.character_encoder == "CAE":
            self.char2embedding = load_data(path["char2embedding"])
        else:
            self.font_img_dict[" "] = self.resize_font_img(self.char_to_font_img(" "))

        self.process = self.test_process if self.slide else self.train_process

        self.character_encoder = args.character_encoder

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        document, label = self.data[i]
        text = self.process(document)

        return text, label

    def train_process(self, document):
        sentence = cutout(document, self.char_len)
        if self.character_encoder == "CAE":
            sentence = [
                self.char2embedding[char].flatten()
                if char in self.chars
                else np.zeros_like(self.char2embedding["a"].flatten())
                for char in sentence
            ]
            sentence = np.asarray(sentence, dtype=np.float32).T
            sentence = np.expand_dims(sentence, 1)
            sentence = torch.from_numpy(sentence)
        else:
            sentence = [
                self.font_img_dict[char]
                if char in self.chars
                else self.font_img_dict[" "]
                for char in sentence
            ]
            sentence = np.asarray(sentence, dtype=np.float32)
            sentence = np.expand_dims(sentence, 1)
            sentence = torch.from_numpy(sentence).div(255)

        return sentence

    def test_process(self, document):
        document_length = len(document)
        num_slide = 1
        num_subseq = document_length - self.char_len + 1

        if self.character_encoder == "CAE":
            document = np.asarray(
                [
                    self.char2embedding[char].flatten()
                    if char in self.chars
                    else np.zeros_like(self.char2embedding["a"].flatten())
                    for char in document
                ],
                dtype=np.float32,
            ).T
            document = np.expand_dims(document, 1)

            subseqs = [
                torch.from_numpy(
                    np.asarray(document[:, :, k : k + self.char_len], dtype=np.float32)
                )
                for k in range(0, num_subseq, num_slide)
            ]
        else:
            document = np.asarray(
                [
                    self.font_img_dict[char]
                    if char in self.chars
                    else self.font_img_dict[" "]
                    for char in document
                ],
                dtype=np.float32,
            )
            document = np.expand_dims(document, 1)
            document = torch.from_numpy(document).div(255)
            subseqs = [
                document[k : k + self.char_len] for k in range(0, num_subseq, num_slide)
            ]

        return subseqs


class Livedoor(Char2Img):
    def __init__(
        self,
        args,
        path,
        test=False,
        slide=False,
    ):
        super().__init__(path["ja_chars"], path["font"])

        data_path = path["livedoor"]["test"] if test else path["livedoor"]["train"]
        data = load_data(data_path)
        self.data = data["data"]
        self.num_class = data["num_class"]

        self.char_len = args.char_len
        self.test = test
        self.slide = slide

        if args.character_encoder == "CAE":
            self.char2embedding = load_data(path["char2embedding"])
        else:
            self.font_img_dict[" "] = self.resize_font_img(self.char_to_font_img(" "))

        self.character_encoder = args.character_encoder

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        sentence, label = self.data[i]
        sentence = cutout(sentence, self.char_len)

        if self.character_encoder == "CAE":
            sentence = [
                self.char2embedding[char].flatten()
                if char in self.chars
                else np.zeros_like(self.char2embedding["a"].flatten())
                for char in sentence
            ]
            sentence = np.asarray(sentence, dtype=np.float32).T
            sentence = np.expand_dims(sentence, 1)
            sentence = torch.from_numpy(sentence)
        else:
            sentence = [
                self.font_img_dict[char]
                if char in self.chars
                else self.font_img_dict[" "]
                for char in sentence
            ]
            sentence = np.asarray(sentence, dtype=np.float32)
            sentence = np.expand_dims(sentence, 1)
            sentence = torch.from_numpy(sentence).div(255)

        return sentence, label


def cutout(sentence, char_len):
    if len(sentence) < char_len:
        sentence += " " * (char_len - len(sentence))
    k = random.randint(0, len(sentence) - char_len)
    cutout_sentence = [c for c in sentence[k : k + char_len]]

    return cutout_sentence


DATASET = {"ja_chars": JapaneseCharacters, "newspaper": Newspaper, "livedoor": Livedoor}
