import numpy as np
from abc import ABC, abstractmethod
from PIL import ImageFont
from PIL import Image
from PIL import ImageDraw
from torch.utils.data import Dataset
from util.load_data import load_data


class Char2Img(ABC, Dataset):
    def __init__(self, chars_path, font_name):
        self.chars = load_data(chars_path)
        self.font_name = font_name
        self.font_size = 64

        self.font = ImageFont.truetype(
            font=font_name, size=int(self.font_size * 0.9), encoding="utf-8"
        )

        self.font_img_dict = {
            char: self.resize_font_img(self.char_to_font_img(char))
            for char in self.chars
        }

    @abstractmethod
    def __len__(self):
        pass

    @abstractmethod
    def __getitem__(self, i):
        pass

    def char_to_font_img(self, char):
        img_size = np.ceil(np.array(self.font.getsize(char)) * 1.1).astype(int)
        img = Image.new("L", tuple(img_size), "black")
        draw = ImageDraw.Draw(img)
        text_offset = (img_size - self.font.getsize(char)) / 2
        draw.text(text_offset, char, font=self.font, fill="#fff")
        return img

    def resize_font_img(self, img):
        arr = np.asarray(img)
        r, c = np.where(arr != 0)
        r.sort()
        c.sort()

        if len(r) == 0:
            b = np.zeros((self.font_size, self.font_size))
        else:
            top = r[0]
            bottom = r[-1]
            left = c[0]
            right = c[-1]

            c_arr = arr[top:bottom, left:right]
            b = np.zeros((self.font_size, self.font_size), dtype=c_arr.dtype)
            r_offset = int((b.shape[0] - c_arr.shape[0]) / 2)
            c_offset = int((b.shape[1] - c_arr.shape[1]) / 2)
            b[
                r_offset : r_offset + c_arr.shape[0],
                c_offset : c_offset + c_arr.shape[1],
            ] = c_arr

        return b
