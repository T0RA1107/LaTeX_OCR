import torch
from torch.utils.data import Dataset
from torchvision.io import read_image, ImageReadMode
import torch.nn.functional as F
import os
import random

random.seed(42)

class LaTeXDataset(Dataset):
    def __init__(
        self,
        data_path_lst,
        all_images_dir,
        all_formulae_lst,
        vocab_txt,
        max_seq_length,
        input_shape
    ):
        """Generates a torch dataset by using preprocessed LaTeX images and formulae

        Args:
            data_path_lst (str):
                preprocessed data list. a set of image file name and formula id for a line.
            all_images_dir (str):
                directory name of LaTeX images.
            all_formulae_lst (str):
                pre-tokenized formulae list. i-th formula stands for the one of id `i`(0-indexed).
            vocab_txt (str):
                text file name of vocabulary list.
            max_seq_length (int):
                max length of LaTeX scription.
            input_shape (int, int):
                input image shape. image is to be zero-padded into this shape.
        """
        self.max_seq_length = max_seq_length
        self.input_shape = input_shape
        self.images = []
        self.formula_ids = []
        self.all_images_dir = all_images_dir
        with open(data_path_lst, "r", newline="\n") as f:
            for s in f.readlines():
                img, formula_id = s.split(" ")
                self.images.append(img)
                self.formula_ids.append(int(formula_id))

        self.all_formulae = list(open(all_formulae_lst, "r", newline="\n"))
        self.word2id = {
            "[BOS]": 0,
            "[EOS]": 1,
            "[PAD]": 2,
            "[UNK]": 3
        }
        word_begin = len(self.word2id)
        with open(vocab_txt, "r", newline="\n") as f:
            for i, word in enumerate(f.readlines()):
                self.word2id[word.rstrip()] = word_begin + i

        self.vocab_size = len(self.word2id)

    def _get_id(self, word):
        if word in self.word2id:
            return self.word2id[word]
        else:
            return self.word2id["[UNK]"]

    def __getitem__(self, index):
        img = self.images[index]
        formula_id = self.formula_ids[index]

        # convert image into Tensor
        img_tensor = read_image(path=os.path.join(self.all_images_dir, img), mode=ImageReadMode.GRAY).to(torch.float32) / 255.0
        # TODO: 形状を固定のものに変換する
        padding_base = torch.zeros((1, *self.input_shape), dtype=torch.float32)
        assert img_tensor.shape[1] <= self.input_shape[0] and img_tensor.shape[2] <= self.input_shape[1], print("input shape: ", self.input_shape, "\nimage shpae: ", img_tensor.shape)
        rh = random.randint(0, self.input_shape[0] - img_tensor.shape[1])
        rw = random.randint(0, self.input_shape[1] - img_tensor.shape[2])
        padding_base[:, rh:rh + img_tensor.shape[1], rw:rw + img_tensor.shape[2]] += img_tensor[:, :, :]
        img_tensor = padding_base

        # convert formula into Tensor
        formula = ("[BOS] " + self.all_formulae[formula_id].rstrip("\n") + " [EOS]").split(" ")
        formula2word_id = []
        l = len(formula)
        for i in range(self.max_seq_length):
            if i < l:
                formula2word_id.append(self._get_id(formula[i]))
            else:
                formula2word_id.append(self.word2id["[PAD]"])
        formula_tensor = F.one_hot(torch.LongTensor(formula2word_id), num_classes=self.vocab_size).to(torch.float32)
        # TODO: KaTeXで上手く変換できないものの扱い(\lbrace(≈\{)など代替表現があるならそれに変換したい)

        return img_tensor, formula_tensor

    def __len__(self):
        return len(self.images)
