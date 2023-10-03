import sys
import torch
from torchvision.io import read_image, ImageReadMode
import numpy as np
import re

import time

model_weight_path = "/Users/tora/Desktop/DL/LaTeX_OCR/check.pth"
vocab_txt = "/Users/tora/Desktop/DL/LaTeX_OCR/data/lukas-data/all2/vocab.txt"
device = "mps"

vocab = [
    "[BOS]",
    "[EOS]",
    "[PAD]",
    "[UNK]"
]
word_begin = len(vocab)
with open(vocab_txt, "r", newline="\n") as f:
    for word in f.readlines():
        vocab.append(word.rstrip())
vocab = np.array(vocab)

model = torch.load(model_weight_path, map_location=device)
model.eval()

def preprocess(img_path):
    img = read_image(path=img_path, mode=ImageReadMode.GRAY).to(torch.float32) / 255.0
    padding_base = torch.zeros((1, model.H, model.W), dtype=torch.float32)
    rh = (model.H - img.shape[1]) // 2
    rw = (model.W - img.shape[2]) // 2
    padding_base[:, rh:rh + img.shape[1], rw:rw + img.shape[2]] += img[:, :, :]
    img_tensor = padding_base[None, :, :, :].permute(0, 2, 3, 1)
    return img_tensor

def inference(img):
    img = img.to(device)
    out = model.generate(img).cpu()
    print(out.shape)
    out = torch.squeeze(out, dim=1)
    print(out)
    out = vocab[out.numpy()].reshape(-1)
    out = [re.sub("\[...\]", "", s) for s in out]
    return "".join(map(str, out))

if __name__ == "__main__":
    img_path = sys.argv[1]
    print(img_path)
    if img_path[-4:] == ".png":
        st_time = time.time()
        img_tensor = preprocess(img_path)
        print(img_tensor.shape)
        res = inference(img_tensor)
        print("output:\n", res)
        end_time = time.time()
        print("inference time: {} (s)".format(int(end_time - st_time)))
