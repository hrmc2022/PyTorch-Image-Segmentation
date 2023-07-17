import argparse
import cv2
import json
import matplotlib.pyplot as plt
from model.UNet import UNet
import numpy as np
import os
from pathlib import Path
import torch
import torchvision.transforms as T
import torch.nn as nn
from utils import imread, imwrite

def color_map(N=256, normalized=False):
    def bitget(byteval, idx):
        return ((byteval & (1 << idx)) != 0)

    dtype = 'float32' if normalized else 'uint8'
    cmap = np.zeros((N, 3), dtype=dtype)
    for i in range(N):
        r = g = b = 0
        c = i
        for j in range(8):
            r = r | (bitget(c, 0) << 7-j)
            g = g | (bitget(c, 1) << 7-j)
            b = b | (bitget(c, 2) << 7-j)
            c = c >> 3

        cmap[i] = np.array([b, g, r])

    cmap = cmap/255 if normalized else cmap
    return cmap

def main(config):
    # Color
    num_classes = 22
    cmap = color_map()
    color = [cmap[i] for i in range(num_classes)]

    labels = ['background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor', 'void']

    # Outdir
    outdir = Path(config["outdir"])
    os.makedirs(outdir, exist_ok=True)
    
    # Device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Image Load
    if os.path.exists(config["image"]):
        orig_image = imread(config["image"])
        h, w, c = orig_image.shape

        if orig_image is None:
            print("[Error] The image is not opened. :{}".format(config["image"]))
            exit()
    else:
        raise FileNotFoundError
    # print(image)
    # cv2.imshow("image", image)
    # cv2.waitKey()
    image = torch.tensor(orig_image, dtype=torch.float32).permute(2, 0, 1)
    
    transform = T.Compose([
        T.Resize((256, 256))
    ])

    image = transform(image)
    image = image.unsqueeze(dim=0)

    # Model
    model = UNet(in_ch=image.shape[1], out_ch=22)
    model.load_state_dict(torch.load(config["model"]))
    model.eval()

    # Predict
    pred = model(image)

    # Reshape
    seg = torch.argmax(pred, dim=1).squeeze(dim=0)
    seg = torch.from_numpy(np.array(color))[seg]
    seg = seg.numpy().astype(np.uint8)
    seg = cv2.resize(seg, (h, w))

    # save
    imwrite(str(outdir / Path(config["image"]).name), orig_image)
    imwrite(str(outdir / (Path(config["image"]).stem + "_seg.png")), seg)

    # cv2.imshow("src", orig_image)
    # cv2.imshow("seg", seg)
    # cv2.waitKey()

    # Plot
    plot_image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
    plot_seg = cv2.cvtColor(seg, cv2.COLOR_BGR2RGB)
    display_list = [plot_image, plot_seg]
    title = ['Input Image', 'Predicted Mask']

    plt.figure(figsize=(15, 15))
    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i+1)
        plt.title(title[i])
        plt.imshow(display_list[i])
        plt.axis('off')
    # plt.show()
    plt.savefig(str(outdir / (Path(config["image"]).stem + "_plot.png")))



    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None)
    args = parser.parse_args()
    
    with open(args.config, mode="r") as f:
        config = json.load(f)

    main(config)
