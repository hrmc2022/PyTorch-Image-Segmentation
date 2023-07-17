import os
import numpy as np
from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image
import torch
import torchvision
import torchvision.transforms as T

class VOCDataset(Dataset):
    def __init__(self, files, image_dir, label_dir, height=512, width=512):
        self.files = files
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.height = height
        self.width = width

    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        image_path = self.image_dir / (self.files[idx] + ".jpg")
        image = Image.open(str(image_path)).convert("RGB").resize((self.width, self.height), Image.ANTIALIAS)
        image = np.asarray(image)
        image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1)

        label_path = self.label_dir / (self.files[idx] + ".png")        
        label = Image.open(str(label_path)).resize((self.width, self.height), Image.ANTIALIAS)
        label = torch.tensor(list(map(int, label.getdata())), dtype=torch.int64).reshape((label.size[1], label.size[0]))
        label = torch.where(label > 20, 21, label)
        label = torch.nn.functional.one_hot(label, num_classes=22)
        label = label.to(torch.float32).permute(2, 0, 1)

        return image, label

def get_files(file, image_dir, label_dir):
    files = []
    with open(file, mode="r") as f:

        for line in f:
            if line != "\n":
                line = line.rstrip("\n")
                if os.path.exists(image_dir / (line + ".jpg")) and os.path.exists(label_dir / (line + ".png")):
                    files.append(line.rstrip("\n"))
    return files

def get_VOC_dataset(data_dir="../data", height=512, width=512):
    data_dir = Path(data_dir)

    image_set_dir = data_dir / "VOCdevkit" / "VOC2007" / "ImageSets" / "Segmentation"
    image_dir = data_dir / "VOCdevkit" / "VOC2007" / "JPEGImages"
    label_dir = data_dir / "VOCdevkit" / "VOC2007" / "SegmentationClass"

    train_file = image_set_dir / "train.txt"
    val_file = image_set_dir / "val.txt"

    train_files = get_files(train_file, image_dir, label_dir)
    val_files = get_files(val_file, image_dir, label_dir)

    train_dataset = VOCDataset(files=train_files, image_dir=image_dir, label_dir=label_dir, height=height, width=width)
    val_dataset = VOCDataset(files=val_files, image_dir=image_dir, label_dir=label_dir, height=height, width=width)
    return train_dataset, val_dataset

if __name__ == "__main__":
    train_dataset, val_dataset = get_VOC_dataset()
    image, label = train_dataset[0]
    print(image.shape)
    print(label.shape)
    label = label.permute(1, 2, 0)

    # for i in range(label.shape[0]):
    #     for j in range(label.shape[1]):
    #         if label[i, j, 0] != 1:
    #             print(label[i, j, :])