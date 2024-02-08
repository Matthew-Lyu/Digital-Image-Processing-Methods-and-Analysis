import os, sys
import torch.utils.data as data
from PIL import Image

# 封装数据

class VeriDataset(data.Dataset):
    def __init__(self, data_dir, list, data_transform=None):
        super(VeriDataset, self).__init__()

        self.data_dir = data_dir
        self.data_transform = data_transform
        reader = open(list)
        lines = reader.readlines()
        self.names = []
        self.labels = []
        for line in lines:
            line = line.strip().split(' ')
            self.names.append(line[0])
            self.labels.append(line[1])

    def __getitem__(self, index):
        # For normalize

        img = Image.open(os.path.join(self.data_dir, self.names[index])).convert('RGB')  # convert gray to rgb
        target = int(self.labels[index])

        if self.data_transform != None:
            img = self.data_transform(img)

        return img, target

    def __len__(self):
        return len(self.names)
