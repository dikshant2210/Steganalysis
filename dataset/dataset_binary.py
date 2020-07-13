import os
import cv2
import random
import torch
from PIL import Image
from torch.utils.data import Dataset


class SteganalysisBinary(Dataset):
    def __init__(self, root_path, transforms):
        self.root_path = root_path
        self.transforms = transforms
        self.images = list()
        self.labels = list()

        folders = os.listdir(self.root_path)
        for f in folders:
            if f == 'Cover':
                files = os.listdir(os.path.join(self.root_path, f))
                for file in files:
                    self.images.append(os.path.join(self.root_path, f, file))
                    self.labels.append(0)
            else:
                files = os.listdir(os.path.join(self.root_path, f))
                for file in files:
                    self.images.append(os.path.join(self.root_path, f, file))
                    self.labels.append(1)

        temp = list(zip(self.images, self.labels))
        random.shuffle(temp)
        self.images, self.labels = zip(*temp)

    def __getitem__(self, item):
        image_path = self.images[item]
        target = self.labels[item]
        image = Image.fromarray(cv2.imread(image_path, cv2.IMREAD_COLOR))
        image = self.transforms(image)
        image = torch.mean(image, axis=0, keepdim=True)
        return image, target

    def __len__(self):
        return len(self.images)
