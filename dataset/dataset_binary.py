import os
import cv2
from torch.utils.data import Dataset


class SteganalysisBinary(Dataset):
    def __init__(self, root_path, transforms):
        super.__init__()
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
        self.labels = self.labels[:1000]
        self.root_path = self.images[:1000]

    def __getitem__(self, item):
        image_path = self.images[item]
        target = self.labels[item]
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        image = self.transforms(image)
        return image, target

    def __len__(self):
        return len(self.images)
