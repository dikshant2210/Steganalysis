import numpy as np
import torch
from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data.sampler import WeightedRandomSampler
from utils.config import *
np.random.seed(46)


transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.Resize(size=(256, 256)),
    transforms.Grayscale(1),
    transforms.ToTensor(),
    # transforms.Normalize(0.5, 0.5)
    ])

data = datasets.ImageFolder('data', transform=transform)
num_data = len(data)
indices_data = list(range(num_data))
np.random.shuffle(indices_data)
split_tt = int(np.floor(test_size * num_data))
train_idx, test_idx = indices_data[split_tt:], indices_data[:split_tt]

num_train = len(train_idx)
indices_train = list(range(num_train))
np.random.shuffle(indices_train)
split_tv = int(np.floor(valid_size * num_train))
train_new_idx, valid_idx = indices_train[split_tv:], indices_train[:split_tv]

# weights = torch.DoubleTensor([0.5, 0.16, 0.17, 0.17])
# train_sampler = WeightedRandomSampler(weights, batch_size)
test_sampler = SubsetRandomSampler(test_idx)
valid_sampler = SubsetRandomSampler(valid_idx)

print(data)
train_loader = torch.utils.data.DataLoader(data, batch_size=batch_size, num_workers=1, shuffle=True)
valid_loader = torch.utils.data.DataLoader(data, batch_size=batch_size, sampler=valid_sampler, num_workers=1)
test_loader = torch.utils.data.DataLoader(data, sampler = test_sampler, batch_size=batch_size, num_workers=1)
classes = [0, 1]
