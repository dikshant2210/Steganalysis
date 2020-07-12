import torch
import numpy as np
from torch.utils.data import DataLoader
from models.srnet.model import Srnet
import torchvision.transforms as transforms
from utils.config import *
from dataset import SteganalysisBinary


train_on_gpu = torch.cuda.is_available()
model = Srnet()

if train_on_gpu:
    model.cuda()

train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.Resize(size=(256, 256)),
    transforms.ToTensor()])

valid_transform = transforms.Compose([
    transforms.Resize(size=(256, 256)),
    transforms.ToTensor()])

train_data = SteganalysisBinary(root_path='data/train', transforms=train_transform)
valid_data = SteganalysisBinary(root_path='data/valid', transforms=valid_transform)

train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(valid_data, batch_size=batch_size)

criterion = torch.nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.003, momentum=0.9)

valid_loss_min = np.Inf  # track change in validation loss

for epoch in range(1, n_epochs+1):

    train_loss = 0.0
    valid_loss = 0.0

    model.train()
    current_batch = 0
    for data, target in train_loader:
        print(data, target)
        current_batch += 1
        if train_on_gpu:
            data, target = data.cuda(), target.cuda()

        optimizer.zero_grad()
        output = model(data)

        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        if current_batch % logging_frequency == 0:
            print("Epoch: {}\tTraining Loss: {}".format(epoch, loss.item()))

    model.eval()
    for data, target in valid_loader:
        if train_on_gpu:
            data, target = data.cuda(), target.cuda()

        output = model(data)

        loss = criterion(output, target)
        valid_loss += loss.item()

    train_loss = train_loss / (len(train_data) / batch_size)
    valid_loss = valid_loss / (len(valid_data) / batch_size)

    print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
        epoch, train_loss, valid_loss))
