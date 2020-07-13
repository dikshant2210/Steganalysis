import torch
import numpy as np
from torch.utils.data import DataLoader
from models.srnet.model import Srnet, SRNET
import torchvision.transforms as transforms
from utils.config import *
from dataset import SteganalysisBinary


train_on_gpu = torch.cuda.is_available()
model = SRNET()

if train_on_gpu:
    model.cuda()

ckpt = torch.load('weights/SRNet_pretrained.pt')
model.load_state_dict(ckpt['model_state_dict'])

train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.Resize(size=(256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))])

valid_transform = transforms.Compose([
    transforms.Resize(size=(256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))])

train_data = SteganalysisBinary(root_path='data/train', transforms=train_transform)
valid_data = SteganalysisBinary(root_path='data/valid', transforms=valid_transform)

print("Training size:{}".format(len(train_data)))

train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(valid_data, batch_size=batch_size)

# criterion = torch.nn.BCELoss()
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

valid_loss_min = np.Inf  # track change in validation loss

for epoch in range(1, n_epochs+1):

    train_loss = 0.0
    valid_loss = 0.0

    model.train()
    current_batch = 0
    for data, target in train_loader:
        current_batch += 1
        if train_on_gpu:
            data, target = data.cuda(), target.cuda()

        optimizer.zero_grad()
        output = model(data)

        # output = output.view(-1)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        if current_batch % logging_frequency == 0:
            print("Epoch: {}\tTraining Loss: {:.6f}".format(epoch, train_loss / current_batch))

    model.eval()
    correct = 0.
    for data, target in valid_loader:
        if train_on_gpu:
            data, target = data.cuda(), target.cuda()

        output = model(data)
        _, pred = torch.max(output, 1)
        correct += torch.sum(pred == target)
        loss = criterion(output, target)
        valid_loss += loss.item()

    train_loss = train_loss / (len(train_data) / batch_size)
    valid_loss = valid_loss / (len(valid_data) / batch_size)

    print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f} \t Validation Accuracy: {:.3f}'.format(
        epoch, train_loss, valid_loss, correct / len(train_data)))
