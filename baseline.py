# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load
from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
from torchvision import datasets, models, transforms
from torch.utils.data.sampler import SubsetRandomSampler
import matplotlib.pyplot as plt
import time
import os
import copy
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import cv2
import random
from sklearn.metrics import f1_score


plt.ion()

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# TODO:// Code need to be formatted and commented properly 

def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001) 



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
        #image = torch.mean(image, axis=0, keepdim=True)
        return image, target

    def __len__(self):
        return len(self.images)


def train_model(model, criterion, optimizer, scheduler, num_epochs=25, val_pred = [],val_true=[]):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    for x, t in zip(preds, labels):
                        val_pred.append(x.item())
                        val_true.append(t.item())
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))
    score = f1_score(val_true, val_pred)
    print('Val F1 Score: {:4f}'.format(score))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, score 

def visualize_model(model, num_images=6):
    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure()

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloaders['val']):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = plt.subplot(num_images//2, 2, images_so_far)
                ax.axis('off')
                ax.set_title('predicted: {}'.format(class_names[preds[j]]))
                imshow(inputs.cpu().data[j])

                if images_so_far == num_images:
                    model.train(mode=was_training)
                    return
        model.train(mode=was_training)

def test_model(model, criterion, test_loader):
    model.eval()
    i = 1 
    print(len(test_loader))
    test_loss = 0.0
    class_correct = list(0. for i in range(2))
    class_total = list(0. for i in range(2))
    train_on_gpu = True
    test_pred = list()
    test_true = list()
    #import pdb;pdb.set_trace()
    for data, target in test_loader:
#         i=i+1
#         if len(target)!=batch_size:
#             continue

        # move tensors to GPU if CUDA is available
        data, target = data.cuda(), target.cuda()
        target[target != 0] = 1
        # forward pass: compute predicted outputs by passing inputs to the model
        output = model(data)
        # calculate the batch loss
        loss = criterion(output, target)
        # update test loss 
        test_loss += loss.item()*data.size(0)
        # convert output probabilities to predicted class
        _, pred = torch.max(output, 1)    
        # compare predictions to true label
        for x, t in zip(pred, target):
            test_pred.append(x.item())
            test_true.append(t.item())
        
        
        correct_tensor = pred.eq(target.data.view_as(pred))
        correct = np.squeeze(correct_tensor.numpy()) if not train_on_gpu else np.squeeze(correct_tensor.cpu().numpy())
        # calculate test accuracy for each object class
    #     print(target)
        
        for i in range(len(data)):       
            label = target.data[i]
            class_correct[label] += correct[i].item()
            class_total[label] += 1
    test_loss = test_loss/len(test_loader.dataset)
    
    print('Test Loss: {:.6f}\n'.format(test_loss))
    for i in range(2):
        if class_total[i] > 0:
            print('Test Accuracy of %5s: %2d%% (%2d/%2d)' % (
                class_names[i], 100 * class_correct[i] / class_total[i],
                np.sum(class_correct[i]), np.sum(class_total[i])))
        else:
            print('Test Accuracy of %5s: N/A (no training examples)' % (class_names[i]))

    print('\nTest Accuracy (Overall): %2d%% (%2d/%2d)' % (
        100. * np.sum(class_correct) / np.sum(class_total),
        np.sum(class_correct), np.sum(class_total)))
    score = f1_score(test_true, test_pred)
    print('TEST F1 Score {:.4f}'.format(score))


batch_size = 32
device = "cuda:0"
class_names = [0,1]

# percentage of training set to use as validation
test_size = 0.3
valid_size = 0.1

# number of epochs to train the model
n_epochs = 10

logging_frequency = 5000

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

train_data = SteganalysisBinary(root_path='/data/train', transforms=train_transform)
valid_data = SteganalysisBinary(root_path='/data/valid', transforms=valid_transform)
test_data = SteganalysisBinary(root_path='/data/test', transforms=valid_transform)

print("Training size:{}".format(len(train_data)))
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(valid_data, batch_size=batch_size)
test_loader = DataLoader(test_data, batch_size=batch_size)
dataloaders = {}
dataloaders["train"] = train_loader
dataloaders["val"] = valid_loader
dataloaders["test"] = test_loader
dataset_sizes = {'train': len(train_data), 'val': len(valid_data), 'test': len(test_loader)}


model_conv = torchvision.models.resnet18(pretrained=True)
for param in model_conv.parameters():
    param.requires_grad = False

# Parameters of newly constructed modules have requires_grad=True by default
num_ftrs = model_conv.fc.in_features
model_conv.fc = nn.Linear(num_ftrs, 2)

model_conv = model_conv.to(device)


criterion = nn.CrossEntropyLoss()

# Observe that only parameters of final layer are being optimized as
# opposed to before.
optimizer_conv = optim.SGD(model_conv.fc.parameters(), lr=0.001, momentum=0.9)
optimizer = optimizer_conv
# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=2, gamma=0.1)

model_conv, score = train_model(model_conv, criterion, optimizer_conv,
                         exp_lr_scheduler, num_epochs=n_epochs)
file_name = "weights/baseline_{:.4f}.pt".format(score)
print("Saving the model to {}".format(file_name))
torch.save(model_conv.state_dict(), file_name)
test_model(model_conv, criterion, dataloaders["test"])


















