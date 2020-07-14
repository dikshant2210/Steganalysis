import torch
from sklearn.metrics import f1_score
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

class_count = torch.sum((torch.tensor(train_data.labels) == 0).long())
print("Class ratio: {}/{}".format(class_count, len(train_data) - class_count))
weights = 1. / torch.tensor([class_count, len(train_data) - class_count]).float()
criterion = torch.nn.CrossEntropyLoss(weight=weights.cuda())
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

max_score = 0.

for epoch in range(1, n_epochs + 1):

    train_loss = 0.0
    valid_loss = 0.0

    model.train()
    current_batch = 0
    train_correct = 0.
    for data, target in train_loader:
        current_batch += 1
        if train_on_gpu:
            data, target = data.cuda(), target.cuda()

        optimizer.zero_grad()
        output, _ = model(data)

        _, pred = torch.max(output, 1)
        train_correct += torch.sum(pred == target)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        if current_batch % logging_frequency == 0:
            print("Epoch: {}\tTraining Loss: {:.6f}".format(epoch, train_loss / current_batch))

    model.eval()
    val_correct = 0.
    val_pred = list()
    val_true = list()
    for data, target in valid_loader:
        if train_on_gpu:
            data, target = data.cuda(), target.cuda()

        output, _ = model(data)
        _, pred = torch.max(output, 1)
        for x, t in zip(pred, target):
            val_pred.append(x.item())
            val_true.append(t.item())
        val_correct += torch.sum(pred == target)
        loss = criterion(output, target)
        valid_loss += loss.item()

    train_loss = train_loss / (len(train_data) / batch_size)
    valid_loss = valid_loss / (len(valid_data) / batch_size)

    score = f1_score(val_true, val_pred)
    print('Epoch: {} \tTraining Loss: {:.6f} \t Training Accuracy: {:.3f} \tValidation Loss: {:.6f} '
          '\t Validation Accuracy: {:.3f} \tF1 Score: {:.4f}'.format(epoch, train_loss,
                                                                     train_correct / len(train_data), valid_loss,
                                                                     val_correct / len(valid_data), score))
    if score > max_score:
        file_name = "weights/srnet_{:.4f}.pt".format(score)
        print("Saving the model to {}".format(file_name))
        torch.save(model.state_dict(), file_name)
        max_score = score
