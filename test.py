import torch
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
from models.srnet.model import SRNET
import torchvision.transforms as transforms
from utils.config import *
from dataset import SteganalysisBinary
from utils import alaska_weighted_auc

train_on_gpu = torch.cuda.is_available()
model = SRNET()

if train_on_gpu:
    model.cuda()

ckpt = torch.load('weights/SRNet_pretrained.pt')
model.load_state_dict(ckpt['model_state_dict'])

test_transform = transforms.Compose([
    transforms.Resize(size=(256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))])

test_data = SteganalysisBinary(root_path='data/test', transforms=test_transform)
print("Testing size:{}".format(len(test_data)))

test_loader = DataLoader(test_data, batch_size=batch_size)

model.eval()
test_correct = 0.
test_pred = list()
test_true = list()
for data, target in test_loader:
    if train_on_gpu:
        data, target = data.cuda(), target.cuda()

    output = model(data)
    pred = output[:, 1]
    print(pred)
    for x, t in zip(pred, target):
        test_pred.append(x.item())
        test_true.append(t.item())
    test_correct += torch.sum(pred == target)

score = f1_score(test_true, test_pred)
auc_score = alaska_weighted_auc(test_true, test_pred)

print("F1-score: {:.4f}, Weighted AUC Score: {:.4f}".format(score, auc_score))
