"""Train CIFAR10 with PyTorch."""
import argparse
import os

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from timm import create_model

import models
from utils import progress_bar

parser = argparse.ArgumentParser(description="PyTorch CIFAR10 Training")
parser.add_argument("model", type=str, help="model name")
parser.add_argument("--lr", default=0.1, type=float, help="learning rate")
args = parser.parse_args()

CK_PATH = f"./checkpoint/{args.model}_{args.lr}_ckpt.pth"
device = "cuda" if torch.cuda.is_available() else "cpu"

# Data
print("==> Preparing data..")
transform_test = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ]
)

testset = torchvision.datasets.CIFAR10(
    root="./data", train=False, download=True, transform=transform_test
)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

classes = ("plane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck")

# Model
print("==> Building model..")
net = create_model(args.model)
net = net.to(device)
if device == "cuda":
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

# Load checkpoint.
print("==> Loading from checkpoint..")
assert os.path.isdir("checkpoint"), "Error: no checkpoint directory found!"
checkpoint = torch.load(CK_PATH)
net.load_state_dict(checkpoint["net"])
best_acc = checkpoint["acc"]
start_epoch = checkpoint["epoch"]

criterion = nn.CrossEntropyLoss()


def test():
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(
                batch_idx,
                len(testloader),
                "Loss: %.3f | Acc: %.3f%% (%d/%d)"
                % (test_loss / (batch_idx + 1), 100.0 * correct / total, correct, total),
            )


print(f"Epoch: {start_epoch}")
test()
