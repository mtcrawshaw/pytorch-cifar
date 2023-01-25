'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse
import pickle

import numpy as np

from models import *


parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--gpu', default=None, type=int, help='gpu index')
parser.add_argument('--epochs', default=200, type=int, help='epochs')
parser.add_argument('--optimizer', default="SGD", type=str, help='optimizer')
parser.add_argument('--beta1', default=0.9, type=float, help='adam beta1')
parser.add_argument('--beta2', default=0.999, type=float, help='adam beta2')
parser.add_argument('--eps', default=1e-8, type=float, help='adam epsilon')
parser.add_argument('--dataset', default='CIFAR10', type=str, help='dataset')
parser.add_argument('--name', default='experiment', type=str, help='name of experiment')
args = parser.parse_args()

device = f'cuda:{args.gpu}' if (torch.cuda.is_available() and args.gpu is not None) else 'cpu'

# Data
print('==> Preparing data..')
if args.dataset == "CIFAR10":
    dset = torchvision.datasets.CIFAR10
    channels = 3
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
elif args.dataset == "MNIST":
    dset = torchvision.datasets.MNIST
    channels = 1
    transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
else:
    raise NotImplementedError

trainset = dset(
    root='./data', train=True, download=True, transform=transform_train
)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=128, shuffle=True, num_workers=2)
testset = dset(
    root='./data', train=False, download=True, transform=transform_test
)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=100, shuffle=False, num_workers=2)


# Model
print('==> Building model..')
# net = VGG('VGG19')
net = ResNet18(channels=channels)
# net = PreActResNet18()
# net = GoogLeNet()
# net = DenseNet121()
# net = ResNeXt29_2x64d()
# net = MobileNet()
# net = MobileNetV2()
# net = DPN92()
# net = ShuffleNetG2()
# net = SENet18()
# net = ShuffleNetV2(1)
# net = EfficientNetB0()
# net = RegNetX_200MF()
# net = SimpleDLA()
net = net.to(device)
if 'cuda' in device:
    cudnn.benchmark = True

criterion = nn.CrossEntropyLoss()
optim_kwargs = {"lr": args.lr, "weight_decay": 5e-4}
if args.optimizer == "SGD":
    optim_kwargs["momentum"] = 0.9
    optimizer = optim.SGD(net.parameters(), **optim_kwargs)
elif args.optimizer == "Adam":
    optim_kwargs["betas"] = (args.beta1, args.beta2)
    optim_kwargs["eps"] = args.eps
    optimizer = optim.Adam(net.parameters(), **optim_kwargs)
else:
    raise NotImplementedError
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)


# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    avg_loss = train_loss / total
    avg_acc = 100. * correct / total
    print(f"Train loss: {avg_loss:.6f}, Train acc: {avg_acc:.3f}")

    return avg_loss, avg_acc


def test(epoch):
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

        avg_loss = test_loss / total
        avg_acc =  100. * correct / total
        print(f"Test loss: {avg_loss:.6f}, test acc: {avg_acc:.3f}")

    return avg_loss, avg_acc


# Run training.
train_losses = []
train_accs = []
test_losses = []
test_accs = []
for epoch in range(args.epochs):

    train_loss, train_acc = train(epoch)
    test_loss, test_acc = test(epoch)
    scheduler.step()

    train_losses.append(train_loss)
    train_accs.append(train_acc)
    test_losses.append(test_loss)
    test_accs.append(test_acc)

# Save results to file.
results = {
    "train_losses": np.array(train_losses),
    "train_accs": np.array(train_accs),
    "test_losses": np.array(test_losses),
    "test_accs": np.array(test_accs),
}
results_file = os.path.join("logs", f"{args.name}.pkl")
results_dir = os.path.dirname(results_file)
if not os.path.isdir(results_dir):
    os.makedirs(results_dir)
with open(results_file, "wb") as f:
    pickle.dump(results, f)
