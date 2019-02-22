import os

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

from cosine_annealing_warm_restart import *
from model import *

batch_size = 128

n_gpu = torch.cuda.device_count()

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

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size*n_gpu, shuffle=True, num_workers=8)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=50, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


criterion = nn.CrossEntropyLoss()
# model = DenseNet201()
model = Inception()
model = model.cuda()
if n_gpu>1:
    model = nn.DataParallel(model)

optimizer = optim.Adam(model.parameters(), lr=3e-4, weight_decay=5e-4)
scheduler = CosineWithRestarts(optimizer, T_max = 10, factor = 2)

best_acc = 94

for epoch in range(300):  # loop over the dataset multiple times
    model.train()
    scheduler.step()
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs
        inputs, labels = data
        inputs = inputs.cuda()
        labels = labels.cuda()
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
    if epoch % 10 == 0:    # print every 2000 mini-batches
        print('[%d] loss: %.3f' %
                (epoch + 1, running_loss / trainloader.__len__()))

    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving.. acc', acc)
        state = {
            'net': model.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        torch.save(state, 'densenet_cifar10.path')
        best_acc = acc

print('Finished Training')
print("best acc", best_acc)
