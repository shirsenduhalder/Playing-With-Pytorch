import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch.optim as optim
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import time

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

data_root = './data/cifar'

trainset = torchvision.datasets.CIFAR10(root=data_root, train=True, transform=transform, download=True)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True, num_workers=4)

testset = torchvision.datasets.CIFAR10(root=data_root, train=False, transform=transform, download=True)
testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=False, num_workers=4)

classes = ('plane', 'car', 'bird', 'cat','deer', 'dog', 'frog', 'horse', 'ship', 'truck')

def imshow(imgs, labels, preds, number=6):
    fig = plt.figure()
    imgs_so_far = 0
    imgs = imgs/2 + 0.5
    imgs = imgs[torch.randperm(imgs.size()[0])].cpu().numpy()
    for i in range(number):
        imgs_so_far += 1
        img = np.transpose(imgs[i], (1, 2, 0))
        label = classes[labels[i]]
        pred = classes[preds[i]]
        ax = plt.subplot(number//2, 2, imgs_so_far)
        ax.axis('off')
        ax.set_title("Original: {}, Pred: {}".format(label, pred))
        plt.imshow(img)

        if imgs_so_far == number:
            return    

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()      
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.pool2 = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(16 * 5 * 5, 100)
        self.fc2 = nn.Linear(100, 64)
        self.fc3 = nn.Linear(64, 10)
    
    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))

        x = x.view(-1, self.flat_num_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def flat_num_features(self, x):
        num_features = 1
        dims = x.size()[1:]
        for dim in dims:
            num_features *= dim
        
        return num_features

net = Net()
net = net.cuda()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

train_start = time.time()

for epoch in range(1):
    running_loss = 0
    for i, (images, labels) in enumerate(trainloader, 0):
        images, labels = images.cuda(), labels.cuda()
        optimizer.zero_grad()
        outputs = net(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        if i%100 == 0 and i > 0:
            print("Epoch: {}, Steps: {}, Loss: {}".format(epoch + 1, i + 1, running_loss/2000))
            running_loss = 0

time_train = time.time() - train_start
print('Finished training in: {}m {:.3f}s'.format(time_train//60, time_train%60))

correct_pred, total = 0, 0

test_start = time.time()
with torch.no_grad():
    for data in testloader:
        images, labels = data
        images, labels = images.cuda(), labels.cuda()
        outputs = net(images)
        _, preds = torch.max(outputs.data, 1)

        total += labels.size(0)
        correct_pred += (preds == labels).sum().item()

time_test = time.time() - test_start
print("Testing on {} images in: {}m {:.3f}s".format(total, time_test//60, time_test%60))
print("Total accuracy: {} %".format(100*correct_pred/total))

test_images, test_labels = next(iter(testloader))
test_images, test_labels = test_images.cuda(), test_labels.cuda()

test_outputs = net(test_images)
_, preds = torch.max(test_outputs.data, 1)
imshow(imgs=test_images, labels=test_labels, preds=preds, number=6)