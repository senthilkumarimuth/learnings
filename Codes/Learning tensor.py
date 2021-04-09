#Learning pytorch
import torch

#Unitialized matrix(whatever present in memory)
x = torch.Tensor(5,3)  #or torch.empty(5,3)
print(x)

#Initialized matrix with random values
y =  torch.rand(5,3)
print(y)

#constructing a matix filled with zeron and dtype long
x = torch.zeros(5,3, dtype = torch.long) #torch.long = torch.int64
print(x)


#construct a tensor directly from data points
y = torch.tensor([1,2])
print(y)


#create a tensor based on an existing tensor
x = x.new_ones(5,3, dtype = torch.double)  #torch.double = torch.float64
print(x)
x = torch.rand_like(x, dtype = torch.float)
print(x)


#getting shape of tensor
print(x.size())


# addition
x = x.new_ones(5,3, dtype = torch.float)
y = y.new_ones(5,3, dtype = torch.float)
print(x+y)

# addition - tensor as argument
result = torch.empty(5,3)
torch.add(x,y, out = result)
print(result)

# inplace addition
x.add_(y)
print(x)

#we can use numpy like indeix slicing
x = torch.rand(5,3)
print(x)
x = x[:,1]
print(x)

#resizing/reshape
x = torch.rand(4,4)
print(x)
y = x.view(16)
print(y)
z = x.view(-1,8)
print(z)
print(z.size())

#handling single number
x = torch.randn(1)
print(x)
print(x.item())
print(type(x.item()))

#converting tensor to numpy

a = torch.ones(6,6)
print(a)
b = a.numpy()
print(b)
a.add_(1)
print(a)
print(b)  #changing tensor on CPU will change the array as well

#convering numpy to torch

import numpy as np
a = np.ones(5)
print(a)
b = torch.from_numpy(a)
print(b)

np.add(a,1, out =a)
print(a)
print(b)  # changing array on CPU will change the tensor as well

#working with autogradient decent

x = torch.ones(2,2, requires_grad = True)
y = x + 2
print(y)

z = y*y*3  # doing a few more operations
out =z.mean()
print(z,out)

out.backward()
x.grad   # d(out)/dx


#stoping gradient tracking

print(x.requires_grad)
y = x.detach()
print(y.requires_grad)

#stopping gradient tracking with the help of 'with'

print(x.requires_grad)
with torch.no_grad():
    print((x ** 2).requires_grad)
    
#SETTING UP A NEUARAL NETWORK

import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    
    def __init__(self):
        super(Net,self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 3) #conv layer1, 1 input and 6 output channels
        self.conv2 = nn.Conv2d(6, 16, 3) #conv layer2, 6 input and 16 output
        #adding fully connected layers
        self.fc1 = nn.Linear(16 * 6 * 6, 120) #6x6 is input size
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84,10)
        
    def forward(self,x):  #need to define forward function, backward function is defined automatically
        #max pooling over 2x2 window
        x = F.max_pool2d(F.relu(self.conv1(x)),(2,2))
        #if the sie is a square you can only specify a single number
        x = F.max_pool2d(F.relu(self.conv2(x)),2)
        x = x.view(-1,self.num_flat_features(x)) #Flattening
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    def num_flat_features(self,x):
        size = x.size()[1:] #all dimentions except the batch dimention
        num_features = 1
        for s in size:
            num_features *=s
        return num_features
    
net = Net()
print(net)

params = list(net.parameters())
print(len(params))
print(params[0].size()) # conv1's weight

#Trying on random dataset

input = torch.randn(1, 1, 32, 32)
out = net(input)
print(out)

#Zero the gradient buffers of all parameters and backprops with random gradients:
    
net.zero_grad()
out.backward(torch.randn(1, 10))

#calculating loss

output = net(input)
target = torch.rand(10) # a dummy target
target = target.view(1,-1) # make it the same shape as output
criterion = nn.MSELoss()

loss = criterion(output, target)
print(loss)

#follow loss in backward direction

print(loss.grad_fn) #MSELoss
print(loss.grad_fn.next_functions[0][0]) #Linear
print(loss.grad_fn.next_functions[0][0].next_functions[0][0]) #ReLu

#Backprop

net.zero_grad()  ## zeroes the gradient buffers of all parameters

print('conv1.bias.grad before backward')
print(net.conv1.bias.grad)

loss.backward()

print('conv1.bias.grad after backward')
print(net.conv1.bias.grad)

#update wieghts: weight = weight - learning_rate * gradient
learning_rate = 0.01
for f in net.parameters():
    f.data.sub_(f.grad.data *learning_rate)

#To make use of SGD,Adam, RMSProb

import torch.optim as optim

#create your optimizer
optimizer = optim.SGD(net.parameters(), lr = 0.01)

#in your training loop:
optimizer.zero_grad
output = net(input)
loss = criterion(output,target)
loss.backward()
optimizer.step() #does the weight update

#TRAINING A REAL CLASSIFICATION MODEL

#load the training data

import torch
import torchvision
import torchvision.transforms as transforms

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True, num_workers=2)
testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                         shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

#showing a few training images

import matplotlib.pyplot as plt
import numpy as np

# functions to show an image

def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

# get some random training images
dataiter = iter(trainloader)
images, labels = dataiter.next()

# show images

# show images
imshow(torchvision.utils.make_grid(images))

# print labels
print(' '.join('%5s' % classes[labels[j]] for j in range(4)))

#defining CNN

import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


net = Net()

#Defining a loss function

import torch.optim as optim

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)


#Initiate training the neural network


for epoch in range(2):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

print('Finished Training')

#save the model

PATH = './cifar_net.pth'
torch.save(net.state_dict(), PATH)

#Testing 

dataiter = iter(testloader)
images, labels = dataiter.next()

# print images
imshow(torchvision.utils.make_grid(images))
print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(4)))

#loading the model

net = Net()
net.load_state_dict(torch.load(PATH))

outputs = net(images)

_, predicted = torch.max(outputs, 1)

print('Predicted: ', ' '.join('%5s' % classes[predicted[j]]
                              for j in range(4)))

#testing against 1000 images

correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the 10000 test images: %d %%' % (
    100 * correct / total))

#where model didn't learn well?

class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs, 1)
        c = (predicted == labels).squeeze()
        for i in range(4):
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1


for i in range(10):
    print('Accuracy of %5s : %2d %%' % (
        classes[i], 100 * class_correct[i] / class_total[i]))