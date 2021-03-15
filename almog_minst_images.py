#!/usr/bin/env python
# coding: utf-8

# In[3]:


import cv2
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import torchvision.datasets as datasets
import torchvision.transforms as transforms

import PIL as Image
import torch.autograd as varible


# In[17]:


mean_image = 0.1
std_image = 0.7


train_transform = transforms.Compose([transforms.ToTensor(), 
                                      transforms.Normalize((mean_image, ), (std_image,))])

prdict_transform = transforms.Compose([transforms.Resize(28, 28), 
                                       transforms.ToTensor(),
                                       transforms.Normalize((mean_image,), (std_image,)) ])

data_tarin = datasets.MNIST(root='./data', 
                           train=True, 
                           transform=train_transform, 
                           download=True)

data_loader = datasets.MNIST(root='./data', 
                           train=False, 
                           transform=prdict_transform, 
                           download=True)


# In[35]:


batch_size = 10
epochs = 100

train_loader = DataLoader(data_tarin, batch_size=batch_size, shuffle=TRUE)
test_loader  = DataLoader(data_test, batch_size=batch_size, shuffle=TRUE)
print(type(train_loader)
      


# 

# In[27]:


print(type(data_tarin))
print(type(data_tarin[0]))
print(len(data_tarin[0]))
print(type(data_tarin[0][0]))
print(type(data_tarin[0][1]))

image = data_loader[0][0].numpy()
image = np.reshape(image,(28,28))
print(image.shape)
plt.figure()
plt.imshow(image)
plt.show()


# In[ ]:


class detectionMNIST(nn.modules)

    def __init__(self, input_image=487, output=10)
        super(seld,detectionMNIST) __init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, stride=1, padding=1)
        self.BatchNorm2d_1 = nn.BatchNorm2d(num_features=8)
        self.MaxPool2d_1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=32, kernel_size=5, stride=1, padding=2)
        self.BatchNorm2d_2 = nn.BatchNorm2d(num_features=32)
        self.relu = nn.ReLU()
        self.MaxPool2d_2 = nn.MaxPool2d(2)
        self.fc1 = nn.linear(32*7*7,600)
        self.dropout2d = nn.Dropout2d(0.5)
        self.fc2 = nn.linear(600, 10)
        
        
    def forwardNet(self, image)     
        
        output = self.conv1(image)
        output= self.BatchNorm2d_1(output) 
        output =self.relu(output)
        output = self.MaxPool2d_1(output)
        
        output = self.conv2(output)
        output = self.BatchNorm2d_2(output)
        output = self.relu(output)
        output = self.MaxPool2d_2(output)
        
        out = out.view(-1,1568)
        
        output = self.fc1(output)
        output = self.relu(output)
        output = self.dropout2d(output)
        output = self.fc2(output)
        return output


# In[ ]:


net_model = detectionMNIST()
optimizer = optim.Adam(net_model, lr=0.001)
loss = nn.CrossEntropyLoss()
# >>> output = loss(input, target)


# In[ ]:


# train
for epoch in range(1,epochs)

    for batch_num in train_loader
        
        






