#!/usr/bin/env python
# coding: utf-8

# In[1]:


import wandb
wandb.init(project="Dog n Cat", entity="")
import os
os.environ["WANDB_API_KEY"] = ""
import torch 
import torchvision
import torch.optim as optim
import torch.nn as nn
import numpy as np
import torchvision.transforms as transforms
import torch.autograd as autograd
import torch.nn.functional as F
#import pandas as pd
import copy
import matplotlib
import matplotlib.pyplot as plt
import math
from sklearn.model_selection import train_test_split

torch.manual_seed(2020)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)


# In[2]:


class ResBlock(nn.Module):
    def __init__(self,inc,outc,stride,direct=None):
        super(ResBlock, self).__init__()
        self.res = nn.Sequential( 
            nn.Conv2d(inc,outc,3,stride,1,bias=False),
            nn.BatchNorm2d(outc),  #使用batchnorm, conv2d就不需要bias了,下同
            nn.ReLU(inplace=True),
            nn.Conv2d(outc,outc,3,1,1,bias=False),
            nn.BatchNorm2d(outc)
        )
        self.direct = direct
    def forward(self, x):
        out = self.res(x)
        if (self.direct != None): # 维度变化
            out = out + self.direct(x)
        else:
            out = out + x
        out = F.relu(out)
        return out

class ResNet34(nn.Module):
    def __init__(self):
        super(ResNet34, self).__init__()
        self.layer0 = nn.Sequential( 
            nn.Conv2d(3,64,7,2,3,bias=False), #[64,112,112]  
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3,2,1)  #[64,56,56]
        )
        self.layer1 = self.Layer(64,64,3)
        self.layer2 = self.Layer(64,128,4)
        self.layer3 = self.Layer(128,256,6)
        self.layer4 = self.Layer(256,512,3)
        self.out = nn.Linear(512,2)
    def Layer(self,inc,outc,blocknum):
        layer = []
        if (inc == outc):
            layer = [ResBlock(inc,outc,1) for i in range(blocknum)]
            return nn.Sequential(*layer)
        direct = nn.Sequential(
            nn.Conv2d(inc,outc,1,2,0,bias=False),
            nn.BatchNorm2d(outc)
        )
        layer.append(ResBlock(inc,outc,2,direct))
        for i in range(1,blocknum):
            layer.append(ResBlock(outc,outc,1))
        return nn.Sequential(*layer)
    def forward(self, x): #input size :  [batchsize, 1, 28, 28]
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = F.avg_pool2d(x,7)
        x = x.view(-1,512)
        x = self.out(x)
        return x


# In[3]:


def train(model, device, loader, optimizer, epoch):
    model.train()
    criterion = nn.CrossEntropyLoss()
    enum = math.ceil(len(loader.sampler)/config.batch_size)
    enum = enum*epoch
    for i, (inputs, labels) in enumerate(loader):
        inputs, labels = inputs.to(device,non_blocking=True), labels.to(device,non_blocking=True)
        optimizer.zero_grad()
        output = model(inputs)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        
        if ((enum+i) % 100) == 0:    
            predict = output.max(1).indices
            correct = torch.sum(predict == labels).item()
            wandb.log({"Train Loss": loss.item(), "Train Acc": correct/len(labels) }  )
            print('[%d, %5d] loss: %.3f Acc:%.3f' % (epoch, enum + i, loss.item(),  correct/len(labels) ))
            test(model,device,test_loader,epoch)


# In[4]:


def test(model, device, loader, epoch):
    model.eval()
    criterion = nn.CrossEntropyLoss(reduction='sum')
    avg_loss = 0
    correct = 0
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(loader):
            inputs, labels = inputs.to(device,non_blocking=True), labels.to(device,non_blocking=True)
            output = model(inputs)
            loss = criterion(output, labels)
            #print(output,labels)
            avg_loss += loss.item()
            predict = output.max(1).indices
            correct += torch.sum(predict == labels).item()
    total = len(loader.sampler)
    avg_loss /= total
    print('Avg Loss : %.3f , Accuracy : %.3f [%d/%d] \n' % (avg_loss, correct/total, correct, total) )
    wandb.log({"Test Loss": avg_loss, "Test Acc": correct/total})
    return correct


# In[5]:


transform = transforms.Compose([
        transforms.RandomResizedCrop(224,scale=(0.6,1)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.485,0.456,0.406), (0.229,0.224,0.225))  # normalized to [-1,1]
    ])
data_all = torchvision.datasets.ImageFolder( "./data/train/", transform)

#80%训练集(20000 cat 10k dog 10k)， 20%验证集(5000 cat 2.5k dog 2.5k)
train_mask, test_mask = train_test_split(np.arange(len(data_all)),  
                                        test_size=0.2,
                                        shuffle=True,
                                        stratify=data_all.targets,
                                        random_state=2020)
train_sampler = torch.utils.data.SubsetRandomSampler(train_mask)
test_sampler = torch.utils.data.SubsetRandomSampler(test_mask)
# num_workers pin_memory drop_last
train_loader = torch.utils.data.DataLoader(data_all, batch_size=50,
                                           drop_last=True,pin_memory=True,
                                           sampler=train_sampler)
test_loader = torch.utils.data.DataLoader(data_all, batch_size=1000,
                                          drop_last=True,pin_memory=True,
                                          sampler=test_sampler)


# In[7]:


net = ResNet34()
model = net.to(device)
#使用wandb存储当前参数
#os.environ["WANDB_RUN_ID"] = "3fsha086"
wandb.init(project="dog-vs-cat",reinit=True)#,resume = true)
wandb.watch(model, log="all")
config = wandb.config          # Initialize config
config.batch_size = 50          # input batch size for training (default: 64)
config.test_batch_size = 1000    # input batch size for testing (default: 1000)
config.maxiter = 12000
config.epochs =  math.ceil(config.maxiter / 
                math.ceil(len(train_sampler)/config.batch_size))
config.lr = 0.005 # learning rate
config.momentum = 0.9
config.weight_decay = 0.0001
#config.log_interval = 10 
 

optimizer = optim.SGD(model.parameters(),
                      lr = config.lr,
                      momentum=config.momentum,
                      weight_decay=config.weight_decay)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[2,5,10,20], gamma=0.5)


# In[ ]:


for epoch in range(config.epochs):
    train(model,device,train_loader,optimizer, epoch)
    scheduler.step()
test(model,device,test_loader,epoch)


# In[14]:


torch.save(model.state_dict(), "12k-iter.pth")

