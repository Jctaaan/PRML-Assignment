#!/usr/bin/env python
# coding: utf-8

# In[1]:


import wandb
wandb.init(project="Autopoetry", entity="jaspertan")
import os
os.environ["WANDB_API_KEY"] = "9b51c1a70a432bca6e85f45f9d7936ed1ae780ff"
import torch 
import torchvision
import torch.optim as optim
import torch.nn as nn
import numpy as np
import torchvision.transforms as transforms
import torch.autograd as autograd
import torch.nn.functional as F
from torch.utils.data import Dataset,DataLoader
#import pandas as pd
import copy
import matplotlib
import matplotlib.pyplot as plt
import math
from sklearn.model_selection import train_test_split

torch.manual_seed(2020)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)


# In[174]:


def prepareData():
    datas = np.load("tang.npz",allow_pickle = True)
    data = datas['data']
    ix2word = datas['ix2word'].item()
    word2ix = datas['word2ix'].item()
    data = torch.from_numpy(data)
    dataloader = DataLoader(data,
                         batch_size=16,
                         shuffle=True,
                         num_workers=0)
    return dataloader, ix2word, word2ix


# In[2]:


def view_poem(data, itx):
    word_data = np.zeros((1,data.shape[1]),dtype=np.str)
    # 这样初始化后值会保留第一一个字符，所以输出中'<START>' 变成了'<'
    row = itx
    for col in range(data.shape[1]):
        word_data[0,col] = ix2word[data[row,col].item()]
    #print(word_data.shape)
    print(' '.join(word_data[0,k] for k in range(word_data.shape[1])) )
    #print(word_data[2])


# In[4]:


class PoetryModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim,layers_num):
        super(PoetryModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.dp = nn.Dropout(config.dropout)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=layers_num, batch_first=True,dropout=config.dropout)
        self.out =  nn.Sequential(
            nn.Linear(self.hidden_dim, 2048),
            nn.ReLU(),
            nn.Linear(2048, 4096),
            nn.Dropout(config.dropout),
            nn.ReLU(),
            nn.Linear(4096, vocab_size)
            )
    def forward(self, input, hidden=None):
        embeds = self.embeddings(input)  # [batch, seq_len] => [batch, seq_len, embed_dim]
        batch_size, seq_len = input.size()
        if hidden is None:
            h_0 = input.data.new(3, batch_size, self.hidden_dim).fill_(0).float()
            c_0 = input.data.new(3, batch_size, self.hidden_dim).fill_(0).float()
        else:
            h_0, c_0 = hidden
        embeds = self.dp(embeds)
        output, hidden = self.lstm(embeds, (h_0, c_0)) 
        output = self.out(output)
        output = output.reshape(batch_size * seq_len, -1)
        return output, hidden


# In[4]:


def train(model, device, loader, optimizer, epoch):
    model.train()
    criterion = nn.CrossEntropyLoss()
    enum = len(dataloader)
    enum = enum*epoch
    for i, inputs in enumerate(loader):
        labels = inputs.clone()
        labels[:,:-1] = labels[:,1:] # 前移一位，后一个字作为前一个的label
        inputs, labels = inputs.long().to(device,non_blocking=True), labels.long().to(device,non_blocking=True)
        optimizer.zero_grad()
        output,hidden = model(inputs)
        labels = labels.view(-1)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        
        if ((enum+i) % 100) == 0:    
            predict = output.max(1).indices
            correct = torch.sum(predict == labels).item()
            wandb.log({"Train Loss": loss.item(), "Train Acc": correct/len(labels) }  )
            print('[%d, %5d] loss: %.3f Acc:%.3f' % (epoch, enum + i, loss.item(),  correct/len(labels) ))
            view_poem(predict.view(config.batch_size,-1),1)
            view_poem(inputs,1)
            #test(model,device,test_loader,epoch)




config = wandb.config          # Initialize config
config.batch_size = 64          # input batch size for training (default: 64)
#config.test_batch_size = 1000    # input batch size for testing (default: 1000)
#config.maxiter = 12000
config.epochs =  320
config.embedding_dim = 128
config.hidden_dim = 1024
config.layers_num = 3
config.dropout = 0.5
config.T_max = 32
config.lr = 0.001 # learning rate
#config.momentum = 0.9
#config.weight_decay = 0.0001
#config.log_interval = 10 
 
# ---  读取数据 ----
datas = np.load("tang.npz",allow_pickle = True)
data = datas['data']  #(57580, 125)
ix2word = datas['ix2word'].item()
word2ix = datas['word2ix'].item()
data = torch.from_numpy(data)
dataloader = DataLoader(data,
                batch_size=config.batch_size,shuffle=True,
                num_workers=0,drop_last=True,pin_memory=True)

net = PoetryModel(len(word2ix),
                  embedding_dim=config.embedding_dim,
                  hidden_dim=config.hidden_dim,
                  layers_num=config.layers_num)

model = net.to(device)
wandb.watch(model, log="all")

optimizer = optim.Adam(model.parameters(), lr=config.lr)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, config.T_max)
#torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[2,5,10,20], gamma=0.5)



# In[ ]:


for epoch in range(config.epochs):
    train(model,device,dataloader,optimizer, epoch)
    scheduler.step()
    if (epoch % 32 == 31):
        torch.save(model.state_dict(), "epoch-%d.pth" %(epoch))
    if (epoch > 32):
        torch.save(model.state_dict(), "epoch-%d.pth" %(epoch))
        if os.path.exists("epoch-%d.pth" %(epoch-1)):
            os.remove("epoch-%d.pth" %(epoch-1)) 
#test(model,device,test_loader,epoch)


# In[10]:


datas = np.load("tang.npz",allow_pickle = True)
data = datas['data']  #(57580, 125)
ix2word = datas['ix2word'].item()
word2ix = datas['word2ix'].item()
data = torch.from_numpy(data)

net = PoetryModel(len(word2ix),
                  embedding_dim=128,
                  hidden_dim=1024,
                  layers_num=3)

model = net.to(device)
model.load_state_dict(torch.load('epoch-31.pth'))


# In[11]:


def p_choose(weights): # 以概率选字
        prob = F.softmax(weights[0],dim=0).cpu().numpy()
        t = np.cumsum(prob)
        index = int(np.searchsorted(t, np.random.rand(1)))
        return index


# In[12]:


def generate(model, start_words, ix2word, word2ix):
    results = list(start_words) 
    start_words_len = len(start_words)
    # 第一个词语是<START>
    input = torch.Tensor([word2ix['<START>']]).view(1, 1).long().to(device)
    hidden = None
    model.eval()
    with torch.no_grad():
        for i in range(64):
            output, hidden = model(input, hidden)
        # 如果在给定的句首中，input为句首中的下一个字
            if i < start_words_len:
                w = results[i]
                input = input.data.new([word2ix[w]]).view(1, 1)
           # 否则将output作为下一个input进行
            else:
                idx = p_choose(output) # 按概率选择下一个字，避免永远相同
                w = ix2word[idx]
                results.append(w)
                input = input.data.new([idx]).view(1, 1)
            if w == '<EOP>':
                del results[-1]
                break
        return results


# In[13]:


def head_generate(model, start_words, poem_len, ix2word, word2ix):
    # 5/7言生成
    start_words = list(start_words) #藏头 
    # 第一个词语是<START>
    input = torch.Tensor([word2ix['<START>']]).view(1, 1).long().to(device)
    hidden = None
    model.eval()
    with torch.no_grad():
        results = ''
        output, hidden = model(input, hidden) # 初始化 '<START>'
        for w in start_words: #每句开头
            input = input.data.new([word2ix[w]]).view(1, 1)
            results += w
            nlen = 1
            while True: #生成一句
                nlen += 1 #现在生成第几个字 
                output, hidden = model(input, hidden)

                if (nlen == 1+2*(poem_len + 1)): #一句生成完毕
                    break
                if (nlen == poem_len + 1): # 半句，强制加"，"
                    w = '，'
                elif (nlen == 2*(poem_len + 1)): # 整句，强制加"。"
                    w = '。'
                else:
                    while True:
                        idx = p_choose(output) 
                        w = ix2word[idx]
                        if (w != '，' and w != '。' and w != '<EOP>' and w != '<s>'):
                            break
                results += w
                input = input.data.new([word2ix[w]]).view(1, 1)
            results += '\n' #一句生成完毕换行
        return results


# In[36]:


print(head_generate(model,'湖光秋月',5,ix2word,word2ix))


# In[15]:


''.join(generate(model,'湖光秋月两相和，',ix2word,word2ix))

