# -*- coding: utf-8 -*-
"""
Created on Mon Jun  7 09:45:39 2021

@author: harveysun

TODO: 把softmax网络以及训练过程重构为class
"""

import torch
import torchvision
from torch import nn
from torch.utils import data
from torchvision import transforms
from matplotlib import pyplot as plt

from IPython import embed

class Dataset:
    '''
    Usage:
        
        dataset = Dataset(batch_size)
        train_iter, test_iter = dataset.load()
        for x, y in train_iter:
            print(x.shape)
            dataset.show_image(x[0].reshape(28,28), label=y[0])
            break;
    '''
    def __init__(self, batch_size):
        self.batch_size = batch_size
        self.train_iter = None
        self.test_iter  = None
        
    def load(self):
        trans = transforms.ToTensor() # trans PIL.Image to torch.Tensor
        train_set = torchvision.datasets.FashionMNIST(
                          root="~", train=True, download=False, transform=trans)
        test_set  = torchvision.datasets.FashionMNIST(
                          root="~", train=False,download=False, transform=trans)
        
        self.train_iter = data.DataLoader(train_set, self.batch_size, shuffle=True, num_workers=4)
        self.test_iter  = data.DataLoader(test_set , self.batch_size, shuffle=True, num_workers=4)
        return self.train_iter, self.test_iter
                
    def show_image(self, img, label=10, size=4):
        titles = [
        't-shirt', 'trouser', 'pullover', 'dress', 'coat', 'sandal', 'shirt',
        'sneaker', 'bag', 'ankle boot', 'NO Title'
        ]
        _, ax = plt.subplots(figsize=(size,size))
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        ax.set_title(titles[int(label)])
        ax.imshow(img.numpy())
 
def accuracy(y_hat, y):
    # 选择置信度最高的那个标签作为预测结果
    pred = y_hat.argmax(axis=1)
    # 统计y_hat中正确预测的个数
    cmp  = pred.type(y.dtype) == y
    return float(cmp.type(y.dtype).sum())


net = nn.Sequential(nn.Flatten(), nn.Linear(28*28, 10))
def init_weights(m):
    # init net's parameters
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)
        
net.apply(init_weights)
loss = nn.CrossEntropyLoss()
trainer = torch.optim.SGD(net.parameters(), lr=0.1)
        

if __name__ == '__main__':
    torch.multiprocessing.freeze_support()
    
    num_epochs = 10
    batch_size = 256
    dataset = Dataset(batch_size)
    train_iter, test_iter = dataset.load()

    for epoch in range(num_epochs):
        net.train() # 训练模式
        for X, y in train_iter:
            y_hat = net(X)
            l = loss(y_hat, y)
            trainer.zero_grad()
            l.backward()
            trainer.step()
        #=============================
        net.eval() # 评估模式，不更新参数
        correct = 0
        num_batch = 0
        epoch_loss = 0
        for X, y in train_iter:
            y_hat = net(X)
            epoch_loss += loss(y_hat, y).item()*len(y)
            correct += accuracy(y_hat, y)  
            num_batch += 1
        total = min(num_batch*batch_size, 60000)
        # 统计本次EPOCH训练结果：正确个数，正确率，loss
        print("EPOCH {}: {} correct, {:.2f}% correction, {:.2f} loss"
              .format(epoch, int(correct), correct/total*100, epoch_loss/total))