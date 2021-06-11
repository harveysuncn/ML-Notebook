# -*- coding: utf-8 -*-
"""
Created on Thu Jun 10 19:22:40 2021

@author: Administrator
"""

import torch
import torchvision
from torch import nn
from torch.utils import data
from torchvision import transforms
from matplotlib import pyplot as plt

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
        

net = nn.Sequential(nn.Flatten(), 
                    nn.Linear(784, 512), nn.ReLU(), 
                    nn.Linear(512, 128), nn.ReLU(),
                    nn.Linear(128, 10))

def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)
    
net.apply(init_weights)

def accuracy(y_hat, y):
    # 选择置信度最高的那个标签作为预测结果
    pred = y_hat.argmax(axis=1)
    # 统计y_hat中正确预测的个数
    cmp  = pred.type(y.dtype) == y
    return float(cmp.type(y.dtype).sum())



if __name__ == '__main__':
    torch.multiprocessing.freeze_support()
    
    batch_size, lr, num_epochs = 256, 0.1, 25
    loss = nn.CrossEntropyLoss()
    trainer = torch.optim.SGD(net.parameters(), lr=lr)
    dataset = Dataset(batch_size)
    train_iter, test_iter = dataset.load()
    
    for epoch in range(num_epochs):
        net.train()
        for X, y in train_iter:
            y_hat = net(X)
            l = loss(y_hat, y)
            trainer.zero_grad()
            l.backward()
            trainer.step()
        net.eval()
        correct = 0
        num_batch = 0
        epoch_loss = 0
        for X, y in test_iter:
            y_hat = net(X)
            epoch_loss += loss(y_hat, y).item()*len(y)
            correct += accuracy(y_hat, y)
            num_batch += 1
        total = min(num_batch*batch_size, 60000)
        print("EPOCH {}: {} correct, {:.2f}% correction, {:.2f} loss"
                  .format(epoch, int(correct), correct/total*100, epoch_loss/total))
    
    