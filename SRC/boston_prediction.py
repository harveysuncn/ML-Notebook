# -*- coding: utf-8 -*-
"""
Created on Tue Jun  8 09:42:48 2021

@author: Administrator

Best loss: 45.87

TODO: 重构
"""

from sklearn.datasets import load_boston
import torch
from torch import nn
from torch.utils import data


def get_data(batch_size):
    X, y = load_boston(return_X_y=True)   
    X_tensor = torch.from_numpy(X).float()
    y_tensor = torch.from_numpy(y).reshape((len(X_tensor),-1)).float()
    dataset  = data.TensorDataset(X_tensor, y_tensor)
    return data.DataLoader(dataset,
                            batch_size,
                            shuffle=True)

        
net = nn.Sequential(nn.Linear(13,1))
nn.init.normal_(net[0].weight, mean=1e-3, std=1e-5)
nn.init.constant_(net[0].bias, val=0)      

loss = nn.MSELoss()
trainer = torch.optim.SGD(net.parameters(), lr=1e-6, weight_decay=1.0)



if __name__ == '__main__':
    total = 506 # total samples
    num_epochs = 200
    batch_size = 20
    for epoch in range(num_epochs):
        net.train()
        for X, y in get_data(batch_size):
            l = loss(net(X), y)
            trainer.zero_grad()
            l.backward()
            trainer.step()
        net.eval()
        epoch_loss = 0
        for X, y in get_data(batch_size):
            epoch_loss += loss(net(X), y).item()*len(y)
        print("EPOCH {}: {:.2f} loss".format(epoch, epoch_loss/total))
    #======================
    print("Weights:", net[0].weight.data)
    print("Bias:", net[0].bias.data)
    
    for X, y in get_data(100):
        y_hat = net(X)
        #print(torch.cat((y_hat,y), dim=1))
        l = loss(y_hat, y)
        print("Average loss: {:.2f}".format(l))
        break