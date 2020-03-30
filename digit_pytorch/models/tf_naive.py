from __future__ import absolute_import
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
__all__ = ['Naive']
class Naive(nn.Module):
    def __init__(self, num_classes, loss={'xent'}, **kwargs):
        super(Naive, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)
    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        print('hhhhhhhh',x)
        print('x shape in network', x.shape)
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        #output = x
        output = F.log_softmax(x)
        #print('output', output)
        f = x.view(x.size(0), -1)
    
        if not self.training:
            #print('return feature')
            return f,output
        return output

