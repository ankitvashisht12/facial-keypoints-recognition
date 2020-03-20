## TODO: define the convolutional neural network architecture

import torch
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        self.conv1 = nn.Conv2d(1, 10, 5)
        
        self.pool1 = nn.MaxPool2d(2,2)
        
        self.pool2 = nn.MaxPool2d(3,2)

        self.conv2 = nn.Conv2d(10, 15, 5)

        self.conv3 = nn.Conv2d(15, 25, 7)
        
        self.conv4 = nn.Conv2d(25, 32, 7)
        
        self.fc1 = nn.Linear(8*8*32, 1026)
        
        self.fc1_drop = nn.Dropout(p=0.4)

        self.fc2 = nn.Linear(1026, 512)
        
        self.fc2_drop = nn.Dropout(p=0.4)

        self.fc3 = nn.Linear(512, 256)
        
        self.fc3_drop = nn.Dropout(p=0.4)

        self.out = nn.Linear(256, 136)
        
        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting
        

        
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool1(F.relu(self.conv2(x)))
        x = self.pool2(F.relu(self.conv3(x)))
        x = self.pool2(F.relu(self.conv4(x)))
        x = x.view(-1, 8*8*32)
        x = F.relu(self.fc1(x))
        x = self.fc1_drop(x)
        x = F.relu(self.fc2(x))
        x = self.fc2_drop(x)
        x = F.relu(self.fc3(x))
        x = self.fc3_drop(x)

        x = self.out(x)
        
        # a modified x, having gone through all the layers of your model, should be returned
        return x
