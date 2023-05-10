import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision ###
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
import os
from torch.utils.data import DataLoader
from torch.utils import data
from torchsummary import summary
from torch.optim import Adam  ###
from torch.autograd import Variable
#from train import Network

#自訂dataset類別
class SportLoader(data.Dataset):
    def __init__(self, mode, transform=None):
        self.mode = mode
        self.img_name = os.listdir('./test/')
        self.transform = transform
        
    def __len__(self):
        return len(self.img_name)
        
    def __getitem__(self, index):
        image_path = self.mode+"/"+self.img_name[index]
        self.img = io.imread(image_path)
        self.img = self.img.transpose(2,0,1).astype(np.float32)  #####
        #print(self.img_name[index])
        if self.transform:
            self.img = self.transform(self.img)
        
        return self.img, self.img_name[index]

test_dataset=SportLoader("test")
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)


class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        # The arguments for commonly used modules:
        # torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        # torch.nn.MaxPool2d(kernel_size, stride, padding)

        # input image size: [3, 224, 224]
        self.cnn_layers = nn.Sequential(
            nn.Conv2d(3, 8, 3, 1, 1),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),

            nn.Conv2d(8, 8, 3, 1, 1),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),

            nn.Conv2d(8, 8, 3, 1, 1),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),
            
            nn.Conv2d(8, 8, 3, 1, 1),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.MaxPool2d(4, 4, 0),            
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(392, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
            #nn.Dropout(0.5),
            nn.Linear(16, 10)
        )

    def forward(self, x):
        # input (x): [batch_size, 3, 224, 224]
        # output: [batch_size, 10]

        # Extract features by convolutional layers.
        x = self.cnn_layers(x)

        # The extracted feature map must be flatten before going to fully-connected layers.
        x = x.flatten(1)

        # The features are transformed by fully-connected layers to obtain the final logits.
        x = self.fc_layers(x)
        return x





model = Network()

# get device 
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")
model = Network().to(device)
print(model)

# the path where checkpoint saved
model_path = 'HW1_311551149.pt'
model.load_state_dict(torch.load(model_path))

predict_img = []
predict = []
model.eval() # set the model to evaluation mode
with torch.no_grad():
    for i, (data, img_name) in enumerate(test_loader):
        inputs = data
        inputs = inputs.to(device)
        outputs = model(inputs)
        _, test_pred = torch.max(outputs, 1) # get the index of the class with the highest probability
        for x in img_name:
            predict_img.append(x)
        for y in test_pred.cpu().numpy():
            predict.append(y) 
#print(len(predict_img))          
with open('HW1_311551149.csv', 'w') as f:
    f.write('names,label\n')
    for i, y in enumerate(predict):
        f.write('{},{}\n'.format(predict_img[i], y))      
