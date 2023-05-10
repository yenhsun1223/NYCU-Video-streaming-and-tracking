import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision ###
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch.utils import data
from torchsummary import summary
from torch.optim import Adam  ###
from torch.autograd import Variable
#from torchvision import transforms

#自訂dataset類別
class SportLoader(data.Dataset):
    def __init__(self, mode, transform=None):
        self.mode = mode
        self.sport = pd.read_csv(mode+'.csv')
        self.img_name = self.sport.iloc[:, 0]
        self.label = self.sport.iloc[:, 1]
        self.transform = transform
        
    def __len__(self):
        return len(self.img_name)
        
    def __getitem__(self, index):
    
        image_path = self.mode+"/"+self.img_name[index]
        self.img = io.imread(image_path)
        self.img = self.img.transpose(2,0,1).astype(np.float32)  #####
        self.target = self.label[index]
        
        if self.transform:
            self.img = self.transform(self.img)
            
        return self.img, self.target

train_dataset=SportLoader("train")
valid_dataset=SportLoader("val")
#test_dataset=SportLoader("test")

# dataset轉為DataLoader
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=True)
#test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True)

print(len(train_dataset), train_dataset[0][1])
print(train_dataset[0][0].shape)

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

summary(model, (3, 224, 224)) 

# training parameters
num_epoch = 300         # number of training epoch
learning_rate = 0.0001       # learning rate

# define a loss function, and optimizer
criterion = nn.CrossEntropyLoss() 
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# the path where checkpoint saved
model_path = 'HW1_311551149.pt'

# start training

best_acc = 0.0
epoch_plot = []
train_loss_plot = []
valid_loss_plot = []
train_acc_plot = []
valid_acc_plot = []

for epoch in range(num_epoch):
    train_acc = 0.0
    train_loss = 0.0
    val_acc = 0.0
    val_loss = 0.0

    # training
    model.train() # set the model to training mode
    for i, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(device), labels.to(device) # move data to device (cpu/cuda)
        optimizer.zero_grad() # set gradient to zero
        outputs = model(inputs) # forward pass (compute output)
        batch_loss = criterion(outputs, labels) # compute loss
        _, train_pred = torch.max(outputs, 1) # get the index of the class with the highest probability  ####
                                              # one is the true value; another is the index of the true value
                                              # we do not care about that true value, hence using "_"
                                              # 1 means dim=1(max value of column), because we need to know the class of this sample
        batch_loss.backward() # compute gradient (backpropagation)
        optimizer.step() # update model with optimizer

        train_acc += (train_pred.cpu() == labels.cpu()).sum().item() ####
        train_loss += batch_loss.item() ####
    
    # validation
    if len(valid_dataset) > 0:
        model.eval() # set the model to evaluation mode
        with torch.no_grad():  # disable gradient calculation
            for i, (inputs, labels) in enumerate(valid_loader):
                inputs, labels = inputs.to(device), labels.to(device)  # same
                outputs = model(inputs)  # same
                batch_loss = criterion(outputs, labels)  # same
                _, val_pred = torch.max(outputs, 1)   ####
            
                val_acc += (val_pred.cpu() == labels.cpu()).sum().item() # get the index of the class with the highest probability  ####
                val_loss += batch_loss.item()  ####

            print('[{:03d}/{:03d}] Train Acc: {:3.6f} Loss: {:3.6f} | Val Acc: {:3.6f} loss: {:3.6f}'.format(
                epoch + 1, num_epoch, train_acc/len(train_dataset), train_loss/len(train_loader), val_acc/len(valid_dataset), val_loss/len(valid_loader)))
            epoch_plot.append(epoch+1)
            train_loss_plot.append(train_loss/len(train_loader))    
            valid_loss_plot.append(val_loss/len(valid_loader))  
            train_acc_plot.append(train_acc/len(train_dataset))
            valid_acc_plot.append(val_acc/len(valid_dataset))

            # if the model improves, save a checkpoint at this epoch
            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(model.state_dict(), model_path, _use_new_zipfile_serialization = False)
                print('saving model with acc {:.3f}'.format(best_acc/len(valid_dataset)))
    else:
        print('[{:03d}/{:03d}] Train Acc: {:3.6f} Loss: {:3.6f}'.format(
            epoch + 1, num_epoch, train_acc/len(train_dataset), train_loss/len(train_loader)))
#print(epoch_plot, train_loss_plot, valid_loss_plot)
#print(train_acc_plot, valid_acc_plot)
plt.figure()
plt.plot(epoch_plot, train_loss_plot, color='blue', label='train')
plt.plot(epoch_plot, valid_loss_plot, color='orange', label='validation')
plt.title("loss curve")
plt.ylabel("loss")
plt.xlabel("epoch")
plt.legend()

plt.figure()
plt.plot(epoch_plot, train_acc_plot, color='blue', label='train')
plt.plot(epoch_plot, valid_acc_plot, color='orange', label='validation')
plt.title("accuracy curve")
plt.ylabel("accuracy")
plt.xlabel("epoch")
plt.legend()

plt.show()
# if not validating, save the last epoch
if len(valid_dataset) == 0:
    torch.save(model.state_dict(), model_path, _use_new_zipfile_serialization = False)
    print('saving model at last epoch')
    
