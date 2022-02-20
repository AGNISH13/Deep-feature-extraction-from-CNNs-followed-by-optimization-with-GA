

import torch
import torchvision
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torchvision.transforms as transf
import numpy as np
import pandas as pd
import torch.nn as  nn
import torch.nn.functional as F
import time


def model(folder_path, out_classes):

    # Configuring Device

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Hyperparameter tuning

    epoch_no = 5
    l_rate = 0.0001
    batch_size_tr = 50
    batch_size_val = 30

    # Transforming data

    train_transform = transf.Compose([
        transf.Resize((224,224)),
        transf.ToTensor()
    ])

    val_transform = transf.Compose([
        transf.Resize((224,224)),
        transf.ToTensor()
    ])

    # Loading the dataset in the system

    train_ds = ImageFolder(folder_path +'/train',transform=train_transform)
    val_ds = ImageFolder(folder_path +'/val', transform = val_transform)
    train_loader = DataLoader(train_ds, batch_size=batch_size_tr, shuffle=True, num_workers=2, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size_val, shuffle=True, num_workers=2, drop_last=True)

    # Specifying reference model

    old_model = torchvision.models.resnet18(pretrained=True)

    # Changing the model for deep feature extraction

    class whole_cnn(nn.Module):
        def __init__(self):
            super(whole_cnn, self).__init__()
            
            # Removing the classifier list from VGG-19 after which the deep features are to be extracted

            self.remv_linear = torch.nn.Sequential(*(list(old_model.children())[:-1]))

            # Re-assigning the removed layers of VGG-19 and storing it in our own CNN to find change in accuracy

            self.flatten = nn.Flatten()
            self.add_linear = torch.nn.Sequential(nn.Linear(512,out_classes,bias=True))
            

        def forward(self,x):
            output = self.remv_linear(x)
            output = self.flatten(output)
            x_deep = output
            output_new = self.add_linear(output)
            return x_deep,output_new

    # Specifying model and performing Loss calculation and Optimization

    model = whole_cnn()
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    criterion.to(device)
    optim = torch.optim.Adam(model.parameters(), lr = l_rate)

    # Training and Validating with our CNN model

    def train_model(model, criterion, optim, epoch_no):
        best_deep_featr_train=[]
        best_labels_train=[]
        best_deep_featr_val=[]
        best_labels_val=[]
        since = time.time()
        best_acc= 0.0
        for epoch in range(epoch_no):
            train_features = np.zeros((50,512))
            train_labels = []
            running_loss = 0.0
            running_acc = 0.0
            model.train()
            for images,labels in train_loader:
                images = images.to(device)
                labels = labels.to(device)
                with torch.set_grad_enabled(True):
                    deep_featr,outputs = model(images)
                    train_features = np.append(train_features, deep_featr.detach().cpu().numpy(), axis=0)  
                    train_labels = np.append(train_labels, labels.cpu().detach().numpy(), axis=0)
                    _ ,preds = torch.max(outputs,1)
                    loss = criterion(outputs,labels)
                    loss.backward()
                    optim.step()
                optim.zero_grad()

            # Calculating and printing all Statistics

                running_loss += loss.item()*batch_size_tr
                running_acc += torch.sum(preds==labels)
            running_val_loss, running_val_acc, val_features, val_labels = model_val(model, criterion, optim)
            epoch_train_loss = running_loss/len(train_ds)
            epoch_train_acc = running_acc.double()/len(train_ds)
            print("Epoch: {}".format(epoch+1))
            print('-'*10)
            print('Train Loss: {:.4f}   Train Acc: {:.4f}'.format(epoch_train_loss,epoch_train_acc))
            epoch_val_loss = running_val_loss/len(val_ds)
            epoch_val_acc = running_val_acc.double()/len(val_ds)
            print('Val Loss: {:.4f}   Val Acc: {:.4f}'.format(epoch_val_loss,epoch_val_acc))
            print()
            if epoch_val_acc > best_acc:
                best_acc = epoch_val_acc
                best_deep_featr_train = train_features
                best_labels_train = train_labels
                best_deep_featr_val = val_features
                best_labels_val = val_labels

        # Printing Time Elapsed and best Validation Accuracy

        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        print("Best model has validation accuracy: {}".format(best_acc))
        return best_deep_featr_train, best_labels_train, best_deep_featr_val, best_labels_val


    def model_val(model, criterion, optim):
        model.eval()
        running_val_loss = 0.0
        running_val_acc = 0.0
        val_features = np.zeros((30,512))
        val_labels = []
        for images,labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)
            deep_featr,outputs = model(images)
            val_features = np.append(val_features, deep_featr.detach().cpu().numpy(), axis=0)   
            val_labels = np.append(val_labels, labels.cpu().detach().numpy(), axis=0)
            _ ,preds = torch.max(outputs,1)
            loss = criterion(outputs,labels)
            running_val_loss += loss.item()*batch_size_val
            running_val_acc += torch.sum(preds==labels)
        return running_val_loss, running_val_acc, val_features, val_labels

    # Calling the function to train our CNN model

    best_deep_featr_train, best_labels_train, best_deep_featr_val, best_labels_val  = train_model(model, criterion, optim, epoch_no)

    # Creating the dataframes for the extracted deep features

    df1=pd.DataFrame(best_deep_featr_train[50:,:])
    df2=pd.DataFrame(best_labels_train)
    df3=pd.DataFrame(best_deep_featr_val[30:,:])
    df4=pd.DataFrame(best_labels_val)
    print(df1.shape,df2.shape,df3.shape,df4.shape)
    df5 = pd.concat([df1,df2], axis=1)
    df6 = pd.concat([df3,df4], axis=1)

    return df5, df6