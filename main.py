import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import time
import os
import argparse
from torchvision import datasets, models, transforms, utils
from torch.autograd import Variable
import torch.nn.functional as F
import copy
from PIL import Image
import matplotlib.pyplot as plt
from collections import OrderedDict
import json
import matplotlib

data_dir = os.getcwd()+ '/flower_data'
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'

use_gpu = torch.cuda.is_available()

train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

valid_transforms = transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

# TODO: Load the datasets with ImageFolder
image_datasets = {}
image_datasets["train"] = datasets.ImageFolder(train_dir, transform=train_transforms)
image_datasets["valid"] = datasets.ImageFolder(valid_dir, transform=valid_transforms)

# TODO: Using the image datasets and the trainforms, define the dataloaders
train_loader = torch.utils.data.DataLoader(image_datasets["train"], batch_size=64, shuffle=True)
valid_loader = torch.utils.data.DataLoader(image_datasets["valid"], batch_size=32)

import json

with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)

def train_model(model, criterion, optimizer, num_epochs, steps, running_loss, train_accuracy):    
    for e in range(num_epochs):

        model.train() # Dropout is turned on for training

        for images, labels in iter(train_loader):

            images, labels = images.to(device), labels.to(device) # Move input and label tensors to the GPU

            steps += 1
            optimizer.zero_grad()
            output = model.forward(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            # get the class probabilities from log-softmax
            ps = torch.exp(output) 
            equality = (labels.data == ps.max(dim=1)[1])
            train_accuracy += equality.type(torch.FloatTensor).mean()

            if steps % print_every == 0:

                model.eval() # Make sure network is in eval mode for inference

                with torch.no_grad():
                    valid_loss, valid_accuracy = validation(model, valid_loader, criterion, device)

                print("Epoch: {}/{}.. ".format(e+1, num_epochs),
                      "Training Loss: {:.3f}.. ".format(running_loss/print_every),
                      "Training Accuracy: {:.3f}".format(train_accuracy/print_every),
                      "Validation Loss: {:.3f}.. ".format(valid_loss/len(valid_loader)),
                      "Validation Accuracy: {:.3f}".format(valid_accuracy/len(valid_loader)))

                running_loss = 0
                train_accuracy = 0
                model.train() # Make sure training is back on
                
    print("\nTraining completed!")

def validation(model, dataloader, criterion, device):
    loss = 0
    accuracy = 0
    with torch.no_grad():
        for images, labels in iter(dataloader):
            
            images, labels = images.to(device), labels.to(device) # Move input and label tensors to the GPU
            
            output = model.forward(images)
            loss += criterion(output, labels).item()

            ps = torch.exp(output) # get the class probabilities from log-softmax
            equality = (labels.data == ps.max(dim=1)[1])
            accuracy += equality.type(torch.FloatTensor).mean()
    
    return loss, accuracy

model = models.resnet101(pretrained=True)

for param in model.parameters():
    param.requires_grad = False # Freeze parameters so we don't backprop through them

classifier = nn.Sequential(OrderedDict([
                          ('dropout1', nn.Dropout(0.3)),
                          ('fc1', nn.Linear(2048, 512)), # 1024 must match
                          ('relu1', nn.ReLU()),
                          ('dropout2', nn.Dropout(0.3)),
                          ('fc2', nn.Linear(512, 102)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))
    
model.fc = classifier

criterion = nn.NLLLoss()

# Observe that only parameters of final layer are being optimized as
# opoosed to before.
optimizer = optim.Adam(model.fc.parameters(), lr=0.001)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)
epochs = 400
print_every = 50
steps = 0
running_loss = 0
train_accuracy = 0
train_model(model, criterion, optimizer, num_epochs=epochs, steps=steps, running_loss= running_loss, train_accuracy= train_accuracy)

model.to("cpu")
model.class_to_idx = image_datasets['train'].class_to_idx
checkpoint = {'input_size': 224*224*3,
              'output_size': 102,
              'model': model,
              'state_dict': model.state_dict(),
              'optimizer_state_dict': optimizer.state_dict,
              'criterion': criterion,
              'epochs': epochs,
              'class_to_idx': model.class_to_idx}

torch.save(checkpoint, 'checkpoint.pt')

def load_checkpoint(filepath):
    checkpoint = torch.load(filepath, map_location='cpu')
    model = checkpoint["model"]
    model.load_state_dict(checkpoint['state_dict'])
    return model

model = load_checkpoint('checkpoint.pt')