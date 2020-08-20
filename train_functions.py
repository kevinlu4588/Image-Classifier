import numpy as np
import torch
from PIL import Image
from torch import nn
import json
import matplotlib.pyplot as plt
from torchvision import datasets, models, transforms
def process_data(data_dir):
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'    
    training_transforms = transforms.Compose([transforms.RandomRotation(30),
                                           transforms.RandomResizedCrop(224),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225]),

                                          ])
    validation_transforms = transforms.Compose([transforms.Resize(255),
                                         transforms.CenterCrop(224),
                                         transforms.ToTensor(),
                                         transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225]),
                                         ])
    testing_transforms = transforms.Compose([transforms.Resize(255),
                                         transforms.CenterCrop(224),
                                         transforms.ToTensor(),
                                         transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225]),
                                         ])

    # TODO: Load the datasets with ImageFolder
    training_dataset = datasets.ImageFolder(train_dir, transform = training_transforms)
    validation_dataset = datasets.ImageFolder(valid_dir, transform = validation_transforms)
    testing_dataset = datasets.ImageFolder(test_dir, transform = testing_transforms)
    # TODO: Using the image datasets and the trainforms, define the dataloaders
    trainloader = torch.utils.data.DataLoader(training_dataset, batch_size = 32, shuffle = True)
    validloader = torch.utils.data.DataLoader(validation_dataset, batch_size = 32, shuffle = True)
    testloader = torch.utils.data.DataLoader(testing_dataset, batch_size = 32, shuffle = True)
    return training_dataset, trainloader, validloader, testloader
def create_model(architecture, hidden_units, learnrate):
    if architecture == "vgg16":
        model = models.vgg16(pretrained = True)
    elif architecture == "alexnet":
        model = models.alexnet(pretrained = True)
    elif architecture == "resnet18":
        model = models.resnet18(pretrained = True)
    for parameters in model.parameters():
        parameters.requires_grad = False
    classifier = nn.Sequential(nn.Linear(25088, hidden_units[0]),
                                 nn.ReLU(),
                                 nn.Dropout(0.5),
                                 nn.Linear(hidden_units[0], hidden_units[1]),
                                 nn.ReLU(),
                                 nn.Dropout(0.5),
                                 nn.Linear(hidden_units[1], 102),
                                 nn.LogSoftmax(dim = 1))
    model.classifier = classifier
    optimizer = torch.optim.Adam(model.classifier.parameters(), lr = learnrate)
    return model, optimizer
def train_model(model, trainloader, validloader, epochs, optimizer, gpu):
    criterion = nn.NLLLoss()
    if torch.cuda.is_available() and gpu:
        model.cuda()
    for e in range(epochs):
        training_loss = 0
        for images, labels in trainloader:
            if torch.cuda.is_available() and gpu:
                images, labels = images.cuda(), labels.cuda()
            optimizer.zero_grad()
            output = model.forward(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            training_loss += loss.item()
            
        else:
            model.eval()
            accuracy = 0
            valid_loss = 0
            with torch.no_grad():
                for images, labels in validloader:
                    if torch.cuda.is_available() and gpu:
                        images, labels = images.cuda(), labels.cuda()
                    output = model.forward(images)
                    loss = criterion(output, labels)
                    valid_loss += loss.item()
                    
                    probs = torch.exp(output)
                    top_p, top_class = probs.topk(1, dim = 1)
                    equals = top_class == labels.view(*top_class.shape)
                    accuracy += torch.mean(equals.type(torch.FloatTensor))
                print(f"Epoch: {e}\n"
                f"Training Loss: {training_loss/len(trainloader)}\n"
                f"Validation Loss: {valid_loss/len(validloader)}\n"
                f"Accuracy: {accuracy.item()/len(validloader)*100}\n")
            
    print("Training Done")    
                
def save_model(model, save_directory, optimizer, train_dataset, epochs, model_name):
    checkpoint = {'model_name': model_name,
              'epochs': epochs,
              'classifier': model.classifier,
              'state_dict': model.state_dict(),
              'optimizer_dict': optimizer.state_dict(),
              'class_to_idx': train_dataset.class_to_idx,
             }
    torch.save(checkpoint, save_directory)
    print(f"Model was saved at {save_directory}")

    
        
    