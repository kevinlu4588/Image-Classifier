import numpy as np
import torch
from PIL import Image
from torch import nn
import json
import matplotlib.pyplot as plt
from torchvision import datasets, models, transforms

def load_checkpoint(checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    if  checkpoint["model_name"] == "vgg16":
        model = models.vgg16(pretrained = True)
    elif checkpoint["model_name"] == "alexnet":
        model = models.alexnet(pretrained = True)
    elif checkpoint["model_name"] == "resnet18":
        model = models.resnet18(pretrained = True)
    model.classifier = checkpoint['classifier']
    model.load_state_dict = checkpoint['state_dict']
    model.class_to_idx = checkpoint['class_to_idx']
    return model
def process_image(image_path):
    image = Image.open(image_path)
    transform = transforms.Compose([transforms.Resize(256),
                                     transforms.CenterCrop(224),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225]),
                                     ])
    image = transform(image)
    return image
def predict(image_path, model, n_topk, gpu):
    model.eval()
    if torch.cuda.is_available() and gpu:
        model.to("cuda")
    else:
        model.cpu()
    image = process_image(image_path)
    image = image.unsqueeze(0)
    with torch.no_grad():
        probs = torch.exp(model.forward(image))
        top_p, top_class = probs.topk(n_topk, dim = 1)
    inv_class_to_idx = {v: k for k, v in model.class_to_idx.items()}
    classes = []
    for idx in top_class.numpy()[0]:
        classes.append(inv_class_to_idx[idx])
    return top_p[0], classes
    