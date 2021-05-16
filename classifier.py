import torch
from torch import nn
from torch import optim
from torchvision import models, transforms, datasets

import numpy as np
import os
import matplotlib.pyplot as plt

from PIL import ImageOps
from PIL import Image
from matplotlib import cm

class Resize(object):
    """
        Utility for image transformer
        Takes an image of specific dimensions
        Outputs an imag of dimensions 224*224
    """
    def __init__(self, size=224):
        self.size = size
        
    def __call__(self, im):
        if(im.height > im.width):
            
            w = int(self.size*im.width/im.height)
            h = self.size
            pad_val = int((224-w)/2)
            pad = (224-w-pad_val,0,pad_val,0)
        else:
            h = int(self.size*im.height/im.width)
            w = self.size
            pad_val = int((224-h)/2)
            pad = (0,224-h-pad_val,0,pad_val)
        return ImageOps.expand(im.resize((w,h),resample=Image.BILINEAR), pad)

def get_model():
    """
        Takes in the state dictionary and loads it into mobile-net
        Returns the model variable
    """
    state_dict = torch.load("checkpoint_9.pth",map_location=torch.device('cpu'))
    model = models.mobilenet_v3_large(pretrained=False)

    model.classifier = nn.Sequential(nn.Linear(960,1280),
                                    nn.ReLU(),
                                    nn.Linear(1280,10),
                                    nn.LogSoftmax(dim=1))

    model.load_state_dict(state_dict['model_state_dict'])
    return model

def get_transform():
    """
        Defines the image transformations to be done on the dataset
    """
    test_transform = transforms.Compose([Resize(),
                                     transforms.CenterCrop(224),
                                     transforms.ToTensor(),
                                     ])
    return test_transform

def get_pose(image):
    """
        Takes an image as an input
        Return
    """
    model = get_model()
    transform = get_transform()
    
    model.eval()
    
    im = Image.fromarray(image)
    image = transform(im)
    h, w, c = image.shape
    
#     print(image.shape)
    image_test = image.view(1,h,w,c)
    
    with torch.no_grad():
        
        out = model(image_test)
        probs = torch.exp(out)
        top_p, top_class = probs.topk(1,dim=1)
        
#     view_classify(image,probs,id_dict)
    pred = top_class.numpy()
    
    return str(pred[0][0])
    
