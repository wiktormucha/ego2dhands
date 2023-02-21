import random
from PIL import Image
import json
import torch
import torch.nn as nn
import numpy as np
import cv2 as cv2
import os


from config import *




def decision(probability):
    return random.random() < probability


class RandomNoise(nn.Module):
    def __init__(self, p=0.5) -> None:
        super().__init__()
        self.p = p
    def forward(self, img):

        assert torch.is_tensor(img)

        noise_factor = 0.1
        flag = decision(probability = self.p)

        if flag == False:
            return img
        
        noisy = img+torch.randn_like(img) * noise_factor
        noisy = torch.clip(noisy,0.,1.)
        return noisy
    

class RandomBoxes(nn.Module):
    def __init__(self, p =0.5) -> None:
        super().__init__()
        self.p = p
    def forward(self, img):

        assert torch.is_tensor(img)

        flag = decision(probability = self.p)

        if flag == False:
            return img
        

        size = 50
        n_k = 5
        
        h,w = size,size
        img = img.numpy()
        img_size = img.shape[1]
        boxes = []
        for k in range(n_k):
            y,x = np.random.randint(0,img_size-w,(2,))
            img[y:y+h,x:x+w] = 0
            boxes.append((x,y,h,w))
        # img = Image.fromarray(img.astype('uint8'), 'RGB')
            print('random_box')
        return torch.from_numpy(img)


class RandomBackground(nn.Module):
    
    def __init__(self, files, bg_path) -> None:
        super().__init__()
        self.files = files
        self.bg_path = bg_path

    def forward(self, img):

        # PIL image to nunpy
        img = np.array(img) 
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
       
        # Remove background
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        a_channel = lab[:,:,1]
        th = cv2.threshold(a_channel,127,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
        masked = cv2.bitwise_and(img, img, mask = th)    
        m1 = masked.copy()
        m1[th==0]=(0,0,0)                          

        # Random background
        bg_name=random.choice(self.files)
        background =cv2.imread(os.path.join(self.bg_path, bg_name))
      
        # Resice random background
        background = cv2.resize(background, (224,224), interpolation = cv2.INTER_AREA)

        background[th != 0] = (0,0,0)

        # Merge background with hand img
        img = m1 + background
        
        # Convert back to PIL format
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)

        return img