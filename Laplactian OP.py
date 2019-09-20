# -*- coding: utf-8 -*-
"""
Created on Thu Aug 29 01:19:03 2019

@author: adiar
"""

import numpy as np
from numpy import array
from numpy import zeros
from matplotlib import pyplot as plt
import argparse

import pandas as pd

import cv2

import os
import os.path as path

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score


dir_path = 'D:/ADM/Blur Detection/saliency-detection'

training_folder = os.path.join(dir_path,'gg_v0','ADM','Train')
test_folder = os.path.join(dir_path,'gg_v0','ADM','Test')

train_motionBlur = os.path.join(training_folder,'MotionBlur')
train_focus = os.path.join(training_folder,'Focused')

threshold = 400
trainy = []
lapop = []

def variance_of_laplacian(image):
    return cv2.Laplacian(image, cv2.CV_16S).var()

im=0
for folder in [train_focus,train_motionBlur]:
    
    if folder == train_focus:
        y_val = 0
    else:
        y_val=1
    
    for filename in os.listdir(folder):
        image = cv2.imread(os.path.join(folder,filename))
        imagegs = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if image is not None:
            trainy.insert(im,y_val)
            v = variance_of_laplacian(imagegs)
            if v > 400:
                 lapop.insert(im,0)
            else:
                 lapop.insert(im,1)
        im = im+1


cm=confusion_matrix(trainy,lapop)
ac=accuracy_score(trainy,lapop)
print(ac)
#print(cm)


             
             
             
            
            
