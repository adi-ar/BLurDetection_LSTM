# -*- coding: utf-8 -*-
"""
Created on Thu Sep 19 20:40:01 2019

@author: adiar
"""

import numpy as np
from numpy import array
from numpy import zeros
from matplotlib import pyplot as plt

import pandas as pd

import cv2

import os
import os.path as path

import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.layers import Dropout
from keras.preprocessing import sequence

import random

#set random seed for reproducability
random.seed(42)

#Number of training set images
no_train_img = 180

#Number of test images
no_test_img = 60


""" Number of salient contours(identified by cv2 saliency detector function) to be considered, this number can influence performance, 
too less will miss out important contours, too many will add noise to training data.
Set to 20 here. """

no_contours = 20

#Pre-define train and test arrays according to variables set here. 5 = number of features.
# 3-d array for lstm input

inputX = zeros([no_train_img,no_contours,5])
testX = zeros([no_test_img,no_contours,5])
trainy = []
testy = []


dir_path = 'D:/ADM/Blur Detection/saliency-detection'

training_folder = os.path.join(dir_path,'gg_v0','ADM','Train')
test_folder = os.path.join(dir_path,'gg_v0','ADM','Test')

train_motionBlur = os.path.join(training_folder,'MotionBlur')
train_focus = os.path.join(training_folder,'Focused')



#Define function for Laplacian value
def variance_of_laplacian(image):
    return cv2.Laplacian(image, cv2.CV_16S).var()


#function to read train folder
def read_train_img(focus_folder, motionBlur_folder):
    im=0
    for folder in [focus_folder,motionBlur_folder]:
    
        if folder == focus_folder:
            y_val = 0
        else:
            y_val=1
    
        for filename in os.listdir(folder):
            image = cv2.imread(os.path.join(folder,filename))
            imagegs = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            if image is not None:
                trainy.insert(im,y_val)
                
            # compute the saliency map
                saliency = cv2.saliency.StaticSaliencySpectralResidual_create()
                (success, saliencyMap) = saliency.computeSaliency(image)
                saliencyMap = (saliencyMap * 255).astype("uint8")
            
            
            #Threshold saliency map for sharp contours
                ret,threshMap = cv2.threshold(saliencyMap.astype("uint8"),127,255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)
                
                #Build contours
                contours, hierarchy = cv2.findContours(threshMap,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
                
                loopRange = min(len(contours),10)
                for i in range(loopRange):
                    
                    x,y,w,h = cv2.boundingRect(contours[i])
                    #print(x,y,w,h)
                    
                    #Extract salient part of image. This will be one step of the input to LSTM
                    imgTemp = imagegs[y:y+h,x:x+w]
                    
                    #Calculate laplacian (blur) value of each contour
                    v = variance_of_laplacian(imgTemp)
                    
                    #not used so far
                    m=cv2.moments(contours[i])
                    
                    #the position, size and laplacian value of each contour is input for detecting overall blur of image
                    val=[x,y,w,h,v]
                           
                    inputX[im,i] = val  
                    
                im = im+1
    print("INputX prepared")
    return inputX, trainy

#function to read test data
def read_test_img(folder,file):
    im=0    
    for filename in os.listdir(folder):
        
        image = cv2.imread(os.path.join(test_images_folder,filename))
        if image is not None:
            img_name = filename
            imagegs = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            
            # compute the saliency map
            saliency = cv2.saliency.StaticSaliencySpectralResidual_create()
            (success, saliencyMap) = saliency.computeSaliency(image)
            saliencyMap = (saliencyMap * 255).astype("uint8")
        
        
            #Threshold saliency map for sharp contours
            ret,threshMap = cv2.threshold(saliencyMap.astype("uint8"),127,255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)
            
            #Build contours
            contours, hierarchy = cv2.findContours(threshMap,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
            
            loopRange = min(len(contours),10)
            for i in range(loopRange):
                #repeat procedure as in training
                
                x,y,w,h = cv2.boundingRect(contours[i])
                #print(x,y,w,h)
        
                imgTemp = imagegs[y:y+h,x:x+w]
                
                v = variance_of_laplacian(imgTemp)
                m=cv2.moments(contours[i])
                
                val=[x,y,w,h,v]
                       
                testX[im,i] = val  
                
                df = pd.read_excel(file)
                
            img_idx = 0
            
    		
            for idx,val1 in df.iterrows():
               if(img_name in val1['Images']):
                   img_idx = idx
                   break
            y_val = df.iloc[img_idx,1]
            testy.append(y_val)
            im=im+1
    return testX, testy



dir_path = 'D:/ADM/Blur Detection/saliency-detection'

training_folder = os.path.join(dir_path,'gg_v0','ADM','Train')
test_folder = os.path.join(dir_path,'gg_v0','ADM','Test')

train_motionBlur = os.path.join(training_folder,'MotionBlur')
train_focus = os.path.join(training_folder,'Focused')

#read test images
inputX, trainy = read_train_img(train_focus,train_motionBlur)



#Create LSTM Model
model=Sequential()
model.add(LSTM(100, input_shape=(20,5)))
model.add(Dropout(0.5))
model.add(Dense(100, activation = 'relu'))
model.add(Dense(1, activation = 'sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

summary = model.summary()


test_images_folder = os.path.join(test_folder,'TestImages')
eval_file= os.path.join(test_folder,'EvaluationFile.xlsx')

testX,testy = read_test_img(test_images_folder, eval_file)

meanAcc = [] 
epochid = 0
noofruns = 50
accuracy = []
epoch = []
for noofepochs in [1,2,5]:
       
    for run in range(noofruns):
        ydata_arr = np.asarray(testy)
        testY = np.reshape(ydata_arr,(ydata_arr.shape[0],1))
        #test_y=keras.utils.to_categorical(testy)
        print(testY[1])
        model.fit(inputX, trainy, validation_data=(testX, testY), epochs=noofepochs, batch_size=16)
        #print("Model fit run")
        scores = model.evaluate(testX, testY, verbose=0)
        #print("Scores evaluate")
        print(scores[1])
        accuracy.insert(run,scores[1]*100)
        
    meanAcc.insert(epochid,sum(accuracy)/len(accuracy))
    epoch.insert(epochid, noofepochs)
    epochid = epochid+1
    
EpochEval=pd.DataFrame(list(zip(epoch,meanAcc)))
print(EpochEval)


pd.DataFrame.to_csv(EpochEval,'Epoch Eval.csv' )

"""
ToDo: Evaluate for combination of values of epochs, batch size, contours. 
Visualize results to get best values
"""


    




    