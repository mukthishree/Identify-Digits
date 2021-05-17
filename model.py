import numpy as np
import cv2
import os
import pandas as pd
import csv
import glob

from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout,Conv2D,Flatten,MaxPooling2D,Activation

from kerastuner.tuners import RandomSearch
from kerastuner.engine.hyperparameters import HyperParameters


images = []
path = "Images/train/" #path to folder containing train images
noOfClasses = set()
classNo = []
numofSamples = []
imageDimensions = (28,28,3)


df = pd.read_csv("train.csv") # a csv file containing images name with label
for index, row in df.iterrows():
    filename = row['filename']
    label = row['label']
    img = cv2.imread(path + str(filename))
    img = cv2.resize(img,(28,28))
    images.append(img)
    classNo.append(label)
    noOfClasses.add(label)

images = np.array(images)
classNo = np.array(classNo)

print(len(images),"....", len(classNo))
X_train,X_test,y_train,y_test = train_test_split(images,classNo,test_size = 0.2)#test_ratio=0.2

def preProcessing(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.equalizeHist(img)
    img = img/255
    return img

X_train = np.array(list(map(preProcessing, X_train)))
X_test = np.array(list(map(preProcessing, X_test)))

X_train = X_train.reshape(X_train.shape[0],X_train.shape[1],X_train.shape[2],1)
X_test = X_test.reshape(X_test.shape[0],X_test.shape[1],X_test.shape[2],1)


model = Sequential()

model.add(Conv2D(64,(3,3),input_shape = X_train.shape[1:]))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64,(3,3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64,(3,3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())

model.add(Dense(64))
model.add(Activation("relu"))

model.add(Dense(32))
model.add(Activation("relu"))

model.add(Dense(10))
model.add(Activation("softmax"))

model.compile(loss = "sparse_categorical_entropy",optimizer="adam", metrics=["accuracy"])

model.fit(X_train,y_train,epochs=5,validation_split=0.3)
model.save("digits_model.h5")

test_loss,test_acc = model.evaluate(X_test,y_test)
print("Test loss:", test_loss)
print("Validation accuracy",test_acc)

predictions =model.predict([X_test])

filenames = glob.glob("Images/test/*.png")# path to test images(to predict)
filenames.sort()

fields = ['filename', 'label'] 
for img in filenames:
    f = img.split("/")[2]
    #print(f)
    image = cv2.imread(img)
    image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    image = cv2.equalizeHist(image)
    image = image/255
    image = image.reshape(-1,28,28,1)
    predictions = model.predict(image)
    predict = np.argmax(predictions)   
    with open("Test_2.csv", 'a',newline='') as csvfile:
        #writer = csv.DictWriter(csvfile, fieldnames=fields) 
        writer = csv.writer(csvfile)
        writer.writerow([str(f),str(predict)])
        
       

















