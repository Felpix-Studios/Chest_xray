import tensorflow as tf
from tensorflow import keras
import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os, sys, random, pickle

DATADIR = os.getcwd()+"\\data"
TRAINDIR=DATADIR+"\\train"
TESTDIR=DATADIR+"\\test"
CATEGORIES = ["NORMAL","PNEUMONIA"]
IMG_SIZE = 200

TRAINING_DATA=[]

def create_training_data():
    for cat in CATEGORIES:
        path = os.path.join(TRAINDIR,cat)
        class_num = CATEGORIES.index(cat)
        for img in os.listdir(path):
            img_array = cv2.imread(os.path.join(path,img), cv2.IMREAD_GRAYSCALE)
            new_array = cv2.resize(img_array,(IMG_SIZE,IMG_SIZE))
            TRAINING_DATA.append([new_array,class_num])

create_training_data()
print(len(TRAINING_DATA))

random.shuffle(TRAINING_DATA)

train_x = []
train_y = []

for features, label in TRAINING_DATA:
    train_x.append(features)
    train_y.append(label)

train_x = np.array(train_x).reshape(-1,IMG_SIZE,IMG_SIZE,1)
train_y = np.array(train_y)

pickle_out = open("train_x.pickle","wb")
pickle.dump(train_x,pickle_out)
pickle_out.close()

pickle_out = open("train_y.pickle","wb")
pickle.dump(train_y,pickle_out)
pickle_out.close()