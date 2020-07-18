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
IMG_SIZE = 220

TEST_DATA=[]

def create_training_data():
    for cat in CATEGORIES:
        path = os.path.join(TESTDIR,cat)
        class_num = CATEGORIES.index(cat)
        for img in os.listdir(path):
            img_array = cv2.imread(os.path.join(path,img), cv2.IMREAD_GRAYSCALE)
            new_array = cv2.resize(img_array,(IMG_SIZE,IMG_SIZE))
            TEST_DATA.append([new_array,class_num])

create_training_data()
print(len(TEST_DATA))


random.shuffle(TEST_DATA)
print(TEST_DATA[0])
test_x = []
test_y = []

for features, label in TEST_DATA:
    test_x.append(features)
    test_y.append(label)

test_x = np.array(test_x).reshape(-1,IMG_SIZE,IMG_SIZE,1)
test_y = np.array(test_y)

pickle_out = open("test_x.pickle","wb")
pickle.dump(test_x,pickle_out)
pickle_out.close()

pickle_out = open("test_y.pickle","wb")
pickle.dump(test_y,pickle_out)
pickle_out.close()