import tensorflow as tf
from tensorflow import keras
import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os, sys, random, pickle

CATEGORIES = ["NORMAL","PNEUMONIA"]
test_x = pickle.load(open("test_x.pickle","rb"))
test_y = pickle.load(open("test_y.pickle","rb"))

def prepare(filepath):
    IMG_SIZE = 200
    img_array = cv2.imread(filepath,cv2.IMREAD_GRAYSCALE)
    new_array = cv2.resize(img_array,(IMG_SIZE,IMG_SIZE))
    return new_array.reshape(-1,IMG_SIZE,IMG_SIZE,1)

model = tf.keras.models.load_model("3x64.model")

prediction = model.predict(test_x)

print(prediction)
print(prediction[0])
print(prediction[0][0])
print(CATEGORIES[int(prediction[0][0])])

for x in range(len(test_x)): 
    print("Predict: "+CATEGORIES[int(prediction[x][0])]+"\tActual: "+CATEGORIES[int(test_y[x])])


