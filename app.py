import tensorflow as tf
from tensorflow import keras
import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os, sys, random, pickle

categories = ["NORMAL","PNEUMONIA"]

train_x = pickle.load(open("train_x.pickle","rb"))
train_y = pickle.load(open("train_y.pickle","rb"))

train_x = train_x/255

test_x = pickle.load(open("test_x.pickle","rb"))
test_y = pickle.load(open("test_y.pickle","rb"))

test_x = test_x/255

model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(64,(3,3),input_shape = train_x.shape[1:], activation="relu"),
    tf.keras.layers.MaxPool2D(pool_size=(2,2)),

    tf.keras.layers.Conv2D(64,(3,3),input_shape = train_x.shape[1:], activation="relu"),
    tf.keras.layers.MaxPool2D(pool_size=(2,2)),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64,activation="relu"),

    tf.keras.layers.Dense(1),
    tf.keras.layers.Activation('sigmoid')
])

model.compile(optimizer="adam",loss="binary_crossentropy",metrics=['accuracy'])
model.fit(train_x,train_y,batch_size=64, epochs=4)

val_loss,val_acc = model.evaluate(train_x,train_y)
print(val_loss,val_acc)

model.save('3x64.model')


test_loss, test_acc = model.evaluate(test_x,  test_y)

print('\nTest accuracy:', test_acc)

#Prediction don't work :(
# probability_model = tf.keras.Sequential([model,tf.keras.layers.Softmax()])
# predictions = probability_model.predict(test_x)
# print(predictions[0])
# print(categories[test_y[0]])
# print(np.argmax(predictions[0]))
# for x in range(len(test_x)):
#     print("Actual: "+categories[test_y[x]]+"\tPrediction: "+categories[np.argmax(predictions[x])])