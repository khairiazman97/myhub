from tensorflow import keras
from keras.datasets import fashion_mnist
from keras.models import Sequential
from keras.layers import Dense,conv2D,Flatten,activation, MaxPooling2D
import matplotlib.pyplot as plt 
import numpy as np 
import pandas as pd

(x_train,y_train), (x_test,y_test) = fashion_mnist.load_data()

#reshape the data
x_train = x_train.reshape(-1,28,28,1)
x_test = x_test.reshape(-1,28,28,1)

#building model
def build_model():#used randomsearch passes this hyperamater() object
    model = Sequential()

    model.add(conv2D(32,(3,3), input_shape= x_train.shape[1:]))
    model.add(activation("relu"))
    model.add(MaxPooling2D(pool_size=(2,2)))
    
    model.add(conv2D(32,(3,3)))
    model.add(activation("relu"))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Flatten())

    model.add(Dense(10))
    model.add(activation("softmax"))
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics="accuracy")
    
    return model

model = build_model()
model.fit(x_train,y_train,batch_size=64,epochs=5, validation_data=(x_test,y_test))