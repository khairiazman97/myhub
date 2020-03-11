from tensorflow import keras
from keras.datasets import fashion_mnist
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, Activation, MaxPooling2D
from kerastuner.tuners import RandomSearch
from kerastuner.engine.hyperparameters import HyperParameters
import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

#Log_dir = f"{int(time.time())}"

#load data plus splitting the dataset into train and loading
(x_train,y_train),(x_test, y_test) = fashion_mnist.load_data()

#reshape the data
x_train = x_train.reshape(-1,28,28,1)
x_test = x_test.reshape(-1,28,28,1)

#building model # used random search pasess the hyperparameter object

def build_model(hp):
    model = Sequential()

    model.add(Conv2D(32,(3,3),input_shape = x_train.shape[1:]))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Conv2D(32,(2,2)))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Flatten())

    model.add(Dense(10))
    model.add(Activation("softmax"))

    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    return model

#fitting the model
#model = build_model()
#model.fit(x_train,y_train,batch_size=64,epochs=5, validation_data=(x_test,y_test))

#sepcify tuner obeject
tuner = RandomSearch(build_model,
                    objective="val_accuracy", 
                    max_trials=1, 
                    executions_per_trial = 1, 
                    #directory = Log_dir
                    )

#specify tuner object
tuner.search(x = x_train,
            y = y_train,
            epochs = 1,
            batch_size = 64,
            Validation_data=(x_test,y_test)
            )
