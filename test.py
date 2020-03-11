#first neural network with keras tutorial
import pandas as pd
from numpy import loadtxt
from tensorflow import keras as t
from keras.models import Sequential
from keras.layers import Dense

#load the dataset using numpy
dataset=loadtxt('pima-indians-diabetes.csv', delimiter=',')

#split the data set to respective x and y
x=dataset[:,0:8]
y=dataset[:, 8]

#split dataset into train and validation

train_x, train_y, test_x, test_y = split(x, y, train_size=0.7, random_state=0)

#step define keras model
model = Sequential()
model.add(Dense(12, input_dim=8, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

#model compiling step
model.compile(loss='mean_squared_error',optimizer='adam', metrics=['accuracy'])

#fitting the training   model
model.fit(train_x,train_y,epochs=150,batch_size=10)

#model performance
_, accuracy=model.evaluate(train_x,train_y,verbose=0)
print('accuracy: %.2f'% (accuracy*100))
#--------------------------------------------------------------------------#
#model prediction
pred= model.predict(train_x)

#round the prediction: if necessary
rounded = [round(train_x[0]) for train_x in pred]

# predict class with the model
pred1=model.predict_classes(train_x)

#summarize first 5 cases only
for i in range(5):
    print("%s =>%d(expected %d)" % (train_x[i].tolist(), pred1[i], train_y[i]))