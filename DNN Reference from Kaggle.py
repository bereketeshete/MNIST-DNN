## Get the basics

# REFERENCE - https://www.kaggle.com/prashant111/comprehensive-guide-to-ann-with-keras
import keras


# load dataset
from keras.datasets import mnist


# split dataset into training and test set
(x_train, y_train), (x_test, y_test) = mnist.load_data()

############################################################################

## Display images

import matplotlib.pyplot as plt

plt.imshow(x_train[7], cmap=plt.cm.binary)

# View the labels

print(y_train[7])

############################################################################

## Data representation in Keras


# View number of dimensions of tensor

print(x_train.ndim)


# View the dimension of tensor

print(x_train.shape)


# View the data type of tensor

print(x_train.dtype)


#############################################################################

## Data normalization in Keras


# Scale the input values to type float32

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')


# Scale the input values within the interval [0, 1]
x_train /= 255
x_test /= 255


# Reshape the input values
x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)

print(x_train.shape)
print(x_test.shape)


from tensorflow.keras.utils import to_categorical

print(y_test[0])
print(y_train[0])

print(y_train.shape)
print(x_test.shape)

y_train = to_categorical(y_train, num_classes=10)
y_test = to_categorical(y_test, num_classes=10)

print(y_test[0])
print(y_train[0])

print(y_train.shape)
print(y_test.shape)

##############################################################################

## Define the model

from keras.models import Sequential
from keras.layers.core import Dense, Activation

model = Sequential()
model.add(Dense(150, input_shape=(784,)))
model.add(Activation('sigmoid'))
model.add(Dense(150))               # hidden layer 1
model.add(Activation('sigmoid'))
model.add(Dense(120))               # hidden layer 2
model.add(Activation('sigmoid'))
model.add(Dense(10, activation='softmax'))

##############################################################################

## Model summary

model.summary()

##############################################################################

## Implementation of Neural Network in Keras


# Compiling the model with compile() method

model.compile(loss="categorical_crossentropy",
optimizer="sgd", metrics = ['accuracy'])


# Training the model with fit() method

model.fit(x_train, y_train, batch_size=64, epochs=20)


# Evaluate model with evaluate() method

test_loss, test_acc = model.evaluate(x_test, y_test)


###############################################################################

## Accuracy of the model

print('Test accuracy:', round(test_acc,4))
