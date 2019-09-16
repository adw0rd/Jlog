from tensorflow.keras.preprocessing.image import load_img, array_to_img
from tensorflow.keras import utils
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.preprocessing.image import load_img, array_to_img

import numpy as np
import matplotlib.pyplot as plt
#%matplotlib inline
classes = ['футболка', 'брюки', 'свитер', 'платье', 'пальто', 'туфли', 'рубашка', 'кроссовки', 'сумка', 'ботинки']

def sample(x_train, y_train):
    plt.figure(figsize=(10, 10))
    for i in range(100, 150):
        plt.subplot(5, 10, i - 100 + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(x_train[i], cmap=plt.cm.binary)
        plt.xlabel(classes[y_train[i]])


def what(n):
    print(classes[np.argmax(predictions[n])])
    plt.imshow(x_train[n].reshape(28, 28), cmap=plt.cm.binary)


(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
# x_train[0].shape
# x_train[0].dtype
sample(x_train, y_train)
x_train = x_train.reshape(60000, 784)
x_train = x_train / 255
y_train = utils.to_categorical(y_train, 10)

model = Sequential()
model.add(Dense(800, input_dim=784, activation='relu'))
model.add(Dense(10, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='SGD', metrics=['accuracy'])
print(model.summary())

model.fit(x_train, y_train, batch_size=200, epochs=100, verbose=1)
predictions = model.predict(x_train)
print(np.argmax(predictions[0]))
print(np.argmax(y_train[0]))

x_test = x_test.reshape(10000, 784)
x_test = x_test / 255
y_test = utils.to_categorical(y_test, 10)
scores = model.evaluate(x_test, y_test, verbose=1)
round(scores[1] * 100, 4)
