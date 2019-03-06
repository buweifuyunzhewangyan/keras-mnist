# -*- coding: utf-8 -*-
"""
Created on Wed Jan 16 10:05:24 2019

@author: Elane
"""

from keras.applications.vgg16 import VGG16
from keras.applications.imagenet_utils import preprocess_input
from keras.layers import Dense,Flatten,Input,Dropout
from keras.models import Sequential
from keras.models import Model
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
#加载数据集
from keras.datasets import mnist
import numpy as np
import cv2
import h5py


inputshape = (224,224)
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train, y_train = X_train[:1000], y_train[:1000]#训练集1000条
X_test, y_test = X_test[:100], y_test[:100]#测试集100条

#变换图像size
X_train = [cv2.cvtColor(cv2.resize(i, inputshape), cv2.COLOR_GRAY2RGB)
           for i in X_train]

X_test = [cv2.cvtColor(cv2.resize(i, inputshape), cv2.COLOR_GRAY2RGB)
           for i in X_test]
X_train = np.concatenate([arr[np.newaxis] for arr in X_train]).astype('float32')
X_test = np.concatenate([arr[np.newaxis] for arr in X_test]).astype('float32')
#数据归一化处理
X_train = preprocess_input(np.array(X_train))
X_test = preprocess_input(np.array(X_test))
print(y_train.shape)

vgg_model = VGG16(include_top=False,weights='imagenet',input_shape=(224,224,3))

add_model = Sequential()
add_model.add(Flatten(input_shape=vgg_model.output_shape[1:]))
add_model.add(Dense(4096,activation='relu'))
add_model.add(Dense(2048,activation='relu'))
add_model.add(Dropout(0.5))
add_model.add(Dense(10,activation='softmax'))

model = Model(vgg_model.input, add_model(vgg_model.output))
model.compile(loss='categorical_crossentropy', optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
            metrics=['accuracy'])

#model.summary()

train_datagen = ImageDataGenerator(
        shear_range=0.2,
        zoom_range=0.2,
        rotation_range=30, 
        width_shift_range=0.1,
        height_shift_range=0.1, 
        horizontal_flip=True)
train_datagen.fit(X_train)

test_datagen = ImageDataGenerator(
        shear_range=0.2,
        zoom_range=0.2,
        rotation_range=30, 
        width_shift_range=0.1,
        height_shift_range=0.1, 
        horizontal_flip=True)
test_datagen.fit(X_test)

def tran_y(y):
    y_ohe = np.zeros(10)
    y_ohe[y] = 1
    return y_ohe

y_train_ohe = np.array([tran_y(y_train[i]) for i in range(len(y_train))])
y_test_ohe = np.array([tran_y(y_test[i]) for i in range(len(y_test))])

batch_size = 128 # tune it
epochs = 10 # increase it
#history = model.fit_generator(
#    train_datagen.flow(X_train, y_train_ohe, batch_size=batch_size),len(X_train),nb_epoch=epochs)
train_datagen = train_datagen.flow(X_train, y_train_ohe, batch_size=batch_size)
 
train = vgg_model.predict_generator(train_datagen,10)
with h5py.File("mnist.h5") as h:
    h.create_dataset("train", data=train)


















