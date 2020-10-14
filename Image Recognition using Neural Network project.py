# -*- coding: utf-8 -*-
"""
Created on Sun Dec 15 16:56:19 2019

@author: madhu
"""
#importing the dataset
import keras 
import theano
import os
import matplotlib.pyplot as plt
import numpy as np

from keras.datasets import cifar10
from keras.utils import np_utils
from matplotlib import pyplot as plt
import numpy as np
from PIL import Image

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.decomposition import PCA

from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
#importing the dataset
cifar10.data_path = r"C:\Users\madhu\OneDrive\Desktop\ML_Folder\cifar-10-batches-py"

import numpy as np
import matplotlib.pyplot as plt
import pickle
#loading the data
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

# convert array of labeled data(from 0 to nb_classes-1) to one-hot vector
Y_train = np_utils.to_categorical(y_train)
Y_test = np_utils.to_categorical(y_test)
num_classes = Y_test.shape[1]
#printing the shape of Y_train data
print(Y_train.shape)
print(Y_train[0])
#scaling the data
def unpickle(file):
 with open(file, 'rb') as f:
  data = pickle.load(f, encoding='latin-1')
  return data
def load_cifar10_data(data_dir):

 train_data = None
 train_labels = []

 for i in range(1, 6):
  data_dic = unpickle(data_dir + "/data_batch_{}".format(i))
  if i == 1:
   train_data = data_dic['data']
  else:
   train_data = np.vstack((train_data, data_dic['data']))
  train_labels += data_dic['labels']

 test_data_dic = unpickle(data_dir + "/test_batch")
 test_data = test_data_dic['data']
 test_labels = test_data_dic['labels']

 train_data = train_data.reshape((len(train_data), 3, 32, 32))
 train_data = np.rollaxis(train_data, 1, 4)
 train_labels = np.array(train_labels)

 test_data = test_data.reshape((len(test_data), 3, 32, 32))
 test_data = np.rollaxis(test_data, 1, 4)
 test_labels = np.array(test_labels)

 return train_data, train_labels, test_data, test_labels

data_dir = r'C:\Users\madhu\OneDrive\Desktop\ML_Folder\cifar-10-batches-py'

train_data, train_labels, test_data, test_labels = load_cifar10_data(data_dir)
#dimentions after scaling
print(train_data.shape)
print(train_labels.shape)

print(test_data.shape)
print(test_labels.shape)

#reshaping scaled data
x_train = train_data.reshape(train_data.shape[0],-1)
x_test = test_data.reshape(test_data.shape[0], -1)
#scaling the data
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train_scaled = sc.fit_transform(x_train)

#tranforming the data
x_test_scaled = sc.transform(x_test)
y_train = train_labels
y_test = test_labels

#using pca
pca = PCA()
#transforming the train data 
pca.fit_transform(x_train)

pca.explained_variance_.shape
k = 0
total = sum(pca.explained_variance_)
current_sum = 0
# Calculating the  optimal k value 

while(current_sum / total < 0.99):
    current_sum += pca.explained_variance_[k]
    k += 1
k

#Applying PCA with  k value obtained  
pca = PCA(n_components=k, whiten=True)
#transforming the data
x_train_pca = pca.fit_transform(x_train)
x_test_pca = pca.transform(x_test)

#Using random forest
rf = RandomForestClassifier()
rf.fit(x_train_pca, y_train)

# calculating the random forest accuracy 
y_pred_rf = rf.predict(x_test_pca)
random_forest_score = accuracy_score(y_test, y_pred_rf)
random_forest_score

#Using KNN algorithm
knn = KNeighborsClassifier()
knn.fit(x_train_pca, y_train)

## calculating the KNN algorithm accuracy
y_pred_knn = knn.predict(x_test_pca)

knn_score = accuracy_score(y_test, y_pred_knn)
knn_score

#SVM
svc = svm.SVC()
svc.fit(x_train_pca, y_train)

#calculating the SVM accyracy 
y_pred_svm = svc.predict(x_test_pca)
svc_score = accuracy_score(y_test, y_pred_svm)
svc_score


# evaluation of all the models
print("RandomForestClassifier : ", random_forest_score)
print("K Nearest Neighbors : ", knn_score)
print("Support Vector Classifier : ", svc_score)

## Creating a model
##neural network
# start building the model - import necessary layers
from keras.models import Sequential
from keras.layers import Dropout, Activation, Conv2D, GlobalAveragePooling2D
from keras.optimizers import SGD

def allcnn(weights=None):
    # define model type - Sequential
    model = Sequential()

    # add model layers 
    model.add(Conv2D(96, (3, 3), padding = 'same', input_shape=(3,32,32)))
    model.add(Activation('relu'))
    model.add(Conv2D(96, (3, 3), padding = 'same'))
    model.add(Activation('relu'))
    model.add(Conv2D(96, (3, 3), padding = 'same', strides = (2,2)))
    model.add(Dropout(0.5))

    model.add(Conv2D(192, (3, 3), padding = 'same'))
    model.add(Activation('relu'))
    model.add(Conv2D(192, (3, 3), padding = 'same'))
    model.add(Activation('relu'))
    model.add(Conv2D(192, (3, 3), padding = 'same', strides = (2,2)))
    model.add(Dropout(0.5))

    model.add(Conv2D(192, (3, 3), padding = 'same'))
    model.add(Activation('relu'))
    model.add(Conv2D(192, (1, 1), padding = 'valid'))
    model.add(Activation('relu'))
    model.add(Conv2D(10, (1, 1), padding = 'valid'))

    # adding GlobalAveragePooling2D layer with Softmax activation
    model.add(GlobalAveragePooling2D())
    model.add(Activation('softmax'))
    
    # load the weights
    if weights:
        model.load_weights(weights)
    
    # return model
    return model
#conda install blas 
    import os
os.environ['THEANO_FLAGS'] = 'optimizer=None'

import theano
theano.config.optimizer="None"
# define hyper parameters
learning_rate = 0.01
weight_decay = 1e-6
momentum = 0.9

# build model using cnn 
model = allcnn()

# define optimizer and compile model
sgd = SGD(lr=learning_rate, decay=weight_decay, momentum=momentum, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

# print the model summary
print (model.summary())

# define additional training parameters ie no of epoch and batch size
epochs = 1
batch_size = 32

#training the model
model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=epochs, batch_size=batch_size, verbose = 1)
# define hyper parameters
learning_rate = 0.01
weight_decay = 1e-6
momentum = 0.9

# define weights and build model
weights = 'all_cnn_weights_0.9088_0.4994.hdf5'
model = allcnn(weights)

# define optimizer and compile model
sgd = SGD(lr=learning_rate, decay=weight_decay, momentum=momentum, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

# print the model summary
print (model.summary())

# test the model with pretrained weights
scores = model.evaluate(X_test, Y_test, verbose=1)
print("Accuracy: %.2f%%" % (scores[1]*100))

