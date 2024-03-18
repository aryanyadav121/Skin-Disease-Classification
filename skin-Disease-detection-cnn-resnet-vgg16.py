#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from keras.layers.normalization import BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
from keras.utils import np_utils
from keras.layers import Dropout
import matplotlib.pyplot as plt
import numpy as np
import os
from tqdm import tqdm
import cv2
from glob import glob
import seaborn as sns
sns.set()

import sklearn
import skimage
from skimage.transform import resize

import random
from skimage.color import rgb2gray
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import accuracy_score


# # Load and Preprocessing

# In[2]:


train = r"E:\MAJOR_PRO\kaggle\train"

test = r"E:\MAJOR_PRO\kaggle\test"



LOAD_FROM_IMAGES = True

def get_data(folder):
    x = []
    y = []
    for folderName in os.listdir(folder):
        if not folderName.startswith("."):
            if folderName in ["benign"]:
                label = 0
            elif folderName in ["malignant"]:
                label = 1
            else:
                label = 2
            for image_filename in tqdm(os.listdir(folder +"/" +folderName+"/")):
                img_file = cv2.imread(folder + "/" +folderName + "/" + image_filename)
                if img_file is not None:
                    img_file = skimage.transform.resize(img_file,(224,224,3), mode = "constant",anti_aliasing=True)
                    #img_file = rgb2gray(img_file)
                    img_arr = np.asarray(img_file)
                    x.append(img_arr)
                    y.append(label)
    x = np.asarray(x)
    y = np.asarray(y)
    return x,y

if LOAD_FROM_IMAGES:
    X_train,y_train = get_data(train)
    X_test, y_test = get_data(test)
    
    np.save("xtrain.npy",X_train)
    np.save("ytrain.npy",y_train)
    np.save("xtest.npy",X_test)
    np.save("ytest.npy",y_test)
else:
    X_train = np.load("xtrain.npy")
    y_train = np.load("ytrain.npy")
    X_test = np.load("xtest.npy")
    y_test = np.load("ytest.npy")


# In[3]:


X_train=X_train.astype('float32')
y_train
X_test=X_test.astype('float32')
y_test


# # Visualization

# In[4]:


def plot_histogram(a):
    plt.figure(figsize=(12,6))
    plt.subplot(1,2,1)
    plt.hist(a.ravel(),bins=255)
    plt.subplot(1,2,2)
    plt.imshow(a,vmin=0,vmax=1)
    plt.show()
    
    
plot_histogram(X_train[2])


# ## Benign

# In[5]:


glob_img = glob("../input/skin-cancer-malignant-vs-benign/train/benign/**")

def plot(images):
    z = random.sample(images,3)
    plt.figure(figsize=(20,20))
    plt.subplot(131)
    plt.imshow(cv2.imread(z[0]))
    plt.subplot(132)
    plt.imshow(cv2.imread(z[1]))
    plt.subplot(133)
    plt.imshow(cv2.imread(z[2]))

plot(glob_img)
    


# ## Malignant

# In[6]:


glob_img = glob("../input/skin-cancer-malignant-vs-benign/train/malignant/**")

def plot(images):
    z = random.sample(images,3)
    plt.figure(figsize=(20,20))
    plt.subplot(131)
    plt.imshow(cv2.imread(z[0]))
    plt.subplot(132)
    plt.imshow(cv2.imread(z[1]))
    plt.subplot(133)
    plt.imshow(cv2.imread(z[2]))

plot(glob_img)


# ## benign

# In[7]:


print("Benign")

glob_img = glob("../input/skin-cancer-malignant-vs-benign/train/benign/**")
i_=0
plt.rcParams["figure.figsize"] =(20.0,20.0)
plt.subplots_adjust(wspace=0,hspace=1)
for i in  glob_img[0:20]:
    img=cv2.imread(i)
    img=cv2.resize(img,(128,128))
    plt.subplot(5,5,i_+1)
    plt.imshow(img);plt.axis("off")
    i_ +=1


# ## malignant

# In[8]:


print("Malignant")

glob_img = glob("../input/skin-cancer-malignant-vs-benign/train/malignant/**")
i_=0
plt.rcParams["figure.figsize"] =(20.0,20.0)
plt.subplots_adjust(wspace=0,hspace=1)
for i in  glob_img[0:20]:
    img=cv2.imread(i)
    img=cv2.resize(img,(128,128))
    plt.subplot(5,5,i_+1)
    plt.imshow(img);plt.axis("off")
    i_ +=1


# In[9]:


plt.figure(figsize=(8,4))

map_characters = {0:"benign",1:"malignant"}
dict_characters = map_characters

df = pd.DataFrame()
df["labels"]=y_train
lab = df["labels"]
dist=lab.value_counts()
sns.countplot(lab)
print(dict_characters)


# ## Data Generator

# In[10]:


generatordata = ImageDataGenerator(zoom_range = 0.2,height_shift_range = 0.1,
                                   width_shift_range = 0.2,rotation_range = 12)


# In[11]:


X_train = X_train/255
X_test = X_test/255

X_train_R= X_train.reshape(len(X_train),224,224,3)
X_test_R= X_test.reshape(len(X_test),224,224,3)
y_train = np_utils.to_categorical(y_train, num_classes= 2)
y_test = np_utils.to_categorical(y_test, num_classes= 2)


# # CNN

# In[12]:


model = models.Sequential()

model.add(layers.Conv2D(64,(3,3),padding="same",activation="relu",kernel_initializer="glorot_uniform",input_shape=(224,224,3)))
model.add(layers.MaxPooling2D((2,2)))
model.add(BatchNormalization())
model.add(Dropout(0.5))

model.add(layers.Conv2D(64,(3,3),padding="same",activation="relu",kernel_initializer="glorot_uniform"))
model.add(layers.MaxPooling2D((2,2)))

model.add(layers.Conv2D(64,(3,3),padding="same",activation="relu",kernel_initializer="glorot_uniform"))
model.add(layers.MaxPooling2D((2,2)))

model.add(layers.Flatten())

model.add(layers.Dense(128,activation="relu",kernel_initializer="normal"))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(layers.Dense(2,activation="softmax"))


# In[13]:


model.summary()


# In[14]:


model.compile(optimizer = "adam" , loss = "binary_crossentropy", metrics=["accuracy"])


# In[15]:


batch_size=32
epochs=100


# #### Thanks @fanconic for the learning rate

# In[16]:


from keras.callbacks import ReduceLROnPlateau
learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', 
                                            patience=5, 
                                            verbose=1, 
                                            factor=0.5, 
                                            min_lr=1e-7)


# In[17]:


from tensorflow.keras.callbacks import EarlyStopping

custom_early_stopping = EarlyStopping(
    monitor='val_accuracy', 
    patience=95,
    restore_best_weights=True
)


# In[18]:


history =  model.fit(generatordata.flow(X_train_R, y_train, batch_size=batch_size),epochs=epochs,
                              verbose=1,
                              validation_data =(X_test_R,y_test),callbacks=[learning_rate_reduction,custom_early_stopping])


# In[19]:


score = model.evaluate(X_test_R, y_test, batch_size=batch_size, verbose=1)

print('Test loss:', score[0])
print('Test accuracy:', score[1])


# # ResNet V2

# In[20]:


from tensorflow.keras.applications.resnet_v2 import ResNet50V2


# In[21]:


input_shape = (224,224,3)
epochs = 100
batch_size = 16


# In[22]:


model = ResNet50V2(include_top=True,
                 weights= None,
                 input_tensor=None,
                 input_shape=input_shape,
                 pooling='max ',
                 classes=2)

model.compile(optimizer = "adam" ,
              loss = "binary_crossentropy", 
              metrics=["accuracy"])


# In[23]:


history = model.fit(X_train_R, y_train, validation_split=0.2,
                    epochs= epochs, batch_size= batch_size, verbose=1,callbacks=[learning_rate_reduction,custom_early_stopping] )


# In[24]:


score = model.evaluate(X_test_R, y_test, batch_size=batch_size, verbose=1)

print('Test loss:', score[0])
print('Test accuracy:', score[1])


# # VGG16

# In[25]:


from tensorflow.keras.applications import vgg16


# In[26]:


def create_vgg16():  
  model = vgg16.VGG16(include_top=True, weights=None, input_tensor=None, input_shape=(224,224,3), pooling="max", classes=2)

  return model


# In[27]:


vgg16_model = create_vgg16()  
vgg16_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])  


# In[28]:


vgg16 = vgg16_model.fit(x=X_train_R,
                        y=y_train, batch_size=32,
                        epochs=100, verbose=1,
                        validation_data=(X_test_R, y_test), shuffle=True,callbacks=[learning_rate_reduction,custom_early_stopping])  


# In[29]:


score = keras_model.h5.evaluate(X_test_R, y_test, batch_size=batch_size, verbose=1)

print('Test loss:', score[0])
print('Test accuracy:', score[1])


# In[ ]:




