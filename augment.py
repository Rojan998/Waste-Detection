import pandas as pd
import numpy as np 
import matplotlib as mlb 
import matplotlib.pyplot as plt 
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import optimizers

train_dir = r'D:\Code\Waste-Detection\data\train\plastic'
train_folder = os.listdir(train_dir)
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
)
for pic in train_folder:
    img = load_img(os.path.join(train_dir,pic))
    x = img_to_array(img)
    x = x.reshape((1,) + x.shape)
    i = 0
    for train_generator in train_datagen.flow(x, batch_size = 1, save_to_dir = "new", save_prefix = "da", save_format = 'jpg'):
        i = i+1
        if i >1:
            break