import os
import pandas as pd
import numpy as np 
import matplotlib as mlb 
import matplotlib.pyplot as plt 

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

test_dir = r"D:\Code\Waste-Detection\data\test"
test_datagen = ImageDataGenerator(rescale = 1./255)
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size = (150, 150),
    batch_size = 20,
    class_mode='categorical'
)
name = os.listdir(test_dir)
print(name)
# load model
model = load_model('Best_ModelAC2.h5')
#model.evaluate(X_test, Y_test)

print(model.evaluate_generator(test_generator))


#print(name[int(model.predict_generator(test_generator, 1).round(0)[0][0])])
