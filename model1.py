import cv2
import glob
import keras
import os, sys
import pickle
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import numpy as np
import pandas as pd
import tensorflow as tf
#import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,accuracy_score,classification_report
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Flatten, Dense, Dropout



parent_path = os.getcwd()
print(parent_path) 

data_path = parent_path + "/SCUT-FBP5500_v2/Images/"
rating_path = parent_path + "/SCUT-FBP5500_v2/All_Ratings/"

print(data_path) 

data = pd.read_csv('/home/amish/Desktop/ml_project/SCUT-FBP5500_v2/train_test_files/All_labels.txt', sep='\t')
model_name = 'Beauty_Prediction'
model_dir = '/home/amish/Desktop/ml_project'
DATA_DIR = '/home/amish/Desktop/ml_project/SCUT-FBP5500_v2/Images/'
LABELS_FILE = 'All_labels.txt'


target_size = (150,150)

with open('X150.pickle', 'rb') as data:
   X = pickle.load(data)
with open('y150.pickle', 'rb') as data:
   y = pickle.load(data)

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_val, y_val, test_size=0.4, random_state=42)

print(f'\n train shape: {X_train.shape}, {y_train.shape}\n val shape: {X_val.shape}, {y_val.shape}\n test shape: {X_test.shape}, {y_test.shape}\n')

test_val_datagen = ImageDataGenerator()
train_datagen = ImageDataGenerator(horizontal_flip=True,
                                   rotation_range=40,
                                   width_shift_range=0.1,
                                   height_shift_range=0.1,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                  )

batch_size = 32

train_generator = train_datagen.flow(X_train, y_train, batch_size = batch_size)
val_generator = test_val_datagen.flow(X_val, y_val, batch_size = batch_size)
test_generator = test_val_datagen.flow(X_test, y_test, batch_size = batch_size, shuffle=False)

basemodel = MobileNetV2(include_top=False, pooling='avg', weights='imagenet')

model = Sequential(name = model_name)
model.add(basemodel)
model.add(Dense(125))
model.add(Dense(25))
model.add(Dense(5))
model.add(Dense(1))


lr=0.001
model.layers[0].trainable = False
model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=lr))
#model.fit(X_train, y_train, epochs=4, batch_size=30, validation_data=(X_test, y_test))

y_pred = model.predict(X_test)

#print(y_pred)
#print(y_test)

y_pred_labels = (y_pred.astype('int32'))
y_test_labels = (y_test.astype('int32')) 

#print(y_pred_labels)
#print(y_test_labels)

cm = confusion_matrix(y_test_labels, y_pred_labels)
print(cm)

cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = cm, display_labels = [0, 1])

print('accuracy Score :'),accuracy_score(y_test_labels,  y_pred_labels) 

print('Report : ')
print(classification_report(y_test_labels,  y_pred_labels))
