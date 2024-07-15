import numpy as np
import pandas as pd
import glob
import os
import pickle
import tensorflow as tf
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout
import cv2
#import matplotlib.pyplot as plt

data = pd.read_csv('/ml_project/SCUT-FBP5500_v2/train_test_files/All_labels.txt', sep='\t')

data.head()

DATA_DIR = '/ml_project/SCUT-FBP5500_v2/Images/'
LABELS_FILE = 'All_labels.txt'

target_size = (150,150)

with open('X150.pickle', 'rb') as data:
   X = pickle.load(data)
with open('y150.pickle', 'rb') as data:
   y = pickle.load(data)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

base_model = VGG16(weights='imagenet', include_top=False, input_shape=(150, 150, 3))


model = Sequential([
    base_model,
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    #Dense(1, activation='softmax')
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit(X_train, y_train, epochs=1, batch_size=30, validation_data=(X_test, y_test))

y_pred = model.predict(X_test)

#print(y_pred)
#print(y_test)

y_pred_labels = np.argmax(y_pred, axis=1) + 1 
y_test_labels = (y_test.astype('int32')) 

#print(y_pred_labels)
#print(y_test_labels)

cm = confusion_matrix(y_test_labels, y_pred_labels)
print(cm)

cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = cm, display_labels = [0, 1])

print('accuracy Score :'),accuracy_score(y_test_labels,  y_pred_labels) 

print('Report : ')
print(classification_report(y_test_labels,  y_pred_labels))

#cm_display.plot()
#plt.show()
