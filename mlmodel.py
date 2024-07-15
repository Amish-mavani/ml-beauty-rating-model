
import cv2
import glob
import pickle
import random
import keras
import os, sys
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import numpy as np
import pandas as pd
import tensorflow as tf
#import seaborn as sns
#import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
#from sklearn import metrics
#from sklearn.metrics import confusion_matrix
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.preprocessing.image import ImageDataGenerator,img_to_array
from tensorflow.keras.applications import VGG16,MobileNetV2
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical


parent_path = os.getcwd()
print(parent_path) 

data_path = parent_path + "/SCUT-FBP5500_v2/Images/"
rating_path = parent_path + "/SCUT-FBP5500_v2/All_Ratings/"

print(data_path) 

data = pd.read_csv('/ml_project/SCUT-FBP5500_v2/train_test_files/All_labels.txt', sep='\t')
model_name = 'Beauty_Prediction'
model_dir = '/ml_project'
DATA_DIR = '/ml_project/SCUT-FBP5500_v2/Images/'
LABELS_FILE = 'All_labels.txt'

def create_dataset(target_size):
    X = []
    y = []
    labels_dict = get_labels_dict()
    img_files = glob.glob(DATA_DIR + '*.jpg')
    print(f'insert {len(img_files)} images into dataset')
    for f in img_files:
        img = preprocess_image(cv2.imread(f), target_size)
        X.append(img)
        y.append(labels_dict[os.path.split(f)[-1]])
    return np.array(X), np.array(y)

def get_labels_dict():
    labels_dict = {}
    with open(DATA_DIR + LABELS_FILE) as fp:
        for line in fp:
            img,label = line.split(' ', 1)
            labels_dict[img] = float(label)
    return labels_dict

def preprocess_image(image,target_size):
    return cv2.resize(cv2.cvtColor(image, cv2.COLOR_BGR2RGB),target_size) / .255

target_size = (200,200)
#target_size = (350,350)
#X,y = create_dataset(target_size)

with open('X.pickle', 'rb') as data:
   X = pickle.load(data)
with open('y.pickle', 'rb') as data:
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
model_path= model_dir + '/' + model_name + '.h5'
if not os.path.isdir(model_dir): os.mkdir(model_dir)

basemodel = MobileNetV2(include_top=False, pooling='avg', weights='imagenet')

model = Sequential([
    basemodel,
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(125, activation='relu'),
    Dropout(0.3),
    Dense(1) 
])

epochs = 5
lr=0.001
model.layers[0].trainable = False
model.compile(loss='mean_squared_error', optimizer=Adam(learning_rate=lr))
#print(model.summary())

base_model = VGG16(weights='imagenet', include_top=False, input_shape=(200, 200, 3))
print(model.trainable_weights)
base_model.trainable = False

model1 = Sequential([
    base_model,
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(5, activation='softmax') 
])

model1.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
#print(model1.summary())

#y_pred = model.predict(X_test)
#y_pred1 = model1.predict(X_test)

#print(y_test)

#print("\n for model1 : MobileNetV2\n")
#print(y_pred)

#y_pred_labels = (y_pred.astype('int32')) + 1 
#y_test_labels = (y_test.astype('int32')) 

#print(y_pred_labels)
#print(y_test_labels)

#cm = confusion_matrix(y_test_labels, y_pred_labels)
#print(cm)

#cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = cm, display_labels = [0, 1])

#print("\n for model2 : VGG16 \n")
#print(y_pred1)

#y_pred_labels = np.argmax(y_pred1, axis=1) + 1 
#y_test_labels = (y_test.astype('int32')) 

#print(y_pred_labels)
#print(y_test_labels)

#cm = confusion_matrix(y_test_labels, y_pred_labels)
#print(cm)

#cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = cm, display_labels = [0, 1])

def preprocess_image1(img):
    img = cv2.resize(img, (200, 200))
    img = img[..., ::-1].astype(np.float32)  # Convert BGR to RGB
    img = tf.keras.applications.vgg16.preprocess_input(img)  # Preprocess input for VGG16
    return img

def predict_beauty_rating(model, img):
    img = preprocess_image1(img)
    img = np.expand_dims(img, axis=0)  
    prediction = model.predict(img)
    return np.argmax(prediction) + 1 

cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret:
        break
    cv2.imshow('Beauty Prediction', frame)

    score = model.predict(np.expand_dims(preprocess_image(frame,(350,350)), axis=0))
    text1 = f'{str(round(score[0][0],3))}'
    
    rating = predict_beauty_rating(model1, frame)
    
    print("Predicted Beauty Rating from model1 : ", score, text1)
    print("Predicted Beauty Rating from model2: ", rating)

    #cv2.imshow('Beauty Prediction', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()



