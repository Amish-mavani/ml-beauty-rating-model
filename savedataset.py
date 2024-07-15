import pickle
import cv2
import glob
import os, sys
import numpy as np
import pandas as pd


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
X,y = create_dataset(target_size)

with open('X.pickle', 'wb') as output:
    pickle.dump(X, output)

with open('y.pickle', 'wb') as output:
    pickle.dump(y, output)

#then you can load it back in another script.
#import pickle
#with open('X.pickle', 'rb') as data:
#   X = pickle.load(data)
#with open('y.pickle', 'rb') as data:
#   y = pickle.load(data)