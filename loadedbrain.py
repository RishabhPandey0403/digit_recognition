import os
from pickletools import optimize
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'#hides error message when running in ubuntu
# baseline cnn model for mnist
from numpy import mean
from numpy import std
from matplotlib import pyplot
from sklearn.model_selection import KFold
from keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import BatchNormalization
from keras.layers import Dense
from keras.layers import Flatten
from tensorflow.keras.optimizers import SGD
import pandas as pd
import numpy as np
import os 
from PIL import Image
import matplotlib.pyplot as plt
import glob
from keras.models import model_from_json
from keras.models import load_model

# load json and create model
# json_file = open('nnbrain.json', 'r')
# loaded_model_json = json_file.read()
# json_file.close()
# loaded_model = model_from_json(loaded_model_json)

json_file = open('nnbrain.json', 'r')

loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)

# load weights into new model
loaded_model.load_weights("nnbrain.h5")
print("Loaded model from disk")

loaded_model.save('nnbrain.hdf5')
loaded_model=load_model('nnbrain.hdf5')
df = pd.read_csv("tamilchar.csv")
# print(df["Tamil Character"][8], df["Unicode"][8])
im= Image.open("pngtest/00411.png")
data = np.asarray(im).astype(int)

l = []
l.append(data)
nptotalxdataset = np.array(l)
predictions = loaded_model.predict(nptotalxdataset.reshape((nptotalxdataset.shape[0], 64, 64, 1)))
classes = np.argmax(predictions, axis = 1)
print(classes)
plt.imshow(im)
plt.title(str(df["Tamil Character"][classes] + ','+ df["Unicode"][classes]))
plt.show()
# print(classes[0][np.argmax(classes,axis=-1)])
