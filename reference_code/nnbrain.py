import os
from pickletools import optimize
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'#hides error message when running in ubuntu
# baseline cnn model for mnist
import tensorflow as tf
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

#tensorflow
# load train and test dataset
def load_dataset():
	totalxdataset = []
	totalydataset = []
	for i in range(16, 229):
		drt = 'pngtrain/usr_'+str(i)#,'/*.tiff'
		if os.path.exists(drt):
			print(drt)
			# os.mkdir("pngtrain/usr_"+str(i))
			for filename in glob.glob(drt+'/*.png'):
				substr = int(str(filename[-10:-7]))
				# print(substr)
				im= Image.open(filename)
				data = np.asarray(im).astype(int)
				totalxdataset.append(data)
				totalydataset.append(substr)
				im.close()
	nptotalxdataset = np.array(totalxdataset)
	nptotalydataset = np.array(totalydataset)

	randomize = np.arange(len(nptotalydataset))
	np.random.shuffle(randomize)
	nptotalxdataset = nptotalxdataset[randomize]
	nptotalydataset = nptotalydataset[randomize]

	# trainX, trainY, testX, testY = []
	trainX = nptotalxdataset[:40546]
	testX = nptotalxdataset[40547:]
	trainY = nptotalydataset[:40546]
	testY = nptotalydataset[40547:]	

	trainX = np.array(trainX)
	testX = np.array(testX)
	trainY = np.array(trainY)
	testY = np.array(testY)

	# load dataset
	# (trainX, trainY), (testX, testY) = mnist.load_data()
	# # reshape dataset to have a single channel
	trainX = trainX.reshape((trainX.shape[0], 64, 64, 1))
	testX = testX.reshape((testX.shape[0], 64, 64, 1))
	# one hot encode target values
	trainY = to_categorical(trainY)
	testY = to_categorical(testY)
	return trainX, trainY, testX, testY

# scale pixels
# def prep_pixels(train, test):
# 	# convert from integers to floats
# 	train_norm = train.astype('float32')
# 	test_norm = test.astype('float32')
# 	# normalize to range 0-1
# 	train_norm = train_norm / 1.0
# 	test_norm = test_norm / 1.0
# 	# return normalized images
# 	return train_norm, test_norm

# define cnn model
def define_model():
	model = Sequential()
	model.add(Conv2D(16,(3,3), activation='relu', kernel_initializer='he_uniform', input_shape=(64,64,1)))
	model.add(BatchNormalization())
	model.add(Conv2D(16,(3,3), activation='relu', kernel_initializer='he_uniform'))
	model.add(BatchNormalization())
	model.add(MaxPooling2D((2,2)))
	model.add(BatchNormalization())
	# model.add(Flatten())
	model.add(Conv2D(32,(3,3), activation='relu', kernel_initializer='he_uniform'))
	model.add(BatchNormalization())
	model.add(Conv2D(32,(3,3), activation='relu', kernel_initializer='he_uniform'))
	model.add(BatchNormalization())
	model.add(MaxPooling2D((2,2)))
	model.add(BatchNormalization())
	# model.add(Flatten())
	model.add(Conv2D(64,(3,3), activation='relu', kernel_initializer='he_uniform'))
	model.add(BatchNormalization())
	model.add(Conv2D(64,(3,3), activation='relu', kernel_initializer='he_uniform'))
	model.add(BatchNormalization())
	model.add(MaxPooling2D((2,2)))
	model.add(BatchNormalization())
	model.add(Flatten())
	model.add(Dense(1064,activation='relu', kernel_initializer='he_uniform'))
	model.add(Dense(512,activation='relu', kernel_initializer='he_uniform'))
	model.add(Dense(156,activation='softmax'))
	opt = SGD(learning_rate=0.03, momentum=0.5)
	model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
	return model

# evaluate a model using k-fold cross-validation
def evaluate_model(trainX, trainY, testX, testY):
		scores, histories = list(), list()
		
		strategy = tf.distribute.MirroredStrategy()
		print("Number of devices: {}".format(strategy.num_replicas_in_sync))

		# Open a strategy scope.
		with strategy.scope():
			# Everything that creates variables should be under the strategy scope.
			# In general this is only model construction & `compile()`.
			model = define_model()
		history = model.fit(trainX, trainY, epochs=10, validation_data=(testX, testY), verbose=0)
		_, acc = model.evaluate(testX, testY, verbose=0)
		print('> %.3f' % (acc * 100.0))
		scores.append(acc)
		histories.append(history)
		model.save('nnbrain.hdf5')
		model_json = model.to_json()
		with open("nnbrain.json", "w") as json_file:
			json_file.write(model_json)
		# serialize weights to HDF5
		model.save_weights("nnbrain.h5")
		print("Saved model to disk")
		return scores, histories
	# prepare cross validation
	# kfold = KFold(n_folds, shuffle=True, random_state=1)
	# # enumerate splits
	# for train_ix, test_ix in kfold.split(dataX):
	# 	# define model
	# 	model = define_model()
	# 	# select rows for train and test
	# 	trainX, trainY, testX, testY = dataX[train_ix], dataY[train_ix], dataX[test_ix], dataY[test_ix]
	# 	# fit model
	# 	history = model.fit(trainX, trainY, epochs=10, batch_size=32, validation_data=(testX, testY), verbose=0)
	# 	# evaluate model
	# 	_, acc = model.evaluate(testX, testY, verbose=0)
	# 	print('> %.3f' % (acc * 100.0))
	# 	# stores scores
	# 	scores.append(acc)
	# 	histories.append(history)


# plot diagnostic learning curves
def summarize_diagnostics(histories):
	for i in range(len(histories)):
		# plot loss
		pyplot.subplot(2, 1, 1)
		pyplot.title('Cross Entropy Loss')
		pyplot.plot(histories[i].history['loss'], color='blue', label='train')
		pyplot.plot(histories[i].history['val_loss'], color='orange', label='test')
		# plot accuracy
		pyplot.subplot(2, 1, 2)
		pyplot.title('Classification Accuracy')
		pyplot.plot(histories[i].history['accuracy'], color='blue', label='train')
		pyplot.plot(histories[i].history['val_accuracy'], color='orange', label='test')
	pyplot.show()

# summarize model performance
def summarize_performance(scores):
	# print summary
	print('Accuracy: mean=%.3f std=%.3f, n=%d' % (mean(scores)*100, std(scores)*100, len(scores)))
	# box and whisker plots of results
	pyplot.boxplot(scores)
	pyplot.show()

# run the test harness for evaluating a model
def run_test_harness():
	# load dataset

	trainX, trainY, testX, testY = load_dataset()
	# prepare pixel data
	# trainX, testX = prep_pixels(trainX, testX)
	# evaluate model
	scores, histories = evaluate_model(trainX, trainY, testX, testY)
	# learning curves
	summarize_diagnostics(histories)
	# summarize estimated performance
	summarize_performance(scores)

# entry point, run the test harness
run_test_harness()