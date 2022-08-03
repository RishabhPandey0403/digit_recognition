import enum
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_digits

## Load in data set
digits = load_digits()

## Checking dataset values
images_labels = list(zip(digits.images, digits.target))
plt.figure(figsize=(5,5))
for index, (image, label) in enumerate(images_labels[:15]):
    plt.subplot(3,5,index+1)
    plt.axis('off')
    plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
    plt.title('%i' % label)

#### Classification

## Flatten images
num_samples = len(digits.images)
data = digits.images.reshape((num_samples,-1))

## Create a support vector classifier
classifier = svm.SVC(gamma=0.001)

## Splitting data into testing and training subsets
x_train, x_test, y_train, y_test = train_test_split(data, digits.target, test_size=0.5, shuffle = False)

## Learning the digits on the training subset
classifier.fit(x_train, y_train)

## Predicting the value
predicted = classifier.predict(x_test)

_, axes = plt.subplots(1,4, figsize=(10,3))
for ax, image, prediction in zip(axes, x_test, predicted):
    ax.set_axis_off()
    image = image.reshape(8,8)
    ax.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
    ax.set_title(f"Prediction: {prediction}")

print(
    f"Classification report for classifier {classifier}:\n"
    f"{metrics.classification_report(y_test, predicted)}\n"
)