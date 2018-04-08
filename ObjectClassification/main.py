import cv2
import os
import re
import numpy as np
from matplotlib import pyplot as plt
from scipy import ndimage
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score,confusion_matrix
from object_classifier import *


brain_filenames = []
face_filenames = []
motorbike_filenames = []
brain_images = []
face_images = []
motorbike_images = []
root,brain_filenames = getFilenames("/home/ashritha/14CO121_Assignment3/brain")
brain_images = readImages(root,brain_filenames)
root,face_filenames = getFilenames("/home/ashritha/14CO121_Assignment3/Faces_easy")
face_images = readImages(root,face_filenames)
root, motorbike_filenames = getFilenames("/home/ashritha/14CO121_Assignment3/Motorbikes")
motorbike_images = readImages(root,motorbike_filenames)
print(len(brain_images))
print(len(face_images))
print(len(motorbike_images))
brain_train = []
brain_test = []
face_train = []
face_test = []
motorbike_train = []
motorbike_test = []
brain_train,brain_test = chooseRandom(brain_images)
face_train,face_test = chooseRandom(face_images)
motorbike_train,motorbike_test = chooseRandom(motorbike_images)
print(len(brain_train))
print(len(face_train))
print(len(motorbike_train))
print(len(brain_test))
print(len(face_test))
print(len(motorbike_test))
train_images = {}
train_images['brain'] = brain_train
train_images['face'] = face_train
train_images['motorbike'] = motorbike_train
test_images = {}
test_images['brain'] = brain_test
test_images['face'] = face_test
test_images['motorbike'] = motorbike_test
num_images = len(train_images['brain'])+len(train_images['face'])+len(train_images['motorbike'])

train_labels = np.array([])
label_count = 0
label_encoding = {}
descriptor_list = []
NUM_CLUSTERS = 250
print(len(train_images))

for label,image_list in train_images.items():
    label_encoding[str(label_count)] = label
    for img in image_list:
        train_labels = np.append(train_labels,label_count)
        desc = extract_features_sift(img)
        descriptor_list.append(desc)
    label_count = label_count + 1

print(len(descriptor_list))
k_means_object,k_means_ret_obj = kmeans(descriptor_list,NUM_CLUSTERS)
number_of_images = len(descriptor_list)
final_histogram = bag_of_words(descriptor_list,k_means_ret_obj,NUM_CLUSTERS,number_of_images)
scale,final_histogram = scale_histogram(final_histogram)
classifier= train(final_histogram,train_labels)

k_means_ret_obj = np.load('kmeans_return_120.npy')
final_histogram = np.load('final_hist_120.npy')

y_true = []
y_predicted = []

for label,image_list in test_images.items():
    print("Processing ",label)
    for img in image_list:
        predicted_label = predict_image(classifier,scale,k_means_object,img,NUM_CLUSTERS)
        print(predicted_label)
        y_true.append(label)
        y_predicted.append(label_encoding[str(int(predicted_label[0]))])
    print(len(y_true))
    print(len(y_predicted))
    score = accuracy_score(y_true,y_predicted)
    print("Accuracy ",score)
    conf_matrix = confusion_matrix(y_true,y_predicted)
    print("Confusion matrix ",conf_matrix)