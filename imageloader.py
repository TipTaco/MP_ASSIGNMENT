# Author: Adrian Shedley
# Date: 23 Aug 2019
#
# Prepared for Machine Perception Assignment
# imageloader.py - load in the images to train from.

import os
import cv2
import numpy as np
from os import listdir

building = list()
building_name = list()
building_grey = list()

directional = list()
directional_name = list()
directional_grey = list()

training = list()
training_aug = list()

kNN = cv2.ml.KNearest_create()

IMG_PATH = '/res/'
TRAIN_VAR_PATH = '/res/training/augmented'
TRAIN_PATH = '/res/training/original'

NUMBERS = ['Zero', 'One', 'Two', 'Three', 'Four', 'Five', 'Six', 'Seven', 'Eight', 'Nine', 'LeftArrow', 'RightArrow']

def init():
    path = os.getcwd() + IMG_PATH
    all_files = listdir(path)
    for file in all_files:
        if file.endswith(".jpg"):
            img = cv2.imread(path + file)

            if img is not None:
                if file.startswith("BS"):
                    building.append(img)
                    building_grey.append(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
                    building_name.append(file)
                elif file.startswith("DS"):
                    directional.append(img)
                    directional_grey.append(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
                    directional_name.append(file)

    path = os.getcwd() + TRAIN_PATH
    all_files = listdir(path)

    for i in range(len(NUMBERS)):
        training.append(list())
        training_aug.append(list())

    for file in all_files:
        # print(file)
        if file.endswith(".jpg"):
            img = cv2.imread(path + '/' + file, 0)
            if img is not None:
                for i, nn in enumerate(NUMBERS):
                    if file.startswith(nn):
                        training[i].append(img[:40,:28])
                        break

    path = os.getcwd() + TRAIN_VAR_PATH
    all_folders = listdir(path)
    for folder in all_folders:
        for augm in listdir(path + '/' + folder):
            if augm.endswith(".jpg"):
                img = cv2.imread(path + '/' + folder + '/' + augm, 0)
                if img is not None:
                    for i, nn in enumerate(NUMBERS):
                        if augm.startswith(nn):
                            training_aug[i].append(img[:40, :28])
                            break

    print("[Image Loader] Loaded", len(building), "building signs and", len(directional), "directional signs")

    # Classifier
    set, lab = training_set()

    # Initiate kNN, train the data, then test it with test data for k=1
    kNN.train(set, cv2.ml.ROW_SAMPLE, lab)

    test, test_lab = test_set()
    _, result, _, _ = kNN.findNearest(test, k=5)
    result = result.reshape(1200)
    print(result.shape, test.shape, test_lab.shape)
    print('kNN classifier for augmented numbers', accuracy(test_lab, result))

   # cv2.imshow('test1 is a ' + str(classify(test[340].reshape(40, 28))), test[340].reshape(40, 28))
    #cv2.waitKey(0)

def deskew(img):
    SZ = 20
    affine_flags = cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR
    m = cv2.moments(img)
    if abs(m['mu02']) < 1e-2:
        return img.copy()
    skew = m['mu11']/m['mu02']
    M = np.float32([[1, skew, -0.5*SZ*skew], [0, 1, 0]])
    img = cv2.warpAffine(img,M,(28, 40),flags=affine_flags)
    return img

def classify(img):
    cv2.imshow('begin', img.astype('uint8'))
    img_resize = cv2.resize(img, (22, 34), interpolation=cv2.INTER_NEAREST)
    img_resize = cv2.copyMakeBorder(img_resize, 3, 3, 3, 3, cv2.BORDER_CONSTANT)
    _, img_resize = cv2.threshold(img_resize, 120, 255, cv2.THRESH_BINARY)
    img_resize = deskew(img_resize)
    cv2.imshow('resize', img_resize.astype('uint8'))
    cv2.waitKey(1)
    print(img_resize.shape)
    vect = np.array(img_resize.reshape(1, 40 * 28))
    print(vect)
    _, result, _, dist = kNN.findNearest(vect, k=5)
    return int(result[0][0]), dist

def test_set():
    return subset(100, 12, 100)

def training_set():
    return subset(0, 12, 100)

def subset(low, digits=12, samples=100):
    set = np.zeros((samples*digits, 40*28), dtype=np.float32)
    set_labels = np.zeros((samples*digits), dtype=np.float32)

    matrixAug = np.array(training_aug)
    #print(training_aug)

    for i in range(digits):
        for j in range(samples):
            _, im = cv2.threshold(matrixAug[i][low + j], 120, 255, cv2.THRESH_BINARY)
            set[i * samples + j] = im.reshape((40*28))
            set_labels[i * samples + j] = i

    #cv2.imshow('test;,', set[0].reshape(40, 28))
    #cv2.waitKey(0)
    #print(set.shape, set_labels.shape)

    return set, set_labels

def accuracy(labels, result):
    # Now we check the accuracy of classification
    # For that, compare the result with test_labels and check which are wrong
    matches = result == labels
    correct = np.count_nonzero(matches)
    accuracy = correct * 100.0 / result.size
    return accuracy