# Author Adrian Shedley, 19142946
# Date: 4 oct 2019
# This is the image classifier that contains a kNN and a HOG descriptor that will handle most of the heavy lifting in
#  terms of classifiying possible regions of interest from the feature extractor.

import os
import os.path
import cv2
import numpy as np
from os import listdir
from os import path as pth

# Running environment variables
IMG_PATH = '/res/'
TRAIN_VAR_PATH = '/res/training/augmented'

TRAIN_SET = 'model/training.npy'
TRAIN_LABELS = 'model/training_labels.npy'
TEST_SET = 'model/test.npy'
TEST_LABELS = 'model/test_labels.npy'

# HOG descriptor parameters
FEAT_WIDTH = 28
FEAT_HEIGHT = 40
WIN_SIZE = (FEAT_WIDTH, FEAT_HEIGHT)                # (28, 40)
BLOCK_SIZE = (FEAT_WIDTH // 2, FEAT_HEIGHT // 2)    # (14, 20)
BLOCK_STRIDE = (FEAT_WIDTH // 4, FEAT_HEIGHT // 4)  # (7, 10)
CELL_SIZE = (FEAT_WIDTH // 4, FEAT_HEIGHT // 4)     # (7, 10)
N_BINS = 9
HOG_FEATURES = 324

# kNN Model parameters
NUMBERS = ['Zero', 'One', 'Two', 'Three', 'Four', 'Five', 'Six', 'Seven', 'Eight', 'Nine', 'LeftArrow', 'RightArrow']
FEATURES = 12
TRAIN_FEATURES = 490
TEST_FEATURES = 5
K = 5

# HOG descriptor and kNN initial setup
hog_des = cv2.HOGDescriptor(WIN_SIZE, BLOCK_SIZE, BLOCK_STRIDE, CELL_SIZE, N_BINS)
kNN = cv2.ml.KNearest_create()


def init(show_test=False):
    """Initialise, load training files"""
    if not (pth.exists(TRAIN_SET) and pth.exists(TRAIN_LABELS) and pth.exists(TEST_SET) and pth.exists(TEST_LABELS)):
        # one or more of the needed files for the kNN model are missing, recreate and save them
        train()

    # Reload the data from file (cleanest way even if we just made it)
    training = np.load(TRAIN_SET)
    training_labels = np.load(TRAIN_LABELS)

    # Train the kNN with the loaded data
    kNN.train(training, cv2.ml.ROW_SAMPLE, training_labels)

    if show_test:
        test_set = np.load(TEST_SET)
        test_labels = np.load(TEST_LABELS)
        _, result, _, _ = kNN.findNearest(test_set, k=K)
        result = result.reshape(FEATURES * TEST_FEATURES)
        print('kNN classifier for augmented numbers with HOG: ', accuracy(test_labels, result))


def train():
    """ Perform Knn digit set building and calculation of HOG features"""
    training_aug = list()
    # Initiate the array to have a List willing to accept each of the Augmented images as pixels
    for i in range(len(NUMBERS) + 1):
        training_aug.append(list())

    # Load the augmented images in greyscale
    path = os.getcwd() + TRAIN_VAR_PATH
    all_folders = listdir(path)
    for folder in all_folders:
        # Go inside each of the translation folders
        for augm in listdir(path + '/' + folder):
            # If the file is a jpg, attempt to load it
            if augm.endswith(".jpg"):
                img = cv2.imread(path + '/' + folder + '/' + augm, 0)
                if img is not None:
                    # Match the image title to an internal index for that number using the dictionary
                    for i, nn in enumerate(NUMBERS):
                        if augm.startswith(nn):
                            # Once we have found what this digit was, add it to the list and move on
                            training_aug[i].append(img[:FEAT_HEIGHT, :FEAT_WIDTH])
                            break

    # Create the datasets (HOG Descriptors) and labels for training and testing
    train_set, train_label = training_hog(training_aug)
    test_set, test_label = test_hog(training_aug)

    #numpy save it to file for use with different runs
    np.save(TRAIN_SET, train_set)
    np.save(TRAIN_LABELS, train_label)
    np.save(TEST_SET, test_set)
    np.save(TEST_LABELS, test_label)


def classify(img):
    """ Given a greyscale image img, return a classification ID and sum of errors (confidence)"""

    if img.shape[0] > 0 and img.shape[1] > 0:
        border = 2
        method = cv2.INTER_AREA

        # Decide if the image is upscaled or downscaled, and then select an interpolation method
        if img.shape[0] < 40 or img.shape[1] < 28:  # Upscale
            method = cv2.INTER_CUBIC

        # Resize the image to (28, 40) with the content scaled to (22, 34) and a black border of 2
        img_resize = cv2.resize(img, (FEAT_WIDTH - border * 2, FEAT_HEIGHT - border * 2), interpolation=method)
        # Add a border width 2 of black
        img_resize = cv2.copyMakeBorder(img_resize, border, border, border, border, cv2.BORDER_CONSTANT)
        vect = hog(img_resize).reshape(1, HOG_FEATURES)
        _, result, _, dist = kNN.findNearest(vect, k=K)
        return int(result[0][0]), dist
    else:
        # Size 0 image, will crash, return false classify code ie 13
        return FEATURES + 1, 100


def hog(img):
    return hog_des.compute(img)


def training_hog(master):
    """Build a set of training features and labels"""
    return subset_hog(0, FEATURES, TRAIN_FEATURES, master)


def test_hog(master):
    """Build a set of test features and labels"""
    return subset_hog(TRAIN_FEATURES, FEATURES, TEST_FEATURES, master)


def subset_hog(low, digits, samples=100, master_set=None):
    """Subset the augmented training digits"""
    set = np.zeros((samples * digits, HOG_FEATURES), dtype=np.float32)
    set_labels = np.zeros((samples * digits), dtype=int)
    matrixAug = np.array(master_set)

    # Loop over each type of digit, and get enough samples from each
    for i in range(digits):
        for j in range(samples):
            # Copy the greyscale
            im = matrixAug[i][low + j]
            # Perform HOG
            hh = hog(im)
            # reshape to 1x324
            set[i * samples + j] = hh.reshape(HOG_FEATURES)
            # Save the HOG
            set_labels[i * samples + j] = int(i)

    return set, set_labels


def accuracy(labels, result):
    # Compare all of the provided labels with the classified labels to get a training percentage
    matches = result == labels
    correct = np.count_nonzero(matches)
    accuracy = correct * 100.0 / result.size
    return accuracy