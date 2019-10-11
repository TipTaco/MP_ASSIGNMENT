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
svm = cv2.ml.SVM_create()
aNN = cv2.ml.ANN_MLP_create()

hog_des = None

FEATURES = 12

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

    for i in range(len(NUMBERS) + 1):
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

    #for i in range(200):
    #    new_img = np.random.randint(256, size=(40, 28), dtype=np.uint8)
    #    training_aug[12].append(new_img)
        #cv2.imshow('new_img', new_img)
        #cv2.imwrite('res/training/noise/noise.' + str(i) + '.jpg', new_img)

    print("[Image Loader] Loaded", len(building), "building signs and", len(directional), "directional signs")

    winSize = (28, 40)
    blockSize = (14, 20)
    blockStride = (7, 10)
    cellSize = (7, 10)
    nbins = 9
    derivAperture = 1
    winSigma = 4.
    histogramNormType = 0
    L2HysThreshold = 2.0000000000000001e-01
    gammaCorrection = 0
    nlevels = 64
    global hog_des
    hog_des = cv2.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins) #, derivAperture, winSigma,
                           # histogramNormType, L2HysThreshold, gammaCorrection, nlevels)


    # Clean classifier KNN
    setH, label = training_hog()
    kNN.train(setH, cv2.ml.ROW_SAMPLE, label)
    kNN.save("knn1.dat")

    # test it
    test, test_lab = test_hog()
    _, result, _, _ = kNN.findNearest(test, k=5)
    result = result.reshape(FEATURES * 100)
    print(result.shape, test.shape, test_lab.shape)
    print('kNN classifier for augmented numbers', accuracy(test_lab, result))

    # Classifier


   # labels = np.zeros((set.shape[0], FEATURES), dtype=np.float32)
   # for i, labl in enumerate(lab):
    #    labels[i][int(labl)] = 1.0

    #aNN.load("aNN1.dat")
    #aNN.setLayerSizes(np.array([28*40, 100, 25, 12], dtype=np.uint16))
    #aNN.setTrainMethod(cv2.ml.ANN_MLP_BACKPROP)
    #aNN.setActivationFunction(cv2.ml.ANN_MLP_SIGMOID_SYM)
    #aNN.train(set, cv2.ml.ROW_SAMPLE, labels)
    #aNN.save("aNN1.dat")


    # Initiate kNN, train the data, then test it with test data for k=1


    #test, test_lab = test_set()
    #_, result, _, _ = kNN.findNearest(test, k=5)
    #result = result.reshape(FEATURES * 100)
    #print(result.shape, test.shape, test_lab.shape)
    #print('kNN classifier for augmented numbers', accuracy(test_lab, result))

    #labels_test = np.zeros((set.shape[0], FEATURES), dtype=np.float32)
   # for i, labl in enumerate(test_lab):
     #   labels_test[i][int(labl)] = 1.0

    ## hog test with SVM
    #hogdata, hoglab = training_hog()

    #svm.setType(cv2.ml.SVM_C_SVC)
    #svm.setC(2.67)
    #svm.setGamma(5.383)
    #svm.setKernel(cv2.ml.SVM_DEGREE)

    #svm.train(hogdata, cv2.ml.ROW_SAMPLE, hoglab)
    #svm.save('svm_data.dat')

    #testhog, testhoglab = test_hog()
    #print(testhog.shape)

    #_, result = svm.predict(testhog)
    #result = result.reshape(1300)
    #print('SVM classifier for augmented numbers', accuracy(testhoglab, result))

    #hh = np.float32(hog(training_aug[12][0])).reshape(-1, 64)
    #hh, _= classifySVM(training_aug[12][0])
    #cv2.imshow('random', training_aug[12][0])
    #print('static is', hh)

    #o, res = aNN.predict(test)

    #crect = 0
    #for i, r in enumerate(res):
    #    mm = np.argmax(r)
    #    if mm == int(test_lab[i]):
    #        crect += 1
    #    else:
    #        print('expect', int(test_lab[i]), 'got', mm)

   # print(crect / len(test))
   # cv2.imshow('test1 is a ' + str(classify(test[340].reshape(40, 28))), test[340].reshape(40, 28))
    #cv2.waitKey(0)

def hog2(img):
    h = hog_des.compute(img)
    return h

def hog(img):
    gradX = cv2.Sobel(img, cv2.CV_32F, 1, 0)
    gradY = cv2.Sobel(img, cv2.CV_32F, 0, 1)
    magnitude, angle = cv2.cartToPolar(gradX, gradY)

    BINS = 16
    bins = np.int32(BINS * (angle / (2*np.pi)))

    C = 14
    R = 20

    bin_cells = bins[:R, :C], bins[R:, :C], bins[:R, C:], bins[R:, C:]
    mag_cells = magnitude[:R, :C], magnitude[R:, :C], magnitude[:R, C:], magnitude[R:, C:]
    hists = [np.bincount(b.ravel(), m.ravel(), BINS) for b, m in zip(bin_cells, mag_cells)]
    hist = np.hstack(hists)
    return hist

def deskew(img):
    affine_flags = cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR
    m = cv2.moments(img)
    if abs(m['mu02']) < 1e-2:
        return img.copy()
    skew = m['mu11']/m['mu02']
    M = np.float32([[1, skew, -0.5*28*skew], [0, 1, 0]])
    img = cv2.warpAffine(img,M,(28, 40),flags=affine_flags)
    return img

def classifySVM(img):
    img_resize = cv2.resize(img, (22, 34), interpolation=cv2.INTER_NEAREST)
    img_resize = cv2.copyMakeBorder(img_resize, 3, 3, 3, 3, cv2.BORDER_CONSTANT)
    img_resize = cv2.threshold(img_resize, 120, 255, cv2.THRESH_BINARY)[1]

    hh = np.float32(hog(img_resize)).reshape(-1, 64)
    #print(hh[0])
    print(hh.shape)
    ret, result = svm.predict(hh)
    return int(result[0][0]), ret

def classifyANN(img):
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
    _, result = aNN.predict(vect)
    return int(np.argmax(result[0]))

def classify(img):
    #cv2.imshow('begin', img.astype('uint8'))
    img_resize = cv2.resize(img, (22, 34), interpolation=cv2.INTER_NEAREST)
    img_resize = cv2.copyMakeBorder(img_resize, 3, 3, 3, 3, cv2.BORDER_CONSTANT)
    #_, img_resize = cv2.threshold(img_resize, 120, 255, cv2.THRESH_BINARY)
    #img_resize = deskew(img_resize)
    img_resize = img_resize.astype('uint8')
    #cv2.imshow('resize', img_resize)
    #print(img_resize)
    #cv2.waitKey(1)
    #print(img_resize.shape)
    vect = hog2(img_resize).reshape(1, 324) # np.array(img_resize.reshape(1, 40 * 28))
    #print(vect.shape)
    #print(vect)
    _, result, _, dist = kNN.findNearest(vect, k=5)
    return int(result[0][0]), dist

def test_set():
    return subset(100, 12, 100)

def training_set():
    return subset(0, 12, 100)

def training_hog():
    return subset_hog(0, 12, 100)

def test_hog():
    return subset_hog(100, 12, 100)

def subset_hog(low, digits, samples=100):
    set = np.zeros((samples * digits, 324), dtype=np.float32)
    set_labels = np.zeros((samples * digits), dtype=int)

    matrixAug = np.array(training_aug)
    # print(training_aug)

    for i in range(digits):
        for j in range(samples):
            #_, im = cv2.threshold(matrixAug[i][low + j], 120, 255, cv2.THRESH_BINARY)
            im = matrixAug[i][low + j]
            hh = hog2(im)
            set[i * samples + j] = hh.reshape(324)
            set_labels[i * samples + j] = int(i)

    return set, set_labels

def subset(low, digits, samples=100):
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