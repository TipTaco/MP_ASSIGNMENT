# Author: Adrian Shedley
# date: 23 aug 2019
#
# Prepared for Machine Perception Assignment
# assignmnet.py - the main file for the assignment with calls to both task one and two
import cv2
from depreciated import imageloader as il
import numpy as np
il.init()

# Take each of the building signs and extract ONE region and associated number
def task2():
    mser = cv2.MSER_create(20, 200)

    for i, img in enumerate(il.directional):
        local = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gaus = cv2.GaussianBlur(local, (3,3), 0.5)
        media = cv2.medianBlur(local, 3)
        bina = np.zeros_like(local)
        cv2.threshold(media, 120, 255, cv2.THRESH_BINARY, bina)
        # bina = cv2.medianBlur(bina, 7)
        gaus = cv2.morphologyEx(gaus, cv2.MORPH_ERODE, np.ones((3,3)))
        cv2.imshow("bina", gaus)

        features, boxes = mser.detectRegions(gaus)

        rgb = img.copy()

        for feat in boxes:
            roi = local[feat[1]:feat[1] + feat[3], feat[0]:feat[0] + feat[2]]
            if roi.shape[0] / roi.shape[1] > 2.0:
                roi = cv2.copyMakeBorder(roi, 0, 0, int(roi.shape[1] / 4), int(roi.shape[1] / 4), cv2.BORDER_REPLICATE)
            claass, dist = il.classify(roi.astype('float32'))
            print('distance', np.sum(dist))
            if (np.sum(dist) < 100000000):
                cv2.rectangle(rgb, (feat[0], feat[1]), (feat[0] + feat[2], feat[1] + feat[3]), (0, 255, 0), thickness=2)
                cv2.putText(rgb, str(claass), (feat[0], feat[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, thickness=2,
                            color=(120, 255, 0))

        #for feat in boxes:
        #    cv2.rectangle(img, (feat[0], feat[1]), (feat[0] + feat[2], feat[1] + feat[3]), (0, 255, 0),
         #                 thickness=2)

        cv2.imshow("Img" + str(i), rgb)
        cv2.waitKey(0)

def task1(index):
    # start by using the first image
    local = il.building_grey[index]
    cv2.imshow("Grey", local)
    # cv2.waitKey(0)

    media = cv2.medianBlur(local, 3)
    cv2.imshow("Median", media)

    bina = np.zeros_like(local)
    cv2.threshold(media, 150, 255, cv2.THRESH_BINARY, bina)

    #bina = cv2.medianBlur(bina, 5)
     # bina = cv2.erode(bina, (9, 9))

   # orig = np.zeros_like(local)
   # cv2.threshold(media, 160, 255, cv2.THRESH_OTSU, orig)

   #output = normalise(np.float64(orig - orig ^ bina))

    mser = cv2.MSER_create(120, 200)
    features, boxes = mser.detectRegions(bina)

    # print(features)

    rgb = il.building[index].copy()

    for feat in boxes:
        roi = il.building_grey[index][feat[1]:feat[1]+feat[3], feat[0]:feat[0]+feat[2]]
        if roi.shape[0]/roi.shape[1] > 2.0:
            roi = cv2.copyMakeBorder(roi, 0, 0, int(roi.shape[1] / 4), int(roi.shape[1] / 4), cv2.BORDER_REPLICATE)
        #claass, dist = il.classify(roi.astype('float32'))
        claass, dist = il.classify(roi.astype('float32'))
        print('distance', np.sum(dist))
        if (np.sum(dist) < 80000000):
            cv2.rectangle(rgb, (feat[0], feat[1]), (feat[0] + feat[2], feat[1] + feat[3]), (0, 255, 0), thickness=2)
            cv2.putText(rgb, str(claass), (feat[0], feat[1]-5), cv2.FONT_HERSHEY_SIMPLEX, 1, thickness=2, color=(120, 255, 0))


    cv2.imshow("binary", bina)
    # cv2.imshow("otsu", orig)
    cv2.imshow("outpit", rgb)
    # cv2.imshow("subtract", output)

    cv2.waitKey(0)


def task1_1(index):
    # start by using the first image
    number = cv2.imread('res/training/original/One1.jpg')
    number_grey = cv2.cvtColor(number, cv2.COLOR_BGR2GRAY)

    grey = il.building_grey[index]
    rgb = il.building[index].copy()
    gaus = cv2.GaussianBlur(grey, (3,3), 0)
    cv2.imshow("gaus", gaus)

    thresh = cv2.threshold(grey, 120, 255, cv2.THRESH_BINARY)[1]
    cv2.imshow("thr", thresh)

    out = grey.copy()
    out[thresh != 255] = 0

    out = cv2.medianBlur(out, 5)
    cv2.imshow('out', out)

    out = cv2.threshold(out, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
    cv2.imshow('out', out)

    out = cv2.GaussianBlur(out, (5,5), 1)

    mser = cv2.MSER_create(5, 40)
    features, boxes = mser.detectRegions(out)

    adapt = cv2.adaptiveThreshold(grey, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 125, 1)
    cv2.imshow('adapt', adapt)

    # _, ccl = cv2.connectedComponents(out, connectivity=4)
    #ccl = np.uint8(cv2.normalize(ccl, None, 0, 255, cv2.NORM_MINMAX))
    #cv2.imshow('ccl', ccl)

    output = cv2.connectedComponentsWithStats(adapt, 4, cv2.CV_32S)
    # Get the results
    # The first cell is the number of labels
    num_labels = output[0]
    # The second cell is the label matrix
    labels = output[1]
    # The third cell is the stat matrix
    stats = output[2]
    # The fourth cell is the centroid matrix
    centroids = output[3]

    ccl = np.uint8(cv2.normalize(labels, None, 0, 255, cv2.NORM_MINMAX))
    cv2.imshow('ccl', ccl)


    for feat in stats:
        if feat[cv2.CC_STAT_AREA] < 80: continue
        roi = il.building_grey[index][feat[1]:feat[1]+feat[3], feat[0]:feat[0]+feat[2]]
        if roi.shape[0]/roi.shape[1] > 2.0:
            roi = cv2.copyMakeBorder(roi, 0, 0, int(roi.shape[1] / 4), int(roi.shape[1] / 4), cv2.BORDER_REPLICATE)
        #claass, dist = il.classify(roi.astype('float32'))
        claass, dist = il.classify(roi.astype('float32'))
        print('distance', np.sum(dist))
        if (np.sum(dist) < 70000000):
            cv2.rectangle(rgb, (feat[0], feat[1]), (feat[0] + feat[2], feat[1] + feat[3]), (0, 255, 0), thickness=2)
            cv2.putText(rgb, str(claass), (feat[0], feat[1]-5), cv2.FONT_HERSHEY_SIMPLEX, 1, thickness=2, color=(120, 255, 0))


    cv2.imshow('rect', rgb)
    cv2.waitKey(0)


def task1_2(index):
    # start by using the first image
    number = cv2.imread('res/training/original/One1.jpg')
    number_grey = cv2.cvtColor(number, cv2.COLOR_BGR2GRAY)

    grey = il.building_grey[index]
    rgb = il.building[index].copy()
    gaus = cv2.GaussianBlur(grey, (3,3), -0.5)
    cv2.imshow("gaus", gaus)

    textKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 30))
    secondPassKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 2))
    output = []
    inv_grey = np.uint8(grey * -1.0 + 255)
    cv2.imshow('inverse', inv_grey)
    blackhat = cv2.morphologyEx(inv_grey, cv2.MORPH_BLACKHAT, textKernel)
    cv2.imshow('blackhot', blackhat)

    thresh = cv2.threshold(blackhat, 100, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.medianBlur(thresh, 3)
    #thresh = cv2.GaussianBlur(thresh, (5,5), 1)
    #thresh = cv2.morphologyEx(thresh, cv2.MORPH_DILATE, secondPassKernel)

    #new_grey = grey.copy()
   #new_grey[thresh == 0] = 0

    cv2.imshow('thre', thresh)
    #cv2.imshow('out', new_grey)

    #new_grey = cv2.threshold(thresh, 150, 255, cv2.THRESH_BINARY)[1]

    #cv2.imshow('out2', new_grey)

    mser = cv2.MSER_create(5, 40)
    features, boxes = mser.detectRegions(thresh)

    for feat in boxes:
        roi = il.building_grey[index][feat[1]:feat[1]+feat[3], feat[0]:feat[0]+feat[2]]
        if roi.shape[0]/roi.shape[1] > 2.0:
            roi = cv2.copyMakeBorder(roi, 0, 0, int(roi.shape[1] / 4), int(roi.shape[1] / 4), cv2.BORDER_REPLICATE)
        #claass, dist = il.classify(roi.astype('float32'))
        claass, dist = il.classifySVM(roi.astype('float32'))
        print('distance', np.sum(dist))
        if (np.sum(dist) < 70000000):
            cv2.rectangle(rgb, (feat[0], feat[1]), (feat[0] + feat[2], feat[1] + feat[3]), (0, 255, 0), thickness=2)
            cv2.putText(rgb, str(claass), (feat[0], feat[1]-5), cv2.FONT_HERSHEY_SIMPLEX, 1, thickness=2, color=(120, 255, 0))

    cv2.imshow('rect', rgb)
    cv2.waitKey(0)

#task2();
#for i in range(len(il.building_grey)):
    #task1(i)

for i in range(len(il.building_grey)):
    task1_2(i)


