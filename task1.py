# Authr Adrian Shedley, Revised 27 Sep 2019

import cv2
import imageloader as il
import numpy as np
il.init()


def sorter_dist(element):
    return element[1]


def sorter_area(element):
    return element[6]


def sorter_x(element):
    return element[2]


def sorter_sumdist(element):
    return element[3]


def simi(roi1, roi2):
    Y_THRESH = 20
    AREA_THRESH = 1.5
    HEIGHT_THRESH = 1.2
    return (abs(roi1[3] - roi2[3]) <= Y_THRESH) and (abs(roi1[6] - roi2[6]) <= max(roi1[6], roi2[6]) *
            AREA_THRESH) and (abs(roi1[5] - roi2[5]) <= max(roi1[5], roi2[5]) * HEIGHT_THRESH) and (roi1 != roi2)


def numbers(feat):
    nums = list()
    for i in range(3):
        nums.append(feat[i])

    nums.sort(key=sorter_x)
    return str(nums[0][0]) + str(nums[1][0]) + str(nums[2][0])


def alignment_filter(rois, in_a_row=3):
    #resort list by size of Roi
    rois.sort(key=sorter_area)
    rois.sort(key=sorter_x)

    temp = list()

    for i, el in enumerate(rois):
        for j, el2 in enumerate(rois[i:]):
            for k, el3 in enumerate(rois[j:]):
                if (simi(el, el2) and simi(el, el3) and simi(el2, el3)):
                    temp.append([el, el2, el3, el[1]+el2[1]+el3[1]])

    temp.sort(key=sorter_sumdist)
    return temp[0]


# Take an image of a building sign and get the numbers
def task1(img, name=None):
    assert img is not None

    # convert to greyscale
    grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Threshold first then invert for contouring
    gaus = cv2.GaussianBlur(grey, (3,3), 2)
    thresh = cv2.adaptiveThreshold(gaus, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 51, 1)
    invert = np.uint8(thresh * -1 + 255)

    # Median filter to reduce shot noise
    #median = cv2.medianBlur(invert, 5)

    im2, conts, heir = cv2.findContours(invert, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    #print('contours shape', conts)

    im2 = img.copy()
    cv2.drawContours(im2, conts, -1, (0, 255, 0), 2)
    #cv2.imshow('out', im2)

    im3 = img.copy()

    toplist = list()

    for i in conts:
        x, y, w, h = cv2.boundingRect(i)
        if w*h > 300 and w*h < 10000:
            if h > 1.1 * w and h < 4*w:
               # cv2.rectangle(im3, (x, y), (x + w, y + h), (0, 255, 0), 2)
                roi = grey[y : y + h, x : x + w]
                #cv2.imshow('roi', roi)
                #cv2.waitKey(0)
                claass, dist = il.classify(roi)
                #print('distance', np.sum(dist))

                toplist.append([claass, np.sum(dist), x, y, w, h, w*h])


    toplist.sort(key=sorter_dist)
    features = alignment_filter(toplist[:10], 3)
    #print(features, features[3])

    for el in features[:3]:
        cv2.rectangle(im3, (el[2], el[3]), (el[2] + el[4], el[3] + el[5]), (0, 255, 0), 2)
        cv2.putText(im3, str(el[0]), (el[2], el[3] - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (120, 255, 0), 2)

    #cv2.imshow('bound' + str(name), im3)

    # show the images
    #cv2.imshow('Grey', grey)
    #cv2.imshow('thresh', thresh)
    #cv2.imshow('invert', invert)
    #cv2.imshow('median', median)

    #cv2.waitKey(5)
    return numbers(features)


# Run the function classification of each of the building images
#for im in il.building:
#    task1(im)