 # Authr Adrian Shedley, Revised 27 Sep 2019

import cv2
import imageloader as il
import numpy as np
il.init()

# Take an image of a building sign and get the numbers
def task1(img):
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
    print('contours shape', conts)

    im2 = img.copy()
    cv2.drawContours(im2, conts, -1, (0, 255, 0), 2)
    cv2.imshow('out', im2)

    im3 = img.copy()

    bounds = list()

    for i in conts:
        x, y, w, h = cv2.boundingRect(i)
        if w*h > 300 and w*h < 10000:
            if h > 1.1 * w and h < 4*w:
                bounds.append((x,y,w,h))
                cv2.rectangle(im3, (x, y), (x + w, y + h), (0, 255, 0), 2)

    group(bounds, grey.shape[1], grey.shape[0])

    cv2.imshow('bound', im3)

    # show the images
    cv2.imshow('Grey', grey)
    cv2.imshow('thresh', thresh)
    cv2.imshow('invert', invert)
    #cv2.imshow('median', median)

    cv2.waitKey(0)


def group(list, width, height):
    target = 3
    inc = 2
    offset = 10

    for i in range(offset, height, inc):
        new_list = getYBetween(list, height - i*inc, height - i*inc - offset)
        blank = np.zeros((height, width), dtype=np.uint8)

        blank = cv2.cvtColor(blank, cv2.COLOR_GRAY2BGR)
        cv2.rectangle(blank, (0, height - i*inc), (0, height - i*inc - offset), color=(255, 0, 0), thickness=1)

        for ele in new_list:
            cv2.rectangle(blank, (ele[0], ele[1]), (ele[0] + ele[2], ele[1] + ele[3]), color=(0,250,0), thickness=2)

        cv2.imshow('black', blank)
        cv2.waitKey(0)


def getYBetween(bounds, y_upper, y_lower):
    output = list()

    for ele in bounds:
        if ele[1] > y_lower and ele[1] <= y_upper:
            output.append(ele)
    return output


# Run the function classification of each of the building images
for im in il.building:
    task1(im)