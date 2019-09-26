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

    for i in conts:
        x, y, w, h = cv2.boundingRect(i)
        if w*h > 300 and w*h < 10000:
            if h > 1.1 * w and h < 4*w:
                cv2.rectangle(im3, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imshow('bound', im3)

    # show the images
    cv2.imshow('Grey', grey)
    cv2.imshow('thresh', thresh)
    cv2.imshow('invert', invert)
    #cv2.imshow('median', median)

    cv2.waitKey(0)


# Run the function classification of each of the building images
for im in il.building:
    task1(im)