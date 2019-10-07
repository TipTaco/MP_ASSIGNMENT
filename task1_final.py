# Author Adrian Shedley
# date: 4 Oct 2019
# Machine Perception task 1 assignment, code has been commented

import numpy as np
import cv2
import classifier as cl

# Initialise the Classifier before anything else
cl.init()


def is_similar(roi1, roi2):
    """Returns true when two features are not the same feature, have similar Y coordinates, similar area and a similar
    height value. These are set byt the absolute Y threshhold in pixels, an area scale factor and a height scale factor.
    The default values for these have been set by experimentation"""
    Y_THRESH = 20
    AREA_THRESH = 1.5
    HEIGHT_THRESH = 1.2
    return (abs(roi1.y - roi2.y) <= Y_THRESH) and (abs(roi1.area - roi2.area) <= max(roi1.area, roi2.area) *
            AREA_THRESH) and (abs(roi1.h - roi2.h) <= max(roi1.h, roi2.h) * HEIGHT_THRESH) and (roi1 != roi2)


def order_features(feat):
    """Returns the 3 numbers in a given sign in left to right format according to their X coordinate"""
    nums = list()
    for i in range(3):
        nums.append(feat[i])

    # Sort by smallest X coordinate
    nums.sort(key=lambda dig: dig.x)
    return nums


def numbers(feat):
    """Returns the building number as a 3 digit number from left to right"""
    nums = order_features(feat)
    return str(nums[0].classify) + str(nums[1].classify) + str(nums[2].classify)


def extracted_area(img, feat):
    """Returns an area from the Top-Left corner of the left digit to the bottom-right corner of the right digit"""
    nums = order_features(feat)
    # nums[0] is left digit, nums[2] is right digit
    img_cropped = img[nums[0].y: nums[2].y + nums[2].h, nums[0].x: nums[2].x + nums[2].w]
    return img_cropped


def alignment_filter(rois):
    #  Sort by smallest area then by smallest X value
    rois.sort(key=lambda dig: dig.area)
    rois.sort(key=lambda dig: dig.x)

    temp = list()

    for i, el in enumerate(rois):
        for j, el2 in enumerate(rois[i:]):
            for k, el3 in enumerate(rois[j:]):
                if is_similar(el, el2) and is_similar(el, el3) and is_similar(el2, el3):
                    temp.append([el, el2, el3, el.error + el2.error + el3.error])

    # Sort by smallest error sum of all 3 features ie index [3]
    temp.sort(key=lambda x: x[3])
    return temp[0]


# Take an image of a building sign and get the numbers
def task1(img, name=None):
    if img is not None:
        # convert to greyscale
        grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Threshold first then invert for contouring
        # gaus = cv2.GaussianBlur(grey, (3,3), 2)
        thresh = cv2.adaptiveThreshold(grey, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 1)
        thresh = cv2.medianBlur(thresh, 3)
        invert = np.uint8(thresh * -1 + 255)

        # Get contours
        _, contours, heir = cv2.findContours(invert, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        im3 = img.copy()
        toplist = list()

        for i in contours:
            x, y, w, h = cv2.boundingRect(i)
            if 300 < w*h < 10000:
                if 1.1*w < h < 4*w:
                    roi = grey[y : y + h, x : x + w]
                    classify, errors = cl.classify(roi)
                    digit = Digit(classify, np.sum(errors), x, y, w, h, w*h)
                    toplist.append(digit)

        # Sort by smallest error
        toplist.sort(key=lambda x: x.error)
        # After sorting, take the most likely (smallest error) top 10 and try to align them
        features = alignment_filter(toplist[:min(10, len(toplist))])

        # Output for visual code, doesn't identify anything here
        for el in features[:3]:
            cv2.rectangle(im3, (el.x, el.y), (el.x + el.w, el.y + el.h), (0, 255, 0), 2)
            cv2.putText(im3, str(el.classify), (el.x, el.y - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (120, 255, 0), 2)

        # return the numbers as a string, and the Region Of Interest they were found in
        return numbers(features), extracted_area(img, features)
    else:
        # Image is none. Return an error sign
        return "???", np.zeros((1,1))


class Digit:
    def __init__(self, classify, error, x, y, w, h, area):
        self.classify = classify
        self.error = error
        self. x = x
        self.y = y
        self.w = w
        self.h = h
        self.area = area
