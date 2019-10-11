# Author Adrian Shedley
# date: 4 Oct 2019
# Machine Perception task 1 assignment, code has been commented

import numpy as np
import cv2
import classifier as cl

# Initialise the Classifier before anything else
cl.init()


def spatially_similar(roi1, roi2):
    """Returns true when two features are not the same feature, have similar Y coordinates, similar area and a similar
    height value. These are set byt the absolute Y threshhold in pixels, an area scale factor and a height scale factor.
    The default values for these have been set by experimentation"""
    Y_THRESH = 20
    AREA_THRESH = 0.5
    HEIGHT_THRESH = 0.4
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


def spatial_grouping(rois):
    #  Sort by smallest area then by smallest X value
    rois.sort(key=lambda dig: dig.area)
    rois.sort(key=lambda dig: dig.x)

    temp = list()

    for i, el in enumerate(rois):
        for j, el2 in enumerate(rois[i:]):
            for k, el3 in enumerate(rois[j:]):
                if spatially_similar(el, el2) and spatially_similar(el, el3) and spatially_similar(el2, el3):
                    temp.append([el, el2, el3, el.error + el2.error + el3.error])

    # Sort by smallest error sum of all 3 features ie index [3]
    temp.sort(key=lambda x: x[3])
    return temp[0]


# Take an image of a building sign and get the numbers
def task1(img, name=None):
    if img is not None:
        # Convert to greyscale
        grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Thresholding with the adaptive thresholding method
        thresh = cv2.adaptiveThreshold(grey, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, blockSize=21, C=1)
        # Median blur to clump groups together
        thresh = cv2.medianBlur(thresh, 3)
        # Invert so that the numbers are foreground and can be contoured
        invert = np.uint8(thresh * -1 + 255)

        # Compute all the contours on the inverted thresholded image
        _, contours, heir = cv2.findContours(invert, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        #im4 = img.copy()
        #cv2.drawContours(im4, contours, -1, (255, 0, 0), 1)
        #cv2.imshow('Contour', im4)
        im3 = img.copy()

        # Allocate a list to store all of the potential digits bounding boxes and classifications in
        digit_list = list()

        # Loop over all the contours, and for those that meet the Area and Aspect Ratio requirements, classify them
        for i in contours:
            x, y, w, h = cv2.boundingRect(i)  # Get the (x,y) and width and height of a bounding box for this contour
            if 300 < w*h < 10000:
                if 1.1*w < h < 4*w:
                    # Save a region of interest from the greyscale image within the bounding box
                    roi = grey[y : y + h, x : x + w]
                    # Get the classification and confidence (errors sum) for this ROI
                    classify, errors = cl.classify(roi)
                    # Create a new Digit class that carries a neat representation of the data
                    digit = Digit(classify, np.sum(errors), x, y, w, h, w*h)
                    # Add the digit to the digit list that will be sorted and spatially grouped
                    digit_list.append(digit)

                    #cv2.putText(im3, str(classify), (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255-10*int(np.sum(errors)), 10*int(np.sum(errors))), 2)
                    #cv2.rectangle(im3, (x, y), (x + w, y +h), (0,255-10*int(np.sum(errors)), 10*int(np.sum(errors))), 2)

        #cv2.imshow('Ranked Classifications', im3)
        #cv2.waitKey(0)

        # Sort by the bets confidence first (Smallest Error)
        digit_list.sort(key=lambda x: x.error)
        # After sorting, take the top 10 best digits by confidence and return the best triple
        features = spatial_grouping(digit_list[:min(10, len(digit_list))])

        # Output for visual code, doesn't identify anything here. Take the 3 digits from the triplet and draw them
        for el in features[:3]:
            cv2.rectangle(im3, (el.x, el.y), (el.x + el.w, el.y + el.h), (0, 255, 0), 2)
            cv2.putText(im3, str(el.classify), (el.x, el.y - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (120, 255, 0), 2)

        # Return the numbers as a string, and the Region Of Interest they were found in
        return numbers(features), extracted_area(img, features)
    else:
        # Image is none. Return an error
        return "???", np.zeros((1,1))


class Digit:
    def __init__(self, classify, error, x, y, w, h, area):
        """This is a data storage class for a classified digit.
            The Digit class contains a digits classification, confidence (error), x pos, y pos
            width, height and precomputed area. """
        self.classify = classify
        self.error = error  # Also the confidence measure
        self. x = x
        self.y = y
        self.w = w
        self.h = h
        self.area = area
