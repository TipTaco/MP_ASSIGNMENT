# Author Adrian Shedley
# date 5 oct 2019

import numpy as np
import cv2
import classifier as cl
import matplotlib.pyplot as plt

cl.init()


def normalise(A):
    """Normalsie a matrix"""
    return (A - A.min()) / (A.max() - A.min())


def thresh_sweep(arr, step, inclusion):
    """Sweep a threshold up from 0,0 by step until inclusion percentage of the image is included"""
    # The range of X values before thresholding stops
    range = int(arr.shape[0] * inclusion)
    arr2 = arr.copy()

    # Sweep the threshold
    for thresh in np.arange(0.0, 1.0, step):
        arr2[arr2 < thresh] = 0.0
        # find the min and max included X values
        where = np.argwhere(arr2 == 0)
        # Test that the range is not exceeded
        if where.shape[0] != 0 and (where.max() - where.min()) > range:
            arr2 = arr.copy()
            # Go one step back, and reset the threshold
            arr2[arr2 < thresh - step] = 0.0
            arr2[arr2 >= thresh - step] = 1.0
            where = np.argwhere(arr2 == 0)
            arr2[where.min():where.max()] = 0.0
            break

    # return the lower and upper X values
    return where.min(), where.max()


def crop_height(img):
    """Take a greyscale image and remove the top and bottom black bars"""
    mini = 0
    maxi = img.shape[0]

    # Loop over the rows from top down, until a pixel of non zero is found
    for i, row in enumerate(img):
        if np.sum(row) > 0:
            mini = max(0, i - 1)
            break
    # Loop from bottom up until a row of non-zero is found
    for i, row in enumerate(np.flip(img, axis=0)):
        if np.sum(row) > 0:
            maxi = img.shape[0] - (i - 1)
            break
    # return the cropped image
    return img[max(0,mini-1):min(img.shape[0], maxi+1), :]


def number_block(col_sum, start_point=0, reverse=False, mini=3):
    """Segment the next block of non-zero columns from an image. Default from the left to right operation
     block must consist of at least mini columns and starts from start_point"""
    start = 0
    stop = 0
    starting = start_point
    ending = col_sum.shape[0]
    step = 1
    # If we are moving from the right hand side, reverse is true
    if reverse:
        starting = col_sum.shape[0]-1
        ending = start_point+1
        step = -1
    # find the start of the block
    for i in range(starting, ending, step):
        if col_sum[i] > 0:
            start = i
            break
    # Find the end of the block
    if reverse:
        starting = col_sum.shape[0]-1
        ending = start+1
    else:
        starting = start
        ending = col_sum.shape[0]
    # If we have no yet exceeded the minimum number of columns included
    for i in range(starting, ending, step):
        if col_sum[i] == 0 and i - start > mini:
            stop = i
            break
        else:
            stop = i
    # returning the start and stop X values depending on the operating mode
    if reverse:
        return stop, start
    else:
        return start, stop


def classify_digits(region, digits=None, reverse=False):
    """Classifies all 3 digits and an arrow in a digit region"""
    region = crop_height(region)  # Shrink the region
    # Sum everything in columns
    col_sum = np.sum(region, axis=0)

    if digits is None:
        digits = region.copy()

    # Section each digit based on the black columns, each start and stop is a digit.
    start1, stop1 = number_block(col_sum, 0, reverse)
    start2, stop2 = number_block(col_sum, stop1 + 1, reverse)
    start3, stop3 = number_block(col_sum, stop2 + 1, reverse)
    start4, stop4 = number_block(col_sum, stop3 + 1, reverse)

    #Extract the digits from the image. Num4 is the arrow
    num1 = digits[:, start1:stop1]
    num2 = digits[:, start2:stop2]
    num3 = digits[:, start3:stop3]
    arrow = digits[:, start4:stop4]

    # Crop the digits down vertically
    num1 = crop_height(num1)
    num2 = crop_height(num2)
    num3 = crop_height(num3)
    arrow = crop_height(arrow)
    # Return the classification ID's for each of the regions.
    return cl.classify(num1)[0], cl.classify(num2)[0], cl.classify(num3)[0], cl.classify(arrow)[0]


def crop_sign(img_rgb, img_grey, digit_regions):
    """Combined function that takes an ordered list of digit regions and crops and perspective transforms the
     image so that the top two corners of the first region and the bottom two corners of the last region are the
     extents of a newly created rectangular image."""
    # Methodolgy inspired by 4 Point OpenCV getPerspective Transform Example:
    #  REF: https://www.pyimagesearch.com/2014/08/25/4-point-opencv-getperspective-transform-example/

    # Make region of interest to save (straight)
    treg = digit_regions[0]  # Top of sign Region
    breg = digit_regions[-1]  # Bottom of sign region (the last element in the list)

    # Starting at the top left corer of the sign, going clockwise
    pt1 = [treg.rx, treg.ry]
    pt2 = [treg.rx + treg.rw, treg.ry]
    pt3 = [breg.rx + breg.rw, breg.ry + breg.rh]
    pt4 = [breg.rx, breg.ry + breg.rh]
    initial_pts = np.float32([pt1, pt2, pt3, pt4])

    # Calculate the size that the output image will become after perspective warp
    new_height = abs(max(pt1[1] - pt4[1], pt2[1] - pt3[1]))  # Difference in  left side height and right side height
    new_width = abs(max(treg.rw, breg.rw))  # Largest of top width and bottom width
    final_pts = np.float32([[0, 0], [new_width - 1, 0], [new_width - 1, new_height - 1], [0, new_height - 1]])

    # Generate a perspective transform matrix that maps initial_pts to final_pts
    M = cv2.getPerspectiveTransform(initial_pts, final_pts)
    warped_grey = cv2.warpPerspective(img_grey, M, (new_width, new_height), flags=cv2.INTER_CUBIC)
    warped_rgb = cv2.warpPerspective(img_rgb, M, (new_width, new_height), flags=cv2.INTER_CUBIC)

    return warped_rgb, warped_grey


def validate_classify(num1, num2, num3, num4):
    """Validate whether or not the 4 numbers given are [DIGIT DIGIT DIGIT ARROW] format. Returns false if not"""
    valid = True
    # If the first three nums are between 0-9
    if num1 > 9 or num2 > 9 or num3 > 9:
        valid = False

    # if num4 (the arrow) is left arrow (10) or right arrow (11)
    if num4 < 10 or num4 > 11:
        valid = False

    return valid


# Take an image of a directional sign and get the numbers
def task2(img, name=None):
    if img is not None:
        # Convert the image to greyscale
        grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Get the sobel gradient on both the X and Y directions, and sum them together
        sobX, sobY = cv2.spatialGradient(grey, None, None, 3)
        sob = np.square(sobX) + np.square(sobY)
        # Sum down each of the columns (axis=0)
        col_sum = np.sum(sob, axis=0)
        col_sum = col_sum / col_sum.max()  # Normalise between 0 and 1.0

        # Get the Lower and Upper X coordinates where an approximation of the sign lies
        minX, maxX = thresh_sweep(col_sum, 0.001, 0.22)

        # Extract a working area that is a little wider than the sign's best match area, ensuring valid bounds
        roi = grey[:, max(0, int(minX*0.9)):min(grey.shape[1], int(maxX*1.1))]
        roi_rgb = img[:, max(0, int(minX*0.9)):min(grey.shape[1], int(maxX*1.1))]

        # Adaptive threshold the macro ROI and then apply contouring
        roi_thresh = cv2.adaptiveThreshold(roi, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 25, 1)
        _, contours, heir = cv2.findContours(roi_thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # First pass classification to find the ARROWS on the sign only. This step is to align the sign
        digit_regions = list()
        for cont in contours:
            # Return the (x,y) and width and height of a bounding rectangle for this contour
            x, y, w, h = cv2.boundingRect(cont)
            # Only consider regions that are the correct shape and aspect initially.
            if 40 < w * h < 5000 and 0.6 * w < h < 1.4 * w:
                # Get the underlying greyscale region for this bounding box for classification
                possible_digit = roi[y: y + h, x: x + w]
                classify, errors = cl.classify(possible_digit)
                # If the confidence of this digit is below the threshold and it is a Left arrow (10) or right arrow (11)
                if np.sum(errors) < 3.0 * cl.FEATURES and (int(classify) == 10 or int(classify) == 11):
                    #cv2.rectangle(roi_rgb, (x, y), (x+w, y+h), (0, 255, 255), thickness=2)
                    # Calculate the full region with digits and arrow from the size of the arrow
                    rx = max(0, x - int(3.9 * w))
                    ry = max(0, y - int(0.6 * h))
                    rw = int(4.9 * w)
                    rh = int(2.2 * h)
                    # Create a Digit Region and populate the data
                    dr = DigitRegion(rx, ry, rw, rh, x, y, w, h, int(classify))
                    digit_regions.append(dr)
                    #cv2.rectangle(roi_rgb, (rx, ry), (rx + rw, ry + rh), (0, 255, 0), thickness=1)
                #else:
                    #cv2.rectangle(roi_rgb, (x, y), (x+w, y+h), (255, 0, 0), thickness=1)

        # Sort the list by the Y Coordinate
        digit_regions.sort(key=lambda dr: dr.ry)

        # Crop the sign down to only include the arrows and numbers
        region_output, cropped = crop_sign(roi_rgb, roi, digit_regions)

        # Upscale the image by 2 times and adaptive threshold it.
        cropped = cv2.resize(cropped, (cropped.shape[1] * 2, cropped.shape[0] * 2), cv2.INTER_LINEAR)
        up_thresh = cv2.adaptiveThreshold(cropped, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 1)
        # Contour the upscaled threshold image
        _, contours, heir = cv2.findContours(up_thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # 3 channel greyscale for drawing coloured rectangles on
        warped_rgb = cv2.cvtColor(cropped, cv2.COLOR_GRAY2BGR)

        # Make a new list to store the new warped regions
        digit_regions = list()
        # Find the digit regions based on the location and size of the arrows
        for cont in contours:
            x, y, w, h = cv2.boundingRect(cont)
            # If the region area is within the threshold and that the width height ratio is acceptable.
            if (np.power(cropped.shape[1], 2))//40 < w * h < 5000 and 0.8 * w < h < 1.2 * w:
                possible_digit = cropped[y: y + h, x: x + w]
                classify, errors = cl.classify(possible_digit)
                # See if the digit is classified as an arrow and is sufficiently low in error. 10 = left, 11 = right
                if np.sum(errors) < 3.0 * cl.FEATURES and (int(classify) == 10 or int(classify) == 11):
                    cv2.rectangle(warped_rgb, (x, y), (x+w, y+h), (0, 255, 255), thickness=2)
                    rx = 0  # Use the full sign width
                    ry = max(0, y - int(.5 * h))
                    rw = cropped.shape[1]  # Using the full sign width
                    rh = int(2.0 * h)
                    # Make it a DigitRegion and add it to the list
                    dr = DigitRegion(rx, ry, rw, rh, x, y, w, h, classify)
                    digit_regions.append(dr)
                    cv2.rectangle(warped_rgb, (rx, ry), (rx + rw, ry + rh), (0, 255, 0), thickness=1)
                else:
                    cv2.rectangle(warped_rgb, (x, y), (x+w, y+h), (255, 0, 0), thickness=1)

        # Order the List by ascending region Y coordinate
        digit_regions.sort(key=lambda dr: dr.ry)

        # Allocate a list to store the text on the sign, eg "113R"
        numbers_on_sign = list()

        # For each of the regions, sweep in from teh left hand side, checking which gives the best classification,
        #  which indicates the true edge of the sign has been found.
        for dr in digit_regions:
            digits = cropped[dr.ry: dr.ry + dr.rh, dr.rx: rx + dr.rw]
            digits = cv2.threshold(digits, 120, 255, cv2.THRESH_OTSU)[1]

            width = int(dr.w * 1.2)
            # Noise reduction matches. Find the best image classification from left hand side inward
            bests = list()
            for sweep in range(0, width//2):
                part = digits[:, sweep:width]
                part = crop_height(part)
                #cv2.imshow('part', part)
                classify, errors = cl.classify(part)
                bests.append([classify, np.sum(errors*errors), sweep])

            # sort the best matches by their lowest distance classification
            bests.sort(key=lambda x: x[1])

            # get the coordinate of the best digit, and crop the digit region
            digits = digits[:, int(bests[0][2]):]

            # first try left to right full height classification
            num1, num2, num3, num4 = classify_digits(digits)

            # Multiple attempts if for some reason the numebrs returned were illogical. E.g. an arrow where a digit is
            if not validate_classify(num1, num2, num3, num4):
                # Try only the top 70% of the digits image, all digits still form blocks here
                num1, num2, num3, num4 = classify_digits(digits[:int(digits.shape[0]*0.70), :], digits)

            #if not validate_classify(num1, num2, num3, num4):
                # Try the full image in back to front order
                #num1, num2, num3, num4 = classify_digits(digits, reverse=True)

            #if not validate_classify(num1, num2, num3, num4):
                #num1, num2, num3, num4 = classify_digits(digits[:int(digits.shape[0] * 0.70), :], digits, reverse=True)

            # Save the output string of numbers
            output = str(num1) + str(num2) + str(num3)

            dir = num4
            # If the direction is Left
            if dir == 10:
                output += "L"
            elif dir == 11:  # if it is right
                output += "R"

            # Add the numbers to the list of directions on this sign
            numbers_on_sign.append(output)

        # return the numbers on the sign as a list and as a region
        return np.array(numbers_on_sign), region_output
    else:
        # Not a valid image, return error
        return "????", np.zeros((1, 1))


class DigitRegion:
    """A data storage class for the region containing 3 digits and an arrow"""
    def __init__(self, rx, ry, rw, rh, x, y, w, h, classify):
        self.rx = rx
        self.ry = ry
        self.rw = rw
        self.rh = rh
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.classify = classify
