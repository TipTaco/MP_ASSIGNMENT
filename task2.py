# Author Adrian Shedley
# date 5 oct 2019

import numpy as np
import cv2
import classifier as cl
import matplotlib.pyplot as plt

cl.init()


def normalise(A):
    return (A - A.min()) / (A.max() - A.min())


def order_y(element):
    return element[1]


def thresh_sweep(arr, step, inclusion):
    range = int(arr.shape[0] * inclusion)
    #print(inclusion)
    arr2 = arr.copy()

    for thresh in np.arange(0.0, 1.0, step):
        arr2[arr2 < thresh] = 0.0
        where = np.argwhere(arr2 == 0)
        if where.shape[0] != 0 and (where.max() - where.min()) > range:
            arr2 = arr.copy()
            arr2[arr2 < thresh - step] = 0.0
            arr2[arr2 >= thresh - step] = 1.0
            where = np.argwhere(arr2 == 0)
            arr2[where.min():where.max()] = 0.0
            break

    return where.min(), where.max()


def crop_height(img):
    mini = 0
    maxi = img.shape[0]

    for i, row in enumerate(img):
        if np.sum(row) > 0:
            mini = max(0, i - 1)
            break

    for i, row in enumerate(np.flip(img, axis=0)):
        if np.sum(row) > 0:
            maxi = img.shape[0] - (i - 1)
            break

    return img[max(0,mini-1):min(img.shape[0], maxi+1), :]


def number_block(col_sum, start_point=0, reverse=False, mini=3):
    start = 0
    stop = 0
    starting = start_point
    ending = col_sum.shape[0]
    step = 1

    if reverse:
        starting = col_sum.shape[0]-1
        ending = start_point+1
        step = -1

    for i in range(starting, ending, step):
        if col_sum[i] > 0:
            start = i
            break

    if reverse:
        starting = col_sum.shape[0]-1
        ending = start+1
    else:
        starting = start
        ending = col_sum.shape[0]

    for i in range(starting, ending, step):
        if col_sum[i] == 0 and i - start > mini:
            stop = i
            break
        else:
            stop = i

    if reverse:
        return stop, start
    else:
        return start, stop


def classify_digits(region, digits=None, reverse=False):
    region = crop_height(region)
    col_sum = np.sum(region, axis=0)

    if digits is None:
        digits = region.copy()

    start1, stop1 = number_block(col_sum, 0, reverse)
    start2, stop2 = number_block(col_sum, stop1 + 1, reverse)
    start3, stop3 = number_block(col_sum, stop2 + 1, reverse)
    start4, stop4 = number_block(col_sum, stop3 + 1, reverse)

    num1 = digits[:, start1:stop1]
    num2 = digits[:, start2:stop2]
    num3 = digits[:, start3:stop3]
    arrow = digits[:, start4:stop4]

    num1 = crop_height(num1)
    num2 = crop_height(num2)
    num3 = crop_height(num3)
    arrow = crop_height(arrow)

    #cv2.imshow('dig1', num1)
    #cv2.imshow('dig2', num2)
    #cv2.imshow('dig3', num3)
    #cv2.imshow('arr', arrow)
    #cv2.waitKey(0)

    return cl.classify(num1)[0], cl.classify(num2)[0], cl.classify(num3)[0], cl.classify(arrow)[0]


def validate_classify(num1, num2, num3, num4):
    valid = True
    if num1 > 9 or num2 > 9 or num3 > 9:
        valid = False

    if num4 < 10 or num4 > 11:
        valid = False

    return valid


# Take an image of a directional sign and get the numbers
def task2(img, name=None):
    assert img is not None

    numbers_on_sign = list()

    # convert to greyscale
    grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sobX, sobY = cv2.spatialGradient(grey, None, None, 3)
    sob = np.square(sobX) + np.square(sobY)

    #cv2.imshow('grey', sob)
    col_sum = np.sum(sob, axis=0)
    col_sum = col_sum / col_sum.max()
    minX, maxX = thresh_sweep(col_sum, 0.001, 0.22)

    #plt.plot(np.arange(0, img.shape[1]), col_sum)
    #plt.show()

    roi = grey[:, int(minX*0.9):int(maxX*1.1)]
    roi_rgb = img[:, int(minX*0.9):int(maxX*1.1)]
    cv2.imshow('roi', roi)

    # Get contours
    roi_thresh = cv2.adaptiveThreshold(roi, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 25, 1) # cv2.threshold(roi, 120, 255, cv2.THRESH_OTSU)[1]
    #roi_thresh = cv2.morphologyEx(roi_thresh, cv2.MORPH_ERODE, np.ones((2,1)))
    cv2.imshow('thresh', roi_thresh)
    im2, conts, heir = cv2.findContours(roi_thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    regions = list()

    for i in conts:
        x, y, w, h = cv2.boundingRect(i)
        if 40 < w * h < 5000:
            if 0.6 * w < h < 1.4 * w:
                feature = roi[y: y + h, x: x + w]
                classify, dist = cl.classify(feature)
                if np.sum(dist) < 3.0 * cl.FEATURES and (int(classify) == 10 or int(classify) == 11):
                    cv2.rectangle(roi_rgb, (x, y), (x+w, y+h), (0, 255, 255), thickness=2)
                    rx = max(0, x - int(3.9 * w))
                    ry = max(0, y - int(0.6 * h))
                    rw = int(4.9 * w)
                    rh = int(2.2 * h)
                    regions.append([rx, ry, rw, rh, x, y, w, h, int(classify)])
                    cv2.rectangle(roi_rgb, (rx, ry), (rx + rw, ry + rh), (0, 255, 0), thickness=1)
                else:
                    cv2.rectangle(roi_rgb, (x, y), (x+w, y+h), (255, 0, 0), thickness=1)

    mser = cv2.MSER_create(2, _min_area=50)

    regions.sort(key=order_y)

    # Make region of interest to save (straight)
    treg = regions[0]
    breg = regions[-1]

    pt1 = [treg[0], treg[1]]
    pt2 = [treg[0] + treg[2], treg[1]]
    pt3 = [breg[0] + breg[2], breg[1] + breg[3]]
    pt4 = [breg[0], breg[1] + breg[3]]

    initial_pts = np.float32([pt1, pt2, pt3, pt4])

    new_height = abs(max(pt1[1] - pt4[1], pt2[1] - pt3[1]))
    new_width = abs(max(pt2[0] - pt1[0], pt3[0] - pt4[0]))

    final_pts = np.float32([[0,0], [new_width-1, 0], [new_width-1, new_height-1], [0, new_height - 1]])

    M = cv2.getPerspectiveTransform(initial_pts, final_pts)
    warped = cv2.warpPerspective(roi, M, (new_width, new_height), flags=cv2.INTER_CUBIC)
    #cv2.imshow('warped', warped)
    #cv2.waitKey(0)

    region_output = warped.copy()

    warped = cv2.resize(warped, (warped.shape[1] * 2, warped.shape[0] * 2), cv2.INTER_LINEAR)
    warped_thresh = cv2.adaptiveThreshold(warped, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 1)
    im2, conts, heir = cv2.findContours(warped_thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    warped_rgb = cv2.cvtColor(warped, cv2.COLOR_GRAY2BGR)

    regions2 = list()

    for i in conts:
        x, y, w, h = cv2.boundingRect(i)
        if (warped.shape[1]*warped.shape[1])//40 < w * h < 5000:
            if 0.8 * w < h < 1.2 * w:
                feature = warped[y: y + h, x: x + w]
                classify, dist = cl.classify(feature)
                if np.sum(dist) < 3.0 * cl.FEATURES and (int(classify) == 10 or int(classify) == 11):
                    cv2.rectangle(warped_rgb, (x, y), (x+w, y+h), (0, 255, 255), thickness=2)
                    rx = 0 # x - int(3.6 * w)
                    ry = max(0, y - int(.5 * h))
                    rw = warped.shape[1] # int(4.7 * w)
                    rh = int(2.0 * h)
                    regions2.append([rx, ry, rw, rh, x, y, w, h])
                    cv2.rectangle(warped_rgb, (rx, ry), (rx + rw, ry + rh), (0, 255, 0), thickness=1)
                else:
                    cv2.rectangle(warped_rgb, (x, y), (x+w, y+h), (255, 0, 0), thickness=1)

    cv2.imshow('warped rgn', warped_rgb)
    regions2.sort(key=order_y)
    #cv2.waitKey(0)"""

    # Classify each of the extracted regions, will be 3 digits then a direction
    for region in regions2:
        rx, ry, rw, rh, x, y, w, h = region[:]

        digits = warped[ry:ry + rh, rx:rx+rw]
        #digits = cv2.resize(digits, (digits.shape[1] * 2, digits.shape[0] * 2), interpolation=cv2.INTER_AREA)
        digits = cv2.threshold(digits, 120, 255, cv2.THRESH_OTSU)[1]
        #digits = cv2.medianBlur(digits, 3)

        width = int(w * 1.2)
        bests = list()
        for i in range(0, width//2):
            part = digits[:, i:width]
            part = crop_height(part)
            cv2.imshow('part', part)
            #cv2.waitKey(0)
            classify, dist = cl.classify(part)
            bests.append([classify, np.sum(dist*dist), i])

        # sort the best matches by their lowest distance classification
        bests.sort(key=lambda x: x[1])

        digits = digits[:, int(bests[0][2]):]
        cv2.imshow('digits', digits)

        # first try left to right full height classification
        num1, num2, num3, num4 = classify_digits(digits)
        if not validate_classify(num1, num2, num3, num4):
            num1, num2, num3, num4 = classify_digits(digits[:int(digits.shape[0]*0.70),:], digits)

        if not validate_classify(num1, num2, num3, num4):
            num1, num2, num3, num4 = classify_digits(digits, reverse=True)

        if not validate_classify(num1, num2, num3, num4):
            num1, num2, num3, num4 = classify_digits(digits[:int(digits.shape[0] * 0.70), :], digits, reverse=True)

        output = ""
        output += str(num1)
        output += str(num2)
        output += str(num3)

        dir = num4
        if dir == 10:
            output += "L"
        elif dir == 11:
            output += "R"

        # print(output)
        numbers_on_sign.append(output)

        #cv2.imshow('region', digits)
        #cv2.waitKey(0)

    #cv2.imshow('rect', roi_rgb)
    #cv2.imshow('roi thresh', roi_thresh)

    return np.array(numbers_on_sign), region_output


