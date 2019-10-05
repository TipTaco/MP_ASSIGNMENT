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


def simi(roi1, roi2):
    Y_THRESH = 20
    AREA_THRESH = 1.5
    HEIGHT_THRESH = 1.2
    return (abs(roi1[3] - roi2[3]) <= Y_THRESH) and (abs(roi1[6] - roi2[6]) <= max(roi1[6], roi2[6]) *
            AREA_THRESH) and (abs(roi1[5] - roi2[5]) <= max(roi1[5], roi2[5]) * HEIGHT_THRESH) and (roi1 != roi2)


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

    return img[mini:maxi, :]


# Take an image of a directional sign and get the numbers
def task2(img, name=None):
    assert img is not None

    numbers_on_sign = list()

    # convert to greyscale
    grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sobX, sobY = cv2.spatialGradient(grey, None, None, 3)
    sob = np.square(sobX) + np.square(sobY)

    cv2.imshow('grey', sob)
    col_sum = np.sum(sob, axis=0)
    col_sum = col_sum / col_sum.max()
    minX, maxX = thresh_sweep(col_sum, 0.001, 0.25)

    #plt.plot(np.arange(0, img.shape[1]), col_sum)
    #plt.show()

    roi = grey[:, int(minX*0.9):int(maxX*1.1)]
    roi_rgb = img[:, int(minX*0.9):int(maxX*1.1)]
    cv2.imshow('roi', roi)

    # Get contours
    roi_thresh = cv2.adaptiveThreshold(roi, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 1) # cv2.threshold(roi, 120, 255, cv2.THRESH_OTSU)[1]
    im2, conts, heir = cv2.findContours(roi_thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    regions = list()

    for i in conts:
        x, y, w, h = cv2.boundingRect(i)
        if 80 < w * h < 5000:
            if 0.8 * w < h < 1.2 * w:
                feature = roi[y: y + h, x: x + w]
                classify, dist = cl.classify(feature)
                if np.sum(dist) < 15 and (int(classify) == 10 or int(classify) == 11):
                    cv2.rectangle(roi_rgb, (x, y), (x+w, y+h), (0, 255, 255), thickness=2)
                    rx = x - int(3.9 * w)
                    ry = y - int(0.4 * h)
                    rw = int(4.9 * w)
                    rh = int(1.8 * h)
                    regions.append([rx, ry, rw, rh, x, y, w, h])
                    cv2.rectangle(roi_rgb, (rx, ry), (rx + rw, ry + rh), (0, 255, 0), thickness=1)
                else:
                    cv2.rectangle(roi_rgb, (x, y), (x+w, y+h), (255, 0, 0), thickness=1)

    mser = cv2.MSER_create(2, _min_area=50)

    regions.sort(key=order_y)

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
    warped = cv2.warpPerspective(roi, M, (new_width, new_height))
    cv2.imshow('warped', warped)
    #cv2.waitKey(0)

    warped_thresh = cv2.adaptiveThreshold(warped, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 1)
    im2, conts, heir = cv2.findContours(warped_thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    warped_rgb = cv2.cvtColor(warped, cv2.COLOR_GRAY2BGR)

    regions2 = list()

    for i in conts:
        x, y, w, h = cv2.boundingRect(i)
        if 80 < w * h < 5000:
            if 0.8 * w < h < 1.2 * w:
                feature = warped[y: y + h, x: x + w]
                classify, dist = cl.classify(feature)
                if np.sum(dist) < 15 and (int(classify) == 10 or int(classify) == 11):
                    cv2.rectangle(warped_rgb, (x, y), (x+w, y+h), (0, 255, 255), thickness=2)
                    rx = x - int(3.6 * w)
                    ry = max(0, y - int(0.45 * h))
                    rw = int(4.6 * w)
                    rh = int(1.8 * h)
                    regions2.append([rx, ry, rw, rh, x, y, w, h])
                    cv2.rectangle(warped_rgb, (rx, ry), (rx + rw, ry + rh), (0, 255, 0), thickness=1)
                else:
                    cv2.rectangle(warped_rgb, (x, y), (x+w, y+h), (255, 0, 0), thickness=1)

    cv2.imshow('warped rgn', warped_rgb)
    regions2.sort(key=order_y)
    #cv2.waitKey(0)

    # Classify each of the extracted regions, will be 3 digits then a direction
    for region in regions2:
        rx, ry, rw, rh, x, y, w, h = region[:]
        digits = warped[ry:ry + rh, rx:rx+rw]

        digits = cv2.threshold(digits, 100, 255, cv2.THRESH_OTSU)[1]

        num1 = digits[:, int(0.1*w): int(1.0 * w)]
        num2 = digits[:, int(1.1*w): int(2.1 * w)]
        num3 = digits[:, int(2.2*w): int(3.3 * w)]
        arrow = digits[:, int(3.5*w): int(4.6 * w)]

        #num2 = cv2.copyMakeBorder(num2, 0, 0, 3, 3, cv2.BORDER_CONSTANT)
        #num3 = cv2.copyMakeBorder(num3, 0, 0, 3, 3, cv2.BORDER_CONSTANT)
       # arrow = cv2.copyMakeBorder(arrow, 0, 0, 3, 3, cv2.BORDER_CONSTANT)

        #num1 = crop_height(num1)
        num2 = crop_height(num2)
        num3 = crop_height(num3)
        arrow = crop_height(arrow)

        cv2.imshow('dig1', num1)
        cv2.imshow('dig2', num2)
        cv2.imshow('dig3', num3)
        cv2.imshow('arr', arrow)

        output = ""
        output += str(cl.classify(num1)[0])
        output += str(cl.classify(num2)[0])
        output += str(cl.classify(num3)[0])

        dir = cl.classify(arrow)[0]
        if dir == 10:
            output += "L"
        elif dir == 11:
            output += "R"

        # print(output)
        numbers_on_sign.append(output)

        cv2.imshow('region', digits)
        #cv2.waitKey(0)

    cv2.imshow('rect', roi_rgb)
    cv2.imshow('roi thresh', roi_thresh)

    return np.array(numbers_on_sign)


    #im2 = roi_rgb.copy()
    #cv2.drawContours(im2, conts, -1, (0, 255, 0), 2)
    #cv2.imshow('out', im2)
    #cv2.waitKey(0)
"""
    im3 = img.copy()
    toplist = list()

    for i in conts:
        x, y, w, h = cv2.boundingRect(i)
        if 300 < w*h < 10000:
            if 1.1*w < h < 4*w:
                roi = grey[y : y + h, x : x + w]
                claass, dist = cl.classify(roi)
                toplist.append([claass, np.sum(dist), x, y, w, h, w*h])

    toplist.sort(key=roi_dist)
    features = alignment_filter(toplist[:10])

    for el in features[:3]:
        cv2.rectangle(im3, (el[2], el[3]), (el[2] + el[4], el[3] + el[5]), (0, 255, 0), 2)
        cv2.putText(im3, str(el[0]), (el[2], el[3] - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (120, 255, 0), 2)"""

