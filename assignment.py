# Author: Adrian Shedley
# date: 23 aug 2019
#
# Prepared for Machine Perception Assignment
# assignmnet.py - the main file for the assignment with calls to both task one and two
import cv2
import imageloader as il
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
        claass, dist = il.classify(roi.astype('float32'))
        print('distance', np.sum(dist))
        if (np.sum(dist) < 80000000):
            cv2.rectangle(rgb, (feat[0], feat[1]), (feat[0] + feat[2], feat[1] + feat[3]), (0, 255, 0), thickness=2)
            cv2.putText(rgb, str(claass), (feat[0], feat[1]-5), cv2.FONT_HERSHEY_SIMPLEX, 1, thickness=2, color=(120, 255, 0))


    cv2.imshow("binary", bina)
    # cv2.imshow("otsu", orig)
    cv2.imshow("outpit", rgb)
    # cv2.imshow("subtract", output)
 ##
    cv2.waitKey(0)


#task2();
for i in range(len(il.building_grey)):
    task1(i)



