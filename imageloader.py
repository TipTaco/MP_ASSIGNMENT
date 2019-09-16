# Author: Adrian Shedley
# Date: 23 Aug 2019
#
# Prepared for Machine Perception Assignment
# imageloader.py - load in the images to train from.

import os
import cv2
from os import listdir

building = list()
building_name = list()
building_grey = list()

directional = list()
directional_name = list()
directional_grey = list()

training = list()

IMG_PATH = '/res/'
TRAIN_PATH = '/res/training/original'
NUMBERS = ['Zero', 'One', 'Two', 'Three', 'Four', 'Five', 'Six', 'Seven', 'Eight', 'Nine', 'LeftArrow', 'RightArrow']

def init():
    path = os.getcwd() + IMG_PATH
    all_files = listdir(path)
    for file in all_files:
        if file.endswith(".jpg"):
            img = cv2.imread(path + file)

            if img is not None:
                if file.startswith("BS"):
                    building.append(img)
                    building_grey.append(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
                    building_name.append(file)
                elif file.startswith("DS"):
                    directional.append(img)
                    directional_grey.append(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
                    directional_name.append(file)

    path = os.getcwd() + TRAIN_PATH
    all_files = listdir(path)

    for i in range(len(NUMBERS)):
        training.append(list())

    for file in all_files:
        # print(file)
        if file.endswith(".jpg"):
            img = cv2.imread(path + '/' + file)
            if img is not None:
                for i, nn in enumerate(NUMBERS):
                    if file.startswith(nn):
                        training[i].append(file)
                        break

    print("[Image Loader] Loaded", len(building), "building signs and", len(directional), "directional signs")
