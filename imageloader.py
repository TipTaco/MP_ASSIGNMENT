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

IMG_PATH = '/res/'

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


    print("[Image Loader] Loaded", len(building), "building signs and", len(directional), "directional signs")
