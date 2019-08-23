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
def task1():
    # start by using the first image
    local = il.building_grey[0]
    cv2.imshow("Grey", local)
    cv2.waitKey(0)


task1()



