# Author Adrian Shedley, Date 6 Oct 2019
# Purpose of this file is to complete the task 1 as per the assignmnet spec sheet
# It also assumes that the directories exist

# READ ALL FILES FROM /home/student/test/task1/
# FILE NAMES ARE: testX.jpg
# REQUIRES OUTPUT: DetectedAreaX.jpg and BuildingX.txt

import cv2
import task1_final as task1
import os
from os import listdir

TASK1_FILES = "/home/student/test/task1/"
TASK1_OUTPUT = os.getcwd() + "/output/task1/"


def main():
    all_files = listdir(TASK1_FILES)
    for file in all_files:
        if file.startswith("test") and file.endswith(".jpg"):
            number = file[4:-4]
            img = cv2.imread(TASK1_FILES + file)
            if img is not None:
                num, roi = task1.task1(img)

                # Output region
                cv2.imwrite(TASK1_OUTPUT + "DetectedArea" + number + ".jpg", roi)

                #Output text file
                f = open(TASK1_OUTPUT + "Building" + number + ".txt", "w+")
                f.write("Building " + num + "\n")
                f.close()

    print("Compelete")


if __name__ == "__main__":
    main()





