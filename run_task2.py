# Author Adrian Shedley, Date 6 Oct 2019
# Purpose of this file is to complete the task 1 as per the assignmnet spec sheet
# It also assumes that the directories exist

# READ ALL FILES FROM /home/student/test/task2/
# FILE NAMES ARE: testX.jpg
# REQUIRES OUTPUT: DetectedAreaX.jpg and BuildingX.txt

import cv2
import task2 as task2
import os
from os import listdir

TASK2_FILES = "/home/student/test/task2/"
TASK2_OUTPUT = os.getcwd() + "/output/task2/"


def main():
    all_files = listdir(TASK2_FILES)
    for file in all_files:
        if file.startswith("test") and file.endswith(".jpg"):
            number = file[4:-4]
            img = cv2.imread(TASK2_FILES + file)
            if img is not None:
                nums, roi = task2.task2(img)

                # Output region
                cv2.imwrite(TASK2_OUTPUT + "DetectedArea" + number + ".jpg", roi)

                #Output text file
                f = open(TASK2_OUTPUT + "Building" + number + ".txt", "w+")
                for num in nums:
                    actual_num = num[:3]
                    direct = num[-1:]
                    directional = "left"
                    if direct == 'R':
                        directional = "right"
                    f.write("Building " + actual_num + " to the " + directional + "\n")
                f.close()

    print("Complete")


if __name__ == "__main__":
    main()





