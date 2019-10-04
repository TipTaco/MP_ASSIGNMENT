# Author Adrian Shedley, date 4 oct 2019
import json
import cv2
import task1_final as task1

def main():
    #filepath
    filename = 'res/Answers.json'

    #Read JSON data into the datastore variable
    if filename:
        with open(filename, 'r') as f:
            datastore = json.load(f)


    correct = 0
    tried = len(datastore["Building"])

    for bs in datastore["Building"]:
        img = cv2.imread('res/' + bs + ".jpg")
        classify = task1.task1(img, bs)

        if classify == datastore["Building"][bs]:
            correct = correct + 1
            print("Correct for", bs, ". Got", classify)
        else:
            print("FAIL for", bs, ". Got", classify, 'expected', datastore["Building"][bs])
            #cv2.waitKey(0)


    print('OVERALL', round(100.0 * correct/tried, 2), "% correct")


if __name__ == "__main__":
    main()