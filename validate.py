# Author Adrian Shedley, date 4 oct 2019
import json
import cv2
import task1_final as task1
import task2

def main():
    #filepath
    filename = 'res/Answers.json'

    #Read JSON data into the datastore variable
    if filename:
        with open(filename, 'r') as f:
            datastore = json.load(f)


    ncorrect = 0
    tried = len(datastore["Directional"])

    """for bs in datastore["Building"]:
        img = cv2.imread('res/' + bs + ".jpg")
        classify = task1.task1(img, bs)

        if classify == datastore["Building"][bs]:
            correct = correct + 1
            print("Correct for", bs, ". Got", classify)
        else:
            print("FAIL for", bs, ". Got", classify, 'expected', datastore["Building"][bs])
            #cv2.waitKey(0)"""

    for ds in datastore["Directional"]:
        img = cv2.imread('res/' + ds + ".jpg")
        classify = task2.task2(img, ds)
        ans = datastore["Directional"][ds]

        correct = True


        if len(classify) == len(ans):
            for i, cla in enumerate(classify):
                if cla != ans[i]:
                    correct = False
                    break
        else:
            correct = False

        if correct:
            ncorrect = ncorrect + 1
            print("Correct for", ds, ". Got", classify)
        else:
            print("FAIL for", ds, ". Got", classify, 'expected', ans)
            cv2.waitKey(0)


    print('OVERALL', round(100.0 * ncorrect/tried, 2), "% correct")


if __name__ == "__main__":
    main()