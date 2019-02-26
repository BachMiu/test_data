import numpy as np
import cv2 as cv
import sys

TRAINING_FILE_NAME            = "haarcascade_frontalface_alt.xml"

def showHelp():
    print("Author: hoaint.13")
    #print("--help/-h\tShow usage help")
    print("usage: face_detector_hoai <image file name>")
    print("Ex: face_detector_hoai demo.jpg")

if __name__ == '__main__':
    if len(sys.argv) != 2:
        showHelp()
        exit(1)

    imageFileName = sys.argv[1]
    face_cascade = cv.CascadeClassifier(TRAINING_FILE_NAME)
    img = cv.imread(imageFileName)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        cv.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
    cv.imshow('img',img)
    #cv.waitKey(0)
    cv.destroyAllWindows()