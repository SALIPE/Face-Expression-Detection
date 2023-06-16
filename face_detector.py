import numpy as np
import cv2 as cv


def face_img_detector(imagefile):
    face_classifier = cv.CascadeClassifier('haarcascade_frontalface_default.xml')

    # image = cv.imread(imagefile)
    nparr = np.fromstring(imagefile, np.uint8)
    # decode image
    image_gray = cv.imdecode(nparr, cv.COLOR_BGR2GRAY)
    # image_gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    faces = face_classifier.detectMultiScale(image_gray, 1.3,5)

    cord1 = 0
    cord2 = 0
    x=0
    y=0
    padding = 40
    for (x,y,w,h) in faces:
        x = x-padding
        y = y-padding
        cord1 = y+h+padding +10
        cord2 = x+w+padding +10
        # cv.rectangle(image_gray,(x,y),(x+w,y+h), (255,0,0),1)

    cropped = image_gray[y:cord1,x:cord2]
    cv.imshow('imagem', cropped)
    cv.waitKey(0)
    cv.destroyAllWindows()

