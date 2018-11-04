import cv2
import numpy as np

#import cascade style sheet to identify face
faceDetect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

cam =cv2.VideoCapture(0)
id = raw_input('Enter ID for User : ')
SampleNumber = 0
while(True):
    ret, img = cam.read()

    #convert RGB image to GRAYSCALE
    grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = faceDetect.detectMultiScale(grey,1.3,5)

    #drawing rectange and store samples
    for(x, y, w, h) in faces:
        SampleNumber = SampleNumber+1
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 2)
        #store captured samples
        cv2.imwrite("dataSet/User."+str(id)+"."+str(SampleNumber)+".jpg", grey[y:y+h, x:x+w])
        cv2.waitKey(100)

    cv2.imshow("face", img)
    cv2.waitKey(1)
    if(SampleNumber>20):
        break

cam.release()
cv2.destroyAllWindows()


