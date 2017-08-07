import cv2
import numpy as np

img =  cv2.imread("sample.jpg")
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
faces=face_cascade.detectMultiScale(gray, 1.3, 5)
for (x,y,w,h) in faces:
    cv2.rectangle(img, (x,y), (x+w, y+h), (255,0,0), 2)
    sub_face = img[y:y+h, x:x+w]
gray_face = cv2.cvtColor(sub_face, cv2.COLOR_BGR2GRAY)
resized_image = cv2.resize(gray_face, (48,48))
hist = np.histogram(resized_image.flatten(), 256, [0,256])[0]
print (hist)
width, height = resized_image.shape[:2]
print (width)
print (height)
cv2.imshow('gray_face', resized_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
