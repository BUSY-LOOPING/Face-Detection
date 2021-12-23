import cv2
from random import randrange

# load some pre-trained data on face frontals from opencv
trained_face_data = cv2.CascadeClassifier('frontalface_default.xml')

#choose an image to detect image
img = cv2.imread('group_img.png')

#must convert to grayscale
grayscaled_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#detect faces
face_coordinates = trained_face_data.detectMultiScale(grayscaled_img)

#draw rectangles around the faces
for (x, y, w, h) in face_coordinates:
    cv2.rectangle(img, (x, y), (x + w, y + h), (randrange(128, 256), randrange(128, 256), randrange(128, 256)), 2)

print(face_coordinates)

#
cv2.imshow('Face Detector', img)


# wait until a key is pressed
cv2.waitKey() 
print("code working")
