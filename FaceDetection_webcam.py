import cv2
from random import randrange

# load some pre-trained data on face frontals from opencv
trained_face_data = cv2.CascadeClassifier('frontalface_default.xml')

#choose an image to detect image
webcam = cv2.VideoCapture(0)

# itereate over frames
while True :
    # read the current frame
    successful_frame_read, frame = webcam.read()

    #must convert to grayscale
    grayscaled_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #detect faces
    face_coordinates = trained_face_data.detectMultiScale(grayscaled_img)

    #draw rectangles around the faces
    for (x, y, w, h) in face_coordinates:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (randrange(128, 256), randrange(128, 256), randrange(128, 256)), 2)

    cv2.imshow("Webcam", frame)

    # wait until a key is pressed
    key = cv2.waitKey(1) 
    if key == 81 or key == 113 :
        break

# release the object
webcam.release()
