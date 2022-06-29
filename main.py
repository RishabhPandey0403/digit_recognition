import cv2 as cv
import numpy as np
import random
'''
- Open laptop camera/webcam ------------------------------------------------------- DONE
- Turn live-feed into grayscale --------------------------------------------------- DONE
- Resize frame dimensions 
- Add textbox countdown from 5
- After 5 seconds, take camera snapshot and save it in the testing_photo folder --- DONE
'''

video = cv.VideoCapture(0)

if not video.isOpened():
    print('Video error')
    exit()

new_filename = (str) (random.random())[2:9] + '.png'

while True:
    ret, frame = video.read()
    # cv.resize(frame, ())
    # Ask shyam about dimensions
    gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    cv.imshow('Camera', gray_frame)
    if cv.waitKey(1) == ord('a'):
        cv.imwrite('image_testing/' + new_filename, gray_frame)
        print(f'Image {new_filename} saved to testing folder!')
        break

video.release()
cv.destroyAllWindows()