import cv2 as cv
import numpy as np
import random
from datetime import datetime

''' 
TASKS + STATUS
- Open laptop camera/webcam ------------------------------------------------------- DONE
- Turn live-feed into grayscale --------------------------------------------------- DONE
- Resize frame dimensions --------------------------------------------------------- Check dimensions with shyam
- Add textbox countdown from 5 ---------------------------------------------------- DONE
- After 5 seconds, take camera snapshot and save it in the testing_photo folder --- DONE
'''

video = cv.VideoCapture(0)

if not video.isOpened():
    print('Video error')
    exit()

new_filename = (str) (random.random())[2:9] + '.png'

start_time = datetime.now()
print(f'Start time is: {start_time}.')

while True:
    ret, frame = video.read()
    cv.resize(frame, (600,600))
    cv.putText(frame, f"Time left until picture taken: {10 - ((datetime.now() - start_time).seconds)}", (50,50), cv.FONT_HERSHEY_SIMPLEX, 1.0, (255,0,0))
    gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    cv.imshow('Camera', gray_frame)
    if cv.waitKey(1) == ord('a'):
        cv.imwrite('image_testing/' + new_filename, gray_frame)
        print(f'Image {new_filename} saved to testing folder!')
        break
    if (10 - ((datetime.now() - start_time).seconds)) < 1:
        cv.imwrite('image_testing/' + new_filename, gray_frame)
        print(f'Image {new_filename} saved to testing folder!')
        break

video.release()
cv.destroyAllWindows()