import cv2 as cv
from matplotlib.pyplot import gray
import numpy as np
import random
from datetime import datetime

''' 
TASKS + STATUS
- Open laptop camera/webcam ------------------------------------------------------- DONE
- Turn live-feed into grayscale --------------------------------------------------- DONE
- Resize frame dimensions --------------------------------------------------------- Check dimensions with shyam
- Add textbox countdown from 5 ---------------------------------------------------- DONE
- Create a rectangle for where you want the A4 to be positioned ------------------- DONE
- After 5 seconds, take camera snapshot and save it in the testing_photo folder --- DONE


- Create a function that extracts the image in the rectangle and saves it --------- TBD
--- (this image will be the user input that will be predicted by the CNN)
- 
'''

video = cv.VideoCapture(0)

if not video.isOpened():
    print('Video error')
    exit()

new_filename = (str) (random.random())[2:9] + '.png'

def crop_image(img, filename):
    cropped = img[img.shape[1]//2 - 300:img.shape[0]//2 - 210, img.shape[1]//2 + 300:img.shape[0]//2 + 210]
    cv.imwrite('processed_image/' + filename, )
    return True
start_time = datetime.now()
print(f'Start time is: {start_time}.')

while True:
    ret, frame = video.read()
    rescaled_frame = cv.resize(frame, (1000,700))
    cv.putText(rescaled_frame, f"Time left until picture taken: {10 - ((datetime.now() - start_time).seconds)}", (50,50), cv.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,0))
    cv.rectangle(rescaled_frame, (rescaled_frame.shape[1]//2 - 300, rescaled_frame.shape[0]//2 - 210), (rescaled_frame.shape[1]//2 + 300, rescaled_frame.shape[0]//2 + 210), (0,0,0), 3)
    gray_frame = cv.cvtColor(rescaled_frame, cv.COLOR_BGR2GRAY)
    cv.imshow('Camera', gray_frame)
    
    # Press 'q' to exit at any moment
    if cv.waitKey(1) == ord('q'):
        break

    # If you want to take + save the image before the 10 second countdown, simply press 'a'
    if cv.waitKey(1) == ord('a'):
        cv.imwrite('preprocessed_image/' + new_filename, gray_frame)
        print(f'Image {new_filename} saved to preprocessed folder!')
        crop_image(gray_frame, new_filename)
        print(f'Image {new_filename} saved to processed folder!')
        break

    # Checks if the countdown value is below 1 and takes a snapshot
    if (10 - ((datetime.now() - start_time).seconds)) < 0:
        cv.imwrite('preprocessed_image/' + new_filename, gray_frame)
        print(f'Image {new_filename} saved to preprocessed folder!')
        crop_image(gray_frame, new_filename)
        print(f'Image {new_filename} saved to processed folder!')
        break

video.release()
cv.destroyAllWindows()