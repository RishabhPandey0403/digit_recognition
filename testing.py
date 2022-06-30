import random
import cv2 as cv

# new_filename = (str) (random.random())[2:9] + '.png'
# print(new_filename)

video = cv.VideoCapture(0)

# new_filename = (str) (random.random())[2:9] + '.png'
# start_time = datetime.now()
# print(f'Start time is: {start_time}.')

while True:
    ret, frame = video.read()
    rescaled_frame = cv.resize(frame, (1000,700))
    cv.rectangle(rescaled_frame, (rescaled_frame.shape[1]//2 - 300, rescaled_frame.shape[0]//2 - 210), (rescaled_frame.shape[1]//2 + 300, rescaled_frame.shape[0]//2 + 210), (0,0,0), 3)
    # cv.line(rescaled_frame, (150,150), (600, 600))
    cv.imshow('Camera', rescaled_frame)
    # Press 'q' to exit at any moment
    if cv.waitKey(1) == ord('q'):
        break


video.release()
cv.destroyAllWindows()