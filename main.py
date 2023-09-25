import cv2
import pickle
import cvzone
import numpy as np

# video feed
cap = cv2.VideoCapture('./assets/carPark.mp4')

with open('CarParkPos', 'rb') as f:
    binary_format = pickle.load(f)

width, height = 107, 48

def checkParkingSpace(imgPro):

    spaceCounter = 0

    for idx, pos in enumerate(binary_format, start=1):  # Start numbering from 1
        x, y = pos

        imgCrop = imgPro[y:y+height, x:x+width]
        pixel = cv2.countNonZero(imgCrop)

        # Default color is red (occupied space)
        color = (0, 0, 255)

        if pixel < 900:
            color = (0, 255, 0)  # Green for free spaces
            spaceCounter += 1

        thickness = 5 if color == (0, 255, 0) else 3  # Adjust thickness based on color
        # Draw rectangle around space (occupied or free)
        cv2.rectangle(img, (x, y), (x + width, y + height), color, thickness)
        # Draw parking slot number for all spaces
        cv2.putText(img, str(idx), (x + 10, y + height - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    # Display overall free space count
    cvzone.putTextRect(img, f'Free: {spaceCounter}/{len(binary_format)}', (100, 50), scale=3, thickness=5, offset=20, colorR=(0, 255, 0))

while True:

    # infinite loop video
    if cap.get(cv2.CAP_PROP_POS_FRAMES) == cap.get(cv2.CAP_PROP_FRAME_COUNT):
        cap.set(cv2.CAP_PROP_POS_FRAMES,0)

    success, img = cap.read()

    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(imgGray, (3,3), 1)

    # converting image to binary (white lines on black bg)
    imgThreshold = cv2.adaptiveThreshold(imgBlur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,25,16)


    # clear out the "noise" pixels
    imgMedian = cv2.medianBlur(imgThreshold, 5)
    kernel = np.ones((3,3), np.uint8)
    imgDilate = cv2.dilate(imgMedian, kernel, iterations=1)

    checkParkingSpace(imgDilate)

# for pos in posList:
        # cv2.rectangle(img, pos, (pos[0]+width, pos[1]+height),(255,0,255),2)

    cv2.imshow("Image", img)
    cv2.imshow("ImageBlur", imgBlur)
    cv2.imshow("ImageThresh", imgMedian)
    cv2.imshow("Imageblur",imgGray)

# slows down the video
    cv2.waitKey(10)
