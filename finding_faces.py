#Find faces from images and label it out

import face_recognition
import argparse
import pickle
import cv2

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path to input image")
ap.add_argument("-d", "--detection-method", type=str, default="cnn",
	help="face detection model to use: either `hog` or `cnn`")
args = vars(ap.parse_args())

image = cv2.imread(args["image"])
rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

print("finding faces....")
boxes = face_recognition.face_locations(rgb, model=args["detection_method"])

print("faces detected(", args["detection_method"], "):",len(boxes))

for (top, right, bottom, left) in boxes:
	cv2.rectangle(image, (left, top), (right, bottom), (0, 0, 255), 2)


cv2.imshow("Image", image)
cv2.waitKey(0)
