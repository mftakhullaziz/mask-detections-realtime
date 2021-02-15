from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import numpy as np
import argparse
import imutils
import time
import cv2
import os
from imutils.video import FPS

def detect_and_predict_mask(frame, faceNet, maskNet):

	(h, w) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),
		(104.0, 177.0, 123.0))
	faceNet.setInput(blob)
	detections = faceNet.forward()

	faces = []
	locs = []
	preds = []

	for i in range(0, detections.shape[2]):
		confidence = detections[0, 0, i, 2]
		if confidence > args["confidence"]:
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")
			(startX, startY) = (max(0, startX), max(0, startY))
			(endX, endY) = (min(w - 1, endX), min(h - 1, endY))
			face = frame[startY:endY, startX:endX]
			face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
			face = cv2.resize(face, (224, 224))
			face = img_to_array(face)
			face = preprocess_input(face)
			faces.append(face)
			locs.append((startX, startY, endX, endY))

	if len(faces) > 0:
		faces = np.array(faces, dtype="float32")
		preds = maskNet.predict(faces, batch_size=32)
	return (locs, preds)

ap = argparse.ArgumentParser()
ap.add_argument("-f", "--face", type=str,
	default="model_face_detection_from_opencv",
	help="path to face detector model directory")
ap.add_argument("-m", "--model", type=str,
	default="model_training/mask_detection_mobilenetV2.model",
	help="path to trained face mask detector model")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
	help="minimum probability to filter weak detections")
args = vars(ap.parse_args())
fps = FPS().start()

print("[INFO] loading face detector model...")
prototxtPath = os.path.sep.join([args["face"], "deploy.prototxt"])
weightsPath = os.path.sep.join([args["face"],
	"res10_300x300_ssd_iter_140000.caffemodel"])
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

print("[INFO] loading face mask detector model...")
maskNet = load_model(args["model"])

print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
time.sleep(2.0)

# If using fullscreen windows
# cv2.namedWindow("Frame", cv2.WND_PROP_FULLSCREEN)
# cv2.setWindowProperty("Frame", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

while True:
	frame = vs.read()
	frame = imutils.resize(frame, width=1200)
	frame = cv2.flip(frame,1)
	(locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet)
	for (box, pred) in zip(locs, preds):
		(startX, startY, endX, endY) = box
		(mask, withoutMask) = pred
		label = "mask" if mask > withoutMask else "no mask"
		color = (0, 255, 0) if label == "mask" else (0, 0, 255)
		# if label == "No Mask":
		# 	notification.notify(
		# 		title = "***No Mask Detected***",
        #         		message = "Wear Mask to stay safe! ",
        #         		app_icon = "images/1.ico",    #ico file should be downloaded
        #         		timeout = 1
        #     		)
		label = "{} , Acc : {:.1f}%".format(label, max(mask, withoutMask) * 100)
		print("Prediction ==> ", label)
		cv2.putText(frame, label, (startX, startY - 10),
			cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.1, color, 1, cv2.LINE_AA)
		cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

		# Alarm when "No Mask" detected
		# if mask < withoutMask:
		# 	path = os.path.abspath("Alarm.wav")
		# 	playsound(path)

	# show the output frame
	cv2.imshow("Realtime Mask Detection", frame)	
	key = cv2.waitKey(1) & 0xFF
	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break

fps.stop()
print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
# print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

# do a bit of cleanup
# cv2.waitKey(1)
cv2.destroyAllWindows()
vs.stop()
