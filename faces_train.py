import os
from PIL import Image
import cv2
import numpy as np
import pickle
#Define directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
image_dir = os.path.join(BASE_DIR, "images")

#Face cascade
face_cascade = cv2.CascadeClassifier('cascades/haarcascade_frontalface_alt2.xml')

#Face recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()

#Define data and label
y_labels = []
x_train = []
current_id = 0
label_ids = {}


for root, dirs, files, in os.walk(image_dir):
	for file in files:
		if file.endswith("png") or file.endswith("jpg"):
			path = os.path.join(root, file)
			label = os.path.basename(os.path.dirname(path)).replace(" ", "_").lower() 
			# print("Path: ",path)
			# print("Label: ",label)

			if not label in label_ids:
				label_ids[label] = current_id
				current_id += 1

			id_ = label_ids[label]
			#print(label_ids)
			#Add data and label
			#labels.append(label) #some number
			#images.append(path) #verify this image, and turn it into Numpy array, GRAY
			
			pil_image = Image.open(path).convert("L")
			size = (550, 550)
			final_image = pil_image.resize(size, Image.ANTIALIAS)

			image_array = np.array(final_image, "uint8")
			#print(image_array)

			faces = face_cascade.detectMultiScale(image_array, scaleFactor=1.5, minNeighbors=5)

			for (x, y, w, h) in faces:
				roi = image_array[y:y+h, x:x+w]
				x_train.append(roi)
				y_labels.append(id_)

# print(x_train)
# print(y_labels)

with open("labels.pickle", "wb") as f:
	pickle.dump(label_ids, f)


recognizer.train(x_train, np.array(y_labels))
recognizer.save("trainer.yml")
