import numpy as np
import pandas as pd
import time
import datetime
import cv2
import pickle

#Define Haar cascade
face_cascade = cv2.CascadeClassifier('cascades/haarcascade_frontalface_alt2.xml')
eyes_cascade = cv2.CascadeClassifier('cascades/haarcascade_eye.xml')

#Face recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainer.yml")

labels = {"person_name": 1}
with open("labels.pickle", "rb") as f:
	og_labels = pickle.load(f)
	labels = {v:k for k,v in og_labels.items()}


# video_src = 'BAR.mp4'
# cap = cv2.VideoCapture(video_src)

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
col_names = ['Name','Date','Time']
attendance = pd.DataFrame(columns = col_names)

while True:
	ret, frame = cap.read()

	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)
	#Checking attendance
	
	for (x, y, w, h) in faces:
		#print(x,y,w,h)
		roi_gray = gray[y:y+h, x:x+w]
		roi_color = frame[y:y+h, x:x+w]

		#Recognizer deep learned model
		id_, conf = recognizer.predict(roi_gray)

		# if conf >= 0.9: #and conf <= 85:
		# 	print(id_)
		# 	print(labels[id_])
		font = cv2.FONT_HERSHEY_SIMPLEX
		name = labels[id_]
		color = (255,255,255)
		stroke = 2
		#cv2.putText(frame, name, (x,y), font, 1, color, stroke, cv2.LINE_AA)
		cv2.putText(frame, str(id_)+": "+name, (x,y), font, 1, color, stroke, cv2.LINE_AA)

		#img_item = 'my-image.png'

		#cv2.imwrite(img_item, roi_gray)

		color = (255,0,0)
		stroke = 2
		end_cord_x = x + w
		end_cord_y = y + h
		cv2.rectangle(frame, (x,y), (end_cord_x, end_cord_y), color, stroke)

		#Detect eyes
		eyes = eyes_cascade.detectMultiScale(roi_gray)
		for (ex, ey, ew, eh) in eyes:
			cv2.rectangle(roi_color, (ex,ey), (ex+ew, ey+eh), (0,255,0), 2)

		

		#Time
		time_s = time.time()
		date = str(datetime.datetime.fromtimestamp(time_s).strftime('%Y-%m-%d'))
		timeStamp = datetime.datetime.fromtimestamp(time_s).strftime('%H:%M:%S')

		#Write csv file
		attendance.loc[len(attendance)] = [name,date,timeStamp]
		#data = pd.DataFrame({"Name": name, "Date": date, "Time": timeStamp}, index=[0])


		#attendance = attendance.append(data, ignore_index=True, sort=False)
		
	
	print(attendance.head())

	cv2.imshow('video', frame)

	if cv2.waitKey(20) & 0xFF == ord("q"):
		break
attendance=attendance.drop_duplicates(keep='first',subset=['Name'])
attendance.to_csv("Checking_attendance.csv")

cv2.destroyAllWindows()
