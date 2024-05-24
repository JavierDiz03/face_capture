import cv2
import os
import time
from tkinter import messagebox
import sys

dataPath = 'C:/Users/javie/Desktop/cosas/ReconocimientoFacial/Data'
imagePaths = os.listdir(dataPath)
print('imagePaths=',imagePaths)

face_recognizer = cv2.face.LBPHFaceRecognizer_create()
face_recognizer.read('modeloLBPHFace.xml')
cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)
faceClassif = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')

desconocido_iniciado = False
tiempo_inicio = 0

conocido_iniciado = False
tiempo_inicio_true = 0

while True:
	ret,frame = cap.read()
	if ret == False: break
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	auxFrame = gray.copy()

	faces = faceClassif.detectMultiScale(gray,1.3,5)

	for (x,y,w,h) in faces:
		rostro = auxFrame[y:y+h,x:x+w]
		rostro = cv2.resize(rostro,(150,150),interpolation= cv2.INTER_CUBIC)
		result = face_recognizer.predict(rostro)

		cv2.putText(frame,'{}'.format(result),(x,y-5),1,1.3,(255,255,0),1,cv2.LINE_AA)
		#result and result[0] < len(imagePaths)
		if result[1] < 60:
			cv2.putText(frame, '{}'.format(imagePaths[result[0]]), (x, y-25), 2, 1.1, (0, 255, 0), 1, cv2.LINE_AA)
			cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
			if not conocido_iniciado:
				conocido_iniciado = True
				tiempo_inicio_true = time.time()
			else:
				if time.time() - tiempo_inicio_true >= 5:
					messagebox.showinfo('Login correct', 'Autentificate user')
					conocido_iniciado = False
					sys.exit()
		else:
			cv2.putText(frame, 'Unkown user', (x, y-20), 2, 0.8, (0, 0, 255), 1, cv2.LINE_AA)
			cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
			if not desconocido_iniciado:
				desconocido_iniciado = True
				tiempo_inicio = time.time()
			else:
				if time.time() - tiempo_inicio >= 5:
					messagebox.showinfo('Incorrect user', 'Unknown user')
					desconocido_iniciado = False
					sys.exit()
	cv2.imshow('frame',frame)
	k = cv2.waitKey(1)
	if k == 27:
		break

cap.release()
cv2.destroyAllWindows()
