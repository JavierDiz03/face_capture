import cv2
import os
import numpy as np

dataPath = 'C:/Users/javie/Desktop/cosas/ReconocimientoFacial/Data'
peopleList = os.listdir(dataPath)
print('Lista de personas: ', peopleList)

labels = []
facesData = []
label = 0

for nameDir in peopleList:
	personPath = dataPath + '/' + nameDir
	print('Reading the images')

	for fileName in os.listdir(personPath):
		print('Faces: ', nameDir + '/' + fileName)
		labels.append(label)
		facesData.append(cv2.imread(personPath+'/'+fileName,0))
	label = label + 1

face_recognizer = cv2.face.LBPHFaceRecognizer_create()

print("Training IA...")
face_recognizer.train(facesData, np.array(labels))

face_recognizer.write('modeloLBPHFace.xml')
print("Face saved...")