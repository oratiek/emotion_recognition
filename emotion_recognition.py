import cv2
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np

def classify(img):
	img = cv2.resize(img,(48,48))
	img = img.astype("float32")/225.0
	img = img_to_array(img)
	img = np.expand_dims(img, axis=0)
	result = model.predict(img)[0]
	return result.argmax(), result

model = load_model("model.h5")
fontType = cv2.FONT_HERSHEY_SIMPLEX
EMOTIONS = ["angry","disgust","scared", "happy", "sad", "surprised","neutral"]
cascade = cv2.CascadeClassifier("cascade.xml")
cap = cv2.VideoCapture(0)
while True:
	ret, frame = cap.read()
	frame = cv2.flip(frame, 1)
	#frame = cv2.resize(frame, (1800,1080))
	img = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
	faces = cascade.detectMultiScale(img, 1.3, 5)
	for x,y,w,h in faces:
		face = img[y:y+h,x:x+w]
		emotion, prob = classify(face)
		print(EMOTIONS[emotion])
		cv2.putText(frame,EMOTIONS[emotion],(x,y), fontType, 3,(255,0,0),2,cv2.LINE_AA)
		cv2.putText(frame,"Angry:"+str(prob[0]),(0,100), fontType, 1,(255,0,0),2,cv2.LINE_AA)
		cv2.putText(frame,"Disgust:"+str(prob[1]),(0,200), fontType, 1,(255,0,0),2,cv2.LINE_AA)
		cv2.putText(frame,"Scared:"+str(prob[2]),(0,300), fontType, 1,(255,0,0),2,cv2.LINE_AA)
		cv2.putText(frame,"Happy:"+str(prob[3]),(0,400), fontType, 1,(255,0,0),2,cv2.LINE_AA)
		cv2.putText(frame,"Sad:"+str(prob[4]),(0,500), fontType, 1,(255,0,0),2,cv2.LINE_AA)
		cv2.putText(frame,"Surprised:"+str(prob[5]),(0,600), fontType, 1,(255,0,0),2,cv2.LINE_AA)
		cv2.putText(frame,"Neutral:"+str(prob[6]),(0,700), fontType, 1,(255,0,0),2,cv2.LINE_AA)
		cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),10)
	cv2.imshow("frame",frame)
	if cv2.waitKey(1) == ord("q"):
		break
