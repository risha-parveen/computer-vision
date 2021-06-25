import cv2
import mediapipe as mp 
import time 

cap=cv2.VideoCapture('videos/4.mp4')

pTime=0
cTime=0

mpFaceDetection=mp.solutions.face_detection
mpDraw=mp.solutions.drawing_utils
faceDetection=mpFaceDetection.FaceDetection(0.75)

while True:
	success, img=cap.read()
	if success:
		img=cv2.resize(img, (0,0), fx=.2,fy=.2, interpolation=cv2.INTER_AREA)
		imgRGB=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
		results=faceDetection.process(imgRGB)

		if results.detections:
			for id, detection in enumerate(results.detections):
				#mpDraw.draw_detection(img,detection)
				#print(detection.location_data.relative_bounding_box)
				bbox=detection.location_data.relative_bounding_box
				h, w, c=img.shape

				bbox=int(bbox.xmin*w) , int(bbox.ymin*h),\
				 int(bbox.width*w), int(bbox.height*h)

				cv2.rectangle(img, bbox, (255,100,255),2)
				cv2.putText(img, str(int(detection.score[0]*100))+"%",(bbox[0], bbox[1]-5),cv2.FONT_HERSHEY_PLAIN,1.5,(255, 100, 255),2)
		

		cTime=time.time()
		fps=1/(cTime-pTime)
		pTime=cTime
		cv2.putText(img, "FPS:"+str(int(fps)),(10,20),cv2.FONT_HERSHEY_PLAIN,1.5,(50,20,20))
		cv2.imshow('image', img)

		if cv2.waitKey(1)==27:
			break
	else:
		break
cap.release()