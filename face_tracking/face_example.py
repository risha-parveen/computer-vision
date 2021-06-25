import cv2
import face_tracking_module as ftm
import time 

cap=cv2.VideoCapture(0)

pTime=0
cTime=0

detector=ftm.FaceDetector(0.5, cam=True)
while True:
	success, img=cap.read()
	if success:
		img, bboxes=detector.findFaces(img)
		print(bboxes)
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