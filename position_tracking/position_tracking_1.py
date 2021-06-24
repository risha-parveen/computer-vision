import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture('videos/vid4.mp4')

mpPose=mp.solutions.pose
pose=mpPose.Pose()
mpDraw=mp.solutions.drawing_utils

pTime=0
cTime=0

while (cap.isOpened()):
	success, img=cap.read()
	if success:
		img=cv2.resize(img, (0,0), fx=.2,fy=.2,interpolation=cv2.INTER_AREA)
		imgRGB=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
		
		results=pose.process(imgRGB)

		if results.pose_landmarks:
			mpDraw.draw_landmarks(img,results.pose_landmarks,mpPose.POSE_CONNECTIONS)
			for id, lm in enumerate(results.pose_landmarks.landmark):
				h, w, c=img.shape
				cx, cy=int(lm.x*w), int(lm.y*h)
					
				if id==0:
					cv2.circle(img,(cx,cy),10,(255,250,0),cv2.FILLED)


		cTime=time.time()
		fps=1/(cTime-pTime)
		pTime=cTime

		cv2.imshow('image',img)
		
		k=cv2.waitKey(1)
		if k==27:
			break
	else:
		break

cap.release()


