import cv2
import mediapipe as mp 
import time 

cap=cv2.VideoCapture('videos/vid (1).mp4')

pTime=0
cTime=0

mpDraw=mp.solutions.drawing_utils
mpFaceMesh=mp.solutions.face_mesh
faceMesh=mpFaceMesh.FaceMesh(max_num_faces=2)
drawSpec=mpDraw.DrawingSpec(thickness=1, circle_radius=1)

while True:
	success, img=cap.read()
	if success:
		
		img=cv2.resize(img, (0,0), fx=.2,fy=.2, interpolation=cv2.INTER_AREA)
		imgRGB=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
		results=faceMesh.process(imgRGB)
		if results.multi_face_landmarks:
			for facelms in results.multi_face_landmarks:
				mpDraw.draw_landmarks(img, facelms,mpFaceMesh.FACE_CONNECTIONS,
					drawSpec,drawSpec)
				for id, lm in enumerate(facelms.landmark):
					#print(lm)
					h, w, c=img.shape
					x,y= int(lm.x*w),int(lm.y*h)
					print(id,x, y)

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