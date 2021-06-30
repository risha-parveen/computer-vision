import cv2
import numpy as np
import time 
import position_tracking_module as ptm 

cap=cv2.VideoCapture('AiTrainer/curls.mp4')

pTime=0
cTime=0

detector=ptm.PositionDetector(detectionCon=0.9)
per=0
direction=1
count=0

while True:
	success,img=cap.read()

	if success:
		img=detector.findPose(img, draw=False)
		lmlist=detector.getPosition(img,draw=False)
		if len(lmlist)!=0:
			angle=detector.findAngle(img,11,13, 15)
			per=np.interp(angle,(210,310),(0,100))
			bar=np.interp(angle, (210,310),(500,200))

			if per==100 and direction==1:
				count+=0.5
				direction=-1
			if per==0 and direction==-1:
				count+=0.5
				direction=1

			cv2.rectangle(img, (40,200),(80,500),(255,255,200),2)
			cv2.rectangle(img, (40,500),(80,int(bar)),(255,255,200),cv2.FILLED)
			cv2.putText(img, str(int(per))+"%",(40,180),cv2.FONT_HERSHEY_PLAIN,1.2,(255,255,255),2)


			cv2.rectangle(img, (10,10),(120,120),(100,100,100),cv2.FILLED)
			cv2.rectangle(img, (10,10),(120,120),(255,255,200),2)

			l=len(str(int(count)))

			cv2.putText(img,str(int(count)),(50-l*5,85-l*2),cv2.FONT_HERSHEY_PLAIN,5-l,(255,255,200),5-l)
			

		cTime=time.time()
		fps=1/(cTime-pTime)
		pTime=cTime
		cv2.putText(img, "FPS:"+str(int(fps)),(img.shape[1]-70,20),cv2.FONT_HERSHEY_PLAIN,1,(50,20,20),2)

		cv2.imshow('img',img)
		if cv2.waitKey(1)==27:
			break
	else:
		break
cap.release()