import cv2
import time 
import numpy as np
import hand_tracking_module as htm 
import math
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

cw, ch=1050,700

cap=cv2.VideoCapture(0)
cap.set(3, cw)
cap.set(4, ch)

pTime=0
cTime=0

detector=htm.HandDetector(detectionCon=0.7)

devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(
    IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))
volRange=volume.GetVolumeRange()

minVol=volRange[0]
maxVol=volRange[1]

vol=0
volbar=400
volPer=0
while True:
	success,img=cap.read()
	detector.findHands(img)
	lmList=detector.findPosition(img, draw=False)
	if len(lmList)!=0:

		x1, y1=lmList[4][1], lmList[4][2]
		x2, y2=lmList[8][1], lmList[8][2]
		x3, y3=(x1+x2)//2, (y1+y2)//2

		cv2.circle(img, (x1, y1), 10, (255, 0, 255), cv2.FILLED)
		cv2.circle(img, (x2, y2), 10, (255, 0, 255), cv2.FILLED)
		cv2.line(img, (x1, y1), (x2,y2),(255, 0,255),3)
		cv2.circle(img, (x3, y3), 10, (255, 0, 255), cv2.FILLED)

		length=math.hypot(x2-x1, y2-y1)
		
		vol=np.interp(length,[30, 200], [minVol, maxVol])
		volbar=np.interp(length,[30, 200], [400,100])
		volPer=np.interp(length,[30,200],[0,100])
		
		volume.SetMasterVolumeLevel(vol, None)

		if length<30:
			cv2.circle(img, (x3, y3), 10, (0, 255, 0), cv2.FILLED)

	cv2.rectangle(img, (50,100),(85, 400), (0, 255,0),2)
	cv2.rectangle(img, (50,int(volbar)),(85, 400),(0,255,0),cv2.FILLED)
	cv2.putText(img, str(int(volPer))+"%" , (50,90), cv2.FONT_HERSHEY_PLAIN,1,(255,255,200),2)

	cTime=time.time()
	fps=1/(cTime-pTime)
	pTime=cTime

	cv2.putText(img, "FPS:"+str(int(fps)), (30,30), cv2.FONT_HERSHEY_PLAIN,1,(0,0,0),2)

	cv2.imshow('img',img)
	if cv2.waitKey(1)==27:
		break
cap.release()
