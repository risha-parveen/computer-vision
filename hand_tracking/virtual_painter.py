import cv2
import time
import os 
import numpy as np
import hand_tracking_module as htm


red=(0,25,225)
green=(0,217,0)
blue=(225,128,0)
black=(10,10,10)
white=(255,255,255)

folderpath="paint"
mylist=os.listdir(folderpath)
overlay=[]
for im in mylist:
	image=cv2.imread(f'{folderpath}/{im}')
	overlay.append(image)

header=overlay[0]

cap=cv2.VideoCapture(0)
detector=htm.HandDetector(detectionCon=0.75)

color=red

brushthickness=15
eraserthickness=25
xp,yp=0,0
imgCanvas=np.zeros((480,640,3),np.uint8)


while True:
	success,img=cap.read()
	w=int((img.shape[0]*960)/img.shape[1])
	img=cv2.resize(img,(960,w))
	imgCanvas=cv2.resize(imgCanvas,(960,w))
	#flip image
	img=cv2.flip(img,1)

	img=detector.findHands(img)
	lmlist=detector.findPosition(img,draw=False)

	

	if len(lmlist)!=0:
		x1,y1= lmlist[8][1:]
		x2,y2= lmlist[12][1:]
		fingers=detector.fingersUp()

		if fingers[1] and fingers[2]:
			xp,yp=x1,y1
			if all(x>=1 for x in fingers)==False:
				cv2.rectangle(img, (x1,y1),(x2,y2),color,cv2.FILLED)
			
			colors=[red,green,blue,black,white]

			for count,c in enumerate(colors):
				if y1<82:
					if count*180<x1<(count+1)*180:
						header=overlay[count]
						color=c

		if fingers[1] and fingers[2]==False:
			if color==white: 
				cv2.circle(img,(x1,y1),25,color,2)
				cv2.line(imgCanvas,(xp,yp),(x1,y1),(0,0,0),eraserthickness)
			
			else:
				cv2.circle(img,(x1,y1),15,color,cv2.FILLED)				
				cv2.line(img,(xp,yp),(x1,y1),color,brushthickness)
				cv2.line(imgCanvas,(xp,yp),(x1,y1),color,brushthickness)
			xp,yp=x1,y1
		if all(x>=1 for x in fingers):
			imgCanvas=np.zeros((480,640,3),np.uint8)
			imgCanvas=cv2.resize(imgCanvas,(960,w))


	# adding images img and imgCanvas ->>	
	imgGray=cv2.cvtColor(imgCanvas, cv2.COLOR_BGR2GRAY)
	_,imgInv=cv2.threshold(imgGray,5,255,cv2.THRESH_BINARY_INV)
	imgInv= cv2.cvtColor(imgInv,cv2.COLOR_GRAY2BGR)
	img=cv2.bitwise_and(img, imgInv)
	img=cv2.bitwise_or(img, imgCanvas)

	img[0:82,0:]=header
	cv2.imshow('image',img)
	if cv2.waitKey(1)==27:
		break
cap.release()