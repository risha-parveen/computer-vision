import cv2
import time
import os 
import hand_tracking_module as htm

cap=cv2.VideoCapture(0)

folderpath='fingers'
mylist=os.listdir(folderpath)

overlaylist=[]

for impath in mylist:
	image=cv2.imread(f'{folderpath}/{impath}')
	overlaylist.append(image)

pTime=0
cTime=0

detector=htm.HandDetector(detectionCon=0.75)

tips=[4, 8, 12, 16, 20 ]

total=0
while True:
	success, img= cap.read()

	img=detector.findHands(img)
	lmlist=detector.findPosition(img,draw=False)
	#print(lmlist)

	if len(lmlist)!=0:
		fingers=[]
		#thumb
		if lmlist[4][1]>lmlist[18][1]:
			if lmlist[4][1]>lmlist[3][1]:
				fingers.append(1)
			else:
				fingers.append(0)
		elif lmlist[4][1]<lmlist[18][1]:
			if lmlist[4][1]<lmlist[3][1]:
				fingers.append(1)
			else:
				fingers.append(0)
		else:
			fingers.append(0)
		#other fingers
		for id in range(1,5):
			if lmlist[tips[id]][2]<lmlist[tips[id]-2][2]:
				fingers.append(1)
			else:
				fingers.append(0)
		total=fingers.count(1)


		h, w, c=overlaylist[total-1].shape
		img[10:h+10, 10:w+10]=overlaylist[total-1]

		cv2.rectangle(img, (30,250),(190,410),(250,200,250),cv2.FILLED)
		cv2.putText(img, str(total), (45,400),cv2.FONT_HERSHEY_PLAIN,13,(255,100,255),15)

	cTime=time.time()
	fps=1/(cTime-pTime)
	pTime=cTime

	cv2.putText(img,"FPS:"+ str(int(fps)),(img.shape[1]-70,20),cv2.FONT_HERSHEY_PLAIN,1,(0,4,9),2)

	cv2.imshow('img',img)
	if cv2.waitKey(1)==27:
		break
cap.release()
