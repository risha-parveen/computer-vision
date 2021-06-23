import cv2
import time
import hand_tracking_module as htm

cap = cv2.VideoCapture(0)
pTime=0
cTime=0
detector=htm.HandDetector()
while True:
	success, img=cap.read()
	img=detector.findHands(img)
	index=5
	lmList=detector.findPosition(img,index=index)
	if len(lmList)!=0:
		print(lmList[index])

	cTime=time.time()
	fps=1/(cTime-pTime)
	pTime=cTime

	cv2.putText(img, str(int(fps)),(10,20),cv2.FONT_HERSHEY_PLAIN,1,(0,4,9))

	cv2.imshow("Image",img)
	k=cv2.waitKey(1)

	if k==27:
		break

cap.release()