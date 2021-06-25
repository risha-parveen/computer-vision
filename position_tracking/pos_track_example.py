import cv2
import time
import position_tracking_module as ptm

cap = cv2.VideoCapture(0)
pTime=0
cTime=0
detector=ptm.PositionDetector(cam=True)
while True:
	success, img=cap.read()
	if success:
		index=12
		img=detector.findPose(img)
		lmlist=detector.getPosition(img, index=index)
		try:
			print(lmlist[index])
		except Exception as e:
			pass
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