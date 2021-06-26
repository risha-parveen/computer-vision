import cv2
import face_mesh_module as fmm
import time 
		
		

cap=cv2.VideoCapture(0)

pTime=0
cTime=0

detector=fmm.FaceMeshDetector(cam=True)
while True:
	success, img=cap.read()
	if success:
		img, faces=detector.findFaceMesh(img)
		if len(faces)!=0:
			print(len(faces))
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
