import cv2
import mediapipe as mp 
import time


class HandDetector():
	def __init__(self, mode=False, maxHands=2,detectionCon=0.5,trackCon=0.5):
		self.mode=mode
		self.maxHands=maxHands
		self.detectionCon=detectionCon
		self.trackCon=trackCon
		self.mpHands = mp.solutions.hands
		self.hands=self.mpHands.Hands(self.mode,self.maxHands,self.detectionCon,self.trackCon)
		self.mpDraw = mp.solutions.drawing_utils
		self.tips=[4, 8, 12, 16, 20 ]

	def findHands(self,img,draw=True):
 		imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
 		self.results = self.hands.process(imgRGB)

 		if self.results.multi_hand_landmarks:
 			for handLms in self.results.multi_hand_landmarks:
 				if draw:
 					self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
 		return img
	
	def findPosition(self,img,handNo=0,draw=True,index=0):
 		self.lmlist=[]
 		if self.results.multi_hand_landmarks:
 			myHand=self.results.multi_hand_landmarks[handNo]
 			for id, lm in enumerate(myHand.landmark):
 				h, w, c= img.shape
 				cx, cy=int(lm.x*w), int(lm.y*h)
 				self.lmlist.append([id,cx,cy])
 				if draw and id==index:
 					cv2.circle(img,(cx,cy),15,(254,23,200),cv2.FILLED)
 		return self.lmlist

	def fingersUp(self):
 		fingers=[]
 		if self.lmlist[4][1]>self.lmlist[18][1]:
 			if self.lmlist[4][1]>self.lmlist[3][1]:
 				fingers.append(1)
 			else:
 				fingers.append(0)
 		elif self.lmlist[4][1]<self.lmlist[18][1]:
 			if self.lmlist[4][1]<self.lmlist[3][1]:
 				fingers.append(1)
 			else:
 				fingers.append(0)
 		else:
 			fingers.append(0)

 		for id in range(1,5):
 			if self.lmlist[self.tips[id]][2]<self.lmlist[self.tips[id]-2][2]:
 				fingers.append(1)
 			else:
 				fingers.append(0)
 		return fingers

def main():
	cap = cv2.VideoCapture(0)
	pTime=0
	cTime=0
	detector=HandDetector()
	while True:
		success, img=cap.read()
		img=detector.findHands(img)
		lmList=detector.findPosition(img,0,True,4)
		if len(lmList)!=0:
			print(lmList[4])

		cTime=time.time()
		fps=1/(cTime-pTime)
		pTime=cTime

		cv2.putText(img, str(int(fps)),(10,20),cv2.FONT_HERSHEY_PLAIN,1,(0,4,9))

		cv2.imshow("Image",img)
		k=cv2.waitKey(1)

		if k==27:
			break

	cap.release()


if __name__=="__main__":
	main()