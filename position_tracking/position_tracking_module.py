import cv2
import mediapipe as mp 
import time
import math

class PositionDetector():
	def __init__(self, mode=False, upBody=False,smooth=True, detectionCon=0.5,trackCon=.5,cam=False):
		self.mode=mode
		self.upBody=upBody
		self.smooth=smooth
		self.detectionCon=detectionCon
		self.trackCon=trackCon
		self.mpPose=mp.solutions.pose
		self.pose=self.mpPose.Pose(self.mode, self.upBody, self.smooth, self.detectionCon,self.trackCon)
		self.mpDraw=mp.solutions.drawing_utils
		self.cam=cam

	def findPose(self, img, draw=True):
		if self.cam==False:
			img=cv2.resize(img, (0,0), fx=.5,fy=.5,interpolation=cv2.INTER_AREA)
		imgRGB=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
		self.results=self.pose.process(imgRGB)

		if self.results.pose_landmarks:
			if draw:
				self.mpDraw.draw_landmarks(img,self.results.pose_landmarks,self.mpPose.POSE_CONNECTIONS)
			
		return img

	def getPosition(self, img, draw=True, index=0):
		self.lmlist=[]
		if self.results.pose_landmarks:
			for id, lm in enumerate(self.results.pose_landmarks.landmark):
				h, w, c=img.shape
				cx, cy=int(lm.x*w), int(lm.y*h)
				self.lmlist.append([id,cx,cy])

				if id==index and draw:
					cv2.circle(img,(cx,cy),10,(255,250,0),cv2.FILLED)
		return self.lmlist

	def findAngle(self, img ,p1, p2, p3, draw=True):
		x1, y1= self.lmlist[p1][1:]
		x2, y2= self.lmlist[p2][1:]
		x3, y3= self.lmlist[p3][1:]

		angle=math.degrees(math.atan2(y3-y2,x3-x2)-math.atan2(y1-y2,x1-x2))

		if angle<0:
			angle+=360

		if draw:
			cv2.line(img, (x1, y1),(x2,y2),(255, 255, 255),3)
			cv2.line(img, (x3, y3),(x2,y2),(255, 255, 255),3)
			cv2.circle(img, (x1, y1), 10, (0,0,255),cv2.FILLED)
			cv2.circle(img, (x2, y2), 10, (0,0,255),cv2.FILLED)
			cv2.circle(img, (x3, y3), 10, (0,0,255),cv2.FILLED)
			cv2.circle(img, (x1, y1), 15,(0, 0,255), 2)
			cv2.circle(img, (x2, y2), 15,(0, 0,255), 2)
			cv2.circle(img, (x3, y3), 15,(0, 0,255), 2)
			# cv2.putText(img,str(int(angle)), (x2+30, y2+30), cv2.FONT_HERSHEY_PLAIN, 2,(0,255,255),2 )
		return angle
		

def main():
	cap = cv2.VideoCapture('videos/vid2.mp4')
	pTime=0
	cTime=0
	detector=PositionDetector()
	while True:
		success, img=cap.read()
		if success:
			index=14

			img=detector.findPose(img)
			lmlist=detector.getPosition(img, index=index)
			try:
				print(lmlist[index])
			except Exception as e:
				pass
			
			cTime=time.time()
			fps=1/(cTime-pTime)
			pTime=cTime

			if success:	
				cv2.imshow('image',img)

			k=cv2.waitKey(1)
			if k==27:
				break
		else:
			break

	cap.release()

if __name__=="__main__":
	main()