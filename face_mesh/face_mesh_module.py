import cv2
import mediapipe as mp 
import time 
		
class FaceMeshDetector():
	def __init__(self, staticmode=False,maxFaces=2,mindetectionCon=0.5,trackCon=0.5,cam=False):
		self.staticmode=staticmode
		self.maxFaces=maxFaces
		self.mindetectionCon=mindetectionCon
		self.trackCon=trackCon
		self.mpDraw=mp.solutions.drawing_utils
		self.mpFaceMesh=mp.solutions.face_mesh
		self.faceMesh=self.mpFaceMesh.FaceMesh(self.staticmode,self.maxFaces,self.mindetectionCon,self.trackCon)
		self.drawSpec=self.mpDraw.DrawingSpec(thickness=1, circle_radius=1)
		self.cam=cam 

	def findFaceMesh(self,img,draw=True):
		if self.cam==False:
			img=cv2.resize(img, (0,0), fx=.2,fy=.2, interpolation=cv2.INTER_AREA)
		self.imgRGB=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
		self.results=self.faceMesh.process(self.imgRGB)

		faces=[]
		if self.results.multi_face_landmarks:		
			for facelms in self.results.multi_face_landmarks:
				if draw:
					self.mpDraw.draw_landmarks(img, facelms,self.mpFaceMesh.FACE_CONNECTIONS,
						self.drawSpec,self.drawSpec)
				face=[]
				for id, lm in enumerate(facelms.landmark):
					#print(lm)
					h, w, c=img.shape
					x,y= int(lm.x*w),int(lm.y*h)
					#print(id,x, y)
					face.append([x,y])
			faces.append([face])
		return img, faces

def main():
	cap=cv2.VideoCapture('videos/vid (1).mp4')

	pTime=0
	cTime=0

	detector=FaceMeshDetector()
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

if __name__=="__main__":
	main()
