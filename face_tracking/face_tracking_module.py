import cv2
import mediapipe as mp 
import time 
		
class FaceDetector():
	def __init__(self,mindetectionCon=0.5,cam=False):
		self.mindetectionCon=mindetectionCon
		self.mpFaceDetection=mp.solutions.face_detection
		self.mpDraw=mp.solutions.drawing_utils
		self.faceDetection=self.mpFaceDetection.FaceDetection(self.mindetectionCon)
		self.cam=cam
		self.color=(255,100,255)

	def findFaces(self,img,draw=True):
		if self.cam==False:
			img=cv2.resize(img, (0,0), fx=.2,fy=.2, interpolation=cv2.INTER_AREA)
		imgRGB=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
		results=self.faceDetection.process(imgRGB)

		bboxes=[]
		if results.detections:
			for id, detection in enumerate(results.detections):
				#mpDraw.draw_detection(img,detection)
				#print(detection.location_data.relative_bounding_box)
				bbox=detection.location_data.relative_bounding_box
				h, w, c=img.shape

				bbox=int(bbox.xmin*w) , int(bbox.ymin*h),\
				 int(bbox.width*w), int(bbox.height*h)

				bboxes.append([id,bbox,detection.score])
				if draw:
					img=self.fancyDraw(img,bbox)
					cv2.putText(img, str(int(detection.score[0]*100))+"%",
						(bbox[0], bbox[1]-10),cv2.FONT_HERSHEY_PLAIN,1.5,self.color,2)
		return img, bboxes
		
	def fancyDraw(self, img, bbox, l=35, t=5, rt=1):
		x,y,w,h=bbox
		x1, y1= x+w,y+h
		cv2.rectangle(img, bbox, self.color,rt)
		#top left corner
		cv2.line(img,(x,y),(x+l,y),self.color,t)
		cv2.line(img,(x,y),(x,y+l),self.color,t)
		#top right
		cv2.line(img,(x1,y),(x1-l,y),self.color,t)
		cv2.line(img,(x1,y),(x1,y+l),self.color,t)
		#bottom left
		cv2.line(img,(x,y1),(x+l,y1),self.color,t)
		cv2.line(img,(x,y1),(x,y1-l),self.color,t)
		#bottom right
		cv2.line(img,(x1,y1),(x1-l,y1),self.color,t)
		cv2.line(img,(x1,y1),(x1,y1-l),self.color,t)

		return img

		
def main():
	cap=cv2.VideoCapture('videos/3.mp4')

	pTime=0
	cTime=0

	detector=FaceDetector(0.75)
	while True:
		success, img=cap.read()
		if success:
			img, bboxes=detector.findFaces(img)
			print(bboxes)
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