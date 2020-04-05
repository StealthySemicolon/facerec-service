import cv2
from threading import Thread

class FaceDetector:
    def __init__(self, startImg):
        self.net = cv2.dnn.readNetFromCaffe("./deploy.prototxt.txt", "./res10_300x300_ssd_iter_140000.caffemodel")
        self.frame = startImg
        blob = cv2.dnn.blobFromImage(cv2.resize(self.frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
        self.net.setInput(blob)
        self.detections = self.net.forward()
    def start(self):
        Thread(target=self.run, args=()).start()
        return self
    def run(self):
        while True:
            blob = cv2.dnn.blobFromImage(cv2.resize(self.frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
            self.net.setInput(blob)
            self.detections = self.net.forward()

detector = FaceDetector()
cap = cv2.VideoCamera(0)

while True:
    ret, img = cap.read()
    