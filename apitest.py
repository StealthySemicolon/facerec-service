import requests
import cv2
import base64
from threading import Thread

URL = "http://localhost:5000/detect"


def detect(img, conf_treshold=0.12):
    img_base64 = base64.b64encode(img).decode("ascii")
    img_shape = img.shape

    post_data = {
        "img_base64": img_base64,
        "img_shape": img_shape,
        "conf_treshold": conf_treshold,
    }
    response = requests.post(URL, json=post_data)
    return response.json()


class ThreadedCapture:
    def __init__(self, camera):
        self.cap = cv2.VideoCapture(camera)
        self.ret, self.frame = self.cap.read()
        self.stopped = False

    def start(self):
        self.thread = Thread(target=self.capture, args=())
        self.thread.start()
        return self

    def capture(self):
        while True:
            if self.stopped:
                return
            self.ret, self.frame = self.cap.read()

    def read(self):
        return self.ret, self.frame.copy()

    def stop(self):
        self.stopped = True


cap = ThreadedCapture(0).start()
ret, img = cap.read()

while True:
    ret, img = cap.read()
    detections = detect(img, 0.50)

    faces = []

    for box in detections:
        (startX, startY, endX, endY) = box
        faces.append(img[startY:endY, startX:endX])

    for i, face in enumerate(faces):
        (startX, startY, endX, endY) = detections[i]
        cv2.rectangle(img, (startX, startY), (endX, endY), (0, 255, 0), 2)

    cv2.imshow("img", img)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.stop()
