import cv2
import numpy as np
from threading import Thread
import base64

net = cv2.dnn.readNetFromCaffe("./deploy.prototxt", "ssd_model.caffemodel")

cap = cv2.VideoCapture("http://192.168.1.3:1024/video")


def detect(img_base64, img_shape, conf_treshold=0.12):
    img = np.frombuffer(base64.b64decode(img_base64), dtype=np.uint8)
    img = img.reshape(img_shape)

    blob = cv2.dnn.blobFromImage(
        cv2.resize(img, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0)
    )

    (h, w) = img_shape[:2]

    net.setInput(blob)
    detections = net.forward()
    output = []

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence < conf_treshold:
            continue
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        output.append(box.astype("int"))

    return output


while True:
    ret, img = cap.read()

    img_shape = img.shape
    img_base64 = base64.b64encode(img).decode("ASCII")
    detections = detect(img_base64, img_shape, 0.20)

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
