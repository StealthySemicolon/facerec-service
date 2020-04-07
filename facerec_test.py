import cv2
import numpy as np
from threading import Thread

net = cv2.dnn.readNetFromCaffe("./deploy.prototxt", "ssd_model.caffemodel")


cap = cv2.VideoCapture("http://192.168.1.3:1024/video")


def detect(img, conf_treshold=0.12):
    blob = cv2.dnn.blobFromImage(
        cv2.resize(img, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0)
    )

    (h, w) = img.shape[:2]

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

    detections = detect(img)
    faces = []

    for box in detections:
        (startX, startY, endX, endY) = box
        faces.append(img[startY:endY, startX:endX])

    for i, face in enumerate(faces):
        try:
            cv2.imshow("face{}".format(i), face)
        except:
            pass

        (startX, startY, endX, endY) = detections[i]
        cv2.rectangle(img, (startX, startY), (endX, endY), (0, 255, 0), 2)

    cv2.imshow("img", img)

    if cv2.waitKey(1) & 0xFF == 27:
        break
