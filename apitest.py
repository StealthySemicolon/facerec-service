import requests
import cv2
import base64

URL = "http://localhost:5000/detect"


def detect(img, conf_treshold=0.12):
    img_base64 = base64.b64encode(img).decode("ascii")
    img_shape = img.shape
    conf_treshold = 0.12

    post_data = {
        "img_base64": img_base64,
        "img_shape": img_shape,
        "conf_treshold": conf_treshold,
    }

    response = requests.post(URL, json=post_data)
    return response.json()


cap = cv2.VideoCapture("http://192.168.1.3:1024/video")
ret, img = cap.read()

while True:
    ret, img = cap.read()
    detections = detect(img, 0.20)

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
