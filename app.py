from flask import Flask, request, jsonify
import base64
import numpy as np
import cv2

app = Flask(__name__)
net = cv2.dnn.readNetFromCaffe("./deploy.prototxt", "ssd_model.caffemodel")


def detect_from_base64(img_base64, img_shape, conf_treshold):
    conf_treshold = conf_treshold if conf_treshold >= 0.05 else 0.05

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


@app.route("/")
def root():
    return """
    Welcome to Sagar's face detection API<br />
    In order to use the API, first encode your image into ASCII base64<br />
    Then, pass a json object with the the following parameters:
    <ul>
    <li>img_base64 - the image base64 string</li>
    <li>img_shape - the shape of the image</li>
    <li>conf_treshold - the confidence level(minimum is 0.05, and anything lower will be capped to this value</li>
    </ul>
    In return, you will get a 2D JSON array. Each element of the array will contain these 4 elements:
    <ul>
    <li>The x-coordinate of the bottom-left corner of the face detection</li>
    <li>The y-coordinate of the bottom-left corner of the face detection</li>
    <li>The x-coordinate of the top-right corner of the face detection</li>
    <li>The y-coordinate of the top-right corner of the face detection</li>
    </ul>
    Thus the shape of the resulting array will be (n, 4) where n is the number of faces detected.<br/>
    The restriction of the confidence level is to minimize the size of the response<br/>
    There is no deliberate restriction on the input image, though the hosting provider might have one.<br/>
    Either way, a larger image will take longer to upload and proccess.
    """


@app.route("/detect", methods=["POST"])
def detect():
    data = request.json
    detections = detect_from_base64(
        data["img_base64"], data["img_shape"], data["conf_treshold"]
    )
    return jsonify(np.array(detections).tolist())


if __name__ == "__main__":
    app.run(host="127.0.0.1", debug=True)
