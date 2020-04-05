from flask import Flask
import cv2

app = Flask(__name__)

@app.route('/')
def root():
    return "Hello World!"

if __name__ == '__main__':
    app.run(debug=True)