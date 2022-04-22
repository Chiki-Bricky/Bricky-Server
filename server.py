from importlib.resources import path
from charset_normalizer import detect
from flask import Flask, request, Response
from RequestUtils import decodeImage
from Detector import Detector
import numpy as np
import cv2
import os 

app = Flask(__name__)
pathToSave = "C:\\Users\\vvpvo\\Desktop\\nsu\\Bricky\\server\\output.jpg"
pathToModel = os.path.join(os.getcwd(),os.path.join('models','best.pt'))
DEBUG = True

detector = Detector(pathToModel)

@app.route('/proccessImage', methods=['POST'])
def test():
    r = request.json
    if r is not None and 'image' in r.keys():
        img = decodeImage(r["image"])
        df = detector(img)
        if DEBUG:
            img = detector.paintDetections(img, df)
            cv2.imwrite(pathToSave, img)
        
        return Response(status = 200)

    else:
        print("shit")
        return Response(status = 422)
    

# start flask app
app.run(host="127.0.0.1", port=5000)