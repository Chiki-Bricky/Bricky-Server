from flask import Flask, request, Response
from RequestUtils import decodeImage, convertDfToJson
from Detector import Detector
import numpy as np
import cv2
import os 

app = Flask(__name__)
pathToSave = "output.jpg"
pathToModel = os.path.join(os.getcwd(),os.path.join('models','train_val_2.pt'))
DEBUG = True

detector = Detector(pathToModel)

@app.route('/proccessImage', methods=['POST'])
def test():
    r = request.json
    if r is not None and 'image' in r.keys():
        img = decodeImage(r["image"])
        shape = img.shape
        df = detector(img)
        if DEBUG:
            img = detector.paintDetections(img, df)
            cv2.imwrite(pathToSave, img)
        
        print(img.shape)
        return convertDfToJson(df, shape)

    else:
        print("shit")
        return Response(status = 422)
    

# start flask app
if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)