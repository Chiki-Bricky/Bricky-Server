from flask import Flask, request, Response

import numpy as np
import cv2
import base64

app = Flask(__name__)
pathToSave = "C:\\Users\\vvpvo\\Desktop\\nsu\\Bricky\\server\\output.jpg"

@app.route('/proccessImage', methods=['POST'])
def test():
    r = request.json
    if r is not None:
        r = r["image"]
        jpg_original = base64.b64decode(r)
        nparr = np.frombuffer(jpg_original, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        cv2.imwrite(pathToSave, img)
        return Response(status = 200)

    else:
        print("shit")
        return Response(status = 422)
    

# start flask app
app.run(host="127.0.0.1", port=5000)