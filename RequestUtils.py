import pandas as pd 
import torch
import numpy as np
import cv2
import base64
from flask import jsonify
from Detector import IMG_SIZE
# decode received image from base64
def decodeImage(received):
    jpg_original = base64.b64decode(received)
    nparr = np.frombuffer(jpg_original, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img


#TODO as we dont yet have classes they will be kinda random
def convertDfToJson(df, shape):
    res = []
    # we can allow iterrows for now but it's really slow though 
    for _, row in df.iterrows():
        tmp = {}
        tmp["xmin"] = round(float(row["xmin"]) / IMG_SIZE, 3)
        tmp["ymin"] = round(float(row["ymin"]) / IMG_SIZE, 3)
        tmp["xmax"] = round(float(row["xmax"]) / IMG_SIZE, 3)
        tmp["ymax"] = round(float(row["ymax"]) / IMG_SIZE, 3)
        tmp['class'] = row["name"]
        tmp['confidence'] = round(row["confidence"], 3)
        res.append(tmp)       
    return jsonify({"bricks": res })