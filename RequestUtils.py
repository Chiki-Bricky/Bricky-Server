import pandas as pd 
import torch
import numpy as np
import cv2
import base64
from flask import jsonify
# decode received image from base64
def decodeImage(received):
    jpg_original = base64.b64decode(received)
    nparr = np.frombuffer(jpg_original, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img


#TODO as we dont yet have classes they will be kinda random
def convertDfToJson(df):
    res = []
    # we can allow iterrows for now but it's really slow though 
    for i, row in df.iterrows():
        tmp = {}
        tmp["xmin"] = int(row["xmin"])
        tmp["ymin"] = int(row["ymin"])
        tmp["xmax"] = int(row["xmax"])
        tmp["ymax"] = int(row["ymax"])
        tmp['class'] = i
        res.append(tmp)       
    return jsonify({"bricks": res })