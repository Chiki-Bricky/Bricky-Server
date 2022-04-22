import pandas as pd 
import torch
import numpy as np
import cv2
import base64

# decode received image from base64
def decodeImage(received):
    jpg_original = base64.b64decode(received)
    nparr = np.frombuffer(jpg_original, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img

