import pandas as pd 
import torch
import numpy as np
import cv2

IMG_SIZE = 640
class Detector():

    def __init__(self, pathToModel):
        self.model = torch.hub.load('ultralytics/yolov5', "custom", path = pathToModel, device="cpu")
        self.model.eval()
    
    def recogize(self, img):
        img = cv2.resize(img, (IMG_SIZE,IMG_SIZE))
        with torch.no_grad():
            preds = self.model(img)
        return preds
    
    def processPreds(self, preds):
        df = preds.pandas().xyxy[0]
        df = df[df.confidence>0.1] 
        # df = df.drop(columns =["confidence","name","class"])
        return df
    
    def __call__(self, img):
        return self.processPreds(self.recogize(img))
    
    def paintDetections(self, img, df):
        coords = df.to_numpy()
        img = cv2.resize(img, (IMG_SIZE,IMG_SIZE))
        a,_, = coords.shape
        for i in range(a):
            color = (0,0,255)
            cv2.rectangle(img, (np.int32(coords[i][0]),np.int32(coords[i][1])),(np.int32(coords[i][2]),np.int32(coords[i][3])), color = color)
            cv2.circle(img, (np.int32(coords[i][0]),np.int32(coords[i][1])),radius=4, color=color, thickness=-1)
        return img 