import cv2
import matplotlib.pyplot as plt
import cvlib as cv
from cvlib.object_detection import draw_bbox
from datetime import datetime as dt 
import base64

def classify(context, event):
       im = cv2.imread('apple-256261_640.jpg')
       jpg_original = base64.b64decode(body)
       bbox, label, conf = cv.detect_common_objects(im)
       output_image = draw_bbox(im, bbox, label, conf)
       msg = {
           "dt": str(dt.now()),
           "labels":label,
           "img":output_image
       }
       return msg   
 
