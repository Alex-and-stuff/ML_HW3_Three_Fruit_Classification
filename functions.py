import cv2
import numpy as np

SAMPLES = 490
CLASS   = 3
DIM     = 32
DEPTH   = 3

def ImportPic(name, data_type = 'train'):
    data = np.zeros((SAMPLES,DIM,DIM,DEPTH), dtype=np.uint8)
    path = f"Data/Data_{data_type}/{name}/{name}_{data_type}_"
    for idx in range(490):
        im = cv2.imread(path+str(idx)+'.png')
        data[idx] = im       
    return data