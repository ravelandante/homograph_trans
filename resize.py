from PIL import Image
import cv2
import os
import ntpath

import numpy as np

for file in os.listdir('out/'):
    filepath = os.path.abspath(file)
    imgOrig = cv2.cvtColor(cv2.imread('out/' + ntpath.basename(filepath)), cv2.COLOR_BGR2RGB)
    num_rows, num_cols = imgOrig.shape[:2]