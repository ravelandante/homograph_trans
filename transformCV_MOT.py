## Fynn Young
## Random Homography Transformations

from PIL import Image
import cv2
import os

import numpy as np
from numpy import genfromtxt
import random

SET_NAME = 'ETH-Bahnhof'
BASE_PATH = 'data/MOT15/train/' + SET_NAME

FRAMES = 2
SHIFT = 100

def randomChoice(factor):
    arr = []
    prob = 1
    for i in range(4):
        ran = np.random.choice([0, 1, 2], p=[prob/3, prob/3, prob/3], size=1)
        if ran[0] == 0:
            arr.append([factor*random.uniform(-SHIFT, SHIFT), 0])
        elif ran[0] == 1:
            arr.append([0, factor*random.uniform(-SHIFT, SHIFT)])
        elif ran[0] == 2:
            arr.append([0, 0])
    return arr


gt = genfromtxt(BASE_PATH + '/gt/gt.txt', delimiter=',')
gt = np.split(gt, np.where(np.diff(gt[:,0]))[0]+1)

for i, file in enumerate(os.listdir(BASE_PATH + '/img1')):
    imgPath = BASE_PATH + '/img1/{:06}.jpg'.format(i + 1)

    imgOrig = cv2.cvtColor(cv2.imread(imgPath), cv2.COLOR_BGR2RGB)
    imgPIL = Image.open(imgPath)
    num_rows, num_cols = imgOrig.shape[:2]

    for row in gt[i]:
        x_min, y_min, x_max, y_max = row[2], row[3], row[2] + row[4], row[3] + row[5]


    for row in gt[i]:
        objID, x_min, y_min, x_max, y_max = row[1], row[2], row[3], row[2] + row[4], row[3] + row[5] # coords of original image
        factor = (((x_max-x_min)*(y_max-y_min))/(num_rows*num_cols))*20
        print(factor)
        # get randomised dest coords
        trans = randomChoice(factor)

        filepath_orig = 'out/norm{}_{}.jpg'.format(i, int(objID))
        imgPIL.crop((x_min, y_min, x_max, y_max)).save(filepath_orig) # crop, save original image

        # perform projective transformation
        src = np.float32([[x_min, y_min], [x_max, y_min], [x_min, y_max], [x_max, y_max]])
        dst = np.float32([[x_min+trans[0][0], y_min+trans[0][1]], [x_max+trans[1][0], y_min+trans[1][1]],
                        [x_min+trans[2][0], y_max+trans[2][1]], [x_max+trans[3][0], y_max+trans[3][1]]])

        projective_matrix = cv2.getPerspectiveTransform(src, dst)
        img_protran = cv2.warpPerspective(imgOrig, projective_matrix, (num_cols,num_rows))

        filepath_trans = 'out/trans{}_{}.jpg'.format(i, int(objID))
        img = Image.fromarray((img_protran).astype(np.uint8)) # convert to PIL image
        #print(dstLL[0], dstLL[1], dstUR[0], dstUR[1])
        img.crop((min(x_min+trans[0][0], x_min+trans[2][0]), min(y_min+trans[0][1], y_min+trans[1][1]),
                max(x_max+trans[1][0], x_max+trans[3][0]), max(y_max+trans[2][1], y_max+trans[3][1]))).save(filepath_trans)  # crop, save
    
    if i == FRAMES - 1:
        break