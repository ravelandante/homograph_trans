## Fynn Young
## Random Homography Transformations

import cv2
import os

import numpy as np
from numpy import genfromtxt
import random

SET_NAME = 'ADL-Rundle-6'
BASE_PATH = 'data/MOT15/train/' + SET_NAME

FRAMES = 2
SHIFT = 80
OUT_SIZE = 512


def randomChoice(factor):
    arr = []
    ran = np.random.randint(0, 2)
    if ran == 0:
        arr = [factor*random.uniform(-SHIFT, SHIFT), 0]
    elif ran == 1:
        arr = [0, factor*random.uniform(-SHIFT, SHIFT)]
    elif ran == 2:
        arr = [0, 0]
    return arr


# calc new coords for crop after rotation
def coordCalc(bounds, angle, center, width, height):
    angle = angle*np.pi/180
    crop = []
    center_x = center[0]
    center_y = height - center[1]
    for coord in bounds:
        x = coord[0] - center_x
        y = height - coord[1] - center_y
        # rotation calc
        coord[0] = center_x + y*np.sin(angle) + x*np.cos(angle)
        coord[1] = height - (center_y + y*np.cos(angle) - x*np.sin(angle))
        # check and correct if out of bounds
        """pad = [0, 0]
        if coord[0] > width:
            coord[0] -= coord[0] - width
            pad = [3, coord[0]]
        elif coord[0] < 0:
            coord[0] += -(coord[0])
            pad = [1, coord[0]]
        if coord[1] > height:
            coord[1] -= coord[1] - height
            pad = [3, coord[1]]
        elif coord[1] < 0:
            coord[1] += -(coord[1])
            pad = [2, coord[1]]"""
    crop.extend([int(min(bounds[0][0], bounds[2][0])), int(min(bounds[2][1], bounds[3][1])),
                int(max(bounds[1][0], bounds[3][0])), int(max(bounds[0][1], bounds[1][1]))])
    return crop


gt = genfromtxt(BASE_PATH + '/gt/gt.txt', delimiter=',')
gt = np.split(gt, np.where(np.diff(gt[:,0]))[0]+1)

for i, file in enumerate(os.listdir(BASE_PATH + '/img1')):
    imgPath = BASE_PATH + '/img1/{:06}.jpg'.format(i + 1)

    imgOrig = cv2.imread(imgPath)
    num_rows, num_cols = imgOrig.shape[:2]
    totalA = 0

    # get total area of all boxes for factor calc
    for row in gt[i]:
        x_min, y_min, x_max, y_max = row[2], row[3], row[2] + row[4], row[3] + row[5]
        totalA += (x_max-x_min)*(y_max-y_min)

    for row in gt[i]:
        objID, x_min, y_min, x_max, y_max = row[1], row[2], row[3], row[2] + row[4], row[3] + row[5] # coords of original image
        factor = (((x_max-x_min)*(y_max-y_min))/totalA)*1.5
        center = ((x_max+x_min)//2, (y_max+y_min)//2)
        angle = random.randint(-80, 80)

        # get randomised dest coords
        dstLL, dstLR = [x_min+randomChoice(factor)[0], y_max+randomChoice(factor)[1]], [x_max+randomChoice(factor)[0], y_max+randomChoice(factor)[1]]
        dstUL, dstUR = [x_min+randomChoice(factor)[0], y_min+randomChoice(factor)[1]], [x_max+randomChoice(factor)[0], y_min+randomChoice(factor)[1]]

        # projective transformation (warp)
        src_mat = np.float32([[x_min, y_max], [x_max, y_max], [x_min, y_min], [x_max, y_min]])
        dst_mat = np.float32([dstLL, dstLR, dstUL, dstUR])

        projective_matrix = cv2.getPerspectiveTransform(src_mat, dst_mat)
        img_protran = cv2.warpPerspective(imgOrig, projective_matrix, (num_cols,num_rows), borderMode = cv2.BORDER_REFLECT)

        # affine rotation transformation
        rot_mat = cv2.getRotationMatrix2D(center, angle, scale=1)
        img_protran = cv2.warpAffine(img_protran, rot_mat, (num_cols,num_rows), borderMode = cv2.BORDER_REFLECT)

        filepath_trans = 'out/trans{}_{}.jpg'.format(i, int(objID))

        aTrans = coordCalc([dstLL, dstLR, dstUL, dstUR], angle, center, num_cols, num_rows)

        width, height = aTrans[2] - aTrans[0], aTrans[3] - aTrans[1]
        padding = 0
        mu = [0, 0, 0, 0]

        if width < height:
            padding = int((height - width)/2)
            m = height - width - 2*padding  # makeup
            # if passing left with padding
            if aTrans[0] - padding <= 0:
                # if passing left without padding
                if aTrans[0] <= 0:
                    mu[0] = 0
                    mu[2] = 2*padding - aTrans[0]
                else:
                    mu[0] = -(aTrans[0])
                    mu[2] = 2*padding - mu[0]
            else:
                mu[0] = padding
            # if passing right with padding
            if aTrans[2] + padding >= num_cols:
                # if passing right without padding
                if aTrans[2] >= num_cols:
                    mu[2] = 0
                    mu[0] = 2*padding + aTrans[2] - num_cols
                else:
                    mu[2] = num_cols - aTrans[2]
                    mu[0] = 2*padding - mu[2]
            else:
                mu[2] = padding
            # if passing top without padding
            if aTrans[1] <= 0:
                mu[1] = 0
                mu[3] = -(aTrans[1])
            # if passing bottom without padding
            if aTrans[3] >= num_rows:
                mu[3] = 0
                mu[1] = aTrans[3] - num_rows
        elif height < width:
            padding = int((width - height)/2)
            m = width - height - 2*padding  # makeup
            # if passing top with padding
            if aTrans[1] - padding <= 0:
                # if passing top without padding
                if aTrans[1] <= 0:
                    mu[1] = 0
                    mu[3] = 2*padding - aTrans[1]
                else:
                    mu[1] = -(aTrans[1])
                    mu[3] = 2*padding - mu[1]
            else:
                mu[1] = padding
            # if passing bottom with padding
            if aTrans[3] + padding >= num_rows:
                # if passing bottom without padding
                if aTrans[3] >= num_rows:
                    mu[3] = 0
                    mu[1] = 2*padding + aTrans[3] - num_rows
                else:
                    mu[3] = num_rows - aTrans[3]
                    mu[1] = 2*padding - mu[3]
            else:
                mu[3] = padding
            # if passing left without padding
            if aTrans[0] <= 0:
                mu[0] = 0
                mu[2] = -(aTrans[0])
            # if passing right without padding
            if aTrans[2] >= num_cols:
                mu[2] = 0
                mu[0] = aTrans[2] - num_cols

        print(width, height, aTrans[0], aTrans[2], padding)
        print(mu)
        img_protran = img_protran[aTrans[1]-mu[1]:aTrans[3]+mu[3], aTrans[0]-mu[0]:aTrans[2]+mu[2]] # crop
        #img_protran = img_protran[aTrans[1]-padding:aTrans[3]+padding+m, aTrans[0]:aTrans[2]] if (width > height) else img_protran[aTrans[1]:aTrans[3], aTrans[0]-padding:aTrans[2]+padding+m]
        fx = (OUT_SIZE/width) if (width > height) else (OUT_SIZE/height)
        img_protran = cv2.resize(img_protran, (0, 0), fx=fx, fy=fx)

        cv2.imwrite(filepath_trans, img_protran)

        # inverse matrices (outdated PIL conversion and crop)
        """inv_trans_mat = cv2.getPerspectiveTransform(dst_mat, src_mat)
        inv_rot_mat = cv2.invertAffineTransform(rot_mat)

        img_protran = cv2.warpAffine(img_protran, inv_rot_mat, (num_cols,num_rows), borderMode = cv2.BORDER_REFLECT)
        img_protran = cv2.warpPerspective(img_protran, projective_matrix, (num_cols,num_rows), cv2.WARP_INVERSE_MAP)

        filepath_trans = 'out/invtrans{}_{}.jpg'.format(i, int(objID))
        img = Image.fromarray((img_protran).astype(np.uint8)) # convert to PIL image

        imgCropOrig = img.crop((x_min, y_min, x_max, y_max)).save(filepath_trans)"""
    
    if i == FRAMES - 1:
        break