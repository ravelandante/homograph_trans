## Fynn Young
## Random Homography Transformations

import cv2
import os

import numpy as np
from numpy import genfromtxt
import random

SET_NAME = 'ADL-Rundle-6'
BASE_PATH = 'data/MOT15/train/' + SET_NAME

FRAMES = 1
SHIFT = 80
OUT_SIZE = 512

np.random.seed(146)


def randomShift(factor):
    arr = []
    ran = np.random.randint(0, 1)
    if ran == 0:
        arr = [factor*np.random.uniform(-SHIFT, SHIFT), 0]
    elif ran == 1:
        arr = [0, factor*np.random.uniform(-SHIFT, SHIFT)]
    return arr

gt = genfromtxt(BASE_PATH + '/gt/gt.txt', delimiter=',')
gt = np.split(gt, np.where(np.diff(gt[:,0]))[0]+1)

for i, file in enumerate(os.listdir(BASE_PATH + '/img1')):
    print('FRAME', i + 1)
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
        factor = (((x_max - x_min)*(y_max - y_min))/totalA)*1.5 # factor for random shift calc
        center = ((x_max + x_min)//2, (y_max + y_min)//2)
        angle = np.random.randint(-80, 80)

        """imgNew = imgOrig[int(y_min):int(y_max), int(x_min):int(x_max)]
        cv2.imshow('image', imgNew)
        cv2.waitKey(0)"""

        # get randomised dest coords
        bot_left, bot_right = [x_min+randomShift(factor)[0], y_max+randomShift(factor)[1]], [x_max+randomShift(factor)[0], y_max+randomShift(factor)[1]]
        top_left, top_right = [x_min+randomShift(factor)[0], y_min+randomShift(factor)[1]], [x_max+randomShift(factor)[0], y_min+randomShift(factor)[1]]

        # projective transformation (warp)
        src_mat = np.float32([[x_min, y_max], [x_max, y_max], [x_min, y_min], [x_max, y_min]])
        dst_mat = np.float32([bot_left, bot_right, top_left, top_right])

        proj_mat = cv2.getPerspectiveTransform(src_mat, dst_mat)
        img_protran = cv2.warpPerspective(imgOrig, proj_mat, (num_cols,num_rows), borderMode = cv2.BORDER_REFLECT)

        # get new coords after perspective and rotation
        bounds = cv2.perspectiveTransform(np.array([src_mat]), proj_mat)
        bounds = bounds[0].astype(int)

        # affine rotation transformation
        rot_mat = cv2.getRotationMatrix2D(center, angle, scale=1)
        while(True):
            print('top')
            for j, p in enumerate(bounds):
                print('old', bounds[j])
                while(True):
                    flag = True
                    bounds[j] = rot_mat.dot(np.array(tuple(bounds[j]) + (1,)))[:2]
                    print('new', bounds[j])
                    if bounds[j][0] > num_cols:
                        rot_mat[0][2] -= 20
                        print('here')
                        flag = False
                    elif bounds[j][0] < 0:
                        rot_mat[0][2] += 20
                        flag = False
                    elif bounds[j][1] > num_rows:
                        rot_mat[1][2] -= 20
                        flag = False
                    elif bounds[j][1] < 0:
                        rot_mat[1][2] += 20
                        flag = False
                    if flag == True:
                        break
            print('perform')
            for j, p in enumerate(bounds):
                bounds[j] = rot_mat.dot(np.array(tuple(bounds[j]) + (1,)))[:2]
            print(bounds)
            img_protran = cv2.warpAffine(img_protran, rot_mat, (num_cols,num_rows), borderMode = cv2.BORDER_REFLECT)
            break

        # line drawing
        cv2.line(img_protran, bounds[3], bounds[1], (0, 255, 0), thickness=2)
        cv2.line(img_protran, bounds[2], bounds[0], (0, 255, 0), thickness=2)
        cv2.line(img_protran, bounds[2], bounds[3], (0, 255, 0), thickness=2)
        cv2.line(img_protran, bounds[1], bounds[0], (0, 255, 0), thickness=2)

        bounds = [min(bounds[0][0], bounds[2][0]), min(bounds[2][1], bounds[3][1]),
                max(bounds[1][0], bounds[3][0]), max(bounds[0][1], bounds[1][1])]

        width, height = bounds[2] - bounds[0], bounds[3] - bounds[1]
        padding = 0
        pad_lst = [0, 0, 0, 0]

        if width < height:
            padding = int((height - width)/2)
            # if passing left with padding
            if bounds[0] - padding <= 0:
                # if passing left without padding
                if bounds[0] <= 0:
                    pad_lst[0] = bounds[0]
                    pad_lst[2] = 2*padding - bounds[0]
                else:
                    pad_lst[0] = -(bounds[0])
                    pad_lst[2] = 2*padding - pad_lst[0]
            else:
                pad_lst[0] = padding
            # if passing right with padding
            if bounds[2] + padding >= num_cols:
                # if passing right without padding
                if bounds[2] >= num_cols:
                    pad_lst[2] = 0
                    pad_lst[0] = 2*padding + bounds[2] - num_cols
                else:
                    pad_lst[2] = num_cols - bounds[2]
                    pad_lst[0] = 2*padding - pad_lst[2]
            elif bounds[0] - padding > 0:
                pad_lst[2] = padding
            # if passing top without padding
            if bounds[1] <= 0:
                pad_lst[1] = bounds[1]
                pad_lst[3] = -(bounds[1])
            # if passing bottom without padding
            elif bounds[3] >= num_rows:
                pad_lst[3] = -(bounds[3] - num_rows)
                pad_lst[1] = bounds[3] - num_rows
        elif height < width:
            padding = int((width - height)/2)
            # if passing top with padding
            if bounds[1] - padding <= 0:
                # if passing top without padding
                if bounds[1] <= 0:
                    pad_lst[1] = bounds[1]
                    pad_lst[3] = 2*padding - bounds[1]
                else:
                    pad_lst[1] = -(bounds[1])
                    pad_lst[3] = 2*padding - pad_lst[1]
            else:
                pad_lst[1] = padding
            # if passing bottom with padding
            if bounds[3] + padding >= num_rows:
                # if passing bottom without padding
                if bounds[3] >= num_rows:
                    pad_lst[3] = 0
                    pad_lst[1] = 2*padding + bounds[3] - num_rows
                else:
                    pad_lst[3] = num_rows - bounds[3]
                    pad_lst[1] = 2*padding - pad_lst[3]
            elif bounds[1] - padding > 0:
                pad_lst[3] = padding
            # if passing left without padding
            if bounds[0] <= 0:
                pad_lst[0] = bounds[0]
                pad_lst[2] = -(bounds[0])
            # if passing right without padding
            elif bounds[2] >= num_cols:
                pad_lst[2] = -(bounds[2] - num_cols)
                pad_lst[0] = bounds[2] - num_cols
        
        img_protran = img_protran[bounds[1]-pad_lst[1]:bounds[3]+pad_lst[3], bounds[0]-pad_lst[0]:bounds[2]+pad_lst[2]] # crop

        width, height, dims = img_protran.shape
        fx, fy = OUT_SIZE/width, OUT_SIZE/height
        img_protran = cv2.resize(img_protran, (OUT_SIZE, OUT_SIZE), fx=fx, fy=fy) # scale up to OUT_SIZE

        filepath_trans = 'out/trans{}_{}.jpg'.format(i + 1, int(objID))
        cv2.imwrite(filepath_trans, img_protran)

        # inverse matrices (outdated PIL conversion and crop)
        """inv_trans_mat = cv2.getPerspectiveTransform(dst_mat, src_mat)
        inv_rot_mat = cv2.invertAffineTransform(rot_mat)

        rot_row = np.array([0,0,1])
        inv_rot_mat = np.vstack((inv_rot_mat, rot_row))
        final_inv_mat = np.multiply(inv_trans_mat, inv_rot_mat)

        img_protran = cv2.warpAffine(img_protran, inv_rot_mat, (num_cols,num_rows), borderMode = cv2.BORDER_REFLECT)
        img_protran = cv2.warpPerspective(img_protran, proj_mat, (num_cols,num_rows), cv2.WARP_INVERSE_MAP)

        filepath_trans = 'out/invtrans{}_{}.jpg'.format(i, int(objID))
        img = Image.fromarray((img_protran).astype(np.uint8)) # convert to PIL image

        imgCropOrig = img.crop((x_min, y_min, x_max, y_max)).save(filepath_trans)"""
        if objID == 1:
            break
    if i == FRAMES - 1:
        break