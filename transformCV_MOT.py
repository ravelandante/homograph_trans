## Fynn Young
## Random Homography Transformations

from tkinter import Y
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

np.random.seed(155)


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

    img_orig = cv2.imread(imgPath)
    height, width = img_orig.shape[:2]
    totalA = 0

    # get total area of all boxes for factor calc
    for row in gt[i]:
        x_min, y_min, x_max, y_max = row[2], row[3], row[2] + row[4], row[3] + row[5]
        totalA += (x_max-x_min)*(y_max-y_min)

    for row in gt[i]:
        objID, x_min, y_min, x_max, y_max = row[1], row[2], row[3], row[2] + row[4], row[3] + row[5] # coords of original image
        factor = (((x_max - x_min)*(y_max - y_min))/totalA)*1.5 # factor for random shift calc
        angle = np.random.randint(-80, 80)

        # translation to centre
        centre = ((x_max + x_min)//2, (y_max + y_min)//2)
        t_centre = (width//2, height//2)
        x_shift = t_centre[0] - centre[0]
        y_shift = t_centre[1] - centre[1]

        trans_mat = np.float32([[1, 0, x_shift], [0, 1, y_shift]])
        img_persp = cv2.warpAffine(img_orig, trans_mat, (width, height), borderMode = cv2.BORDER_REFLECT)

        x_min += x_shift
        x_max += x_shift
        y_min += y_shift 
        y_max += y_shift
        centre = ((x_max + x_min)//2, (y_max + y_min)//2)

        """imgNew = img_orig[int(y_min):int(y_max), int(x_min):int(x_max)]
        cv2.imshow('image', imgNew)
        cv2.waitKey(0)"""

        # get randomised dest coords
        bot_left, bot_right = [x_min+randomShift(factor)[0], y_max+randomShift(factor)[1]], [x_max+randomShift(factor)[0], y_max+randomShift(factor)[1]]
        top_left, top_right = [x_min+randomShift(factor)[0], y_min+randomShift(factor)[1]], [x_max+randomShift(factor)[0], y_min+randomShift(factor)[1]]

        # perspective transformation (warp)
        src_mat = np.float32([[x_min, y_max], [x_max, y_max], [x_min, y_min], [x_max, y_min]])
        dst_mat = np.float32([bot_left, bot_right, top_left, top_right])

        persp_mat = cv2.getPerspectiveTransform(src_mat, dst_mat)
        img_persp = cv2.warpPerspective(img_persp, persp_mat, (width,height), borderMode = cv2.BORDER_REFLECT)

        # affine rotation transformation
        rot_mat = cv2.getRotationMatrix2D(centre, angle, scale=1)
        img_persp = cv2.warpAffine(img_persp, rot_mat, (width,height), borderMode = cv2.BORDER_REFLECT)

        # get new coords after perspective and rotation
        corners = cv2.perspectiveTransform(np.array([src_mat]), persp_mat)
        corners = corners[0].astype(int)

        for j, p in enumerate(corners):
            corners[j] = rot_mat.dot(np.array(tuple(corners[j]) + (1,)))[:2]
                
        bounds = [min(corners[0][0], corners[2][0]), min(corners[2][1], corners[3][1]),
                max(corners[1][0], corners[3][0]), max(corners[0][1], corners[1][1])]

        n_width, n_height = bounds[2] - bounds[0], bounds[3] - bounds[1]

        # line drawing
        cv2.line(img_persp, corners[3], corners[1], (0, 255, 0), thickness=2)
        cv2.line(img_persp, corners[2], corners[0], (0, 255, 0), thickness=2)
        cv2.line(img_persp, corners[2], corners[3], (0, 255, 0), thickness=2)
        cv2.line(img_persp, corners[1], corners[0], (0, 255, 0), thickness=2)

        padding = 0
        pad_lst = [0, 0, 0, 0]

        # NOTE: bounds[0] = x_min, bounds[1] = y_min, bounds[2] = x_max, bounds[3] = y_max

        if n_width < n_height:
            padding = int((n_height - n_width)/2)
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
            if bounds[2] + padding >= width:
                # if passing right without padding
                if bounds[2] >= width:
                    pad_lst[2] = 0
                    pad_lst[0] = 2*padding + bounds[2] - width
                else:
                    pad_lst[2] = width - bounds[2]
                    pad_lst[0] = 2*padding - pad_lst[2]
            elif bounds[0] - padding > 0:
                pad_lst[2] = padding
            # if passing top without padding
            if bounds[1] <= 0:
                pad_lst[1] = bounds[1]
                pad_lst[3] = -(bounds[1])
            # if passing bottom without padding
            elif bounds[3] >= height:
                pad_lst[3] = -(bounds[3] - height)
                pad_lst[1] = bounds[3] - height
        elif n_height < n_width:
            padding = int((n_width - n_height)/2)
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
            if bounds[3] + padding >= height:
                # if passing bottom without padding
                if bounds[3] >= height:
                    pad_lst[3] = 0
                    pad_lst[1] = 2*padding + bounds[3] - height
                else:
                    pad_lst[3] = height - bounds[3]
                    pad_lst[1] = 2*padding - pad_lst[3]
            elif bounds[1] - padding > 0:
                pad_lst[3] = padding
            # if passing left without padding
            if bounds[0] <= 0:
                pad_lst[0] = bounds[0]
                pad_lst[2] = -(bounds[0])
            # if passing right without padding
            elif bounds[2] >= width:
                pad_lst[2] = -(bounds[2] - width)
                pad_lst[0] = bounds[2] - width
        
        img_persp = img_persp[bounds[1]-pad_lst[1]:bounds[3]+pad_lst[3], bounds[0]-pad_lst[0]:bounds[2]+pad_lst[2]] # crop

        n_width, n_height, dims = img_persp.shape
        fx, fy = OUT_SIZE/n_width, OUT_SIZE/n_height
        img_persp = cv2.resize(img_persp, (OUT_SIZE, OUT_SIZE), fx=fx, fy=fy) # scale up to OUT_SIZE

        filepath_trans = 'out/trans{}_{}.jpg'.format(i + 1, int(objID))
        cv2.imwrite(filepath_trans, img_persp)

        # inverse matrices (outdated PIL conversion and crop)
        """inv_trans_mat = cv2.getPerspectiveTransform(dst_mat, src_mat)
        inv_rot_mat = cv2.invertAffineTransform(rot_mat)

        rot_row = np.array([0,0,1])
        inv_rot_mat = np.vstack((inv_rot_mat, rot_row))
        final_inv_mat = np.multiply(inv_trans_mat, inv_rot_mat)

        img_persp = cv2.warpAffine(img_persp, inv_rot_mat, (width,height), borderMode = cv2.BORDER_REFLECT)
        img_persp = cv2.warpPerspective(img_persp, persp_mat, (width,height), cv2.WARP_INVERSE_MAP)

        filepath_trans = 'out/invtrans{}_{}.jpg'.format(i, int(objID))
        img = Image.fromarray((img_persp).astype(np.uint8)) # convert to PIL image

        imgCropOrig = img.crop((x_min, y_min, x_max, y_max)).save(filepath_trans)"""
    if i == FRAMES - 1:
        break