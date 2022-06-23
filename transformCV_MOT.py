"""
* Applying random homograph transforms (affine rotation, perspective transform) to images from the MOT15 dataset
* Padding, cropping, resizing transformed image to IN_SIZE for neural network input
* Calculating inverse transform matrix and resizing output image to OUT_SIZE

Author: Fynn Young
        23/06/2022
"""

import cv2
import os

import numpy as np
from numpy import genfromtxt

SET_NAME = 'ADL-Rundle-6'   # name of dataset
BASE_PATH = 'data/MOT15/train/' + SET_NAME
SAVE_PATH = 'load_dataset/MOT_data/train'

DRAW_BOXES = True       # whether to draw bounding boxes
DISPLAY = True          # whether to display images at the end of each loop

FRAMES = 1              # num of frames to process (-1 to process all)
IN_SIZE = (512, 512)
OUT_SIZE = (128, 256)
BORDER_MODE = cv2.BORDER_CONSTANT
BORDER_VALUE = (127, 127, 127)

#np.random.seed(146)

def random_shift(points):
    """Calculates random shift for 4 points of bounding box
    Args:
        points (list of lists): 4 corners of original bounding box
    Returns:
        list of lists: 4 corners of new bounding box
    """
    points = np.array(points)
    width = points[1][0] - points[0][0]
    height = points[1][1] - points[2][1]
    orig_points = np.array(points)
    x_shift = 0.23*width
    y_shift = 0.23*height

    # bot_left
    points[0][0] += np.random.randint(0, x_shift)
    points[0][1] += np.random.randint(-y_shift/2, y_shift/2)
    # bot_right
    points[1][0] += np.random.randint(-x_shift, 0)
    points[1][1] += np.random.randint(-y_shift/2, y_shift/2)
    # top_left
    points[2][0] += np.random.randint(-x_shift, 0)
    points[2][1] += np.random.randint(-y_shift, 0)
    # top_right
    points[3][0] += np.random.randint(0, x_shift)
    points[3][1] += np.random.randint(-y_shift, 0)

    return points, points - orig_points

def calc_edges(corners, out_size=IN_SIZE):
    """Calculates parameters (bounds, padding) for cropping and resizing the image to the out_size
    Args:
        corners (list of lists): 4 corners of bounding box
        out_size (tuple of ints): (w, h) of output image
    Returns:
        list: 4 bounding coordinates for crop (x_min, y_min, x_max, y_max)
    """
    bounds = [np.min(corners, axis=0)[0], np.min(corners, axis=0)[1],   # find min/max bounds (x_min, y_min, x_max, y_max)
                np.max(corners, axis=0)[0], np.max(corners, axis=0)[1]]

    aspect = out_size[0]/out_size[1]                                    # aspect ratio of output image
    n_width, n_height = bounds[2] - bounds[0], bounds[3] - bounds[1]
    padding = 0
    pad_lst = [0, 0, 0, 0]

    # NOTE: bounds[0] = x_min, bounds[1] = y_min, bounds[2] = x_max, bounds[3] = y_max

    # padding of arbitrarily sized image to square for scaling to out_size
    if n_width > n_height:
        padding = int((aspect*n_width - n_height)/2)
        pad_lst = [0, padding, 0, padding]
    elif n_height > n_width:
        padding = int((aspect*n_height - n_width)/2)
        pad_lst = [padding, 0, padding, 0]
    
    return [bounds[0] - pad_lst[0], bounds[1] - pad_lst[1], bounds[2] + pad_lst[2], bounds[3] + pad_lst[3]]


gt = genfromtxt(BASE_PATH + '/gt/gt.txt', delimiter=',')    # open/organise ground-truth tracking data
gt = np.split(gt, np.where(np.diff(gt[:,0]))[0]+1)

for i, _ in enumerate(os.listdir(BASE_PATH + '/img1')):
    img_path = BASE_PATH + '/img1/{:06}.jpg'.format(i + 1)
    img_orig = cv2.imread(img_path)

    print('FRAME', i + 1)

    height, width = img_orig.shape[:2]  # full image dimensions

    for row in gt[i]:
        obj_ID, x_min, y_min, x_max, y_max = row[1], row[2], row[3], row[2] + row[4], row[3] + row[5]   # coords of original image
        angle = np.random.randint(-80, 80)

        img_new = img_orig[int(y_min):int(y_max), int(x_min):int(x_max)]

        # translation to box_centre to prevent image corners going past image edges when rotating
        box_centre = ((x_max + x_min)//2, (y_max + y_min)//2)
        img_centre = (width//2, height//2)
        x_shift, y_shift = int(img_centre[0] - box_centre[0]), int(img_centre[1] - box_centre[1])

        trans_mat = np.float32([[1, 0, x_shift], [0, 1, y_shift]])
        img_persp = cv2.warpAffine(img_orig, trans_mat, (width, height), borderMode=BORDER_MODE, borderValue=BORDER_VALUE)

        x_min += x_shift    # realign bounding box coords to new translation
        x_max += x_shift
        y_min += y_shift
        y_max += y_shift
        box_centre = ((x_max + x_min)//2, (y_max + y_min)//2)

        # get randomised dest coords
        corners, diff = random_shift([[x_min, y_max], [x_max, y_max], [x_min, y_min], [x_max, y_min]])
        corners = corners.astype(int)

        # perspective transformation (warp)
        src_mat = np.float32([[x_min, y_max], [x_max, y_max], [x_min, y_min], [x_max, y_min]])
        dst_mat = np.float32(corners)

        persp_mat = cv2.getPerspectiveTransform(src_mat, dst_mat)
        img_persp = cv2.warpPerspective(img_persp, persp_mat, (width,height), borderMode=BORDER_MODE, borderValue=BORDER_VALUE)

        # affine rotation transformation
        rot_mat = cv2.getRotationMatrix2D(box_centre, angle, scale=1)
        img_persp = cv2.warpAffine(img_persp, rot_mat, (width,height), borderMode=BORDER_MODE, borderValue=BORDER_VALUE)

        # get new corner coords after rotation
        for j, _ in enumerate(corners):
            corners[j] = rot_mat.dot(np.array(tuple(corners[j]) + (1,)))[:2]

        crop = calc_edges(corners)  # calculate padding and bounds

        # draw bounding boxes
        if DRAW_BOXES:
            points = np.array([corners[3], corners[1], corners[0], corners[2]])
            points = points.reshape((-1, 1, 2))
            cv2.polylines(img_persp, [points], True, (0, 255, 0), thickness=1)

        img_persp = img_persp[crop[1]:crop[3], crop[0]:crop[2]]         # crop using calculated bounds and padding

        n_width, n_height, dims = img_persp.shape                       # dimensions of image before scaling
        fx, fy = IN_SIZE[0]/n_width, IN_SIZE[1]/n_height                # scaling factors

        img_persp = cv2.resize(img_persp, IN_SIZE, fx=fx, fy=fy)        # scale up to out_size

        filepath_trans = '{}/{:04}_{:04}.jpg'.format(SAVE_PATH, i + 1, int(obj_ID))
        cv2.imwrite(filepath_trans, img_persp)

        with open('load_dataset/MOT_data/MOT_labels.csv', 'a') as f:    # write filenames to csv file for custom dataset
            if i == 0 and obj_ID == 1:
                f.truncate(14)
            f.write('\n{:04}_{:04}.jpg,0'.format(i + 1, int(obj_ID)))

        # inverse matrices
        inv_rot_mat = cv2.getRotationMatrix2D((IN_SIZE[0]//2, IN_SIZE[0]//2), -angle, scale=1)
        src_mat = dst_mat - diff
        for j, _ in enumerate(corners):
            corners[j][0] -= crop[0]                                                # find new coords of warped corners
            corners[j][1] -= crop[1]
            corners[j] = corners[j][0]*fx, corners[j][1]*fy                         # rescale coords to IN_SIZE

            diff[j] = diff[j][0]*fx, diff[j][1]*fy                                  # rescale difference between src_mat, dst_mat to IN_SIZE
            corners[j] = inv_rot_mat.dot(np.array(tuple(corners[j]) + (1,)))[:2]    # find new coords after rotation back

            src_mat[j][0] = corners[j][0] - diff[j][0]                              # find original bounding box coords in new frame using difference
            src_mat[j][1] = corners[j][1] - diff[j][1]

        inv_dst_mat = np.float32([corners])
        inv_src_mat = np.float32([src_mat])
        inv_persp_mat = cv2.getPerspectiveTransform(inv_dst_mat, inv_src_mat)

        rot_row = np.array([0,0,1])
        inv_rot_mat = np.vstack((inv_rot_mat, rot_row))                             # add 3rd row to inv_rot_mat to be equal in shape to inv_persp_mat
        final_inv_mat = np.dot(inv_persp_mat, inv_rot_mat)                          # dot product to get final inverse matrix

        img_rev = cv2.warpPerspective(img_persp, final_inv_mat, IN_SIZE, borderMode=BORDER_MODE, borderValue=BORDER_VALUE)

        crop = calc_edges(corners, OUT_SIZE)                    # calculate padding and bounds
        img_rev = img_rev[crop[1]:crop[3], crop[0]:crop[2]]     # crop using calculated bounds and padding

        n_width, n_height, dims = img_persp.shape               # dimensions of image before scaling
        fx, fy = OUT_SIZE[0]/n_width, OUT_SIZE[1]/n_height      # scaling factors

        img_rev = cv2.resize(img_rev, OUT_SIZE, fx=fx, fy=fy)

        if DISPLAY:                                                                  
            cv2.imshow('original', img_new)
            cv2.imshow('warped', img_persp)
            cv2.imshow('reversed', img_rev)
            cv2.waitKey(0)

    if i == FRAMES - 1:
        break
