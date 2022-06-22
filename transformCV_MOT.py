## Fynn Young
## Random Homograph Image Transformations

import cv2
import os

import numpy as np
from numpy import genfromtxt

SET_NAME = 'ADL-Rundle-6'   # name of dataset
BASE_PATH = 'data/MOT15/train/' + SET_NAME
SAVE_PATH = 'load_dataset/MOT_data/train'

FRAMES = 1         # num of frames to process (-1 to process all)
IN_SIZE = (512, 512)
OUT_SIZE = (128, 256)
BOUNDING = True     # whether to draw bounding boxes

BORDER_MODE = cv2.BORDER_CONSTANT
BORDER_VALUE = (127, 127, 127)

np.random.seed(146)


def random_shift(points):
    width = points[1][0] - points[0][0]
    height = points[1][1] - points[2][1]
    x_shift = 0.23*width
    y_shift = 0.23*height
    orig_points = points
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
    return points, np.float32(np.array(points)-np.array(orig_points))


def calc_edges(corners, out_size):
    bounds = [min(corners[0][0], corners[1][0], corners[2][0], corners[3][0]), 
                min(corners[0][1], corners[1][1],corners[2][1], corners[3][1]),
                max(corners[0][0], corners[1][0], corners[2][0], corners[3][0]),
                max(corners[0][1], corners[1][1],corners[2][1], corners[3][1])]

    aspect = out_size[0]/out_size[1]    # aspect ratio of output image
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

for i, file in enumerate(os.listdir(BASE_PATH + '/img1')):
    img_path = BASE_PATH + '/img1/{:06}.jpg'.format(i + 1)
    img_orig = cv2.imread(img_path)

    print('FRAME', i + 1)

    height, width = img_orig.shape[:2]  # full image dimensions

    for row in gt[i]:
        obj_ID, x_min, y_min, x_max, y_max = row[1], row[2], row[3], row[2] + row[4], row[3] + row[5]   # coords of original image
        angle = np.random.randint(-80, 80)

        # translation to box_centre to prevent image corners going past image edges when rotating
        box_centre = ((x_max + x_min)//2, (y_max + y_min)//2)
        img_centre = (width//2, height//2)
        x_shift, y_shift = int(img_centre[0] - box_centre[0]), int(img_centre[1] - box_centre[1])

        trans_mat = np.float32([[1, 0, x_shift], [0, 1, y_shift]])
        img_persp = cv2.warpAffine(img_orig, trans_mat, (width, height), borderMode=BORDER_MODE, borderValue=BORDER_VALUE)

        x_min += x_shift    # realign bounding coords to new translation
        x_max += x_shift
        y_min += y_shift
        y_max += y_shift
        box_centre = ((x_max + x_min)//2, (y_max + y_min)//2)

        # get randomised dest coords
        corners, diff = random_shift([[x_min, y_max], [x_max, y_max], [x_min, y_min], [x_max, y_min]])
        corners = np.array(corners).astype(int)

        # perspective transformation (warp)
        src_mat = np.float32([[x_min, y_max], [x_max, y_max], [x_min, y_min], [x_max, y_min]])
        dst_mat = np.float32(corners)

        persp_mat = cv2.getPerspectiveTransform(src_mat, dst_mat)
        img_persp = cv2.warpPerspective(img_persp, persp_mat, (width,height), borderMode=BORDER_MODE, borderValue=BORDER_VALUE)

        # affine rotation transformation
        rot_mat = cv2.getRotationMatrix2D(box_centre, angle, scale=1)
        img_persp = cv2.warpAffine(img_persp, rot_mat, (width,height), borderMode=BORDER_MODE, borderValue=BORDER_VALUE)

        # get new corner coords after rotation
        for j, p in enumerate(corners):
            corners[j] = rot_mat.dot(np.array(tuple(corners[j]) + (1,)))[:2]

        crop = calc_edges(corners, IN_SIZE)  # calculate padding and bounds

        # draw bounding boxes
        if BOUNDING:
            points = np.array([corners[3], corners[1], corners[0], corners[2]])
            points = points.reshape((-1, 1, 2))
            cv2.polylines(img_persp, [points], True, (0, 255, 0), thickness=1)

        img_persp = img_persp[crop[1]:crop[3], crop[0]:crop[2]]   # crop using calculated bounds and padding

        n_width, n_height, dims = img_persp.shape              # dimensions of image before scaling
        fx, fy = IN_SIZE[0]/n_width, IN_SIZE[1]/n_height  # scaling factors

        img_persp = cv2.resize(img_persp, IN_SIZE, fx=fx, fy=fy)   # scale up to out_size

        filepath_trans = '{}/{:04}_{:04}.jpg'.format(SAVE_PATH, i + 1, int(obj_ID))
        cv2.imwrite(filepath_trans, img_persp)

        with open('load_dataset/MOT_data/MOT_labels.csv', 'a') as f:    # write filenames to csv file for custom dataset
            if i == 0 and obj_ID == 1:
                f.truncate(14)
            f.write('\n{:04}_{:04}.jpg,0'.format(i + 1, int(obj_ID)))

        # inverse matrices
        inv_rot_mat = cv2.getRotationMatrix2D((IN_SIZE[0]//2, IN_SIZE[0]//2), -angle, scale=1)

        print(corners)
        for j, corner in enumerate(corners):
            corners[j][0] = corners[j][0] - (crop[0])
            corners[j][1] = corners[j][1] - (crop[1])
            corners[j] = corners[j][0]*fx, corners[j][1]*fy
            corners[j] = inv_rot_mat.dot(np.array(tuple(corners[j]) + (1,)))[:2]
        print(corners)

        inv_src_mat = np.float32([[0, IN_SIZE[0]], [IN_SIZE[0], IN_SIZE[0]], [0, 0], [IN_SIZE[0], 0]])
        inv_dst_mat = np.float32([corners])

        inv_persp_mat = cv2.getPerspectiveTransform(inv_dst_mat, inv_src_mat)

        img_persp = cv2.warpAffine(img_persp, inv_rot_mat, (IN_SIZE[0],IN_SIZE[0]), borderMode=BORDER_MODE, borderValue=BORDER_VALUE)
        #img_persp = cv2.warpPerspective(img_persp, inv_persp_mat, (OUT_SIZE,OUT_SIZE))
        cv2.imshow('img', img_persp)
        cv2.waitKey(0)
        exit()

        #rot_row = np.array([0,0,1])
        #inv_rot_mat = np.vstack((inv_rot_mat, rot_row))
        #final_inv_mat = np.multiply(inv_trans_mat, inv_rot_mat)
        
    if i == FRAMES - 1:
        break
