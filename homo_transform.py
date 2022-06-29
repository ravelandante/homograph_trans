import cv2
import numpy as np


DRAW_BOXES = False                           # whether to draw bounding boxes
SHOW_INVERSE = False                         # whether to display images at the end of each loop

IN_SIZE = (256, 256)
OUT_SIZE = (128, 256)
BORDER_MODE = cv2.BORDER_CONSTANT
BORDER_VALUE = (127, 127, 127)


def random_shift(points):
    """Calculates random shift for 4 points of bounding box
    Args:
        points (list of lists): 4 corners of original bounding box
    Returns:
        2D numpy array: 4 corners of new bounding box
        2D numpy array: x and y differences between source and dest points
    """
    points = np.array(points)
    orig_points = np.array(points)
    width, height = points[1][0] - points[0][0], points[1][1] - points[2][1]
    x_shift, y_shift = 0.23*width, 0.23*height

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

    return points, (points - orig_points)


def centre_shift(bounds, width, height):
    """Calculates transform matrix to translate bounding box to image center
    Args:
        bounds (list of floats): x_min, y_min, x_max, y_max of bounding box
        width (int): width of whole image
        height(int): height of whole image
    Returns:
        2D numpy array: transform matrix
        list: new bounds of translated box
    """
    x_min, y_min, x_max, y_max = bounds
    box_centre = ((x_max + x_min)//2, (y_max + y_min)//2)
    img_centre = (width//2, height//2)

    x_shift, y_shift = int(img_centre[0] - box_centre[0]), int(img_centre[1] - box_centre[1])

    x_min += x_shift    # realign bounding box coords to new translation
    x_max += x_shift
    y_min += y_shift
    y_max += y_shift

    return np.float32([[1, 0, x_shift], [0, 1, y_shift]]), [int(c) for c in [x_min, y_min, x_max, y_max]]


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
    pad_lst = [0]*4

    # NOTE: bounds[0] = x_min, bounds[1] = y_min, bounds[2] = x_max, bounds[3] = y_max

    # padding of arbitrarily sized image to square for scaling to out_size
    if n_width > n_height:
        padding = int((aspect*n_width - n_height)/2)
        pad_lst = [0, padding, 0, padding]
    elif n_height > n_width:
        padding = int((aspect*n_height - n_width)/2)
        pad_lst = [padding, 0, padding, 0]
    return [int(c) for c in [bounds[0] - pad_lst[0], bounds[1] - pad_lst[1], bounds[2] + pad_lst[2], bounds[3] + pad_lst[3]]]


def img_transform(img_orig, row, width, height):
    bounds = []
    # NOTE: bounds[0] = x_min, bounds[1] = y_min, bounds[2] = x_max, bounds[3] = y_max
    obj_ID, bounds = row[1], [row[2], row[3], row[2] + row[4], row[3] + row[5]]             # coords of original image
    angle = np.random.randint(-80, 80)

    # translation of bounding box to image centre to prevent rotated image corners getting cut off by image bounds
    trans_mat, bounds = centre_shift(bounds, width, height)
    img_trans = cv2.warpAffine(img_orig, trans_mat, (width, height), borderMode=BORDER_MODE, borderValue=BORDER_VALUE)

    # get randomised dest coords
    corners, diff = random_shift([[bounds[0], bounds[3]], [bounds[2], bounds[3]], [bounds[0], bounds[1]], [bounds[2], bounds[1]]])
    corners = corners.astype(int)

    # perspective transform (warp)
    src_mat = np.float32([[bounds[0], bounds[3]], [bounds[2], bounds[3]], [bounds[0], bounds[1]], [bounds[2], bounds[1]]])
    dst_mat = np.float32(corners)

    persp_mat = cv2.getPerspectiveTransform(src_mat, dst_mat)
    img_persp = cv2.warpPerspective(img_trans, persp_mat, (width,height), borderMode=BORDER_MODE, borderValue=BORDER_VALUE)

    # affine rotation transform
    rot_mat = cv2.getRotationMatrix2D(((bounds[2] + bounds[0])//2, (bounds[3] + bounds[1])//2), angle, scale=1)
    img_persp = cv2.warpAffine(img_persp, rot_mat, (width,height), borderMode=BORDER_MODE, borderValue=BORDER_VALUE)

    # get new corner coords after rotation
    for j, _ in enumerate(corners):
        corners[j] = rot_mat.dot(np.array(tuple(corners[j]) + (1,)))[:2]

    crop = calc_edges(corners)  # calculate padding and bounds
    # pad image with constant BORDER_VALUE if not enough space around bounding box to pad normally
    border = [0]*4
    for k in range(len(crop)):
        if crop[k] < 0:
            border[k] = -crop[k]
            crop[k] = 0
    img_persp = cv2.copyMakeBorder(img_persp, border[1], border[3], border[0], border[2], borderType=BORDER_MODE, value=BORDER_VALUE)

    # draw bounding boxes
    if DRAW_BOXES:
        points = np.array([corners[3], corners[1], corners[0], corners[2]])
        points = points.reshape((-1, 1, 2))
        cv2.polylines(img_persp, [points], True, (0, 255, 0), thickness=1)
        
    img_persp = img_persp[crop[1]:crop[3], crop[0]:crop[2]]         # crop using calculated bounds and padding

    # catch zero division (usually if cropping bound is < 0)
    try:
        n_width, n_height, _ = img_persp.shape                      # dimensions of image before scaling
        fx, fy = IN_SIZE[0]/n_width, IN_SIZE[1]/n_height            # scaling factors
    except ZeroDivisionError:
        print('Zero Division', '\nobj_ID:', obj_ID, 'dims:', (n_width, n_height), 'corners:', crop)
        #continue

    img_persp = cv2.resize(img_persp, IN_SIZE, fx=fx, fy=fy)        # scale up to out_size

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
    inv_rot_mat = np.vstack((inv_rot_mat, rot_row))                                         # add 3rd row to inv_rot_mat to be equal in shape to inv_persp_mat
    final_inv_mat = np.dot(inv_persp_mat, inv_rot_mat)                                      # dot product to get final inverse matrix

    return img_persp, final_inv_mat