


import os
import operator
import subprocess
import time
import argparse
import cv2
from osgeo import gdal
import numpy as np


# def get_counts(filename):
#     fields = os.path.basename(filename).split('.')
#     if fields[0].startswith('neg'):
#         return
#     basename = '.'.join(fields[1:-2])
#     if basename not in counts:
#         counts[basename] = 0
#     counts[basename] += 1


def convert_bbox_to_percent(size, box_center, box_wh):
    '''Input = image size: (w,h), box: [x0, x1, y0, y1]'''
    dw = 1. / size
    dh = 1. / size
    x = box_center[0] * dw
    y = box_center[1] * dh
    w = box_wh * dw
    h = box_wh * dh

    return x, y, w, h


def extract_info_from_filename(filename):
    """
    For example in COWC_train_list_detection.txt we would see a line like:

    Utah_AGRC/train/neg.Utah_AGRC-HRO_15.0cm_12TVL220180-CROP.05573.03863.030.png

    (1) `neg` means this is a negative sample. It would say `car` otherwise.
    (2) 12TVL220180-CROP is the original image name (see file_name_translate.txt)
    (3) 05573.03863.030 means this image was taken centered from the pixel offset 05573,03863 in the image. 
        The 030 means this patch was rotated 30 degrees from its original.
    :param filename: 
    :type filename: 
    :return: 
    :rtype: 
    """
    fields = os.path.basename(filename).split('.')
    basename = '.'.join(fields[1:-4])
    meta = {'label': fields[0],
            'offset_xy': (int(fields[-3]), int(fields[-4])),
            'angle': fields[-2]}
    return basename, meta


def gen_bounds(xr, xc=416):
    nx = xr / (xc - 2 * bbox_wh)
    xf = ((nx + 1.0) * xc - xr) / nx
    xa = xc - xf
    bounds = []
    for ix in range(nx + 1):
        x_left = ix * xa
        x_right = (xa + xf) + ix * xa
        bounds.append((int(round(x_left)), int(round(x_right))))
        assert int(round(x_right)) - int(round(x_left)) == xc
    assert bounds[-1][1] == xr
    return np.array(bounds)


def get_poly_points(img_mask):

    pixel_coords = []


if __name__ == '__main__':

    src_dir = '/media/RED6/DATA/geos/cowc/datasets'
    patch_dir = os.path.join(src_dir, 'patch_sets/detection')
    truth_dir = os.path.join(src_dir, 'ground_truth_sets')
    dst_dir = '/home/maddoxw/temp/cowc'
    anno_dir = os.path.join(dst_dir, 'annotations')

    if not os.path.exists(anno_dir):
        os.makedirs(anno_dir)

    loc_pres = ('Columbus', 'Potsdam', 'Selwyn', 'Toronto', 'Utah', 'Vaihingen')
    loc_posts = ('CSUAV_AFRL', 'ISPRS', 'LINZ', 'ISPRS', 'AGRC', 'ISPRS')
    loc_dirs = ('_'.join([a, b]) for a, b in zip(loc_pres, loc_posts))

    file_name_translate = {}
    with open(os.path.join(truth_dir, 'file_name_translate.txt')) as ifs:
        for line in ifs.readlines():
            if line == '\n':
                continue
            src, tgt = line.strip().split()
            file_name_translate[src] = tgt

    metadata = {}
    for s in ('train', 'test'):
        with open(os.path.join(patch_dir, 'COWC_'+s+'_list_detection.txt')) as ifs:
            for line in ifs.readlines():
                basename, meta = extract_info_from_filename(line.split()[0])
                if basename not in metadata:
                    metadata[basename] = {}
                if meta['label'] not in metadata[basename]:
                    metadata[basename][meta['label']] = []
                if meta['offset_xy'] not in metadata[basename][meta['label']]:
                    metadata[basename][meta['label']].append(meta['offset_xy'])

    clip_size = 416
    bbox_wh = 48
    used_tiles = 0
    max_tiles = 0
    image_files_seen = []
    for loc, loc_dir in zip(loc_pres, list(loc_dirs)):
        for basename, truth_name in file_name_translate.iteritems():
            if not basename.startswith(loc):
                continue
            print basename
            image_filename = os.path.join(truth_dir, loc_dir, truth_name + '.png')
            assert os.path.exists(image_filename)

            image = cv2.imread(image_filename)
            x_size = image.shape[0]
            y_size = image.shape[1]

            ds = gdal.Open(image_filename)
            assert y_size == ds.RasterXSize
            assert x_size == ds.RasterYSize

            x_bounds = gen_bounds(x_size, xc=clip_size)
            y_bounds = gen_bounds(y_size, xc=clip_size)

            x_centers = np.sum(x_bounds, axis=1) / 2
            y_centers = np.sum(y_bounds, axis=1) / 2

            x_lims = np.vstack((x_bounds.T, x_centers)).T
            y_lims = np.vstack((y_bounds.T, y_centers)).T

            patches = []
            for y_lim in y_lims:
                for x_lim in x_lims:
                    patches.append([[x_lim[0], x_lim[2], x_lim[1]], [y_lim[0], y_lim[2], y_lim[1]]])
            patches = np.array(patches)

            car_centers = np.array(metadata[basename]['car'])
            patch_dict = {}
            for car_center in car_centers:
                patch_idx = np.argmin(np.linalg.norm(patches[:, :, 1] - car_center, axis=1))
                if patch_idx not in patch_dict:
                    patch_dict[patch_idx] = []
                patch_dict[patch_idx].append(car_center)

            for patch_idx, features in patch_dict.iteritems():
                patch = patches[patch_idx]

                chip_name = '_'.join([basename, str(patch[0, 1]), str(patch[1, 1])])
                chip_image_filename = os.path.join(anno_dir, chip_name + '.png')
                image_new = image[patch[0, 0]:patch[0, 2], patch[1, 0]:patch[1, 2]]
                cv2.imwrite(chip_image_filename, image_new)

                chip_labels_filename = os.path.join(anno_dir, chip_name + '.txt')
                with open(chip_labels_filename, 'w') as ofs:

                    for feature in features:
                        bbox_center = feature - patch[:, 0]
                        dbox = convert_bbox_to_percent(clip_size, bbox_center, bbox_wh)
                        if dbox[0] < dbox[2] / 2 or dbox[1] < dbox[3] / 2 or dbox[0] + dbox[2] / 2 > 1 or dbox[1] + dbox[3] / 2 > 1:
                            continue

                        # if dbox[0] < dbox[2]/2:
                        #     print 'x', feature[0], patch[0, 0], bbox_center[0]
                        # if dbox[1] < dbox[3]/2:
                        #     print 'y', feature[1], patch[1, 0], bbox_center[1]
                        # if dbox[0] + dbox[2]/2 > 1:
                        #     print 'x', feature[0], patch[0, 1], bbox_center[0]
                        # if dbox[1] + dbox[3]/2 > 1:
                        #     print 'y', feature[1], patch[1, 1], bbox_center[1]

                        classNum = 0
                        ofs.write('{} {} {} {} {}\n'.format(classNum, dbox[0], dbox[1], dbox[2], dbox[3]))

            used_tiles += len(patch_dict)
            max_tiles += len(x_bounds) * len(y_bounds)

    print used_tiles, max_tiles, used_tiles/float(max_tiles)

    # sorted_x = sorted(metadata.items(), key=operator.itemgetter(0))
    # for key, val in sorted_x:
    #     print val, key