# import the necessary packages
import os
import numpy as np
import argparse
import cv2
from osgeo import gdal, osr, ogr


def create_edge_chip(poly, large_raster, gt, border):

    if large_raster is None:
        return False

    if 'left' in border and 'right' in border:
        return False

    if 'top' in border and 'bot' in border:
        return False

    poly_max = np.max(poly, axis=0)
    poly_min = np.min(poly, axis=0)
    poly_wh = (poly_max - poly_min)
    poly_left, poly_bot = poly_min - poly_wh / 2
    poly_right, poly_top = poly_max + poly_wh / 2
    assert poly_left < poly_right
    assert poly_bot < poly_top

    contour_x = (poly[:, 0] - gt[0]) / gt[1]
    contour_y = (poly[:, 1] - gt[3]) / gt[5]
    contour = np.array([contour_x, contour_y]).T.astype(int)

    contour_max = np.max(contour, axis=0)
    contour_min = np.min(contour, axis=0)
    contour_wh = (contour_max - contour_min)

    rast_left1, rast_top1 = contour_min - contour_wh / 2
    rast_right1, rast_bot1 = contour_max + contour_wh / 2

    rast_left = int((poly_left - gt[0]) / gt[1])
    rast_right = int((poly_right - gt[0]) / gt[1])
    rast_top = int((poly_top - gt[3]) / gt[5])
    rast_bot = int((poly_bot - gt[3]) / gt[5])
    assert rast_left < rast_right
    assert rast_bot > rast_top

    rep_str = ""
    if 'left' in border:
        rast_left -= contour_wh[0]
        rep_str += '_left_' + str(rast_left)
    if 'right' in border:
        rast_right += contour_wh[0]
        rep_str += '_right_' + str(rast_right)
    if 'top' in border:
        rast_top -= contour_wh[1]
        rep_str += '_top_' + str(rast_top)
    if 'bot' in border:
        rast_bot += contour_wh[1]
        rep_str += '_bot_' + str(rast_bot)

    new_origin = np.array([rast_left, rast_top])
    contour_new = contour - new_origin
    chip_raster = large_raster[rast_top:rast_bot, rast_left:rast_right]
    cv2.drawContours(chip_raster, [contour_new], -1, (0, 255, 0), 1)

    savefile = imagefile.replace(".tif", rep_str + ".tif")
    title = os.path.basename(savefile)
    # cv2.imwrite(savefile, chip_raster)
    cv2.imshow(title, chip_raster)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return True
# First make sure the contents of Vaihingen_dsm_tiles_geoinfo.zip are
# extracted into gts_for_participants and rename the prefix:
# dsm_09cm_matching --> top_mosaic_09cm
# Unfortunately these world files aren't enough, and you're still going to
# need the projection system from the raw tif's in the Image dir.
# (i.e. the ones that are like 820 MB a piece.)

# construct the argument parser and parse the arguments
# ap = argparse.ArgumentParser()
# ap.add_argument("-q", "--query", required=True, help="Path to the query image")
# args = vars(ap.parse_args())
# image = cv2.imread(args["query"])

src_dir = '/media/Borg_LS/DATA/geos/cowc/raw'
dset_paths = {'Vaihingen': {'rasters': 'ISPRS_semantic_labeling_Vaihingen/gts_for_participants',
                            'georaster': 'Images/10030061.tif'},
              'Potsdam': {'rasters': '5_Labels_for_participants',
                          'georaster': '2_Ortho_RGB/top_potsdam_2_10_RGB.tif'}}

truth_colors = {'car': [0, 255, 255], 'building': [255, 0, 0]}

# crop_size = 800
# image = image[:crop_size, :crop_size]
# savefile = imagefile.replace(".tif", "_" + str(crop_size) + ".tif")
# cv2.imwrite(savefile, image)

# percentage = 0.002
# for c in cnts:
#     peri = cv2.arcLength(c, True)
#     poly = cv2.approxPolyDP(c, percentage * peri, True)
#     poly3 = cv2.approxPolyDP(c, 2, True)
#     print len(c), percentage * peri, len(poly), len(poly3)

# fname = '/media/Borg_LS/DATA/geos/cowc/datasets/ground_truth_sets/Vaihingen_ISPRS/TOP_Mosaic_09cm_scaled_15cm_Gray_Annotated_Cars.png'
# fname = '/media/Borg_LS/DATA/geos/cowc/datasets/ground_truth_sets/Potsdam_ISPRS/top_potsdam_5_10_RGB_Annotated_Cars.png'
# image = cv2.imread(fname)
# cv2.imshow("Ground Truth", image)
# cv2.waitKey(0)

for dset, paths in dset_paths.iteritems():

    if dset == 'Vaihingen':
        top_mosaic_fname = os.path.join(src_dir, 'Vaihingen/Ortho/TOP_Mosaic_09cm.tif')
        top_mosaic_ds = gdal.Open(top_mosaic_fname)
        top_mosaic_gt = top_mosaic_ds.GetGeoTransform(can_return_null=False)
        # top_mosaic_gd = top_mosaic_ds.ReadAsArray().T
        top_mosaic_im = cv2.imread(top_mosaic_fname)
        top_mosaic_ds = None
    else:
        top_mosaic_im = None
        # top_mosaic_gd = None
        top_mosaic_gt = None

    # srcimagefile = os.path.join(src_dir, dset, paths['georaster'])
    # ds = gdal.Open(srcimagefile)
    # srs = osr.SpatialReference()
    # srs.ImportFromWkt(ds.GetProjectionRef())
    # ds = None
    # cti = osr.CoordinateTransformation(srs, srs.CloneGeogCS())

    rasterpath = os.path.join(src_dir, dset, paths['rasters'])
    imagefiles = [os.path.join(rasterpath, f) for f in os.listdir(rasterpath) if f.endswith('tif')]

    for imagefile in imagefiles:
        res = False
        # if not imagefile.endswith('28.tif'):
        #     continue

        ds = gdal.Open(imagefile)
        gt = ds.GetGeoTransform(can_return_null=False)
        # image = ds.ReadAsArray()
        x_size = ds.RasterXSize
        y_size = ds.RasterYSize
        ds = None

        image = cv2.imread(imagefile)
        contours = []
        for labelname, truth_color in truth_colors.iteritems():

            truth_range = np.array(truth_color, dtype=np.uint8)
            mask = cv2.inRange(image, truth_range, truth_range)
            cnts = cv2.findContours(mask.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[1]
            contours += [cv2.approxPolyDP(cnt, 0, True) for cnt in cnts]

        # cv2.drawContours(image, contours, -1, (0, 0, 0), 2)
        # cv2.imshow("Truth", image)
        # cv2.waitKey(0)

        n_contours = len(contours)
        edge_contours = []
        for ii in range(n_contours):
            shp = contours[ii].shape
            contour = contours[ii].reshape(shp[0], shp[2])

            poly_x = contour[:, 0] * gt[1] + gt[0]
            poly_y = contour[:, 1] * gt[5] + gt[3]
            polygon = np.array([poly_x, poly_y]).T
            # lonlats = cti.TransformPoints(polygon)

            img_min = np.min(contour, axis=0)
            img_max = np.max(contour, axis=0)

            border = []
            if img_min[0] == 0:
                border.append('left')
            if img_max[0] >= image.shape[1] - 1:
                border.append('right')
            if img_min[1] == 0:
                border.append('top')
            if img_max[1] >= image.shape[0] - 1:
                border.append('bot')

            create_edge_chip(polygon, top_mosaic_im, top_mosaic_gt, border)
