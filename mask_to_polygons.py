# import the necessary packages
import os
import numpy as np
import argparse
import cv2
from osgeo import gdal, osr, ogr

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

src_dir = '/media/Borg_LS/DATA/geos/cowc/datasets/raw'
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

fname = '/media/Borg_LS/DATA/geos/cowc/datasets/ground_truth_sets/Vaihingen_ISPRS/TOP_Mosaic_09cm_scaled_15cm_Gray_Annotated_Cars.png'
fname = '/media/Borg_LS/DATA/geos/cowc/datasets/ground_truth_sets/Potsdam_ISPRS/top_potsdam_5_10_RGB_Annotated_Cars.png'
image = cv2.imread(fname)
cv2.imshow("Game Boy Screen", image)
cv2.waitKey(0)

for dset, paths in dset_paths.iteritems():

    srcimagefile = os.path.join(src_dir, dset, paths['georaster'])
    ds = gdal.Open(srcimagefile)
    srs = osr.SpatialReference()
    srs.ImportFromWkt(ds.GetProjectionRef())
    ds = None
    cti = osr.CoordinateTransformation(srs, srs.CloneGeogCS())

    imagefiles = []
    rasterpath = os.path.join(src_dir, dset, paths['rasters'])
    for path, _, files in os.walk(rasterpath):
        for f in files:
            if f.endswith('tif'):
                imagefiles.append(os.path.join(path, f))

    for imagefile in imagefiles:

        ds = gdal.Open(imagefile)
        gt = ds.GetGeoTransform(can_return_null=False)
        ds = None

        image = cv2.imread(imagefile)
        contours = []
        for labelname, truth_color in truth_colors.iteritems():

            truth_range = np.array(truth_color, dtype=np.uint8)
            mask = cv2.inRange(image, truth_range, truth_range)
            cnts = cv2.findContours(mask.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[1]
            contours += [cv2.approxPolyDP(cnt, 2, True) for cnt in cnts]

        n_contours = len(contours)
        for ii in range(n_contours):
            print os.path.basename(imagefile)
            shp = contours[ii].shape
            contour = contours[ii].reshape(shp[0], shp[2])
            print contour
            contour_x = gt[0] + gt[1] * contour[:, 0]
            contour_y = gt[3] + gt[5] * contour[:, 1]
            polygon = np.array([contour_x, contour_y]).T
            print polygon
            lonlats = cti.TransformPoints(polygon)
            print lonlats

        cv2.drawContours(image, contours, -1, (0, 0, 0), 1)
        cv2.imshow("Game Boy Screen", image)
        cv2.waitKey(0)
