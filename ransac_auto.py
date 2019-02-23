from __future__ import print_function
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
from time import time
import glob
from itertools import cycle
import argparse


colors = [(102, 0, 51), (255, 0, 127), (255, 102, 178), (178, 102, 255), (102, 102, 255), (0, 102, 204), (0, 204, 102), (51, 255, 51), (255, 255, 51), (255, 128, 0), (0, 102, 102), (255, 51, 51), (153, 0, 0), (178, 255, 102), (0, 102, 0), (0, 0, 102), (51, 0, 102), (60, 60, 60), (255, 178, 102), (153, 104, 255), (255, 204, 153), (204, 153, 255), (204, 255, 153)]

tile_names = glob.glob('tiles/tile*.jpg')
board_names = glob.glob('tiles/board*.jpg')

scene_name = 'board (2)'

images = [cv.imread(x, cv.IMREAD_GRAYSCALE) for x in tile_names]
img_scene = cv.imread('tiles/' + scene_name + '.jpg', cv.IMREAD_GRAYSCALE)

start = time()


minHessian = 400
detector = cv.xfeatures2d.SURF_create(hessianThreshold=minHessian)
keypoints_scene, descriptors_scene = detector.detectAndCompute(img_scene, None)

# image to draw the tiles on it
mma_image = cv.cvtColor(img_scene, cv.COLOR_GRAY2RGB)


for c, img_object in enumerate(images):
    if img_object is None or img_scene is None:
        print('Could not open or find the images!')
        exit(0)

    # -- Step 1b: Detect the keypoints using SURF Detector, compute the descriptors
    keypoints_obj, descriptors_obj = detector.detectAndCompute(img_object, None)

    #-- Step 2: Matching descriptor vectors with a FLANN based matcher
    # Since SURF is a floating-point descriptor NORM_L2 is used
    matcher = cv.DescriptorMatcher_create(cv.DescriptorMatcher_FLANNBASED)
    knn_matches = matcher.knnMatch(descriptors_obj, descriptors_scene, 2)

    #-- Filter matches using the Lowe's ratio test
    ratio_thresh = 0.75
    good_matches = []
    for m,n in knn_matches:
        if m.distance < ratio_thresh * n.distance:
            good_matches.append(m)


    #-- Draw matches
    img_matches = np.empty((max(img_object.shape[0], img_scene.shape[0]), img_object.shape[1]+img_scene.shape[1], 3), dtype=np.uint8)
    cv.drawMatches(img_object, keypoints_obj, img_scene, keypoints_scene, good_matches, img_matches, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    #-- Localize the object
    obj = np.empty((len(good_matches),2), dtype=np.float32)
    scene = np.empty((len(good_matches),2), dtype=np.float32)

    for i in range(len(good_matches)):
        #-- Get the keypoints from the good matches
        obj[i,0] = keypoints_obj[good_matches[i].queryIdx].pt[0]
        obj[i,1] = keypoints_obj[good_matches[i].queryIdx].pt[1]
        scene[i,0] = keypoints_scene[good_matches[i].trainIdx].pt[0]
        scene[i,1] = keypoints_scene[good_matches[i].trainIdx].pt[1]
    H, _ = cv.findHomography(obj, scene, cv.RANSAC)

    #-- Get the corners from the image_1 ( the object to be "detected" )
    obj_corners = np.empty((4,1,2), dtype=np.float32)
    obj_corners[0,0,0] = 0
    obj_corners[0,0,1] = 0
    obj_corners[1,0,0] = img_object.shape[1]
    obj_corners[1,0,1] = 0
    obj_corners[2,0,0] = img_object.shape[1]
    obj_corners[2,0,1] = img_object.shape[0]
    obj_corners[3,0,0] = 0
    obj_corners[3,0,1] = img_object.shape[0]
    scene_corners = cv.perspectiveTransform(obj_corners, H)

    # -- Draw lines between the corners (the mapped object in the scene - image_2 )
    cv.line(img_matches, (int(scene_corners[0,0,0] + img_object.shape[1]), int(scene_corners[0,0,1])),\
        (int(scene_corners[1,0,0] + img_object.shape[1]), int(scene_corners[1,0,1])), (0,255,0), 4)
    cv.line(img_matches, (int(scene_corners[1,0,0] + img_object.shape[1]), int(scene_corners[1,0,1])),\
        (int(scene_corners[2,0,0] + img_object.shape[1]), int(scene_corners[2,0,1])), (0,255,0), 4)
    cv.line(img_matches, (int(scene_corners[2,0,0] + img_object.shape[1]), int(scene_corners[2,0,1])),\
        (int(scene_corners[3,0,0] + img_object.shape[1]), int(scene_corners[3,0,1])), (0,255,0), 4)
    cv.line(img_matches, (int(scene_corners[3,0,0] + img_object.shape[1]), int(scene_corners[3,0,1])),\
        (int(scene_corners[0,0,0] + img_object.shape[1]), int(scene_corners[0,0,1])), (0,255,0), 4)


    ''' Colouring lines between the corners only on image_scene '''
    color = colors[c % len(colors)]  # RGB, where (0,0,0) - black
    cv.line(mma_image, (int(scene_corners[0, 0, 0]), int(scene_corners[0, 0, 1])), \
            (int(scene_corners[1, 0, 0]), int(scene_corners[1, 0, 1])), color, 4)
    cv.line(mma_image, (int(scene_corners[1, 0, 0]), int(scene_corners[1, 0, 1])), \
            (int(scene_corners[2, 0, 0]), int(scene_corners[2, 0, 1])), color, 4)
    cv.line(mma_image, (int(scene_corners[2, 0, 0]), int(scene_corners[2, 0, 1])), \
            (int(scene_corners[3, 0, 0]), int(scene_corners[3, 0, 1])), color, 4)
    cv.line(mma_image, (int(scene_corners[3, 0, 0]), int(scene_corners[3, 0, 1])), \
            (int(scene_corners[0, 0, 0]), int(scene_corners[0, 0, 1])), color, 4)



    #-- Show detected matches
    # cv.imshow('Good Matches & Object detection', img_matches)
    plt.imshow(img_matches, cmap='gray', interpolation='nearest')
    plt.xticks([])
    plt.yticks([])
    plt.show()


# All tiles on one image
plt.imshow(mma_image, cmap='gray', interpolation='nearest')
plt.xticks([])
plt.yticks([])
plt.show()

end = time()
print("Overall time for", len(images), "tiles:", end - start, "sec")

