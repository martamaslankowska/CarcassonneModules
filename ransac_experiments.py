from __future__ import print_function
import cv2
import numpy as np
from matplotlib import pyplot as plt
from time import time
import glob
from PIL import Image
from resizeimage import resizeimage
from sklearn.cluster import KMeans



minHessian = 400


def find_keypoints(img_tile, img_board, minHessian=400):
    detector = cv2.xfeatures2d.SURF_create(hessianThreshold=minHessian)
    keypoints_tile, descriptors_tile = detector.detectAndCompute(img_tile, None)
    keypoints_board, descriptors_board = detector.detectAndCompute(img_board, None)
    return keypoints_tile, descriptors_tile, keypoints_board, descriptors_board


def find_good_matches(des_tile, des_board):
    # Since SURF is a floating-point descriptor NORM_L2 is used
    matcher = cv2.DescriptorMatcher_create(cv2.DescriptorMatcher_FLANNBASED)
    knn_matches = matcher.knnMatch(des_tile, des_board, 2)

    # -- Filter matches using the Lowe's ratio test
    ratio_thresh = 0.75
    good_matches = []
    for m, n in knn_matches:
        if m.distance < ratio_thresh * n.distance:
            good_matches.append(m)

    return good_matches


def draw_good_matches(img_tile, img_board, keypts_tile, keypts_board, good_matches):
    img_matches = np.empty((max(img_tile.shape[0], img_board.shape[0]), img_tile.shape[1] + img_board.shape[1], 3),
                           dtype=np.uint8)
    cv2.drawMatches(img_tile, keypts_tile, img_board, keypts_board, good_matches, img_matches,
                   flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    return img_matches


def find_homography_tile2board(good_matches, keypts_tile, keypts_board):
    # -- Localize the object
    tile = np.empty((len(good_matches), 2), dtype=np.float32)
    board = np.empty((len(good_matches), 2), dtype=np.float32)

    for i in range(len(good_matches)):
        # -- Get the keypoints from the good matches
        tile[i, 0] = keypts_tile[good_matches[i].queryIdx].pt[0]
        tile[i, 1] = keypts_tile[good_matches[i].queryIdx].pt[1]
        board[i, 0] = keypts_board[good_matches[i].trainIdx].pt[0]
        board[i, 1] = keypts_board[good_matches[i].trainIdx].pt[1]
    H, _ = cv2.findHomography(tile, board, cv2.RANSAC)
    return H


def find_corners(img_tile, H_tile2board):
    # -- Get the corners from the image_1 (the tile to be "detected")
    tile_corners = np.empty((4, 1, 2), dtype=np.float32)
    tile_corners[0, 0, 0] = 0
    tile_corners[0, 0, 1] = 0
    tile_corners[1, 0, 0] = img_tile.shape[1]
    tile_corners[1, 0, 1] = 0
    tile_corners[2, 0, 0] = img_tile.shape[1]
    tile_corners[2, 0, 1] = img_tile.shape[0]
    tile_corners[3, 0, 0] = 0
    tile_corners[3, 0, 1] = img_tile.shape[0]
    board_corners = cv2.perspectiveTransform(tile_corners, H_tile2board)

    return tile_corners, board_corners


def draw_tile_on_board_corners(img_matches, img_tile, board_corners):
    cv2.line(img_matches, (int(board_corners[0, 0, 0] + img_tile.shape[1]), int(board_corners[0, 0, 1])), \
             (int(board_corners[1, 0, 0] + img_tile.shape[1]), int(board_corners[1, 0, 1])), (0, 255, 0), 4)
    cv2.line(img_matches, (int(board_corners[1, 0, 0] + img_tile.shape[1]), int(board_corners[1, 0, 1])), \
             (int(board_corners[2, 0, 0] + img_tile.shape[1]), int(board_corners[2, 0, 1])), (0, 255, 0), 4)
    cv2.line(img_matches, (int(board_corners[2, 0, 0] + img_tile.shape[1]), int(board_corners[2, 0, 1])), \
             (int(board_corners[3, 0, 0] + img_tile.shape[1]), int(board_corners[3, 0, 1])), (0, 255, 0), 4)
    cv2.line(img_matches, (int(board_corners[3, 0, 0] + img_tile.shape[1]), int(board_corners[3, 0, 1])), \
             (int(board_corners[0, 0, 0] + img_tile.shape[1]), int(board_corners[0, 0, 1])), (0, 255, 0), 4)

    plt.imshow(img_matches, cmap='gray', interpolation='nearest')
    plt.xticks([])
    plt.yticks([])
    plt.show()


def wrap_board2tile(img_tile, img_board, tile_corners, board_corners):
    H, status = cv2.findHomography(board_corners, tile_corners, cv2.RANSAC, 5.0)
    # Warp source image to destination based on homography
    im_out = cv2.warpPerspective(img_board, H, ((int)(img_tile.shape[1]), (int)(img_tile.shape[0])))
    return im_out


def visualize_image(img, cm=None):
    if cm:
        plt.imshow(img, cmap=cm, interpolation='nearest')
    else:
        plt.imshow(img, interpolation='nearest')
    plt.xticks([])
    plt.yticks([])
    plt.show()



''' Scene homography '''

img_tile = cv2.imread('tiles/tile (13).jpg', cv2.IMREAD_GRAYSCALE)
img_board = cv2.imread('tiles/board test.jpg', cv2.IMREAD_GRAYSCALE)

keypts_tile, des_tile, keypts_board, des_board = find_keypoints(img_tile, img_board)
good_matches = find_good_matches(des_tile, des_board)
H_tile2board = find_homography_tile2board(good_matches, keypts_tile, keypts_board)

tile_corners, board_corners = find_corners(img_tile, H_tile2board)

img_matches = draw_good_matches(img_tile, img_board, keypts_tile, keypts_board, good_matches)
draw_tile_on_board_corners(img_matches, img_tile, board_corners)

img_board_transformed = wrap_board2tile(img_tile, img_board, tile_corners, board_corners)

img_tile_color_transformed = wrap_board2tile(img_tile, cv2.cvtColor(cv2.imread('tiles/board test.jpg'), cv2.COLOR_BGR2RGB), tile_corners, board_corners)


# Display images
# cv2.imshow("Source Image", img_board)
# cv2.imshow("Destination Image", img_tile)
# cv2.imshow("Warped Source Image", img_board_transformed)

visualize_image(img_tile_color_transformed)




H, status = cv2.findHomography(board_corners, tile_corners, cv2.RANSAC, 5.0)
img_out = cv2.warpPerspective(img_board, H, ((int)(img_tile.shape[1]), (int)(img_tile.shape[0])))
visualize_image(img_out, 'gray')


src = img_board
dst = img_tile

# Find the corners after the transform has been applied
height, width = src.shape[:2]
corners = np.array([
  [0, 0],
  [0, height - 1],
  [width - 1, height - 1],
  [width - 1, 0]
])
corners = cv2.perspectiveTransform(np.float32([corners]), H)[0]

# Find the bounding rectangle
bx, by, bwidth, bheight = cv2.boundingRect(corners)

# Compute the translation homography that will move (bx, by) to (0, 0)
H_trans = np.array([
  [ 1, 0, -bx ],
  [ 0, 1, -by ],
  [ 0, 0,   1 ]
])

# Combine the homographies
pth = H_trans.dot(H)

# Apply the transformation to the image
warped = cv2.warpPerspective(src, pth, (bwidth, bheight),
                             flags=cv2.INTER_CUBIC,
                             borderMode=cv2.BORDER_CONSTANT)


corners = np.array([
  [0, 0],
  [0, height - 1],
  [width - 1, height - 1],
  [width - 1, 0]
])
corners = cv2.perspectiveTransform(np.float32([corners]), pth)[0]
bx2, by2, bwidth2, bheight2 = cv2.boundingRect(corners)

print(bx, by, bwidth, bheight)
print(bx2, by2, bwidth2, bheight2)



img_test = cv2.warpPerspective(img_board, pth, (bwidth2, bheight2))
visualize_image(img_test, 'gray')


# Display image - to big...
# cv2.imshow("Whole Image", img_test)

# im = Image.fromarray(img_test)
# im.save("board_transformation.jpeg")


cv2.waitKey(0)