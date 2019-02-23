from __future__ import print_function
import cv2
import numpy as np
from matplotlib import pyplot as plt
from time import time
import glob
from PIL import Image
from resizeimage import resizeimage
from sklearn.cluster import KMeans
from skimage.measure import compare_ssim
import imutils


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


def k_means(img_name, k):
    img = cv2.imread(img_name + '.jpg')
    Z = img.reshape((-1, 3))
    Z = np.float32(Z)

    # define criteria, number of clusters(K) and apply kmeans()
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    ret, label, center = cv2.kmeans(Z, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)


    # kmeans = KMeans(n_clusters=k)
    # kmeans.fit(img)
    # colors = kmeans.cluster_centers_

    # Now convert back into uint8, and make original image
    center = np.uint8(center)
    res = center[label.flatten()]
    res2 = res.reshape((img.shape))
    return res2


def centroid_histogram(clt):
    # grab the number of different clusters and create a histogram
    # based on the number of pixels assigned to each cluster
    numLabels = np.arange(0, len(np.unique(clt.labels_)) + 1)
    (hist, _) = np.histogram(clt.labels_, bins=numLabels)

    # normalize the histogram, such that it sums to one
    hist = hist.astype("float")
    hist /= hist.sum()

    # return the histogram
    return hist


def plot_colors(hist, centroids):
    # initialize the bar chart representing the relative frequency
    # of each of the colors
    bar = np.zeros((50, 300, 3), dtype="uint8")
    startX = 0

    # loop over the percentage of each cluster and the color of
    # each cluster
    for (percent, color) in zip(hist, centroids):
        # plot the relative percentage of each cluster
        endX = startX + (percent * 300)
        cv2.rectangle(bar, (int(startX), 0), (int(endX), 50),
                      color.astype("uint8").tolist(), -1)
        startX = endX

    # return the bar chart
    return bar


def visualize_image(img, cm=None):
    if cm:
        plt.imshow(img, cmap=cm, interpolation='nearest')
    else:
        plt.imshow(img)
    plt.xticks([])
    plt.yticks([])
    plt.show()


def draw_tile_centers(img, tile_centers, col=(255, 0, 127)):
    cv2.circle(img, (tile_centers[0], tile_centers[1]), 5, col, -1)
    return img

''' Scene homography '''
tile_name = 'tile1'
board_name = 'board test'

img_tile = cv2.imread('tiles/' + tile_name + '.jpg', cv2.IMREAD_GRAYSCALE)
img_board = cv2.imread('tiles/' + board_name + '.jpg', cv2.IMREAD_GRAYSCALE)

keypts_tile, des_tile, keypts_board, des_board = find_keypoints(img_tile, img_board)
good_matches = find_good_matches(des_tile, des_board)
H_tile2board = find_homography_tile2board(good_matches, keypts_tile, keypts_board)

tile_corners, board_corners = find_corners(img_tile, H_tile2board)

img_matches = draw_good_matches(img_tile, img_board, keypts_tile, keypts_board, good_matches)
draw_tile_on_board_corners(img_matches, img_tile, board_corners)

img_board_transformed = wrap_board2tile(img_tile, img_board, tile_corners, board_corners)
img_tile_color_transformed = wrap_board2tile(img_tile, cv2.cvtColor(cv2.imread('tiles/board test.jpg'), cv2.COLOR_BGR2RGB), tile_corners, board_corners)

im = Image.fromarray(img_tile_color_transformed)
im.save("tiles/coloured_tile.jpg")

# Display images
# cv2.imshow("Source Image", img_board)
# cv2.imshow("Destination Image", img_tile)
# cv2.imshow("Warped Source Image", img_board_transformed)

# visualize_image(img_tile_color_transformed)

# ''' Testing figure finding '''
# K = 5
#
# im = Image.fromarray(img_tile_color_transformed)
# im.save("tiles/coloured_tile.jpg")
# k_means_tile = k_means('tiles/coloured_tile', K)
# visualize_image(k_means_tile)


# ''' Finding color ranges '''
# # define the lower and upper boundaries of the colors in the HSV color space
# # lower = {'red':(166, 84, 141), 'green':(66, 122, 129), 'blue':(97, 100, 117), 'yellow':(23, 59, 119), 'orange':(0, 50, 80)}
# # upper = {'red':(186,255,255), 'green':(86,255,255), 'blue':(117,255,255), 'yellow':(54,255,255), 'orange':(20,255,255)}
#
# lower = {'pink': (255, 153, 255)}
# upper = {'pink': (243, 66, 226)}
#
# frame = img_tile_color_transformed
# # blurred = cv2.GaussianBlur(frame, (11, 11), 0)
# # hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
# # visualize_image(hsv)
#
# frame_kmeans = k_means_tile
#
# # for each color in dictionary check object in frame
# for key, value in upper.items():
#     # construct a mask for the color from dictionary`1, then perform
#     # a series of dilations and erosions to remove any small
#     # blobs left in the mask
#     kernel = np.ones((9, 9), np.uint8)
#     mask = cv2.inRange(frame_kmeans, lower[key], upper[key])
#     mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
#     mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
#
#     # find contours in the mask and initialize the current
#     # (x, y) center of the ball
#     cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
#     center = None
#
#     # only proceed if at least one contour was found
#     if len(cnts) > 0:
#         # find the largest contour in the mask, then use
#         # it to compute the minimum enclosing circle and
#         # centroid
#         c = max(cnts, key=cv2.contourArea)
#         ((x, y), radius) = cv2.minEnclosingCircle(c)
#         M = cv2.moments(c)
#         center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
#
#         # only proceed if the radius meets a minimum size. Correct this value for your obect's size
#         if radius > 0.5:
#             # draw the circle and centroid on the frame,
#             # then update the list of tracked points
#             cv2.circle(frame, (int(x), int(y)), int(radius), upper[key], 2)
#             print(f'Color - {key}')
#
# visualize_image(frame)


# ''' Extra KMeans '''
# image = img_tile_color_transformed.reshape((img_tile_color_transformed.shape[0] * img_tile_color_transformed.shape[1], 3))
# clt = KMeans(n_clusters=K)
# clt.fit(image)
# hist = centroid_histogram(clt)
# bar = plot_colors(hist, clt.cluster_centers_)
#
# plt.figure()
# plt.axis("off")
# plt.imshow(bar)
# plt.show()


# visualize_image(img_board_transformed)


''' Image differences '''
imageA = cv2.imread('C:/Users/Marta/Documents/Nauka/PyCharm/CarcassonneTests/tiles/' + tile_name + '.jpg')
imageB = cv2.imread('C:/Users/Marta/Documents/Nauka/PyCharm/CarcassonneTests/tiles/coloured_tile.jpg')
grayA = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)
grayB = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY)

visualize_image(grayA, 'gray')
visualize_image(grayB, 'gray')

# compute the Structural Similarity Index (SSIM) between the two
# images, ensuring that the difference image is returned
(score, diff) = compare_ssim(grayA, grayB, full=True)
diff = (diff * 255).astype("uint8")
# print("SSIM: {}".format(score))

# threshold the difference image, followed by finding contours to
# obtain the regions of the two input images that differ
thresh = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
# cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# cnts = imutils.grab_contours(cnts)

# visualize_image(thresh, 'gray')

im = Image.fromarray(thresh)
im.save("tiles/difference_tile.jpg")

img_len = thresh.shape[0]
padding = int(0.08 * img_len)
new_img_len = img_len - 2*padding
cropped_image = thresh[padding:padding+new_img_len, padding:padding+new_img_len]

n_white_pix = np.sum(thresh == 255)
print(f'BEFORE CROP: {n_white_pix} / {img_len * img_len} is {n_white_pix / (img_len * img_len) * 100}%')
visualize_image(thresh, 'gray')

n_white_pix_cropped = np.sum(cropped_image == 255)
print(f'AFTER CROP: {n_white_pix_cropped} / {new_img_len * new_img_len} is {n_white_pix_cropped / (new_img_len * new_img_len) * 100}%')
visualize_image(cropped_image, 'gray')

img = cv2.medianBlur(cropped_image, 21)
visualize_image(img, 'gray')
n_white_pix_cropped = np.sum(img == 255)
print(f'AFTER CROP: {n_white_pix_cropped} / {new_img_len * new_img_len} is {n_white_pix_cropped / (new_img_len * new_img_len) * 100}%')

percentage = int(n_white_pix_cropped / (new_img_len * new_img_len) * 100)




ret, thresh = cv2.threshold(img, 127, 255, 0)
_, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, 2)

per = 0

for cnt in contours:

    area = cv2.contourArea(cnt)
    perimeter = cv2.arcLength(cnt, True)
    if (per < perimeter):
        per = perimeter

        epsilon = 0.1*cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)

        hull = cv2.convexHull(cnt)

        x,y,w,h = cv2.boundingRect(cnt)
        # cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),5)

        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        # cv2.drawContours(img, [box], 0, (0, 0, 255), 5)

# cv2.rectangle(img,(x,y),(x+w,y+h),(255,255,255),6)


if percentage > 10:
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    cv2.drawContours(img, [box], 0, (255, 153, 255), 3)
    # cv2.drawContours(img, [hull], 0, (255, 255, 255), 3)
    visualize_image(img)

    x_center, y_center = int(np.min(box[:,0]) + (np.max(box[:,0]) - np.min(box[:,0]))/2), int(np.min(box[:,1]) + (np.max(box[:,1]) - np.min(box[:,1]))/2)
    img = draw_tile_centers(img, (x_center, y_center))
    visualize_image(img)
