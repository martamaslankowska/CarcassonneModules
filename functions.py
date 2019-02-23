from __future__ import print_function
import cv2
import numpy as np
from matplotlib import pyplot as plt
import math


colors = [(102, 0, 51), (255, 0, 127), (255, 102, 178), (178, 102, 255), (102, 102, 255), (0, 102, 204), (0, 204, 102), (51, 255, 51), (255, 255, 51), (255, 128, 0), (0, 102, 102), (255, 51, 51), (153, 0, 0), (178, 255, 102), (0, 102, 0), (0, 0, 102), (51, 0, 102), (60, 60, 60), (255, 178, 102), (153, 104, 255), (255, 204, 153), (204, 153, 255), (204, 255, 153)]


def find_board_keypoints(img_board, detector):
    keypoints_board, descriptors_board = detector.detectAndCompute(img_board, None)
    return keypoints_board, descriptors_board


def find_tile_keypoints(img_tile, detector):
    keypoints_tile, descriptors_tile = detector.detectAndCompute(img_tile, None)
    return keypoints_tile, descriptors_tile


def find_board_keypoints_BRISK(img_board, detector, extractor):
    keypoints_board = detector.detect(img_board)
    keypoints_board, descriptors_board = extractor.compute(img_board, keypoints_board)
    return keypoints_board, descriptors_board


def find_tile_keypoints_BRISK(img_tile, detector, extractor):
    keypoints_tile = detector.detect(img_tile)
    keypoints_tile, descriptors_tile = extractor.compute(img_tile, keypoints_tile)
    return keypoints_tile, descriptors_tile


def find_all_keypoints(img_tile, img_board, detector):
    keypoints_tile, descriptors_tile = detector.detectAndCompute(img_tile, None)
    keypoints_board, descriptors_board = detector.detectAndCompute(img_board, None)
    return keypoints_tile, descriptors_tile, keypoints_board, descriptors_board


def match_features_SURF(des_tile, des_board):
    # Since SURF is a floating-point descriptor NORM_L2 is used
    matcher = cv2.DescriptorMatcher_create(cv2.DescriptorMatcher_FLANNBASED)
    knn_matches = matcher.knnMatch(des_tile, des_board, 2)
    return knn_matches


def match_features_AKAZE(des_tile, des_board):
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    matches = bf.knnMatch(des_tile, des_board, k=2)
    return matches


def match_features_BRISK(des_tile, des_board):
    bf = cv2.BFMatcher(cv2.NORM_L2SQR)
    matches = bf.match(des_tile, des_board)
    return matches


def find_good_matches(knn_matches, ratio=0.75):
    # -- Filter matches using the Lowe's ratio test
    good_matches = []
    for m, n in knn_matches:
        if m.distance < ratio * n.distance:
            good_matches.append(m)
    return good_matches


def find_good_matches_BRISK(matches):
    distances = [match.distance for match in matches]
    min_dist = min(distances)
    avg_dist = sum(distances) / len(distances)
    min_multiplier_tolerance = 10
    min_dist = min_dist or avg_dist * 1.0 / min_multiplier_tolerance
    good_matches = [match for match in matches if
                    match.distance <= min_multiplier_tolerance * min_dist]
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
    try:
        H, H_mask = cv2.findHomography(tile, board, cv2.RANSAC)
    except:
        print('Error :(')
        H, H_mask = None, None
    return H, H_mask


def filter_good_matches_BRISK(H_mask, good_matches):
    best_matches = []
    for i, match in enumerate(good_matches):
        if H_mask[i]:
            best_matches.append(match)
    return best_matches


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


def draw_all_matches_on_board(img_board_color, board_corners, c):
    color = colors[c % len(colors)]  # RGB, where (0,0,0) - black
    cv2.line(img_board_color, (int(board_corners[0, 0, 0]), int(board_corners[0, 0, 1])), \
             (int(board_corners[1, 0, 0]), int(board_corners[1, 0, 1])), color, 4)
    cv2.line(img_board_color, (int(board_corners[1, 0, 0]), int(board_corners[1, 0, 1])), \
             (int(board_corners[2, 0, 0]), int(board_corners[2, 0, 1])), color, 4)
    cv2.line(img_board_color, (int(board_corners[2, 0, 0]), int(board_corners[2, 0, 1])), \
             (int(board_corners[3, 0, 0]), int(board_corners[3, 0, 1])), color, 4)
    cv2.line(img_board_color, (int(board_corners[3, 0, 0]), int(board_corners[3, 0, 1])), \
             (int(board_corners[0, 0, 0]), int(board_corners[0, 0, 1])), color, 4)


def visualize_image(img, cm=None):
    if cm:
        plt.imshow(img, cmap=cm, interpolation='nearest')
    else:
        plt.imshow(img, interpolation='nearest')
    plt.xticks([])
    plt.yticks([])
    plt.show()


def cover_found_tiles(img_board, board_corners, pad=0.15):
    sum_of_points = []
    for ind in range(4):
        sum_of_points.append(board_corners[ind, 0, 0] + board_corners[ind, 0, 1])

    left_top_corner, right_bottom_corner = sum_of_points.index(min(sum_of_points)), sum_of_points.index(max(sum_of_points))

    i = left_top_corner
    tile_up_length = abs(int(board_corners[(i+1)%4, 0, 0]) - int(board_corners[i%4, 0, 0]))
    tile_right_line = abs(int(board_corners[(i+2)%4, 0, 1]) - int(board_corners[(i+1)%4, 0, 1]))
    tile_down_length = abs(int(board_corners[(i+2)%4, 0, 0]) - int(board_corners[(i+3)%4, 0, 0]))
    tile_left_line = abs(int(board_corners[(i+3)%4, 0, 1]) - int(board_corners[i%4, 0, 1]))

    avg_tile_length = int((tile_up_length + tile_down_length + tile_right_line + tile_left_line)/4)

    padding = int(pad * avg_tile_length)
    cv2.rectangle(img_board, (int(board_corners[left_top_corner, 0, 0] + padding), int(board_corners[left_top_corner, 0, 1] + padding)),
                  (int(board_corners[right_bottom_corner, 0, 0] - padding), int(board_corners[right_bottom_corner, 0, 1] - padding)), (0, 0, 0), -1)

    pad_center = int(0.5 * avg_tile_length)
    board_corners[left_top_corner, 0] += pad_center
    board_corners[right_bottom_corner, 0] -= pad_center
    x_center = int((board_corners[left_top_corner, 0, 0] + board_corners[right_bottom_corner, 0, 0])/2)
    y_center = int((board_corners[left_top_corner, 0, 1] + board_corners[right_bottom_corner, 0, 1])/2)

    return (x_center, y_center, avg_tile_length)


def draw_tile_centers(img, tile_centers, col=(255, 0, 127)):
    for point in tile_centers:
        cv2.circle(img, (point[0], point[1]), 10, col, -1)
    visualize_image(img)


def print_rotation(H):
    theta = - math.atan2(H[0, 1], H[0, 0]) * 180 / math.pi
    rotation = -1
    if abs(theta) < 40:
        rotation = 0
    elif 50 < theta < 130:
        rotation = 90
    elif abs(theta) > 140:
        rotation = 180
    elif -130 < theta < -50:
        rotation = 270
    print(f'Rotation: {rotation} degrees')
    return rotation
