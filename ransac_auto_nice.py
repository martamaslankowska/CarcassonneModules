from functions import *
import glob
import pickle
import math


''' Load all images '''
tile_names = glob.glob('tiles/tile*.jpg')
# tile_names = ['tiles/tile (3).jpg']
scene_name = 'board test'

images = [cv2.imread(x, cv2.IMREAD_GRAYSCALE) for x in tile_names]
img_board = cv2.imread('tiles/' + scene_name + '.jpg', cv2.IMREAD_GRAYSCALE)
img_board_color = cv2.cvtColor(img_board, cv2.COLOR_GRAY2RGB)

detector = cv2.xfeatures2d.SURF_create(hessianThreshold=400)
# detector = cv2.AKAZE_create()
# detector = cv2.BRISK_create(thresh=10, octaves=1)
# extractor = cv2.BRISK_create()

# ''' Count keypoints for board - useless if we want to cover tiles '''
# keypts_board, des_board, detector = find_board_keypoints(img_board)

''' Prepare to count tiles and their location in matrix '''
tile_centers_and_avg = []

''' Count keypoints and matches for all tiles'''
for c, img_tile in enumerate(images):
    keypts_board, des_board = find_board_keypoints(img_board, detector)
    keypts_tile, des_tile = find_tile_keypoints(img_tile, detector)
    # keypts_board, des_board = find_board_keypoints_BRISK(img_board, detector, extractor)
    # keypts_tile, des_tile = find_tile_keypoints_BRISK(img_tile, detector, extractor)
    matches = match_features_SURF(des_tile, des_board)
    good_matches = find_good_matches(matches, ratio=0.8)
    H_tile2board, H_mask = find_homography_tile2board(good_matches, keypts_tile, keypts_board)
    # good_matches = filter_good_matches_BRISK(H_mask, good_matches)

    if H_tile2board is not None and int(0.07 * len(keypts_tile)) <= len(good_matches):
        tile_corners, board_corners = find_corners(img_tile, H_tile2board)

        img_matches = draw_good_matches(img_tile, img_board, keypts_tile, keypts_board, good_matches)
        draw_tile_on_board_corners(img_matches, img_tile, board_corners)
        draw_all_matches_on_board(img_board_color, board_corners, c)

        print_rotation(H_tile2board)

        tile_centers_and_avg.append(cover_found_tiles(img_board, board_corners))
    else:
        print('No success...')

visualize_image(img_board_color)

img = cv2.cvtColor(img_board, cv2.COLOR_GRAY2RGB)
draw_tile_centers(img, tile_centers_and_avg)


# with open("tile_centers_and_averages_piotrowice1.txt", "wb") as fp:
#     pickle.dump(tile_centers_and_avg, fp)
