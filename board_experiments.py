from functions import *
import pickle
import math
import random
import time


def point_distance(p1, p2):
    return math.sqrt(pow(abs(p1[0] - p2[0]), 2) + pow(abs(p1[1] - p2[1]), 2))


with open("tile_centers_and_averages_piotrowice1.txt", "rb") as fp:   # Unpickling
    tile_centers_and_avg = pickle.load(fp)

tile_centers_and_avg.sort(key=lambda p: p[1])

diff_by_y = [0] + [abs(p1[1] - p2[1]) for p1, p2 in zip(tile_centers_and_avg[:-1], tile_centers_and_avg[1:])]
avg_tile_len = int(sum([p[2] for p in tile_centers_and_avg])/len(tile_centers_and_avg))

grouping_y_indices = [i for i, p in enumerate(diff_by_y) if p > int(0.5 * avg_tile_len)] + [len(tile_centers_and_avg)]
matrix_levels = [[] for i in range(len(grouping_y_indices))]

# Forming 2D matrix with tiles on good levels
j = 0
for i in range(len(diff_by_y)):
    if i == grouping_y_indices[j]:
        j = j+1
    matrix_levels[j].append(tile_centers_and_avg[i])
    matrix_levels[j].sort(key=lambda p: p[0])

avg_level_tile_len = []
for i in range(len(matrix_levels)):
    avg_level_tile_len.append(int(sum([p[2] for p in matrix_levels[i]])/len(matrix_levels[i])))
    for j in range(1, len(matrix_levels[i])):
        if matrix_levels[i][j][0] - matrix_levels[i][j-1][0] > int(1.3 * avg_level_tile_len[i]):
            matrix_levels[i].insert(j, (matrix_levels[i][j-1][0] + avg_level_tile_len[i],  matrix_levels[i][j][1], 0))
min_avg_tile_len = avg_level_tile_len[0]

img_board = cv2.imread('tiles/board_6.jpg', cv2.IMREAD_GRAYSCALE)
img = cv2.cvtColor(img_board, cv2.COLOR_GRAY2RGB)
draw_tile_centers(img, tile_centers_and_avg)

flatten_matrix = [item for sublist in matrix_levels for item in sublist]
differences = list(set(flatten_matrix) - set(tile_centers_and_avg))  # + [(int(img_board.shape[1]/2), int(img_board.shape[0]/2))]
draw_tile_centers(img, differences, col=(51, 255, 51))


# Forming real matrix with tiles in good columns
center_tile = min(flatten_matrix, key=lambda x: point_distance(x, (int(img_board.shape[1]/2), int(img_board.shape[0]/2))))
center_tile_xrange = (center_tile[0] - int(0.4 * min_avg_tile_len), center_tile[0] + int(0.4 * min_avg_tile_len))

center_vertical_tiles = [[] for i in range(len(matrix_levels))]

# Detect center vertical line of tiles
for i in range(len(matrix_levels)):
    for j in range(len(matrix_levels[i])):
        if center_vertical_tiles[i] == []:
            if center_tile_xrange[0] <= matrix_levels[i][j][0] and matrix_levels[i][j][0] <= center_tile_xrange[1]:
                center_vertical_tiles[i].append(matrix_levels[i][j])
    if center_vertical_tiles[i] == []:
        center_vertical_tiles[i].append((center_tile[0], matrix_levels[i][0][1], 0))

draw_tile_centers(img, [item for sublist in center_vertical_tiles for item in sublist], col=(156, 0, 203))
draw_tile_centers(img, [center_tile], col=(54, 0, 123))

max_tiles_in_row = (int((img_board.shape[1] - (center_vertical_tiles[0][0][0] + int(0.5 * min_avg_tile_len))) / min_avg_tile_len),
                    int((center_vertical_tiles[0][0][0] - int(0.5 * min_avg_tile_len)) / min_avg_tile_len))
center_index = max_tiles_in_row[0]

matrix = [[None for _ in range(max_tiles_in_row[0] + max_tiles_in_row[1] + 1)] for _ in range(len(center_vertical_tiles))]

# Add all of tiles in proper places
for i in range(len(center_vertical_tiles)):
    matrix[i][center_index] = center_vertical_tiles[i][0]

for i in range(len(matrix_levels)):
    try:
        i_center_tile_index = matrix_levels[i].index(center_vertical_tiles[i][0])
    except ValueError:
        i_center_tile_index = -1

    if i_center_tile_index >= 0:
        for j in range(len(matrix_levels[i])):
            if i_center_tile_index > j:
                matrix[i][center_index - (i_center_tile_index - j)] = matrix_levels[i][j]
            if i_center_tile_index < j:
                matrix[i][center_index + j - i_center_tile_index] = matrix_levels[i][j]
    else:  # if center tile in this row has no neighbours
        found, avg_tile_len = False, avg_level_tile_len[i]
        if matrix_levels[i][0][0] > center_vertical_tiles[i][0][0]:  # tiles are on right side
            n = 0
            while not found:
                n = n+1
                tile_center = center_vertical_tiles[i][0][0] + avg_tile_len * n
                if tile_center - int(0.3 * avg_tile_len) <= matrix_levels[i][0][0] <= tile_center + int(0.3 * avg_tile_len):
                    found = True
            for j in range(len(matrix_levels[i])):
                matrix[i][center_index + n + j] = matrix_levels[i][j]
        else:  # tiles are on left side
            n = max_tiles_in_row[0]
            while not found:
                n = n - 1
                tile_center = center_vertical_tiles[i][0][0] - avg_tile_len * n
                if tile_center - int(0.3 * avg_tile_len) <= matrix_levels[i][len(matrix_levels[i])-1][0] <= tile_center + int(0.3 * avg_tile_len):
                    found = True
            for j in range(len(matrix_levels[i])):
                matrix[i][center_index - (n + j)] = matrix_levels[i][len(matrix_levels[i]) - j - 1]




matrix.insert(0, [None for _ in range(len(matrix[0]))])
matrix.append([None for _ in range(len(matrix[0]))])



print(matrix)


# Draw the board in pyplot :)

# Make a 9x9 grid...
nrows, ncols = len(matrix), len(matrix[0])
image = np.zeros(nrows * ncols)
image = image.reshape((nrows, ncols))

for i in range(image.shape[0]):
    for j in range(image.shape[1]):
        if matrix[i][j] is not None and matrix[i][j][2] > 0:
            image[i][j] = random.randint(100, 256)

row_labels = range(1, nrows+1)
col_labels = range(1, ncols+1)
plt.matshow(image)
plt.xticks(range(ncols), col_labels)
plt.yticks(range(nrows), row_labels)
plt.show()













print(tile_centers_and_avg)



